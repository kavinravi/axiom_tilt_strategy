"""Factor-screen v1: replace LightGBM ranker with classic 4-factor composite.

Per Friday t (PIT), compute each S&P 500 member's score across 4 factors:
  - Value     : E/P = netinc / marketcap (clipped at 0 for negative earnings).
                Higher = cheaper = better.
  - Quality   : ROE = netinc / equity (most recent quarter, PIT via panel merge_asof).
                Higher = better.
  - Momentum  : cumulative return over [t-252d, t-21d] (12-month lookback,
                skip last month to avoid short-term reversal). Higher = better.
  - Low Vol   : -1 * std(daily ret over [t-252d, t-1d]) * sqrt(252). Lower vol
                gets higher z-score.

Cross-sectional z-score each factor within the active S&P 500 universe on date t.
Composite = mean of the 4 z-scores. Top-30 by composite = the picks.

Outputs per walk-NNN to artifacts/rl_factor_v1/walk-NNN/scoreboard.parquet,
same schema as the LightGBM ranker scoreboards (drop-in replacement).

usage: python experiments/factor_screen_v1.py [--walks 1 17]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import processed_dir, repo_root
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.ranker import friday_only

log = get_logger(__name__)

REPO_ROOT = repo_root()
PANEL_DIR = processed_dir() / "panel"
TRAIN_PANEL_DIR = processed_dir() / "training_panel"
OUT_ROOT = REPO_ROOT / "artifacts" / "rl_factor_v1"
TOP_K = 30
MOM_LOOKBACK_DAYS = 252      # 12 months
MOM_SKIP_DAYS = 21           # skip most recent month
VOL_LOOKBACK_DAYS = 252      # 12 months trailing vol


def load_panel_years(years: list[int], cols=None) -> pd.DataFrame:
    frames = []
    for y in years:
        for p in sorted((PANEL_DIR / f"year={y}").glob("*.parquet")):
            df = pd.read_parquet(p, columns=cols)
            df["date"] = pd.to_datetime(df["date"])
            df["permno"] = df["permno"].astype("int64")
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_train_panel_year(year: int) -> pd.DataFrame:
    """training_panel has fwd_ret_5d (the label) — used for scoreboard rows."""
    files = sorted((TRAIN_PANEL_DIR / f"year={year}").glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["permno"] = df["permno"].astype("int64")
    return df


def compute_daily_factors(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing momentum + vol for every (permno, date) in `daily`.

    `daily` must be sorted by (permno, date) and span enough history before
    the earliest target date (we need ~252 trading days of lookback).
    """
    daily = daily.sort_values(["permno", "date"]).reset_index(drop=True)
    daily["log_ret"] = np.log1p(daily["ret"].fillna(0.0))

    # Rolling 252-day cum return, then shift 21 days (skip recent month → mom_12_1)
    grp = daily.groupby("permno", sort=False)
    daily["cum_ret_252"] = grp["log_ret"].transform(
        lambda x: x.rolling(MOM_LOOKBACK_DAYS, min_periods=MOM_LOOKBACK_DAYS).sum()
    )
    daily["mom_12_1"] = grp["cum_ret_252"].transform(
        lambda x: np.expm1(x.shift(MOM_SKIP_DAYS))
    )

    # Rolling 252-day std of daily returns → annualized vol
    daily["vol_252"] = grp["ret"].transform(
        lambda x: x.rolling(VOL_LOOKBACK_DAYS, min_periods=VOL_LOOKBACK_DAYS).std()
    ) * np.sqrt(252)

    return daily[["permno", "date", "mom_12_1", "vol_252"]]


def zscore(series: pd.Series) -> pd.Series:
    """Robust z-score: subtract median, divide by MAD-style scale."""
    s = series.astype(float)
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return s * 0
    return (s - mu) / sd


def build_friday_scoreboard(walk_id: int, test_year_end: int) -> pd.DataFrame:
    """For walk-NNN, build top-30 factor-screen scoreboard spanning
    2002 Friday through (test_year_end)-12 Friday.

    Returns dataframe with columns (permno, date, score, fwd_ret_5d, mcap).
    """
    # 1. Load daily panel for the entire window plus 2001 lookback for first momentum/vol.
    panel_years = list(range(2001, test_year_end + 1))
    log.info("walk %d: loading panel %d years (%d-%d) ...",
             walk_id, len(panel_years), panel_years[0], panel_years[-1])
    daily = load_panel_years(
        panel_years,
        cols=["permno", "date", "prc", "shrout", "ret", "pe",
              "netinc", "equity", "marketcap", "in_universe"],
    )
    log.info("  daily rows: %d", len(daily))

    # 2. Compute rolling momentum + vol per permno.
    log.info("  computing rolling momentum + vol ...")
    factors_daily = compute_daily_factors(daily)
    daily = daily.merge(factors_daily, on=["permno", "date"], how="left")

    # 3. Restrict to Fridays inside walk's window [2002-01-01, test_year_end-12-31].
    fri_panel = pd.concat([
        load_train_panel_year(y)[["permno", "date", "fwd_ret_5d"]]
        for y in range(2002, test_year_end + 1)
    ], ignore_index=True)

    daily_fri = daily.merge(fri_panel, on=["permno", "date"], how="inner")
    daily_fri = daily_fri.dropna(subset=["fwd_ret_5d"]).copy()
    # CRITICAL: training_panel has DAILY rows, not Friday-only. Apply friday filter explicitly.
    daily_fri = friday_only(daily_fri).reset_index(drop=True)
    log.info("  Friday rows (in walk window): %d", len(daily_fri))

    # 4. Restrict to S&P 500 members on each date.
    daily_fri = daily_fri[daily_fri["in_universe"]].copy()
    log.info("  S&P-member Friday rows: %d", len(daily_fri))

    # 5. Per-Friday factor construction.
    daily_fri["mcap"] = daily_fri["marketcap"]  # use Sharadar marketcap (more reliable than prc*shrout when present)
    daily_fri.loc[daily_fri["mcap"].isna(), "mcap"] = (
        np.abs(daily_fri.loc[daily_fri["mcap"].isna(), "prc"])
        * daily_fri.loc[daily_fri["mcap"].isna(), "shrout"]
    )

    # Value: E/P = netinc / mcap, clip negative earnings to 0 (no value premium for losing money)
    daily_fri["ep"] = (daily_fri["netinc"] / daily_fri["mcap"]).clip(lower=0)

    # Quality: ROE = netinc / equity. Cap at sensible bounds.
    daily_fri["roe"] = (daily_fri["netinc"] / daily_fri["equity"]).clip(lower=-1.0, upper=2.0)
    # Drop rows where equity is non-positive (book value can be negative; meaningless ROE)
    daily_fri.loc[daily_fri["equity"] <= 0, "roe"] = np.nan

    # Low Vol: lower vol → higher score, so negate
    daily_fri["lowvol_signal"] = -daily_fri["vol_252"]

    # Momentum: already mom_12_1 (higher = better)

    # 6. Per-date z-scores of each factor, then composite.
    def _per_date_z(g):
        g = g.copy()
        g["z_ep"] = zscore(g["ep"])
        g["z_roe"] = zscore(g["roe"])
        g["z_mom"] = zscore(g["mom_12_1"])
        g["z_vol"] = zscore(g["lowvol_signal"])
        # Composite = mean of available factor z-scores (skip NaN factors per stock)
        g["score"] = g[["z_ep", "z_roe", "z_mom", "z_vol"]].mean(axis=1, skipna=True)
        return g

    log.info("  computing per-Friday z-scores + composite ...")
    scored = daily_fri.groupby("date", group_keys=False).apply(_per_date_z, include_groups=False)
    # The groupby(.apply) above strips the 'date' column; add it back from the index.
    scored = scored.reset_index(drop=False) if "date" not in scored.columns else scored
    if "date" not in scored.columns:
        # Fall back: zip with the original Friday list — the apply preserves row order within groups
        scored = daily_fri.assign(score=scored["score"].values,
                                  z_ep=scored["z_ep"].values,
                                  z_roe=scored["z_roe"].values,
                                  z_mom=scored["z_mom"].values,
                                  z_vol=scored["z_vol"].values)

    # 7. Drop rows where composite is NaN (stock has no factor coverage at all)
    scored = scored.dropna(subset=["score"]).copy()

    # 8. Per-Friday top-K by composite
    scored = (scored.sort_values(["date", "score"], ascending=[True, False])
                    .groupby("date", sort=False, group_keys=False)
                    .head(TOP_K)
                    .reset_index(drop=True))
    log.info("  top-%d scoreboard rows: %d (over %d Fridays)",
             TOP_K, len(scored), scored["date"].nunique())

    # 9. Output schema matches rl_env.build_scoreboard_from_scored_panel: permno, date, score, fwd_ret_5d, mcap
    out = scored[["permno", "date", "score", "fwd_ret_5d", "mcap",
                  "z_ep", "z_roe", "z_mom", "z_vol"]].copy()
    return out


def main():
    configure_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk-start", type=int, default=1)
    parser.add_argument("--walk-end", type=int, default=17)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for walk_id in range(args.walk_start, args.walk_end + 1):
        t0 = time.time()
        test_year = 2008 + walk_id
        out_dir = OUT_ROOT / f"walk-{walk_id:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "scoreboard.parquet"
        if out_path.exists():
            log.info("walk %d: scoreboard exists, skipping", walk_id)
            continue
        sb = build_friday_scoreboard(walk_id, test_year_end=test_year)
        sb.to_parquet(out_path, compression="zstd", index=False)
        elapsed = time.time() - t0
        log.info("walk %d: wrote %d rows -> %s [%.0fs]",
                 walk_id, len(sb), out_path.relative_to(REPO_ROOT), elapsed)


if __name__ == "__main__":
    main()
