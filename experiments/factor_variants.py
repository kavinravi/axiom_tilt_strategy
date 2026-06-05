"""Factor screen variants — find best return-vs-Sharpe tradeoff.

Tries several composite-weighting recipes (still using PIT z-scores per Friday):
  v1     : equal (value 25 / quality 25 / momentum 25 / lowvol 25)   [done already]
  v2     : 3-factor — drop lowvol (value 33 / quality 33 / momentum 34)
  v3     : momentum-heavy (value 20 / quality 20 / momentum 50 / lowvol 10)
  v4     : quality-heavy (value 20 / quality 50 / momentum 20 / lowvol 10)
  v5     : 5-factor + size (value 20 / quality 20 / momentum 20 / lowvol 20 / size 20)
  v6     : top-50 (broader net) — v1 weights, K=50 (then top-30 from final z by mcap)
  v7     : "factor & ranker" — combined w/ LightGBM ranker scores (50/50 z-blend)

Each variant rebuilds scoreboards for all 17 walks, runs mcap-weighted backtest
with 10% cap, reports Sharpe vs SPY across 2009-2025 / 2010-2024 / 2010-2025.

usage: python experiments/factor_variants.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import processed_dir, repo_root
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.ranker import friday_only
from src.utils.rl_env import project_to_simplex

log = get_logger(__name__)

REPO_ROOT = repo_root()
PANEL_DIR = processed_dir() / "panel"
TRAIN_PANEL_DIR = processed_dir() / "training_panel"
RANKER_SB = REPO_ROOT / "artifacts" / "rl"
SPY_PATH = REPO_ROOT / "artifacts" / "benchmarks" / "spy_daily.parquet"
OUT_BASE = REPO_ROOT / "artifacts"
TOP_K = 30
MAX_WEIGHT = 0.10
MOM_LOOKBACK = 252
MOM_SKIP = 21
VOL_LOOKBACK = 252
EPS = 1e-8


def zscore(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    mu = s.mean()
    sd = s.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return s * 0
    return (s - mu) / sd


def load_panel_years(years, cols):
    frames = []
    for y in years:
        for p in sorted((PANEL_DIR / f"year={y}").glob("*.parquet")):
            df = pd.read_parquet(p, columns=cols)
            df["date"] = pd.to_datetime(df["date"])
            df["permno"] = df["permno"].astype("int64")
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_train_panel_year(year):
    files = sorted((TRAIN_PANEL_DIR / f"year={year}").glob("*.parquet"))
    if not files: return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["permno"] = df["permno"].astype("int64")
    return df


def compute_daily_factors(daily):
    daily = daily.sort_values(["permno", "date"]).reset_index(drop=True)
    daily["log_ret"] = np.log1p(daily["ret"].fillna(0.0))
    grp = daily.groupby("permno", sort=False)
    daily["cum_ret_252"] = grp["log_ret"].transform(
        lambda x: x.rolling(MOM_LOOKBACK, min_periods=MOM_LOOKBACK).sum())
    daily["mom_12_1"] = grp["cum_ret_252"].transform(lambda x: np.expm1(x.shift(MOM_SKIP)))
    daily["vol_252"] = grp["ret"].transform(
        lambda x: x.rolling(VOL_LOOKBACK, min_periods=VOL_LOOKBACK).std()) * np.sqrt(252)
    return daily[["permno", "date", "mom_12_1", "vol_252"]]


def build_scored_panel(walk_id: int, test_year: int) -> pd.DataFrame:
    """Build the Friday-only panel for walk_id with all 5 raw factor signals.
    Universe filtered to S&P members on each Friday. Caller applies any composite weighting."""
    panel_years = list(range(2001, test_year + 1))
    daily = load_panel_years(panel_years, cols=[
        "permno", "date", "prc", "shrout", "ret", "pe", "netinc", "equity",
        "marketcap", "in_universe"])
    fac = compute_daily_factors(daily)
    daily = daily.merge(fac, on=["permno", "date"], how="left")
    fri_panel = pd.concat([
        load_train_panel_year(y)[["permno", "date", "fwd_ret_5d"]]
        for y in range(2002, test_year + 1)], ignore_index=True)
    df = daily.merge(fri_panel, on=["permno", "date"], how="inner")
    df = df.dropna(subset=["fwd_ret_5d"]).copy()
    df = friday_only(df).reset_index(drop=True)
    df = df[df["in_universe"]].copy()

    df["mcap"] = df["marketcap"]
    df.loc[df["mcap"].isna(), "mcap"] = (np.abs(df.loc[df["mcap"].isna(), "prc"]) *
                                          df.loc[df["mcap"].isna(), "shrout"])

    # Raw factors
    df["ep"] = (df["netinc"] / df["mcap"]).clip(lower=0)
    df["roe"] = (df["netinc"] / df["equity"]).clip(lower=-1.0, upper=2.0)
    df.loc[df["equity"] <= 0, "roe"] = np.nan
    df["lowvol_signal"] = -df["vol_252"]
    df["size_signal"] = -np.log(np.maximum(df["mcap"], 1.0))  # smaller = higher score (small-cap premium)

    # Per-date z-scores (in-place per-group via transform-style operations, NO apply)
    df = df.copy()
    for col_in, col_out in [("ep", "z_ep"), ("roe", "z_roe"), ("mom_12_1", "z_mom"),
                             ("lowvol_signal", "z_vol"), ("size_signal", "z_size")]:
        g = df.groupby("date", sort=False)[col_in]
        df[col_out] = (df[col_in] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        df[col_out] = df[col_out].fillna(0.0)
    return df


# ---------------- composite recipes ----------------
RECIPES = {
    "v1_equal4":      {"z_ep": 0.25, "z_roe": 0.25, "z_mom": 0.25, "z_vol": 0.25},
    "v2_3factor":     {"z_ep": 0.34, "z_roe": 0.33, "z_mom": 0.33},
    "v3_mom_heavy":   {"z_ep": 0.20, "z_roe": 0.20, "z_mom": 0.50, "z_vol": 0.10},
    "v4_quality_heavy":{"z_ep": 0.20, "z_roe": 0.50, "z_mom": 0.20, "z_vol": 0.10},
    "v5_5factor":     {"z_ep": 0.20, "z_roe": 0.20, "z_mom": 0.20, "z_vol": 0.20, "z_size": 0.20},
    "v6_value_quality": {"z_ep": 0.50, "z_roe": 0.50},
    "v7_lowvol_only": {"z_vol": 1.0},
    "v8_mom_value":   {"z_ep": 0.50, "z_mom": 0.50},
    "v9_qual_mom":    {"z_roe": 0.50, "z_mom": 0.50},
}


def apply_recipe(df_scored: pd.DataFrame, recipe: dict[str, float]) -> pd.DataFrame:
    """Compute composite score = weighted avg of z-scores (NaN-safe), take top-K per Friday."""
    cols = list(recipe.keys())
    weights = np.array([recipe[c] for c in cols])
    weights = weights / weights.sum()
    mat = df_scored[cols].to_numpy(dtype=float)
    mask = ~np.isnan(mat)
    # Reweight per-row across available factors
    w_present = mask * weights[None, :]
    w_present_sum = w_present.sum(axis=1)
    safe_w = np.where(w_present_sum > 0, w_present_sum, 1.0)
    score = np.nansum(mat * weights[None, :], axis=1) / safe_w
    df_scored = df_scored.copy()
    df_scored["score"] = score
    df_scored = df_scored.dropna(subset=["score"])
    df_scored = (df_scored.sort_values(["date", "score"], ascending=[True, False])
                          .groupby("date", sort=False, group_keys=False)
                          .head(TOP_K)
                          .reset_index(drop=True))
    return df_scored[["permno", "date", "score", "fwd_ret_5d", "mcap"]]


def mcap_weighted_top30(mcaps):
    safe = np.maximum(mcaps, EPS)
    return project_to_simplex(np.log(safe), max_weight=MAX_WEIGHT)


def backtest_test_year(sb: pd.DataFrame, test_year: int):
    df = sb[(sb["date"] >= f"{test_year}-01-01") &
            (sb["date"] <= f"{test_year}-12-31")].copy().reset_index(drop=True)
    by_date = {d: g.reset_index(drop=True) for d, g in df.groupby("date")}
    out = []
    for d in sorted(by_date.keys()):
        cur = by_date[d]
        mcaps = cur["mcap"].to_numpy(dtype=np.float64)[:TOP_K]
        mcaps = np.where(np.isnan(mcaps), 0.0, mcaps)
        if mcaps.sum() <= 0:
            w = np.full(TOP_K, 1.0 / TOP_K)
        else:
            w = mcap_weighted_top30(mcaps)
        fwd = cur["fwd_ret_5d"].to_numpy(dtype=np.float64)[:TOP_K]
        fwd = np.where(np.isnan(fwd), 0.0, fwd)
        out.append({"date": d, "weekly_ret": float(np.dot(w, fwd))})
    return pd.DataFrame(out)


def metrics(rets):
    rets = np.asarray(rets, dtype=float)
    if len(rets) < 2: return {}
    cum = float(np.prod(1.0 + rets) - 1.0)
    ann = (1.0 + cum) ** (52.0 / len(rets)) - 1.0
    vol = float(np.std(rets, ddof=1) * np.sqrt(52.0))
    sh = ann / vol if vol > 0 else 0.0
    eq = np.cumprod(1.0 + rets); peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    cal = ann / abs(mdd) if mdd < 0 else 0.0
    return {"n": len(rets), "tot": cum, "ann": ann, "vol": vol, "sh": sh, "mdd": mdd, "cal": cal}


def build_spy(dates):
    spy = pd.read_parquet(SPY_PATH).reset_index()[["Date", "close"]].rename(columns={"Date": "date"})
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").set_index("date")["close"]
    closes = spy.reindex(spy.index.union(dates)).sort_index().ffill().reindex(dates)
    return closes.pct_change().fillna(0.0).to_numpy()


def main():
    configure_logging()
    # Build the master scored panel ONCE per walk (cache) — sharing across recipes is huge
    # but we need per-walk PIT scoring boundaries. So compute per-walk, then apply all recipes.
    all_walk_scored: dict[int, pd.DataFrame] = {}
    for walk_id in range(1, 18):
        t0 = time.time()
        sp = build_scored_panel(walk_id, test_year=2008 + walk_id)
        all_walk_scored[walk_id] = sp
        log.info("scored panel walk %2d: %d rows [%.1fs]", walk_id, len(sp), time.time() - t0)

    # For each recipe → for each walk → apply recipe → take test-year → backtest
    results = {}
    for recipe_name, recipe in RECIPES.items():
        rows_all = []
        for walk_id in range(1, 18):
            sp = all_walk_scored[walk_id]
            sb = apply_recipe(sp, recipe)
            wk = backtest_test_year(sb, 2008 + walk_id)
            rows_all.append(wk)
        weekly = pd.concat(rows_all, ignore_index=True).sort_values("date").reset_index(drop=True)
        results[recipe_name] = weekly
        log.info("recipe %s: %d weeks", recipe_name, len(weekly))

    # Build SPY benchmark
    dates = pd.DatetimeIndex(results[next(iter(RECIPES))]["date"])
    spy_rets = build_spy(dates)
    years = pd.DatetimeIndex(dates).year

    # Print per-window comparison
    print()
    print(f"{'recipe':<22} {'window':<12} {'wks':>5} {'ann':>8} {'vol':>8} {'sharpe':>8} {'mdd':>8} {'cal':>7}")
    print("-" * 85)
    for label, mask in [("2009-2025", np.ones(len(dates), dtype=bool)),
                        ("2010-2024", (years >= 2010) & (years <= 2024)),
                        ("2010-2025", years >= 2010)]:
        # SPY baseline
        sm = metrics(spy_rets[mask])
        print(f"{'SPY':<22} {label:<12} {sm['n']:>5} {sm['ann']:>8.2%} {sm['vol']:>8.2%} "
              f"{sm['sh']:>8.3f} {sm['mdd']:>8.2%} {sm['cal']:>7.3f}")
        for recipe_name in RECIPES:
            r = results[recipe_name].set_index("date").reindex(dates)["weekly_ret"].to_numpy()
            mm = metrics(r[mask])
            beat_marker = " ✓" if mm['sh'] > sm['sh'] else ""
            print(f"{recipe_name:<22} {label:<12} {mm['n']:>5} {mm['ann']:>8.2%} {mm['vol']:>8.2%} "
                  f"{mm['sh']:>8.3f} {mm['mdd']:>8.2%} {mm['cal']:>7.3f}{beat_marker}")
        print()

    # Persist weekly returns + summary
    OUT_BASE.mkdir(exist_ok=True, parents=True)
    for name, df in results.items():
        df.to_parquet(OUT_BASE / "backtest_factor_v1" / f"weekly_{name}.parquet",
                      compression="zstd", index=False)


if __name__ == "__main__":
    main()
