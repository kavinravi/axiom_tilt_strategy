"""Persist v6_value_quality scoreboards per-walk to artifacts/rl_factor_v6/walk-NNN/.

These are drop-in replacements for artifacts/rl/walk-NNN/scoreboard.parquet
(same schema: permno, date, score, fwd_ret_5d, mcap) so any downstream RL/backtest
that reads scoreboard.parquet picks up the v6 factor selections instead of the
LightGBM ranker.

usage: python experiments/build_v6_scoreboards.py
"""
from __future__ import annotations

import json
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
OUT_ROOT = REPO_ROOT / "artifacts" / "rl_factor_v6"
TOP_K = 30


def load_panel_years(years, cols):
    frames = []
    for y in years:
        for p in sorted((PANEL_DIR / f"year={y}").glob("*.parquet")):
            df = pd.read_parquet(p, columns=cols)
            df["date"] = pd.to_datetime(df["date"])
            df["permno"] = df["permno"].astype("int64")
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_v6_scoreboard(walk_id: int, test_year_end: int) -> pd.DataFrame:
    """Build top-30 v6 scoreboard for walk_id, spanning 2002 → test_year_end."""
    panel_years = list(range(2001, test_year_end + 1))
    daily = load_panel_years(panel_years, cols=[
        "permno", "date", "prc", "shrout", "marketcap", "in_universe",
        "netinc", "equity"])

    # Friday panel rows (PIT panel filtered to Fridays with valid fwd_ret_5d)
    fri_panel = pd.concat([
        pd.concat([pd.read_parquet(f) for f in sorted((TRAIN_PANEL_DIR / f"year={y}").glob("*.parquet"))],
                  ignore_index=True)[["permno", "date", "fwd_ret_5d"]]
        for y in range(2002, test_year_end + 1)
    ], ignore_index=True)
    fri_panel["date"] = pd.to_datetime(fri_panel["date"])
    fri_panel["permno"] = fri_panel["permno"].astype("int64")

    df = daily.merge(fri_panel, on=["permno", "date"], how="inner")
    df = df.dropna(subset=["fwd_ret_5d"]).copy()
    df = friday_only(df).reset_index(drop=True)
    df = df[df["in_universe"]].copy()

    df["mcap"] = df["marketcap"]
    df.loc[df["mcap"].isna(), "mcap"] = (np.abs(df.loc[df["mcap"].isna(), "prc"]) *
                                          df.loc[df["mcap"].isna(), "shrout"])
    df["ep"] = (df["netinc"] / df["mcap"]).clip(lower=0)
    df["roe"] = (df["netinc"] / df["equity"]).clip(lower=-1.0, upper=2.0)
    df.loc[df["equity"] <= 0, "roe"] = np.nan

    # Per-date z-scores
    g_ep = df.groupby("date", sort=False)["ep"]
    df["z_ep"] = (df["ep"] - g_ep.transform("mean")) / g_ep.transform("std").replace(0, np.nan)
    df["z_ep"] = df["z_ep"].fillna(0.0)
    g_roe = df.groupby("date", sort=False)["roe"]
    df["z_roe"] = (df["roe"] - g_roe.transform("mean")) / g_roe.transform("std").replace(0, np.nan)
    df["z_roe"] = df["z_roe"].fillna(0.0)

    # Composite 50/50
    df["score"] = 0.5 * df["z_ep"] + 0.5 * df["z_roe"]

    # Top-K per Friday
    sb = (df.sort_values(["date", "score"], ascending=[True, False])
            .groupby("date", sort=False, group_keys=False)
            .head(TOP_K)
            .reset_index(drop=True))

    # Drop-in schema for downstream RL: permno, date, score, fwd_ret_5d + standard panel cols
    out_cols = ["permno", "date", "score", "fwd_ret_5d", "mcap"]
    # Also include macro + a few panel features so the RL env's obs builder can find them
    # (artifacts/rl/walk-NNN/scoreboard had macro_vixcls/dgs10/t10y2y + payoutratio/ncfdiv/bidlo/sgna/retearn).
    # We don't have all of those here; fill with NaN — env will tolerate via nan_to_num.
    for c in ["macro_vixcls", "macro_dgs10", "macro_t10y2y",
              "payoutratio", "ncfdiv", "bidlo", "sgna", "retearn"]:
        if c in df.columns:
            sb[c] = df.loc[sb.index, c].values  # not ideal — better to join properly
        else:
            sb[c] = np.nan
    return sb[out_cols + ["macro_vixcls", "macro_dgs10", "macro_t10y2y",
                           "payoutratio", "ncfdiv", "bidlo", "sgna", "retearn"]]


def main():
    configure_logging()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    # We also need the macro + panel columns for the RL env's observation builder.
    # Rather than re-extracting them here, do a defensive build (RL env tolerates NaN via nan_to_num).
    for walk_id in range(1, 18):
        t0 = time.time()
        out_dir = OUT_ROOT / f"walk-{walk_id:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "scoreboard.parquet"
        if out_path.exists():
            log.info("walk %d: exists, skipping", walk_id)
            continue
        sb = build_v6_scoreboard(walk_id, test_year_end=2008 + walk_id)
        sb.to_parquet(out_path, compression="zstd", index=False)
        log.info("walk %2d: wrote %d rows over %d Fridays [%.1fs]",
                 walk_id, len(sb), sb["date"].nunique(), time.time() - t0)


if __name__ == "__main__":
    main()
