"""Build sp_fcfa scoreboards per walk, with all features the RL env needs.

sp_fcfa = the winning factor pair from the 36-combo V × Q sweep:
  - Value:   S/P = revenue / marketcap (clipped at 0)
  - Quality: FCFA = fcf / assets (clipped to [-1, 2], NaN if assets <= 0)
  - composite = 0.5 * z(S/P) + 0.5 * z(FCFA)
  - top-30 per Friday, S&P 500 PIT universe

The scoreboard includes the env's required features:
  - permno, date, score, fwd_ret_5d, mcap   (selection + label + weights)
  - macro_vixcls, macro_dgs10, macro_t10y2y (env obs macro)
  - payoutratio, ncfdiv, bidlo, sgna, retearn (env obs TOP_FEATURES)

Output: artifacts/rl_factor_spfcfa/walk-NNN/scoreboard.parquet
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import processed_dir, repo_root
from src.utils.ranker import friday_only
from src.utils.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

REPO_ROOT = repo_root()
PANEL_DIR = processed_dir() / "panel"
TRAIN_PANEL_DIR = processed_dir() / "training_panel"
OUT_ROOT = REPO_ROOT / "artifacts" / "rl_factor_spfcfa"
TOP_K = 30


def load_panel(years, cols):
    frames = []
    for y in years:
        for p in sorted((PANEL_DIR / f"year={y}").glob("*.parquet")):
            df = pd.read_parquet(p, columns=cols)
            df["date"] = pd.to_datetime(df["date"])
            df["permno"] = df["permno"].astype("int64")
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_train(years, cols):
    frames = []
    for y in years:
        for p in sorted((TRAIN_PANEL_DIR / f"year={y}").glob("*.parquet")):
            df = pd.read_parquet(p, columns=cols)
            df["date"] = pd.to_datetime(df["date"])
            df["permno"] = df["permno"].astype("int64")
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_scoreboard(walk_id: int, test_year_end: int) -> pd.DataFrame:
    panel_years = list(range(2001, test_year_end + 1))
    daily = load_panel(panel_years, cols=[
        "permno", "date", "prc", "shrout", "marketcap", "in_universe",
        "revenue", "fcf", "assets",
        # TOP_FEATURES for RL env obs:
        "payoutratio", "ncfdiv", "bidlo", "sgna", "retearn",
    ])
    # macro columns live in training_panel
    train_cols = ["permno", "date", "fwd_ret_5d",
                  "macro_vixcls", "macro_dgs10", "macro_t10y2y"]
    fri = load_train(range(2002, test_year_end + 1), cols=train_cols)

    df = daily.merge(fri, on=["permno", "date"], how="inner")
    df = df.dropna(subset=["fwd_ret_5d"]).copy()
    df = friday_only(df).reset_index(drop=True)
    df = df[df["in_universe"]].copy()

    # mcap fallback
    df["mcap"] = df["marketcap"]
    df.loc[df["mcap"].isna(), "mcap"] = (np.abs(df.loc[df["mcap"].isna(), "prc"]) *
                                          df.loc[df["mcap"].isna(), "shrout"])

    # Compute factor signals
    df["sp"] = (df["revenue"] / df["mcap"]).clip(lower=0)
    df["fcfa"] = (df["fcf"] / df["assets"]).clip(lower=-1.0, upper=2.0)
    df.loc[df["assets"] <= 0, "fcfa"] = np.nan

    # Per-Friday z-scores (transform style, no apply)
    for col_in, col_out in [("sp", "z_sp"), ("fcfa", "z_fcfa")]:
        g = df.groupby("date", sort=False)[col_in]
        df[col_out] = (df[col_in] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        df[col_out] = df[col_out].fillna(0.0)
    df["score"] = 0.5 * df["z_sp"] + 0.5 * df["z_fcfa"]

    # Top-K per Friday
    sb = (df.sort_values(["date", "score"], ascending=[True, False])
            .groupby("date", sort=False, group_keys=False)
            .head(TOP_K)
            .reset_index(drop=True))

    # Output schema for RL env (drop-in compatible)
    cols_out = ["permno", "date", "score", "fwd_ret_5d", "mcap",
                "macro_vixcls", "macro_dgs10", "macro_t10y2y",
                "payoutratio", "ncfdiv", "bidlo", "sgna", "retearn"]
    return sb[cols_out].copy()


def main():
    configure_logging()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for walk_id in range(1, 18):
        t0 = time.time()
        out_dir = OUT_ROOT / f"walk-{walk_id:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "scoreboard.parquet"
        if out_path.exists():
            log.info("walk %d: exists, skipping", walk_id)
            continue
        sb = build_scoreboard(walk_id, test_year_end=2008 + walk_id)
        sb.to_parquet(out_path, compression="zstd", index=False)
        log.info("walk %2d: wrote %d rows over %d Fridays [%.1fs]",
                 walk_id, len(sb), sb["date"].nunique(), time.time() - t0)


if __name__ == "__main__":
    main()
