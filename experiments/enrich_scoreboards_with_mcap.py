"""Add an `mcap` column to all 17 walks' scoreboards.

mcap = |prc| * shrout from CRSP panel, joined on (permno, date). The env can
then read `cur['mcap']` for the new mcap-baseline mode.

Loads each scoreboard, joins with panel for the years it covers, writes back.
Backup of the original is saved as scoreboard.pre_mcap.parquet.

usage: python -m experiments.enrich_scoreboards_with_mcap
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import processed_dir, repo_root

REPO_ROOT = repo_root()
SCOREBOARD_ROOT = REPO_ROOT / "artifacts" / "rl"
PANEL_DIR = processed_dir() / "panel"


def load_panel_mcap_years(years: list[int]) -> pd.DataFrame:
    frames = []
    for y in years:
        for p in sorted((PANEL_DIR / f"year={y}").glob("*.parquet")):
            df = pd.read_parquet(p, columns=["permno", "date", "prc", "shrout"])
            df["date"] = pd.to_datetime(df["date"])
            df["permno"] = df["permno"].astype("int64")
            df["mcap"] = np.abs(df["prc"].astype(float)) * df["shrout"].astype(float)
            frames.append(df[["permno", "date", "mcap"]])
    if not frames:
        return pd.DataFrame(columns=["permno", "date", "mcap"])
    return pd.concat(frames, ignore_index=True)


def enrich_walk(walk_id: int) -> dict:
    sb_path = SCOREBOARD_ROOT / f"walk-{walk_id:03d}" / "scoreboard.parquet"
    if not sb_path.exists():
        return {"walk": walk_id, "status": "no_scoreboard"}

    sb = pd.read_parquet(sb_path)
    if "mcap" in sb.columns:
        return {"walk": walk_id, "status": "already_has_mcap"}

    sb["date"] = pd.to_datetime(sb["date"])
    sb["permno"] = sb["permno"].astype("int64")
    years = sorted({int(y) for y in sb["date"].dt.year.unique()})

    mcap = load_panel_mcap_years(years)
    sb_enriched = sb.merge(mcap, on=["permno", "date"], how="left")

    miss_rate = sb_enriched["mcap"].isna().mean()
    # Backup before overwriting
    backup = sb_path.with_suffix(".pre_mcap.parquet")
    if not backup.exists():
        shutil.copy2(sb_path, backup)

    # Forward-fill mcap NaNs per permno (a few delisted days may have NaN prc)
    # then fall back to a small positive sentinel so log(mcap) is defined.
    sb_enriched["mcap"] = (sb_enriched
                           .sort_values(["permno", "date"])
                           .groupby("permno")["mcap"]
                           .ffill())
    sb_enriched["mcap"] = sb_enriched["mcap"].fillna(1.0)

    sb_enriched.to_parquet(sb_path, compression="zstd", index=False)
    return {
        "walk": walk_id,
        "status": "enriched",
        "rows": len(sb_enriched),
        "nan_rate_pre_ffill": float(miss_rate),
    }


def main():
    for w in range(1, 18):
        r = enrich_walk(w)
        print(f"  walk {w:2d}: {r}")


if __name__ == "__main__":
    main()
