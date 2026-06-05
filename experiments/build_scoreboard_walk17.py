"""Build the full (train + val + test) scoreboard for walk-17.

Walk-17: train 2002-2023, val 2024, test 2025. Outputs:
  artifacts/rl/walk-017/scoreboard.parquet

Mirrors notebook 07's scoreboard build (loads ranker + walk-PCA, scores Friday
panel rows joined with PCA-projected text embeds, takes top-30 per Friday).
Single-shot — covers all three years (no separate extend_scoreboards step needed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.io import processed_dir, repo_root
from src.utils.ranker import friday_only, load_walk_pca, project_text_to_pca
from src.utils.rl_env import build_scoreboard_from_scored_panel
from src.utils.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

WALK_ID = 17
START = "2002-01-01"
END = "2025-12-31"
TOP_K = 30


def _load_years(dir_: Path, start: str, end: str, cols=None) -> pd.DataFrame:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    for y in range(s.year, e.year + 1):
        for p in sorted((dir_ / f"year={y}").glob("*.parquet")):
            df = pd.read_parquet(p, columns=cols)
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= s) & (df["date"] <= e)]
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    configure_logging()
    repo = repo_root()
    panel_dir = processed_dir() / "training_panel"
    embed_dir = processed_dir() / "finbert_stockday_embed"
    ranker_path = repo / "artifacts" / "ranker" / f"walk-{WALK_ID:03d}" / "model.joblib"
    out_path = repo / "artifacts" / "rl" / f"walk-{WALK_ID:03d}" / "scoreboard.parquet"

    if not ranker_path.exists():
        sys.exit(f"missing ranker: {ranker_path}")
    if out_path.exists():
        log.info("walk-17 scoreboard already exists at %s -- skipping", out_path)
        return

    log.info("Loading walk-17 ranker bundle ...")
    bundle = joblib.load(ranker_path)
    model = bundle["model"]
    features = bundle["feature_names"]
    log.info("  %d ranker features", len(features))

    log.info("Loading walk-17 PCA ...")
    pca, n_pca = load_walk_pca(WALK_ID)
    log.info("  n_pca = %d", n_pca)

    log.info("Loading panel %s -> %s ...", START, END)
    panel = _load_years(panel_dir, START, END)
    log.info("  panel rows: %d", len(panel))

    log.info("Loading FinBERT stockday embeds %s -> %s ...", START, END)
    embed = _load_years(embed_dir, START, END, cols=["permno", "date", "vec"])
    log.info("  embed rows: %d", len(embed))

    log.info("Projecting embeds to PCA ...")
    embed_pca = project_text_to_pca(embed, pca)

    log.info("Friday-only filter + merge ...")
    fri = friday_only(panel).merge(embed_pca, on=["permno", "date"], how="inner")
    fri = fri.dropna(subset=["fwd_ret_5d"]).copy()
    log.info("  scored rows: %d | unique Fridays: %d", len(fri), fri["date"].nunique())

    log.info("Predicting with ranker ...")
    X = pd.DataFrame({c: fri[c] if c in fri.columns else np.nan for c in features})
    fri["score"] = model.predict(X)

    sb = build_scoreboard_from_scored_panel(fri, top_k=TOP_K)
    log.info("Building scoreboard: %d rows over %d Fridays", len(sb), sb["date"].nunique())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sb.to_parquet(out_path, compression="zstd", index=False)
    log.info("wrote -> %s", out_path)

    by_year = sb["date"].dt.year.value_counts().sort_index()
    log.info("rows per year:\n%s", by_year.to_string())


if __name__ == "__main__":
    main()
