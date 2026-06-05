"""Fit PCA for walk-17 (train 2002-2023) at the locked dim from walk-1.

Standalone replacement for the full notebook 04 loop — only does walk-17.
Mirrors notebook 04's _fit_and_persist logic. Walk-17 is the first walk that
tests 2025, which the parent project hadn't built yet.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.utils.io import processed_dir, repo_root
from src.utils.pca import assemble_training_matrix
from src.utils.logging_utils import configure_logging, get_logger

log = get_logger(__name__)


def main():
    configure_logging()
    WALK_ID = 17
    TRAIN_START = "2002-01-01"
    TRAIN_END = "2023-12-31"
    LOCKED_N_PCA = 79  # locked at walk-1 (95% var + 1 safety)

    artifacts_dir = repo_root() / "artifacts" / "pca-text"
    walk_dir = artifacts_dir / f"walk-{WALK_ID:03d}"
    walk_dir.mkdir(parents=True, exist_ok=True)

    if (walk_dir / "pca.joblib").exists():
        log.info("walk-17 pca.joblib already exists at %s -- skipping", walk_dir)
        return

    embed_dir = processed_dir() / "finbert_stockday_embed"
    universe_ids = pd.read_parquet(processed_dir() / "universe_ids.parquet")

    log.info("Loading FinBERT stockday embeds %s -> %s ...", TRAIN_START, TRAIN_END)
    X, meta = assemble_training_matrix(
        embed_dir=embed_dir,
        universe_ids=universe_ids,
        start=TRAIN_START,
        end=TRAIN_END,
    )
    log.info("  X shape: %s | meta rows: %d", X.shape, len(meta))

    if len(meta) < 100_000:
        raise AssertionError(f"walk-17 has only {len(meta)} samples (< 100K threshold)")

    log.info("Fitting PCA(n=%d, svd_solver='full') ...", LOCKED_N_PCA)
    pca = PCA(n_components=LOCKED_N_PCA, svd_solver="full").fit(X)
    var_captured = float(pca.explained_variance_ratio_.sum())
    log.info("  variance captured: %.6f", var_captured)

    joblib.dump(pca, walk_dir / "pca.joblib")
    summary = {
        "walk_id": WALK_ID,
        "window_start": TRAIN_START,
        "window_end": TRAIN_END,
        "n_train_samples": int(X.shape[0]),
        "n_meta_rows": int(len(meta)),
        "hidden_dim": int(X.shape[1]),
        "locked_n_pca": LOCKED_N_PCA,
        "target_variance": 0.95,
        "variance_captured": var_captured,
        "use_synthetic": False,
    }
    (walk_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("walk-17 done -> %s", walk_dir)


if __name__ == "__main__":
    main()
