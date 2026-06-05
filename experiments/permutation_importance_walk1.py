"""Per-feature permutation importance on walk-1's val 2008 NDCG@30.

For the walk-1 LightGBM ranker (already trained), for each of the ~190 features:
  1. Shuffle that single feature's values in val data (random permutation, fixed seed).
  2. Re-predict val scores.
  3. Compute val NDCG@30.
  4. Permutation importance (PI) = baseline_NDCG - shuffled_NDCG.

PI > 0: shuffling broke something → feature helps prediction (KEEP).
PI = 0: shuffling didn't change anything → feature is dead (drop, no cost).
PI < 0: shuffling improved prediction → feature was *hurting* (drop, biggest benefit).

Outputs:
  artifacts/ranker/walk-001/permutation_importance.csv
  reports/permutation_importance_walk1.md (summary + distribution)

usage: python experiments/permutation_importance_walk1.py [--n-repeats N]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.io import processed_dir, repo_root
from src.utils.ranker import (
    assemble_walk_features,
    drop_zero_info_columns,
    friday_only,
    load_walk_pca,
    project_text_to_pca,
)

WALK_ID = 1
VAL_START = "2008-01-01"
VAL_END = "2008-12-31"
TRAIN_START = "2002-01-01"
TRAIN_END = "2007-12-31"
TOP_K = 30

REPO_ROOT = repo_root()
PANEL_DIR = processed_dir() / "training_panel"
EMBED_DIR = processed_dir() / "finbert_stockday_embed"
RANKER_DIR = REPO_ROOT / "artifacts" / "ranker" / f"walk-{WALK_ID:03d}"


def load_years(dir_: Path, start: str, end: str, cols=None) -> pd.DataFrame:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    for y in range(s.year, e.year + 1):
        for p in sorted((dir_ / f"year={y}").glob("*.parquet")):
            df = pd.read_parquet(p, columns=cols)
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= s) & (df["date"] <= e)]
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ndcg_at_k(scores: np.ndarray, labels: np.ndarray, groups: list[int], k: int = TOP_K) -> float:
    """Cross-sectional NDCG@K averaged across groups (= dates).
    labels = excess return (continuous gain); idcg uses sorted-by-true labels."""
    ndcg_sum = 0.0
    n_groups = 0
    idx = 0
    for g in groups:
        if g < 2:
            idx += g
            continue
        s = scores[idx:idx + g]
        l = labels[idx:idx + g]
        # NDCG with continuous gains: gain = max(0, l) to handle negative excess returns
        gains = np.maximum(l, 0.0)
        # DCG = sum gains by predicted-score order (top-k positions)
        order = np.argsort(-s)
        kk = min(k, g)
        positions = np.arange(1, kk + 1)
        discounts = 1.0 / np.log2(positions + 1)
        dcg = float(np.sum(gains[order[:kk]] * discounts))
        # IDCG = same but sorted by true gains
        ideal = np.argsort(-gains)
        idcg = float(np.sum(gains[ideal[:kk]] * discounts))
        if idcg > 0:
            ndcg_sum += dcg / idcg
            n_groups += 1
        idx += g
    return ndcg_sum / max(n_groups, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-repeats", type=int, default=3,
                        help="Number of shuffle repetitions per feature (for noise estimate)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load trained ranker bundle
    bundle = joblib.load(RANKER_DIR / "model.joblib")
    model = bundle["model"]
    train_features = bundle["feature_names"]
    print(f"Loaded ranker: {len(train_features)} features")

    # Load walk-1 PCA + assemble val features (same logic as notebook 06)
    pca, n_pca = load_walk_pca(WALK_ID)
    print(f"Loaded walk-1 PCA: n_components={n_pca}")

    val_panel = load_years(PANEL_DIR, VAL_START, VAL_END)
    val_embed = load_years(EMBED_DIR, VAL_START, VAL_END, cols=["permno", "date", "vec"])
    val_embed_pca = project_text_to_pca(val_embed, pca)
    Xvl, yvl, gvl, mvl = assemble_walk_features(val_panel, val_embed_pca)

    # Drop zero-info columns (notebook 06 does this on train; replicate so val matches)
    tr_panel = load_years(PANEL_DIR, TRAIN_START, TRAIN_END)
    tr_embed = load_years(EMBED_DIR, TRAIN_START, TRAIN_END, cols=["permno", "date", "vec"])
    tr_embed_pca = project_text_to_pca(tr_embed, pca)
    Xtr, _, _, _ = assemble_walk_features(tr_panel, tr_embed_pca)
    Xtr, Xvl = drop_zero_info_columns(Xtr, Xvl)
    print(f"Post-drop_zero_info: train cols {Xtr.shape[1]}, val cols {Xvl.shape[1]}")
    assert Xvl.shape[1] == len(train_features), \
        f"col mismatch: {Xvl.shape[1]} val vs {len(train_features)} ranker"

    # Reorder val features to match the ranker's training order, cast to float64 (NA→NaN)
    Xvl = Xvl[train_features].copy()
    # Cast all columns to float64 — handles pandas Int64/NAType which LightGBM rejects via numpy
    for c in Xvl.columns:
        Xvl[c] = pd.to_numeric(Xvl[c], errors="coerce").astype("float64")
    y_arr = yvl.to_numpy()
    print(f"Val: {len(Xvl)} rows, {len(gvl)} groups (Fridays)")

    # Baseline NDCG (no shuffling)
    baseline_scores = model.predict(Xvl.to_numpy())
    baseline_ndcg = ndcg_at_k(baseline_scores, y_arr, gvl, k=TOP_K)
    print(f"\nBaseline val NDCG@{TOP_K}: {baseline_ndcg:.6f}\n")

    # Per-feature permutation importance
    rng = np.random.default_rng(args.seed)
    rows = []
    X_arr = Xvl.to_numpy()
    n_feat = X_arr.shape[1]
    t0 = time.time()
    for j, feat in enumerate(train_features):
        ndcgs = []
        for rep in range(args.n_repeats):
            X_perm = X_arr.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            s = model.predict(X_perm)
            ndcgs.append(ndcg_at_k(s, y_arr, gvl, k=TOP_K))
        mean_shuffled = float(np.mean(ndcgs))
        std_shuffled = float(np.std(ndcgs, ddof=1)) if args.n_repeats > 1 else 0.0
        pi = baseline_ndcg - mean_shuffled
        rows.append({
            "feature": feat,
            "perm_importance": pi,
            "shuffled_ndcg_mean": mean_shuffled,
            "shuffled_ndcg_std": std_shuffled,
        })
        if (j + 1) % 25 == 0:
            elapsed = time.time() - t0
            rate = (j + 1) / elapsed
            eta = (n_feat - j - 1) / rate
            print(f"  {j+1}/{n_feat} ({rate:.2f} feat/s, ETA {eta:.0f}s)")

    df = pd.DataFrame(rows).sort_values("perm_importance", ascending=False).reset_index(drop=True)
    df["baseline_ndcg"] = baseline_ndcg

    out_csv = RANKER_DIR / "permutation_importance.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote -> {out_csv.relative_to(REPO_ROOT)}")

    # Bucketed summary
    n_total = len(df)
    n_helpful = (df["perm_importance"] > 1e-5).sum()
    n_neutral = ((df["perm_importance"].abs() <= 1e-5)).sum()
    n_harmful = (df["perm_importance"] < -1e-5).sum()

    print(f"\nPermutation importance summary (n={n_total}):")
    print(f"  Helpful (PI > 0): {n_helpful}")
    print(f"  Neutral (PI ≈ 0): {n_neutral}")
    print(f"  Harmful (PI < 0): {n_harmful}")
    print(f"\nPI percentiles:")
    for p in [0, 5, 10, 25, 50, 75, 90, 95, 100]:
        print(f"  p{p:>3}: {df['perm_importance'].quantile(p/100):+.6f}")

    # By type
    def categorize(name):
        if name.startswith("pca_"): return "pca_text"
        if name.startswith("macro_"): return "macro"
        if name in ("text_novelty", "days_since_filing", "doc_count_7d"): return "text_aux"
        if name in ("prc", "openprc", "askhi", "bidlo", "vol", "shrout", "cfacpr",
                    "cfacshr", "ret", "dlret", "dlstcd"): return "crsp_price"
        return "sharadar"
    df["type"] = df["feature"].apply(categorize)
    print("\nBy type:")
    agg = df.groupby("type").agg(
        n=("perm_importance", "count"),
        n_helpful=("perm_importance", lambda x: (x > 1e-5).sum()),
        n_harmful=("perm_importance", lambda x: (x < -1e-5).sum()),
        pi_sum=("perm_importance", "sum"),
        pi_median=("perm_importance", "median"),
    )
    print(agg.to_string())

    # Top helpful + top harmful
    print("\nTop 20 most-helpful features:")
    print(df.nlargest(20, "perm_importance")[["feature", "perm_importance", "type"]].to_string(index=False))
    print("\nBottom 20 features (most harmful or most dead):")
    print(df.nsmallest(20, "perm_importance")[["feature", "perm_importance", "type"]].to_string(index=False))


if __name__ == "__main__":
    main()
