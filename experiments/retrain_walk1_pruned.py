"""Retrain walk-1 LightGBM ranker on the pruned 154-feature set.

Uses the same FROZEN_HPS and walk-1 train/val/test splits as notebook 06's
process_walk(1), but restricts the feature matrix to `pruned_feature_set_v1.json`.

Outputs:
  artifacts/ranker/walk-001_pruned/model.joblib
  artifacts/ranker/walk-001_pruned/feature_importance.csv
  artifacts/ranker/walk-001_pruned/summary.json

Then prints side-by-side: original walk-1 vs pruned walk-1
  - val NDCG@30
  - test rank IC mean / IR
  - test hit rate
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import early_stopping

from src.utils.io import processed_dir, repo_root
from src.utils.ranker import (
    assemble_walk_features,
    build_ranker,
    compute_excess_return_buckets,
    drop_zero_info_columns,
    evaluate_ranker,
    friday_only,
    load_walk_pca,
    project_text_to_pca,
)

WALK_ID = 1
TRAIN_START, TRAIN_END = "2002-01-01", "2007-12-31"
VAL_START, VAL_END = "2008-01-01", "2008-12-31"
TEST_START, TEST_END = "2009-01-01", "2009-12-31"
TOP_K = 30
N_BUCKETS = 32

REPO_ROOT = repo_root()
PANEL_DIR = processed_dir() / "training_panel"
EMBED_DIR = processed_dir() / "finbert_stockday_embed"
PRUNE_JSON = REPO_ROOT / "experiments" / "pruned_feature_set_v1.json"
OUT_DIR = REPO_ROOT / "artifacts" / "ranker" / "walk-001_pruned"
ORIG_DIR = REPO_ROOT / "artifacts" / "ranker" / "walk-001"

# FROZEN_HPS from notebook 06 cell C (winning Optuna config for walk-1).
# Filter to actual LightGBM hyperparams (drop metric/log fields).
_RAW_HPS = json.loads((ORIG_DIR / "hp.json").read_text())
_LGBM_HP_KEYS = {"num_leaves", "learning_rate", "min_data_in_leaf",
                  "feature_fraction", "bagging_fraction", "lambda_l2"}
FROZEN_HPS = {k: v for k, v in _RAW_HPS.items() if k in _LGBM_HP_KEYS}
print(f"Frozen HPs (filtered): {FROZEN_HPS}")


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


def build_xy(start: str, end: str, pca):
    panel = load_years(PANEL_DIR, start, end)
    embed = load_years(EMBED_DIR, start, end, cols=["permno", "date", "vec"])
    embed_pca = project_text_to_pca(embed, pca)
    return assemble_walk_features(panel, embed_pca)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pruned feature list
    pruned = json.loads(PRUNE_JSON.read_text())
    keep_features = pruned["kept_features"]
    print(f"\nKept feature set: {len(keep_features)} (from {pruned['n_total']})")

    # Build train/val/test feature matrices using walk-1 PCA
    pca, n_pca = load_walk_pca(WALK_ID)
    print(f"\nLoaded walk-1 PCA: n={n_pca}")

    print("Building train/val/test feature matrices...")
    Xtr, ytr, gtr, mtr = build_xy(TRAIN_START, TRAIN_END, pca)
    Xvl, yvl, gvl, mvl = build_xy(VAL_START, VAL_END, pca)
    Xte, yte, gte, mte = build_xy(TEST_START, TEST_END, pca)
    Xtr, Xvl, Xte = drop_zero_info_columns(Xtr, Xvl, Xte)
    print(f"  train {Xtr.shape}, val {Xvl.shape}, test {Xte.shape}")

    # Restrict to kept features (intersection — defensive)
    keep = [c for c in keep_features if c in Xtr.columns]
    drop_count = len(keep_features) - len(keep)
    if drop_count > 0:
        print(f"  warning: {drop_count} kept features not in post-zero-info set")
    Xtr = Xtr[keep].copy()
    Xvl = Xvl[keep].copy()
    Xte = Xte[keep].copy()
    print(f"  restricted to {len(keep)} features")

    # Bucket labels for LambdaRank
    btr = compute_excess_return_buckets(mtr, n_buckets=N_BUCKETS).astype(int).values
    bvl = compute_excess_return_buckets(mvl, n_buckets=N_BUCKETS).astype(int).values

    # Fit (use build_ranker so label_gain is set to the 64-bucket extension)
    print("\nFitting LightGBM ranker (via build_ranker, label_gain handled)...")
    t0 = time.time()
    model = build_ranker(FROZEN_HPS)
    model.fit(
        Xtr, btr, group=gtr,
        eval_set=[(Xvl, bvl)], eval_group=[gvl], eval_at=[TOP_K],
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
    )
    fit_sec = time.time() - t0
    print(f"  fit time: {fit_sec:.1f}s, best_iteration: {model.best_iteration_}")

    val_ndcg = float(model.best_score_["valid_0"][f"ndcg@{TOP_K}"])
    print(f"  val NDCG@{TOP_K}: {val_ndcg:.6f}")

    # Test evaluation (evaluate_ranker returns keys without 'test_' prefix; rename for compat)
    raw_m = evaluate_ranker(model, Xte, yte, mte["date"],
                            top_k=TOP_K, entity_ids=mte["permno"])
    test_m = {f"test_{k}": v for k, v in raw_m.items()}
    print(f"  test rank IC mean: {test_m['test_rank_ic_mean']:.6f}")
    print(f"  test rank IC IR:   {test_m['test_rank_ic_ir']:.6f}")
    print(f"  test hit rate:     {test_m['test_hit_rate']:.4f}")
    print(f"  test decile spread bps: {test_m['test_decile_spread_bps']:.2f}")

    # Save
    joblib.dump({"model": model, "feature_names": Xtr.columns.tolist()},
                OUT_DIR / "model.joblib")
    fi = pd.DataFrame({
        "feature": Xtr.columns,
        "gain": model.booster_.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    fi.to_csv(OUT_DIR / "feature_importance.csv", index=False)
    summary = {
        "walk_id": WALK_ID,
        "n_features": int(len(keep)),
        "n_pca": int(n_pca),
        "n_train_rows": int(len(Xtr)),
        "n_val_rows": int(len(Xvl)),
        "n_test_rows": int(len(Xte)),
        "best_iteration": int(model.best_iteration_),
        "val_ndcg_at_30": val_ndcg,
        **{k: (float(v) if hasattr(v, "__float__") else v) for k, v in test_m.items()},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # Comparison
    orig = json.loads((ORIG_DIR / "summary.json").read_text())
    print("\n=== SIDE-BY-SIDE: walk-1 original (190 feat) vs pruned (154 feat) ===")
    print(f"  {'metric':<28} {'original':>12} {'pruned':>12} {'Δ':>10}")
    for key in ["n_features", "best_iteration", "val_ndcg_at_30",
                "test_rank_ic_mean", "test_rank_ic_ir", "test_hit_rate",
                "test_decile_spread_bps"]:
        a, b = orig.get(key, None), summary.get(key, None)
        if a is None or b is None:
            continue
        if isinstance(a, (int, np.integer)):
            print(f"  {key:<28} {a:>12} {b:>12} {b-a:>+10}")
        else:
            print(f"  {key:<28} {a:>12.6f} {b:>12.6f} {b-a:>+10.6f}")

    print(f"\nwrote -> {OUT_DIR.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
