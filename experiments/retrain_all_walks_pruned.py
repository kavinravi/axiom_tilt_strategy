"""Retrain ranker for all 17 walks using the pruned 154-feature set.

Outputs to artifacts/ranker/walk-NNN_pruned/. Each walk's PCA stays the same;
only the feature subset changes (same 154 features across all walks).

Mirrors notebook 06's process_walk() logic. Logs per-walk val NDCG and test
rank IC, compares against the original walk's summary.json.

usage: python experiments/retrain_all_walks_pruned.py
"""
from __future__ import annotations

import json
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
    load_walk_pca,
    project_text_to_pca,
)

WALK_TRAIN_END_YEARS = list(range(2007, 2024))  # walk 1..17 → train ends 2007..2023
TOP_K = 30
N_BUCKETS = 32

REPO_ROOT = repo_root()
PANEL_DIR = processed_dir() / "training_panel"
EMBED_DIR = processed_dir() / "finbert_stockday_embed"
RANKER_ROOT = REPO_ROOT / "artifacts" / "ranker"
PRUNE_JSON = REPO_ROOT / "experiments" / "pruned_feature_set_v1.json"

# Use walk-1's frozen HPs across all walks (matches parent's autoresearch design).
_RAW_HPS = json.loads((RANKER_ROOT / "walk-001" / "hp.json").read_text())
_LGBM_HP_KEYS = {"num_leaves", "learning_rate", "min_data_in_leaf",
                  "feature_fraction", "bagging_fraction", "lambda_l2"}
FROZEN_HPS = {k: v for k, v in _RAW_HPS.items() if k in _LGBM_HP_KEYS}
print(f"Frozen HPs: {FROZEN_HPS}\n")


def _walk_windows(walk_id: int) -> tuple[str, str, str, str, str, str]:
    train_end_year = 2007 + walk_id - 1
    return (
        "2002-01-01", f"{train_end_year}-12-31",
        f"{train_end_year + 1}-01-01", f"{train_end_year + 1}-12-31",
        f"{train_end_year + 2}-01-01", f"{train_end_year + 2}-12-31",
    )


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


def main():
    pruned = json.loads(PRUNE_JSON.read_text())
    kept_features = pruned["kept_features"]
    print(f"Pruned set: {len(kept_features)} features\n")

    rows = []
    for walk_id in range(1, 18):
        t0 = time.time()
        tr_s, tr_e, vl_s, vl_e, te_s, te_e = _walk_windows(walk_id)
        out_dir = RANKER_ROOT / f"walk-{walk_id:03d}_pruned"
        out_dir.mkdir(parents=True, exist_ok=True)

        orig_summary = json.loads((RANKER_ROOT / f"walk-{walk_id:03d}" / "summary.json").read_text())

        # Load walk PCA + assemble feature matrices
        pca, n_pca = load_walk_pca(walk_id)

        def _build(start, end):
            panel = load_years(PANEL_DIR, start, end)
            embed = load_years(EMBED_DIR, start, end, cols=["permno", "date", "vec"])
            embed_pca = project_text_to_pca(embed, pca)
            return assemble_walk_features(panel, embed_pca)

        Xtr, ytr, gtr, mtr = _build(tr_s, tr_e)
        Xvl, yvl, gvl, mvl = _build(vl_s, vl_e)
        Xte, yte, gte, mte = _build(te_s, te_e)
        Xtr, Xvl, Xte = drop_zero_info_columns(Xtr, Xvl, Xte)

        keep = [c for c in kept_features if c in Xtr.columns]
        Xtr = Xtr[keep]; Xvl = Xvl[keep]; Xte = Xte[keep]

        btr = compute_excess_return_buckets(mtr, n_buckets=N_BUCKETS).astype(int).values
        bvl = compute_excess_return_buckets(mvl, n_buckets=N_BUCKETS).astype(int).values

        model = build_ranker(FROZEN_HPS)
        model.fit(
            Xtr, btr, group=gtr,
            eval_set=[(Xvl, bvl)], eval_group=[gvl], eval_at=[TOP_K],
            callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
        )
        val_ndcg = float(model.best_score_["valid_0"][f"ndcg@{TOP_K}"])
        raw_m = evaluate_ranker(model, Xte, yte, mte["date"],
                                top_k=TOP_K, entity_ids=mte["permno"])
        test_m = {f"test_{k}": v for k, v in raw_m.items()}

        # Save
        joblib.dump({"model": model, "feature_names": Xtr.columns.tolist()},
                    out_dir / "model.joblib")
        summary = {
            "walk_id": walk_id,
            "n_features": int(len(keep)),
            "n_pca": int(n_pca),
            "best_iteration": int(model.best_iteration_),
            "val_ndcg_at_30": val_ndcg,
            **test_m,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

        ic_o = orig_summary.get("test_rank_ic_mean", float("nan"))
        ic_n = summary["test_rank_ic_mean"]
        ir_o = orig_summary.get("test_rank_ic_ir", float("nan"))
        ir_n = summary["test_rank_ic_ir"]
        ndcg_o = orig_summary.get("val_ndcg_at_30", float("nan"))
        ndcg_n = val_ndcg
        elapsed = time.time() - t0
        rows.append({
            "walk": walk_id, "test_year": 2008 + walk_id,
            "val_ndcg_orig": ndcg_o, "val_ndcg_new": ndcg_n, "val_ndcg_dlt": ndcg_n - ndcg_o,
            "ic_orig": ic_o, "ic_new": ic_n, "ic_dlt": ic_n - ic_o,
            "ir_orig": ir_o, "ir_new": ir_n, "ir_dlt": ir_n - ir_o,
        })
        print(f"walk {walk_id:2d} ({2008+walk_id}): "
              f"ndcg {ndcg_o:.4f} -> {ndcg_n:.4f} ({ndcg_n-ndcg_o:+.4f})  "
              f"IC {ic_o:+.4f} -> {ic_n:+.4f} ({ic_n-ic_o:+.4f})  "
              f"IR {ir_o:+.3f} -> {ir_n:+.3f} ({ir_n-ir_o:+.3f})  "
              f"[{elapsed:.0f}s]")

    df = pd.DataFrame(rows)
    df.to_csv(RANKER_ROOT / "pruned_vs_original_summary.csv", index=False)
    print("\n=== AGGREGATE ===")
    print(f"  Mean ΔIC:  {df['ic_dlt'].mean():+.4f}  (positive => better)")
    print(f"  Mean ΔIR:  {df['ir_dlt'].mean():+.4f}")
    print(f"  Mean ΔNDCG: {df['val_ndcg_dlt'].mean():+.4f}")
    print(f"  Walks with IC improvement: {(df['ic_dlt'] > 0).sum()}/17")
    print(f"  Walks with IR improvement: {(df['ir_dlt'] > 0).sum()}/17")
    print(f"  wrote -> {(RANKER_ROOT / 'pruned_vs_original_summary.csv').relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
