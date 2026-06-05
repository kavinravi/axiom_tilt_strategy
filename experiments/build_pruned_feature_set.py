"""Build the pruned feature set for walk-1 ranker retrain.

Hybrid rule (user-chosen 2026-06-01):
  1. Drop features with permutation_importance <= 0.0001 (harmful + dead-weight + tiny positive)
  2. Drop USD-duplicates in Sharadar: any `*usd` column where its non-usd counterpart
     also exists. These are near-collinear (S&P 500 is mostly USD-reporting).

Outputs a JSON file with the kept feature list + a markdown report.

usage: python experiments/build_pruned_feature_set.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.utils.io import repo_root

REPO_ROOT = repo_root()
PI_CSV = REPO_ROOT / "artifacts" / "ranker" / "walk-001" / "permutation_importance.csv"
OUT_JSON = REPO_ROOT / "experiments" / "pruned_feature_set_v1.json"
OUT_REPORT = REPO_ROOT / "reports" / "feature_pruning_v1.md"

PI_THRESHOLD = 0.0001


def find_usd_duplicates(features: list[str]) -> list[str]:
    """Return USD-suffix features whose non-USD counterpart is also in `features`."""
    feat_set = set(features)
    dups = []
    for f in features:
        if f.endswith("usd"):
            base = f[:-3]
            if base in feat_set:
                dups.append(f)
    return dups


def main():
    df = pd.read_csv(PI_CSV)
    print(f"Loaded {len(df)} feature PI rows from {PI_CSV.relative_to(REPO_ROOT)}")

    # Rule 1: PI threshold
    drop_by_pi = df[df["perm_importance"] <= PI_THRESHOLD]["feature"].tolist()
    print(f"\nRule 1: PI <= {PI_THRESHOLD} → {len(drop_by_pi)} features to drop")

    # Rule 2: USD duplicates
    all_features = df["feature"].tolist()
    usd_dups = find_usd_duplicates(all_features)
    print(f"Rule 2: USD-duplicate features → {len(usd_dups)} features to drop")

    # Union (some features may be flagged by both)
    drop_set = set(drop_by_pi) | set(usd_dups)
    keep = [f for f in all_features if f not in drop_set]
    print(f"\nTotal dropped: {len(drop_set)} (overlap {len(set(drop_by_pi) & set(usd_dups))})")
    print(f"Total kept:    {len(keep)} (started from {len(all_features)})")

    # Categorize the kept set
    def cat(name):
        if name.startswith("pca_"): return "pca_text"
        if name.startswith("macro_"): return "macro"
        if name in ("text_novelty", "days_since_filing", "doc_count_7d"): return "text_aux"
        if name in ("prc", "openprc", "askhi", "bidlo", "vol", "shrout", "cfacpr",
                    "cfacshr", "ret", "dlret", "dlstcd"): return "crsp_price"
        return "sharadar"

    df["type"] = df["feature"].apply(cat)
    df["keep"] = ~df["feature"].isin(drop_set)
    df["drop_reason"] = df["feature"].apply(
        lambda f: ("usd_dup, " if f in usd_dups else "")
                + ("low_PI" if f in set(drop_by_pi) else "")
    ).str.rstrip(", ")
    df.loc[df["keep"], "drop_reason"] = ""

    by_type = df.groupby(["type", "keep"]).size().unstack(fill_value=0)
    by_type.columns = ["dropped", "kept"] if False in by_type.columns and True in by_type.columns else by_type.columns
    print("\nBy type (kept / dropped):")
    print(by_type.to_string())

    # Output
    out = {
        "n_total": len(all_features),
        "n_kept": len(keep),
        "n_dropped": len(drop_set),
        "pi_threshold": PI_THRESHOLD,
        "rule": "Drop PI <= 0.0001 OR USD-duplicate of a non-USD feature",
        "kept_features": keep,
        "dropped_features": sorted(drop_set),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nwrote -> {OUT_JSON.relative_to(REPO_ROOT)}")

    # Markdown report
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Walk-1 feature pruning v1\n",
        f"**Date:** 2026-06-01\n",
        f"**Rule:** Drop features where `permutation_importance <= {PI_THRESHOLD}` OR feature is a USD-duplicate of a non-USD counterpart.\n",
        f"\n## Summary\n",
        f"- Total features (walk-1 ranker): {len(all_features)}",
        f"- Dropped by low PI:    {len(drop_by_pi)}",
        f"- Dropped by USD-dup:   {len(usd_dups)}",
        f"- Total dropped (union): {len(drop_set)}",
        f"- **Total kept: {len(keep)}**",
        f"\n## By type\n",
        f"```\n{by_type.to_string()}\n```",
        f"\n## Dropped features\n```",
    ]
    for f in sorted(drop_set):
        row = df[df["feature"] == f].iloc[0]
        lines.append(f"  {f:<25} PI={row['perm_importance']:+.6f}  type={row['type']}  reason={row['drop_reason']}")
    lines.append("```\n")
    OUT_REPORT.write_text("\n".join(lines))
    print(f"wrote -> {OUT_REPORT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
