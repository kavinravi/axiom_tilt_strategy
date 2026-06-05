"""Train + persist ONE production K-selector model on all history.

CLI:  python -m src.strategy.train [--out trading/models/k_selector.txt]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import pandas as pd

from src.strategy.constants import K_CANDIDATES, REGIME_FEATURES
from src.strategy.historical import (
    build_k_labels, load_data, load_spy_at, macro_by_date, per_k_weights_and_returns,
)
from src.strategy.k_selector import build_regime_features, save_model, train_model
from src.utils.io import repo_root

DEFAULT_OUT = repo_root() / "trading" / "models" / "k_selector.txt"


def train_production_model(out_path: Path | str = DEFAULT_OUT) -> dict:
    """Train one LGBM on all history through the latest date; persist model + meta.
    Returns the meta dict."""
    out_path = Path(out_path)
    df = load_data()
    k_returns = {K: per_k_weights_and_returns(df, K)[1] for K in K_CANDIDATES}
    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in k_returns.values()])))
    labels, _ = build_k_labels(k_returns, all_dates)
    spy_at = load_spy_at(all_dates)
    regime = build_regime_features(all_dates, spy_at, macro_by_date(df, all_dates))

    valid = labels.notna()
    model = train_model(
        regime[valid.to_numpy()].to_numpy(),
        labels[valid].astype(int).to_numpy(),
        num_class=len(K_CANDIDATES),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, out_path)
    meta = {
        "train_date_utc": datetime.now(timezone.utc).isoformat(),
        "n_train_fridays": int(valid.sum()),
        "first_date": str(all_dates.min().date()),
        "last_date": str(all_dates.max().date()),
        "features": REGIME_FEATURES,
        "K_candidates": K_CANDIDATES,
        "label": "argmax K of per-K weekly fwd_ret_5d",
        "lightgbm_version": lightgbm.__version__,
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    meta = train_production_model(out_path=args.out)
    print(f"Saved model -> {args.out}")
    print(f"Saved meta  -> {Path(args.out).with_suffix('.meta.json')}")
    print(f"Trained on {meta['n_train_fridays']} Fridays "
          f"{meta['first_date']}..{meta['last_date']}")


if __name__ == "__main__":
    main()
