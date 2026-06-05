"""Reconstruct per-Friday per-stock weights for the regime K-selector ENSEMBLE.

Mirrors `regime_k_selector.py` end-to-end but additionally saves:
  - artifacts/backtest_factor_v1/k_ensemble_weights.parquet
      columns: date, permno, weight
      one row per (Friday, held permno) with positive weight in the ensemble
  - artifacts/backtest_factor_v1/k_ensemble_probas.parquet
      columns: date, K10_prob, K20_prob, K30_prob, K50_prob
      one row per OOS Friday; the LGBM's class probabilities

The weights are the convex-combination:
    w_combined(stock_i) = sum_K  proba(K | regime) * w_K(stock_i)
where w_K is the top-K mcap-weighted (10% cap, water-fill) portfolio for K.
"""
from __future__ import annotations

import time

import lightgbm as lgb
import pandas as pd

from src.strategy import build_regime_features, make_k_classifier
from src.strategy.constants import K_CANDIDATES
from src.strategy.historical import (
    build_k_labels, load_data, load_spy_at, macro_by_date, per_k_weights_and_returns,
)
from src.utils.io import repo_root
from src.utils.logging_utils import configure_logging, get_logger

log = get_logger(__name__)
REPO_ROOT = repo_root()
OUT_DIR = REPO_ROOT / "artifacts" / "backtest_factor_v1"


def main():
    configure_logging()
    t0 = time.time()
    log.info("Loading panel...")
    df = load_data()
    log.info("Loaded %d Friday-stock rows. Building K-portfolios...", len(df))

    k_weights = {}
    k_returns = {}
    for K in K_CANDIDATES:
        log.info("  K=%d", K)
        kw, kr = per_k_weights_and_returns(df, K)
        k_weights[K] = kw
        k_returns[K] = kr

    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in k_returns.values()])))

    # Regime features + labels (shared core)
    spy_at = load_spy_at(all_dates)
    mbd = macro_by_date(df, all_dates)
    regime_df = build_regime_features(all_dates, spy_at, mbd)
    labels, k_mat = build_k_labels(k_returns, all_dates)

    # Walk-forward retrain (identical to regime_k_selector.py)
    log.info("Training LGBM walks ...")
    years = all_dates.year
    proba_rows = []
    for walk_id in range(1, 18):
        train_end = 2007 + walk_id - 1
        val_year = train_end + 1
        test_year = train_end + 2
        train_mask = years <= train_end
        val_mask = years == val_year
        test_mask = years == test_year
        if test_mask.sum() < 10: continue
        Xtr = regime_df[train_mask]; ytr = labels[train_mask]
        Xvl = regime_df[val_mask]; yvl = labels[val_mask]
        Xte = regime_df[test_mask]
        valid_tr = ytr.notna(); Xtr = Xtr[valid_tr]; ytr = ytr[valid_tr].astype(int)
        valid_vl = yvl.notna(); Xvl = Xvl[valid_vl]; yvl = yvl[valid_vl].astype(int)
        if len(Xtr) < 100 or len(Xvl) < 5: continue
        clf = make_k_classifier(num_class=len(K_CANDIDATES))
        clf.fit(Xtr.to_numpy(), ytr.to_numpy(),
                eval_set=[(Xvl.to_numpy(), yvl.to_numpy())],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
        test_dates = all_dates[test_mask]
        proba = clf.predict_proba(Xte.to_numpy())
        for i, d in enumerate(test_dates):
            proba_rows.append({
                "date": d,
                "K10_prob": float(proba[i, 0]),
                "K20_prob": float(proba[i, 1]),
                "K30_prob": float(proba[i, 2]),
                "K50_prob": float(proba[i, 3]),
            })
        log.info("walk %2d (test %d) done", walk_id, test_year)

    proba_df = pd.DataFrame(proba_rows).sort_values("date").reset_index(drop=True)
    proba_df["date"] = pd.to_datetime(proba_df["date"])
    log.info("Got %d OOS Fridays with probabilities", len(proba_df))

    # Build the ensemble weights: w_combined(stock) = sum_K proba_K * w_K(stock)
    # For each OOS Friday, gather the 4 K-portfolios' weights and blend them.
    log.info("Building ensemble weight panel ...")
    ensemble_weight_rows = []
    proba_idx = proba_df.set_index("date")
    for d in proba_df["date"]:
        proba_vec = proba_idx.loc[d, ["K10_prob", "K20_prob", "K30_prob", "K50_prob"]].to_numpy()
        # Get each K's weights at this date
        combined = {}  # permno -> weight
        for j, K in enumerate(K_CANDIDATES):
            wK = k_weights[K]
            wK_d = wK[wK["date"] == d]
            for _, row in wK_d.iterrows():
                permno = int(row["permno"])
                combined[permno] = combined.get(permno, 0.0) + proba_vec[j] * row["weight"]
        for permno, weight in combined.items():
            if weight > 1e-8:
                ensemble_weight_rows.append({"date": d, "permno": permno, "weight": weight})

    ens_w = pd.DataFrame(ensemble_weight_rows)
    log.info("Ensemble weight panel: %d (date, permno) rows", len(ens_w))

    # Sanity check: weights sum to ~1 each Friday
    sums = ens_w.groupby("date")["weight"].sum()
    log.info("Weight sums per Friday: min=%.6f, max=%.6f, mean=%.6f",
             sums.min(), sums.max(), sums.mean())

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ens_w.to_parquet(OUT_DIR / "k_ensemble_weights.parquet", compression="zstd", index=False)
    proba_df.to_parquet(OUT_DIR / "k_ensemble_probas.parquet", compression="zstd", index=False)
    log.info("Wrote weights → %s", OUT_DIR / "k_ensemble_weights.parquet")
    log.info("Wrote probas → %s", OUT_DIR / "k_ensemble_probas.parquet")
    log.info("Total time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
