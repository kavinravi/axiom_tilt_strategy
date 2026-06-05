"""Regime-conditional K selector.

ML classifier picks the optimal K (concentration level) of sp_fcfa per Friday
based on regime features. Candidate K values: 10, 20, 30, 50.

Hypothesis: concentrate in calm bullish regimes (K=10 → more alpha),
diversify in volatile/bearish regimes (K=50 → defensive).

Same regime features as before (VIX, yields, trailing SPY ret/vol).
Walk-forward retrained per walk.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.strategy import build_regime_features, make_k_classifier
from src.strategy.constants import K_CANDIDATES
from src.strategy.historical import load_data, load_spy_at, macro_by_date, per_k_weights_and_returns
from src.utils.io import repo_root
from src.utils.logging_utils import get_logger

log = get_logger(__name__)
REPO_ROOT = repo_root()


print("Loading panel ...")
df = load_data()


print("Building per-K weekly returns ...")
k_returns = {K: per_k_weights_and_returns(df, K)[1] for K in K_CANDIDATES}
all_dates = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in k_returns.values()])))
k_mat = pd.DataFrame({f"K{K}": k_returns[K].reindex(all_dates).values for K in K_CANDIDATES}, index=all_dates)

# Build regime features (shared core)
spy_at = load_spy_at(all_dates)
mbd = macro_by_date(df, all_dates)
regime_df = build_regime_features(all_dates, spy_at, mbd)

# Label: argmax K per Friday
k_to_idx = {K: i for i, K in enumerate(K_CANDIDATES)}
labels = k_mat.idxmax(axis=1).str[1:].astype(int).map(k_to_idx)


def metrics(rets):
    r = np.asarray(rets, dtype=float)
    cum = float(np.prod(1.0 + r) - 1.0)
    ann = (1.0 + cum) ** (52.0 / len(r)) - 1.0
    vol = float(np.std(r, ddof=1) * np.sqrt(52.0))
    sh = ann / vol if vol > 0 else 0.0
    eq = np.cumprod(1.0 + r); peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    return {"ann": ann, "vol": vol, "sh": sh, "mdd": mdd}


# Walk-forward train + inference
print("Training walk-forward K-selector ...")
years = all_dates.year
argmax_rets = []
ensemble_rets = []
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
    argmax_idx = clf.predict(Xte.to_numpy()).astype(int)
    for i, d in enumerate(test_dates):
        K_pick = K_CANDIDATES[argmax_idx[i]]
        argmax_rets.append({"date": d, "K": K_pick, "weekly_ret": float(k_returns[K_pick].get(d, 0.0))})
        # probability-weighted ensemble of K-strategies (NOT weights, just returns weighted)
        ensemble_ret = sum(proba[i, j] * k_returns[K_CANDIDATES[j]].get(d, 0.0)
                            for j in range(len(K_CANDIDATES)))
        ensemble_rets.append({"date": d, "weekly_ret": float(ensemble_ret)})
    log.info("walk %2d (test %d): K pick freq = %s",
             walk_id, test_year,
             pd.Series([K_CANDIDATES[i] for i in argmax_idx]).value_counts().to_dict())

arg_df = pd.DataFrame(argmax_rets).sort_values("date").reset_index(drop=True)
ens_df = pd.DataFrame(ensemble_rets).sort_values("date").reset_index(drop=True)
arg_df["date"] = pd.to_datetime(arg_df["date"])
ens_df["date"] = pd.to_datetime(ens_df["date"])

test_dates_all = pd.DatetimeIndex(arg_df["date"])
spy_aligned = spy_at.reindex(test_dates_all).pct_change().fillna(0.0).to_numpy()
det_sp = pd.read_parquet(REPO_ROOT / "artifacts" / "backtest_factor_v1" / "weekly_sp_fcfa.parquet")
det_sp["date"] = pd.to_datetime(det_sp["date"])
det_aligned = det_sp.set_index("date").reindex(test_dates_all)["weekly_ret"].to_numpy()
years_t = test_dates_all.year
mask = years_t >= 2010

# Static K reference points (sp_fcfa K=10 alone, K=30 alone)
k10_static = k_returns[10].reindex(test_dates_all).to_numpy()
k30_static = k_returns[30].reindex(test_dates_all).to_numpy()

print(f"\n=== Regime K-selector (LGBM picks K_in {{10,20,30,50}} per Friday) ===")
print(f"  2010-2025 BAR ({mask.sum()} weeks):")
print(f"    {'strategy':<40} {'ann':>8} {'vol':>8} {'sharpe':>8} {'mdd':>8}")
ms = metrics(spy_aligned[mask])
m10 = metrics(k10_static[mask])
m30 = metrics(k30_static[mask])
ma = metrics(arg_df["weekly_ret"].to_numpy()[mask])
me = metrics(ens_df["weekly_ret"].to_numpy()[mask])
print(f"    {'SPY':<40} {ms['ann']:>8.2%} {ms['vol']:>8.2%} {ms['sh']:>8.3f} {ms['mdd']:>8.2%}")
print(f"    {'sp_fcfa K=30 static (current best det)':<40} {m30['ann']:>8.2%} {m30['vol']:>8.2%} {m30['sh']:>8.3f} {m30['mdd']:>8.2%}")
print(f"    {'sp_fcfa K=10 static (best from K sweep)':<40} {m10['ann']:>8.2%} {m10['vol']:>8.2%} {m10['sh']:>8.3f} {m10['mdd']:>8.2%}")
print(f"    {'regime-LGBM K argmax':<40} {ma['ann']:>8.2%} {ma['vol']:>8.2%} {ma['sh']:>8.3f} {ma['mdd']:>8.2%}")
print(f"    {'regime-LGBM K ensemble (proba-weighted)':<40} {me['ann']:>8.2%} {me['vol']:>8.2%} {me['sh']:>8.3f} {me['mdd']:>8.2%}")

out_dir = REPO_ROOT / "artifacts" / "backtest_factor_v1"
out_dir.mkdir(parents=True, exist_ok=True)
arg_df.to_parquet(out_dir / "weekly_regime_K_argmax.parquet", compression="zstd", index=False)
ens_df.to_parquet(out_dir / "weekly_regime_K_ensemble.parquet", compression="zstd", index=False)
