"""Pure backtest helpers: metrics, turnover, net returns, year windows, and the
generic walk-forward LGBM probability pass. No I/O, no globals."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.strategy.k_selector import make_k_classifier

PERIODS_PER_YEAR = 52
COST_BPS = 5.0


def metrics(rets) -> dict[str, float]:
    """Annualized return, vol, Sharpe, Sortino, max drawdown for a weekly series.
    Zero-vol cases return 0.0 (never inf/nan) so tables stay clean."""
    r = np.asarray(rets, dtype=np.float64)
    if len(r) == 0:
        return {"ann": 0.0, "vol": 0.0, "sharpe": 0.0, "sortino": 0.0, "mdd": 0.0}
    cum = float(np.prod(1.0 + r) - 1.0)
    ann = (1.0 + cum) ** (PERIODS_PER_YEAR / len(r)) - 1.0
    vol = float(np.std(r, ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if len(r) > 1 else 0.0
    sharpe = float(ann / vol) if vol > 0 else 0.0
    downside = r[r < 0]
    dvol = float(np.std(downside, ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if len(downside) > 1 else 0.0
    sortino = float(ann / dvol) if dvol > 0 else 0.0
    eq = np.cumprod(1.0 + r)
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    return {"ann": ann, "vol": vol, "sharpe": sharpe, "sortino": sortino, "mdd": mdd}


def turnover_series(weight_dicts: list[dict]) -> np.ndarray:
    """One-way turnover per period: 0.5 * sum_i |w_t(i) - w_{t-1}(i)|, with names
    absent on either side treated as 0. First period builds from cash."""
    tu = np.zeros(len(weight_dicts), dtype=np.float64)
    prev: dict = {}
    for t, cur in enumerate(weight_dicts):
        names = set(cur) | set(prev)
        tu[t] = 0.5 * sum(abs(cur.get(n, 0.0) - prev.get(n, 0.0)) for n in names)
        prev = cur
    return tu


def net_returns(gross, turnover, cost_bps: float = COST_BPS) -> np.ndarray:
    """gross - (cost_bps/1e4) * turnover, elementwise."""
    g = np.asarray(gross, dtype=np.float64)
    tu = np.asarray(turnover, dtype=np.float64)
    return g - (cost_bps / 1e4) * tu


def window_mask(dates: pd.DatetimeIndex, start_year: int, end_year: int) -> np.ndarray:
    """Boolean mask for start_year <= year <= end_year (inclusive)."""
    y = pd.DatetimeIndex(dates).year.to_numpy()
    return (y >= start_year) & (y <= end_year)


def walk_forward_proba(regime_df: pd.DataFrame, labels: pd.Series,
                       all_dates: pd.DatetimeIndex, num_class: int) -> pd.DataFrame:
    """Walk-forward LGBM class probabilities, identical scheme to
    experiments/regime_k_selector.py (walks 1..17, 1y val / 1y test, early
    stopping). Returns a frame indexed by OOS date with one column per class
    ('c0'..'c{num_class-1}')."""
    years = all_dates.year
    rows = []
    for walk_id in range(1, 18):
        train_end = 2007 + walk_id - 1
        val_year = train_end + 1
        test_year = train_end + 2
        train_mask = years <= train_end
        val_mask = years == val_year
        test_mask = years == test_year
        if test_mask.sum() < 10:
            continue
        Xtr, ytr = regime_df[train_mask], labels[train_mask]
        Xvl, yvl = regime_df[val_mask], labels[val_mask]
        Xte = regime_df[test_mask]
        vtr = ytr.notna(); Xtr, ytr = Xtr[vtr], ytr[vtr].astype(int)
        vvl = yvl.notna(); Xvl, yvl = Xvl[vvl], yvl[vvl].astype(int)
        if len(Xtr) < 100 or len(Xvl) < 5:
            continue
        clf = make_k_classifier(num_class=num_class)
        clf.fit(Xtr.to_numpy(), ytr.to_numpy(),
                eval_set=[(Xvl.to_numpy(), yvl.to_numpy())],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
        proba = clf.predict_proba(Xte.to_numpy())
        for i, d in enumerate(all_dates[test_mask]):
            rows.append({"date": d, **{f"c{j}": float(proba[i, j]) for j in range(num_class)}})
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.set_index("date")
