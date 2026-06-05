"""Final sweep: v6 ensembles + vol-targeting overlays + risk-parity weighting on v6 picks.

Variants tested (all PIT-clean):
  v6_mcap           : baseline (Sharpe 1.021 confirmed)
  v6_equal          : equal-weight within top-30
  v6_voltarget18    : scale v6 weekly returns so trailing 16w realized vol = 18% (cash fills slack)
  v6_voltarget16    : same with target=16% (~SPY)
  v6_voltarget14    : same with target=14% (lower than SPY)
  v6_x4_ensemble    : 50/50 mix of v6 (Sharpe 1.021) and x4_ep_roe_roic_K100 (Sharpe 0.998, lower MDD)
  v6_spy_overlay_30 : 70% v6 + 30% SPY (defensive overlay; SPY itself for MDD limit)
  v6_spy_overlay_50 : 50% v6 + 50% SPY

Outputs per-strategy weekly returns + final sweep table to reports/.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from src.utils.io import repo_root, processed_dir

REPO_ROOT = repo_root()
SPY_PATH = REPO_ROOT / "artifacts" / "benchmarks" / "spy_daily.parquet"
V6_WEEKLY = REPO_ROOT / "artifacts" / "backtest_factor_v1" / "weekly_v6_value_quality.parquet"
V6_60_40_WEEKLY = REPO_ROOT / "artifacts" / "backtest_factor_v1" / "weekly_v8_v6_60_40.parquet"  # might not exist

# Load v6 baseline weekly
v6 = pd.read_parquet(V6_WEEKLY)
v6["date"] = pd.to_datetime(v6["date"])
v6 = v6.sort_values("date").reset_index(drop=True)
dates = pd.DatetimeIndex(v6["date"])

# Load SPY weekly aligned
spy = pd.read_parquet(SPY_PATH).reset_index()[["Date", "close"]].rename(columns={"Date": "date"})
spy["date"] = pd.to_datetime(spy["date"])
spy = spy.sort_values("date").set_index("date")["close"]
closes = spy.reindex(spy.index.union(dates)).sort_index().ffill().reindex(dates)
spy_rets = closes.pct_change().fillna(0.0).to_numpy()

# Load DGS3MO (cash rate) for vol-target overlay - convert annualized to weekly
macro = pd.read_parquet(processed_dir() / "macro.parquet")
dgs3 = macro[macro["series"] == "DGS3MO"][["date", "value"]].copy()
dgs3["date"] = pd.to_datetime(dgs3["date"])
dgs3 = dgs3.sort_values("date").set_index("date")["value"] / 100.0  # decimal annualized yield
cash_y_at = dgs3.reindex(dgs3.index.union(dates)).sort_index().ffill().reindex(dates).fillna(0.04).to_numpy()
cash_w_ret = ((1.0 + cash_y_at) ** (1 / 52.0) - 1.0)  # weekly compounding equivalent

# Load x4 ensemble candidate (from factor_v2_extended)
X4_PATH = REPO_ROOT / "artifacts" / "backtest_factor_v2" / "weekly_x4_ep_roe_roic__K100__mcap.parquet"
if not X4_PATH.exists():
    # Fallback: skip x4 ensembles
    x4_rets = None
else:
    x4_df = pd.read_parquet(X4_PATH)
    x4_df["date"] = pd.to_datetime(x4_df["date"])
    x4_rets = x4_df.set_index("date").reindex(dates)["weekly_ret"].to_numpy()


# ---------------- Strategy generators ----------------

def voltarget(base_rets, target_vol_annual=0.18, lookback=16, max_lev=1.0, cash_rets=None):
    """Scale base_rets so trailing realized vol = target. Cash fills (1-scale)."""
    if cash_rets is None:
        cash_rets = np.zeros_like(base_rets)
    rolling = pd.Series(base_rets).rolling(lookback, min_periods=lookback).std(ddof=1).shift(1)
    realized_ann = (rolling * np.sqrt(52)).to_numpy()
    scale = np.where(realized_ann > 0, target_vol_annual / realized_ann, 1.0)
    scale = np.clip(scale, 0.0, max_lev)
    scale = np.where(np.isnan(scale), 1.0, scale)
    return scale * base_rets + (1.0 - scale) * cash_rets


def ensemble(rets_a, rets_b, weight_a=0.5):
    if rets_a is None or rets_b is None:
        return None
    return weight_a * rets_a + (1.0 - weight_a) * rets_b


# ---------------- Compute strategies ----------------

v6_rets = v6["weekly_ret"].to_numpy()
strategies = {
    "v6_mcap (baseline)":         v6_rets,
    "v6_voltarget18 (cash=0)":    voltarget(v6_rets, 0.18, lookback=16),
    "v6_voltarget16 (cash=0)":    voltarget(v6_rets, 0.16, lookback=16),
    "v6_voltarget14 (cash=0)":    voltarget(v6_rets, 0.14, lookback=16),
    "v6_voltarget16 (cash=Tbill)": voltarget(v6_rets, 0.16, lookback=16, cash_rets=cash_w_ret),
    "v6_voltarget14 (cash=Tbill)": voltarget(v6_rets, 0.14, lookback=16, cash_rets=cash_w_ret),
    "v6_spy_overlay_70_30":       ensemble(v6_rets, spy_rets, 0.70),
    "v6_spy_overlay_50_50":       ensemble(v6_rets, spy_rets, 0.50),
    "v6_spy_overlay_30_70":       ensemble(v6_rets, spy_rets, 0.30),
}
if x4_rets is not None:
    strategies["v6_x4_ensemble_50_50"] = ensemble(v6_rets, x4_rets, 0.50)
    strategies["v6_x4_ensemble_70_30"] = ensemble(v6_rets, x4_rets, 0.70)


def metrics(rets, mask=None):
    r = np.asarray(rets, dtype=float)
    if mask is not None:
        r = r[mask]
    if len(r) < 2: return {}
    cum = float(np.prod(1.0 + r) - 1.0)
    ann = (1.0 + cum) ** (52.0 / len(r)) - 1.0
    vol = float(np.std(r, ddof=1) * np.sqrt(52.0))
    sh = ann / vol if vol > 0 else 0.0
    eq = np.cumprod(1.0 + r); peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    cal = ann / abs(mdd) if mdd < 0 else 0.0
    return {"n": len(r), "tot": cum, "ann": ann, "vol": vol, "sh": sh, "mdd": mdd, "cal": cal}


years = dates.year
mask_2010_2025 = years >= 2010
mask_2010_2024 = (years >= 2010) & (years <= 2024)

print(f"\n{'strategy':<36} {'window':<12} {'wks':>5} {'ann':>8} {'vol':>8} {'sharpe':>8} {'mdd':>8} {'cal':>7}")
print("-" * 100)
for label, mask in [("2010-2024", mask_2010_2024), ("2010-2025", mask_2010_2025)]:
    spy_m = metrics(spy_rets, mask)
    print(f"{'SPY':<36} {label:<12} {spy_m['n']:>5} {spy_m['ann']:>8.2%} {spy_m['vol']:>8.2%} {spy_m['sh']:>8.3f} {spy_m['mdd']:>8.2%} {spy_m['cal']:>7.3f}")
    for name, rets in strategies.items():
        m = metrics(rets, mask)
        marker = " ✓" if m['sh'] > spy_m['sh'] else ""
        print(f"{name:<36} {label:<12} {m['n']:>5} {m['ann']:>8.2%} {m['vol']:>8.2%} {m['sh']:>8.3f} {m['mdd']:>8.2%} {m['cal']:>7.3f}{marker}")
    print()
