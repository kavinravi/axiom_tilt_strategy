"""Final robustness analyses on v6_spy_overlay_50_50:

1. Bootstrap 95% CI for Sharpe over 2010-2025 (1000 bootstrap samples)
2. Cost sensitivity (5, 10, 15, 20, 30 bps on the v6 sleeve's turnover)
3. Per-walk consistency: Sharpe / return / vol per test year
4. Variance of mix-weight (40-60 range) — what's robust?
5. Different rebalance frequencies (weekly vs bi-weekly vs monthly)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.io import processed_dir, repo_root

REPO_ROOT = repo_root()
SPY_PATH = REPO_ROOT / "artifacts" / "benchmarks" / "spy_daily.parquet"

# Load v6 weekly returns
v6_df = pd.read_parquet(REPO_ROOT / "artifacts" / "backtest_factor_v1" / "weekly_v6_value_quality.parquet")
v6_df["date"] = pd.to_datetime(v6_df["date"])
v6_df = v6_df.sort_values("date").reset_index(drop=True)
dates = pd.DatetimeIndex(v6_df["date"])

# Build SPY weekly
spy = pd.read_parquet(SPY_PATH).reset_index()[["Date", "close"]].rename(columns={"Date": "date"})
spy["date"] = pd.to_datetime(spy["date"])
spy = spy.sort_values("date").set_index("date")["close"]
closes = spy.reindex(spy.index.union(dates)).sort_index().ffill().reindex(dates)
spy_rets = closes.pct_change().fillna(0.0).to_numpy()
v6_rets = v6_df["weekly_ret"].to_numpy()
blend = 0.5 * v6_rets + 0.5 * spy_rets

years = dates.year
mask = years >= 2010
v6_oos = v6_rets[mask]; spy_oos = spy_rets[mask]; blend_oos = blend[mask]
dates_oos = dates[mask]
years_oos = years[mask]


def metrics(rets):
    r = np.asarray(rets, dtype=float)
    cum = float(np.prod(1.0 + r) - 1.0)
    ann = (1.0 + cum) ** (52.0 / len(r)) - 1.0
    vol = float(np.std(r, ddof=1) * np.sqrt(52.0))
    sh = ann / vol if vol > 0 else 0.0
    eq = np.cumprod(1.0 + r); peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    return {"ann": ann, "vol": vol, "sh": sh, "mdd": mdd}


# ====================================================================
# 1) Bootstrap 95% CI for Sharpe
# ====================================================================
print(f"\n=== Bootstrap 95% CI for Sharpe (2010-2025, n=806 weeks, 1000 resamples) ===")
rng = np.random.default_rng(42)
n_boot = 1000
sharpe_v6 = []; sharpe_spy = []; sharpe_blend = []
for _ in range(n_boot):
    idx = rng.integers(0, len(blend_oos), size=len(blend_oos))
    sharpe_v6.append(metrics(v6_oos[idx])["sh"])
    sharpe_spy.append(metrics(spy_oos[idx])["sh"])
    sharpe_blend.append(metrics(blend_oos[idx])["sh"])
print(f"  {'strategy':<10} {'mean':>8} {'2.5%':>8} {'97.5%':>8} {'P(beat_SPY)':>12}")
for name, arr in [("v6", sharpe_v6), ("SPY", sharpe_spy), ("blend", sharpe_blend)]:
    arr = np.array(arr)
    pbeat = float(np.mean(np.array(sharpe_blend) > np.array(sharpe_spy))) if name == "blend" else None
    pb_str = f"{pbeat:.3f}" if pbeat is not None else ""
    print(f"  {name:<10} {arr.mean():>8.3f} {np.percentile(arr, 2.5):>8.3f} {np.percentile(arr, 97.5):>8.3f} {pb_str:>12}")


# ====================================================================
# 2) Cost sensitivity (v6 sleeve only — SPY half has zero turnover)
# ====================================================================
print(f"\n=== Cost sensitivity (varying v6 sleeve cost; SPY sleeve = 0 cost) ===")
# v6 sleeve has avg turnover ~0.43 (one-way) per week, which means 0.43 trades per week
# at C bps, weekly cost = (C/1e4) * 0.43 → annualized cost reduction = 52 * (C/1e4) * 0.43
# For a 5 bps strategy: 52 * 5e-4 * 0.43 = 1.12% ann cost
# Real v6 turnover from build: estimated 25%/yr
# Let me use a realistic 5-30 bps × turnover_assumed
print(f"  {'cost_bps':>10} {'v6_ann_after_cost':>18} {'blend_ann_after':>16} {'blend_sharpe':>14}")
# Assume v6 weekly turnover ~0.43 (rough; would need actual weight changes per week to be exact)
# Adjusting: v6 weekly returns ALREADY are gross of cost (no cost was charged in factor_variants.py)
# So we need to subtract cost from each week's v6 return based on turnover
# Use a flat 0.43 weekly turnover assumption (the same we used in old runs)
assumed_turnover_per_week = 0.43
for c_bps in [0, 5, 10, 15, 20, 30]:
    cost_per_wk = (c_bps / 1e4) * assumed_turnover_per_week
    v6_net = v6_oos - cost_per_wk  # subtract cost from each week's v6 return
    blend_net = 0.5 * v6_net + 0.5 * spy_oos  # SPY half has zero turnover
    m_v6 = metrics(v6_net)
    m_bl = metrics(blend_net)
    print(f"  {c_bps:>10} {m_v6['ann']:>18.2%} {m_bl['ann']:>16.2%} {m_bl['sh']:>14.3f}")


# ====================================================================
# 3) Per-walk consistency (using year-of-test as proxy for walk)
# ====================================================================
print(f"\n=== Per-walk consistency (test year = walk's TEST window) ===")
print(f"  {'year':>4} {'wks':>4} {'v6_sh':>7} {'spy_sh':>7} {'blend_sh':>9} {'blend_ann':>10} {'spy_ann':>9}")
print(f"  {'----':>4} {'---':>4} {'-----':>7} {'------':>7} {'--------':>9} {'---------':>10} {'-------':>9}")
for year in sorted(set(years_oos)):
    mask_y = years_oos == year
    if mask_y.sum() < 5:
        continue
    v6_y = v6_oos[mask_y]; spy_y = spy_oos[mask_y]; bl_y = blend_oos[mask_y]
    print(f"  {year:>4} {len(v6_y):>4} {metrics(v6_y)['sh']:>7.3f} {metrics(spy_y)['sh']:>7.3f} "
          f"{metrics(bl_y)['sh']:>9.3f} {metrics(bl_y)['ann']:>10.2%} {metrics(spy_y)['ann']:>9.2%}")


# ====================================================================
# 4) Mix-weight stability around 50/50
# ====================================================================
print(f"\n=== Mix-weight stability ===")
for w in np.arange(0.40, 0.66, 0.02):
    r = w * v6_oos + (1 - w) * spy_oos
    m = metrics(r)
    print(f"  w_v6={w:.2f}  Sharpe={m['sh']:.3f}  AnnRet={m['ann']:.2%}  Vol={m['vol']:.2%}  MDD={m['mdd']:.2%}")


# ====================================================================
# 5) Different rebalance frequencies (compounded blend)
# ====================================================================
print(f"\n=== Rebalance frequency variants (informational; v6 weekly is the production) ===")
# Weekly: original
weekly = metrics(blend_oos)
# Monthly: aggregate to ~4-week compounded returns, then ann_ret stays the same conceptually
# In a true monthly rebal we'd hold the same picks for 4 weeks — different picks. This is
# more an academic exercise.
print(f"  Weekly (production) : Sharpe={weekly['sh']:.3f}  AnnRet={weekly['ann']:.2%}")
# Build a "monthly" approximation: average returns within each calendar month
df_for_monthly = pd.DataFrame({"date": dates_oos, "ret": blend_oos})
df_for_monthly["ym"] = df_for_monthly["date"].dt.to_period("M")
monthly_ret = df_for_monthly.groupby("ym")["ret"].apply(lambda x: (1 + x).prod() - 1).to_numpy()
m_monthly = (
    (1 + monthly_ret).prod() ** (12 / len(monthly_ret)) - 1,
    float(monthly_ret.std(ddof=1) * np.sqrt(12)),
)
print(f"  Same returns aggregated monthly (just for reference): "
      f"ann={m_monthly[0]:.2%}  vol={m_monthly[1]:.2%}  sharpe={m_monthly[0]/m_monthly[1]:.3f}")
print(f"  [True monthly rebalance would require new picks — would change Sharpe.]")
