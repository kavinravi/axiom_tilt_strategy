"""Per-year breakdown of v6_value_quality vs SPY. Sanity check on driver years.
Also: side-by-side cumulative growth (1$ invested 2010-01) and yearly Sharpe."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.io import repo_root

REPO_ROOT = repo_root()

w = pd.read_parquet(REPO_ROOT / "artifacts" / "backtest_factor_v1" / "weekly_v6_value_quality.parquet")
w["date"] = pd.to_datetime(w["date"])

spy_daily = pd.read_parquet(REPO_ROOT / "artifacts" / "benchmarks" / "spy_daily.parquet").reset_index()
spy_daily.columns = ["date", "close"] if list(spy_daily.columns)[:2] != ["date", "close"] else spy_daily.columns
spy_daily["date"] = pd.to_datetime(spy_daily.iloc[:, 0] if "date" not in spy_daily.columns else spy_daily["date"])
spy = spy_daily[["date", "close"]].sort_values("date").set_index("date")["close"]
dates = pd.DatetimeIndex(w["date"])
closes = spy.reindex(spy.index.union(dates)).sort_index().ffill().reindex(dates)
spy_rets = closes.pct_change().fillna(0.0).to_numpy()

w["spy_return"] = spy_rets
w["year"] = w["date"].dt.year

print(f"{'year':>6} {'wks':>5} {'v6_ret':>9} {'spy_ret':>9} {'v6_vol':>8} {'spy_vol':>8} {'v6_sh':>7} {'spy_sh':>7} {'edge':>7}")
print("-" * 80)
beats_sharpe = 0
beats_return = 0
total_years = 0
for year, sub in sorted(w[w["year"] >= 2009].groupby("year")):
    rets_v6 = sub["weekly_ret"].to_numpy()
    rets_spy = sub["spy_return"].to_numpy()
    v6_total = float(np.prod(1.0 + rets_v6) - 1.0)
    spy_total = float(np.prod(1.0 + rets_spy) - 1.0)
    v6_vol = float(np.std(rets_v6, ddof=1) * np.sqrt(52))
    spy_vol = float(np.std(rets_spy, ddof=1) * np.sqrt(52))
    v6_sh = (v6_total * 52 / len(rets_v6)) / v6_vol if v6_vol > 0 else 0
    spy_sh = (spy_total * 52 / len(rets_spy)) / spy_vol if spy_vol > 0 else 0
    edge = v6_sh - spy_sh
    print(f"{year:>6} {len(sub):>5} {v6_total:>9.2%} {spy_total:>9.2%} {v6_vol:>8.2%} {spy_vol:>8.2%} "
          f"{v6_sh:>7.3f} {spy_sh:>7.3f} {edge:>+7.3f}")
    total_years += 1
    beats_sharpe += int(v6_sh > spy_sh)
    beats_return += int(v6_total > spy_total)

print(f"\nv6 beats SPY on annual Sharpe: {beats_sharpe}/{total_years} years")
print(f"v6 beats SPY on annual return: {beats_return}/{total_years} years")

# Cumulative growth of $1 from 2010-01
w2 = w[w["year"] >= 2010].copy().sort_values("date").reset_index(drop=True)
w2["v6_eq"] = (1 + w2["weekly_ret"]).cumprod()
w2["spy_eq"] = (1 + w2["spy_return"]).cumprod()
print(f"\nCumulative growth of $1 (2010-01-01 → 2025-12-19):")
print(f"  v6_value_quality: ${w2['v6_eq'].iloc[-1]:.2f}")
print(f"  SPY:              ${w2['spy_eq'].iloc[-1]:.2f}")
print(f"  v6 / SPY ratio:   {w2['v6_eq'].iloc[-1] / w2['spy_eq'].iloc[-1]:.2f}x")

# Information ratio of v6 vs SPY (active returns / tracking error)
active = w2["weekly_ret"] - w2["spy_return"]
ir = (active.mean() * 52) / (active.std(ddof=1) * np.sqrt(52))
print(f"\nInformation ratio (v6 vs SPY, 2010-2025): {ir:.3f}")
print(f"Mean active return (annualized):          {active.mean()*52:+.2%}")
print(f"Tracking error (annualized):              {active.std(ddof=1)*np.sqrt(52):.2%}")
print(f"% weeks v6 > SPY:                          {(active > 0).mean():.1%}")
