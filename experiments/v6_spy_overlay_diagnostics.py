"""Diagnostics on v6_spy_overlay_50_50 — the winning strategy.

Checks:
1. Per-year breakdown vs SPY (does it win every year? noisy?)
2. Correlation between v6 and SPY excess returns (explains why blend works)
3. Drawdown decomposition (when does the worst drawdown happen?)
4. Information ratio + tracking error
5. Sweep mix weights from 0% to 100% v6 to find Sharpe-optimal blend
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.io import repo_root

REPO_ROOT = repo_root()
v6 = pd.read_parquet(REPO_ROOT / "artifacts" / "backtest_factor_v1" / "weekly_v6_value_quality.parquet")
v6["date"] = pd.to_datetime(v6["date"])
v6 = v6.sort_values("date").reset_index(drop=True)

# Build SPY weekly aligned
spy_daily = pd.read_parquet(REPO_ROOT / "artifacts" / "benchmarks" / "spy_daily.parquet").reset_index()
spy_daily.columns = list(spy_daily.columns)
spy = spy_daily[["Date", "close"]].rename(columns={"Date": "date"})
spy["date"] = pd.to_datetime(spy["date"])
spy = spy.sort_values("date").set_index("date")["close"]
dates = pd.DatetimeIndex(v6["date"])
closes = spy.reindex(spy.index.union(dates)).sort_index().ffill().reindex(dates)
spy_w = closes.pct_change().fillna(0.0).to_numpy()
v6_w = v6["weekly_ret"].to_numpy()

# 50/50 blend
blend = 0.5 * v6_w + 0.5 * spy_w

# ---- 1) Per-year breakdown ----
df = pd.DataFrame({"date": dates, "v6": v6_w, "spy": spy_w, "blend": blend})
df["year"] = df["date"].dt.year

print(f"\n{'year':>6} {'wks':>4} {'v6_ret':>9} {'spy_ret':>9} {'blend_ret':>10} {'blend_vol':>10} {'blend_sh':>9} {'spy_sh':>8}")
print("-" * 80)
beats_blend = 0
total = 0
for year, sub in sorted(df[df["year"] >= 2010].groupby("year")):
    v6r = float((1 + sub["v6"]).prod() - 1)
    spr = float((1 + sub["spy"]).prod() - 1)
    blr = float((1 + sub["blend"]).prod() - 1)
    blvol = float(sub["blend"].std(ddof=1) * np.sqrt(52))
    spvol = float(sub["spy"].std(ddof=1) * np.sqrt(52))
    blsh = blr / blvol if blvol > 0 else 0
    spsh = spr / spvol if spvol > 0 else 0
    print(f"{year:>6} {len(sub):>4} {v6r:>9.2%} {spr:>9.2%} {blr:>10.2%} {blvol:>10.2%} {blsh:>9.3f} {spsh:>8.3f}")
    total += 1
    beats_blend += int(blr > spr)

print(f"\nblend (50/50) beats SPY on return: {beats_blend}/{total} years (2010-2025)")

# ---- 2) Correlation analysis ----
print(f"\n=== Correlation analysis (2010-2025) ===")
sub = df[df["year"] >= 2010]
corr_returns = sub[["v6", "spy"]].corr().iloc[0, 1]
print(f"  corr(v6_return, spy_return)        : {corr_returns:.4f}")
excess_v6 = sub["v6"].to_numpy() - sub["spy"].to_numpy()
print(f"  v6 excess return vol (annualized) : {excess_v6.std(ddof=1) * np.sqrt(52):.2%}")
print(f"  v6 mean excess (annualized)       : {excess_v6.mean() * 52:+.2%}")
print(f"  IR (info ratio v6 vs SPY)         : {(excess_v6.mean() * 52) / (excess_v6.std(ddof=1) * np.sqrt(52)):.3f}")

# ---- 3) Drawdown decomposition ----
def drawdown_periods(rets, dates_arr, top_n=3):
    eq = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1.0
    # Find largest drawdown periods
    in_dd = dd < -0.05
    periods = []
    cur_start = None
    for i, isdd in enumerate(in_dd):
        if isdd and cur_start is None:
            cur_start = i
        elif not isdd and cur_start is not None:
            seg = dd[cur_start:i]
            min_idx = int(np.argmin(seg))
            periods.append({
                "start": dates_arr[cur_start], "end": dates_arr[i-1],
                "trough_date": dates_arr[cur_start + min_idx],
                "max_dd": float(seg.min()),
                "duration_weeks": i - cur_start,
            })
            cur_start = None
    if cur_start is not None:
        seg = dd[cur_start:]
        min_idx = int(np.argmin(seg))
        periods.append({
            "start": dates_arr[cur_start], "end": dates_arr[-1],
            "trough_date": dates_arr[cur_start + min_idx],
            "max_dd": float(seg.min()),
            "duration_weeks": len(seg),
        })
    periods.sort(key=lambda x: x["max_dd"])
    return periods[:top_n]

print(f"\n=== Top 3 drawdown periods for blend (50/50) on 2010-2025 ===")
sub_arr = sub.reset_index(drop=True)
for p in drawdown_periods(sub_arr["blend"].to_numpy(), pd.to_datetime(sub_arr["date"]).dt.date.to_numpy()):
    print(f"  {p['start']} → {p['trough_date']} → {p['end']}: MDD {p['max_dd']:.2%}, duration {p['duration_weeks']} weeks")

print(f"\n=== Top 3 drawdown periods for SPY on 2010-2025 ===")
for p in drawdown_periods(sub_arr["spy"].to_numpy(), pd.to_datetime(sub_arr["date"]).dt.date.to_numpy()):
    print(f"  {p['start']} → {p['trough_date']} → {p['end']}: MDD {p['max_dd']:.2%}, duration {p['duration_weeks']} weeks")

# ---- 4) Mix weight sweep ----
print(f"\n=== Mix-weight sweep (2010-2025): find Sharpe-optimal v6/SPY mix ===")
print(f"  {'w_v6':>6} {'ann':>8} {'vol':>8} {'sharpe':>8} {'mdd':>8}")
weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
sub_arr = df[df["year"] >= 2010].reset_index(drop=True)
v6_only = sub_arr["v6"].to_numpy()
spy_only = sub_arr["spy"].to_numpy()
best_sh, best_w = 0, 0
for w in weights:
    r = w * v6_only + (1 - w) * spy_only
    cum = float(np.prod(1 + r) - 1.0)
    ann = (1 + cum) ** (52 / len(r)) - 1.0
    vol = float(np.std(r, ddof=1) * np.sqrt(52))
    sh = ann / vol if vol > 0 else 0
    eq = np.cumprod(1 + r); peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    marker = " ← Sharpe-max" if sh > best_sh else ""
    if sh > best_sh:
        best_sh = sh; best_w = w
    print(f"  {w:>6.2f} {ann:>8.2%} {vol:>8.2%} {sh:>8.3f} {mdd:>8.2%}{marker}")
print(f"\nSharpe-optimal mix: w_v6 = {best_w:.2f}, Sharpe = {best_sh:.3f}")

# Save final result
df.to_parquet(REPO_ROOT / "artifacts" / "backtest_factor_v1" / "weekly_v6_spy_overlay_50_50.parquet",
              compression="zstd", index=False)
print(f"\nSaved per-week returns to artifacts/backtest_factor_v1/weekly_v6_spy_overlay_50_50.parquet")
