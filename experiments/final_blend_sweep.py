"""Final sweep: more blend variants + dynamic-mix exploration.

Strategies tested:
- v6 + SPY at various ratios (already know 50/50 best)
- v8 (60/40 EP/ROE) + SPY blend
- v9 (40/60 EP/ROE) + SPY blend
- x4 (defensive, K=100) + SPY blend
- Three-way: v6 + SPY + Tbills (cash)
- Equal-weight v6 picks + SPY blend (vs mcap-weight)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.io import processed_dir, repo_root

REPO_ROOT = repo_root()
BACKTEST_ROOT = REPO_ROOT / "artifacts" / "backtest_factor_v1"
BACKTEST_V2 = REPO_ROOT / "artifacts" / "backtest_factor_v2"
SPY_PATH = REPO_ROOT / "artifacts" / "benchmarks" / "spy_daily.parquet"

# Load all candidate strategies' weekly returns
def load_strat(name, root=BACKTEST_ROOT):
    p = root / f"weekly_{name}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


v6 = load_strat("v6_value_quality")
v8 = load_strat("v8_v6_60_40", root=BACKTEST_ROOT)
if v8 is None: v8 = load_strat("v8_mom_value")  # might be named differently
v9 = load_strat("v9_v6_40_60", root=BACKTEST_ROOT)
if v9 is None: v9 = load_strat("v9_qual_mom")

# x4 = ep_roe_roic K=100 from factor_v2_extended
x4 = load_strat("x4_ep_roe_roic__K100__mcap", root=BACKTEST_V2)

print("Loaded strategies:")
for nm, df in [("v6", v6), ("v8", v8), ("v9", v9), ("x4", x4)]:
    if df is not None:
        print(f"  {nm}: {len(df)} weeks, dates {df['date'].min().date()} → {df['date'].max().date()}")

dates = pd.DatetimeIndex(v6["date"])
# Build SPY weekly
spy = pd.read_parquet(SPY_PATH).reset_index()[["Date", "close"]].rename(columns={"Date": "date"})
spy["date"] = pd.to_datetime(spy["date"])
spy = spy.sort_values("date").set_index("date")["close"]
closes = spy.reindex(spy.index.union(dates)).sort_index().ffill().reindex(dates)
spy_rets = closes.pct_change().fillna(0.0).to_numpy()

# Build T-bill weekly
macro = pd.read_parquet(processed_dir() / "macro.parquet")
dgs3 = macro[macro["series"] == "DGS3MO"][["date", "value"]].copy()
dgs3["date"] = pd.to_datetime(dgs3["date"])
dgs3 = dgs3.sort_values("date").set_index("date")["value"] / 100.0
cash_y = dgs3.reindex(dgs3.index.union(dates)).sort_index().ffill().reindex(dates).fillna(0.04)
cash_rets = ((1.0 + cash_y) ** (1 / 52.0) - 1.0).to_numpy()


def metrics(rets, mask=None):
    r = np.asarray(rets, dtype=float)
    if mask is not None: r = r[mask]
    if len(r) < 2: return {}
    cum = float(np.prod(1.0 + r) - 1.0)
    ann = (1.0 + cum) ** (52.0 / len(r)) - 1.0
    vol = float(np.std(r, ddof=1) * np.sqrt(52.0))
    sh = ann / vol if vol > 0 else 0.0
    eq = np.cumprod(1.0 + r); peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    return {"ann": ann, "vol": vol, "sh": sh, "mdd": mdd}


years = dates.year
mask = years >= 2010

# Helper to align a strategy's returns to the canonical date index
def align(df):
    if df is None: return None
    return df.set_index("date").reindex(dates)["weekly_ret"].to_numpy()

v6_w = align(v6); v8_w = align(v8); v9_w = align(v9); x4_w = align(x4)

# Test many combinations
candidates = {}
# Pure
candidates["SPY only"] = spy_rets
candidates["v6 only"] = v6_w
# Two-way blends with SPY
for w in [0.4, 0.5, 0.6]:
    candidates[f"v6/{int(100*w)} + SPY/{int(100*(1-w))}"] = w * v6_w + (1 - w) * spy_rets
# Two-way blends v6 vs v8 vs v9 with SPY
if v8_w is not None:
    for w in [0.4, 0.5, 0.6]:
        candidates[f"v8/{int(100*w)} + SPY/{int(100*(1-w))}"] = w * v8_w + (1 - w) * spy_rets
if x4_w is not None:
    for w in [0.5, 0.6, 0.7]:
        candidates[f"x4/{int(100*w)} + SPY/{int(100*(1-w))}"] = w * x4_w + (1 - w) * spy_rets
# Three-way: v6 + SPY + cash
for w_v6, w_spy in [(0.40, 0.40), (0.45, 0.45), (0.50, 0.40), (0.40, 0.50), (0.50, 0.45)]:
    w_cash = 1.0 - w_v6 - w_spy
    candidates[f"v6/{int(100*w_v6)}+SPY/{int(100*w_spy)}+cash/{int(100*w_cash)}"] = (
        w_v6 * v6_w + w_spy * spy_rets + w_cash * cash_rets)
# Four-way: v6 + v8 + SPY + cash (diversification across active variants)
if v8_w is not None:
    candidates["v6/30+v8/30+SPY/40"] = 0.30 * v6_w + 0.30 * v8_w + 0.40 * spy_rets
    candidates["v6/25+v8/25+SPY/40+cash/10"] = 0.25 * v6_w + 0.25 * v8_w + 0.40 * spy_rets + 0.10 * cash_rets

# v6 + x4 + SPY (combining high-return + defensive + passive)
if x4_w is not None:
    candidates["v6/25+x4/25+SPY/50"] = 0.25 * v6_w + 0.25 * x4_w + 0.50 * spy_rets
    candidates["v6/30+x4/20+SPY/50"] = 0.30 * v6_w + 0.20 * x4_w + 0.50 * spy_rets
    candidates["v6/40+x4/10+SPY/50"] = 0.40 * v6_w + 0.10 * x4_w + 0.50 * spy_rets

# Sort and print
print(f"\n{'strategy':<45} {'ann':>8} {'vol':>8} {'sharpe':>8} {'mdd':>8}")
print("-" * 85)
rows = []
for name, r in candidates.items():
    m = metrics(r, mask)
    rows.append({"name": name, **m})
rows.sort(key=lambda x: -x["sh"])
for row in rows:
    marker = " ✓" if row['sh'] > metrics(spy_rets, mask)['sh'] else ""
    print(f"{row['name']:<45} {row['ann']:>8.2%} {row['vol']:>8.2%} {row['sh']:>8.3f} {row['mdd']:>8.2%}{marker}")
