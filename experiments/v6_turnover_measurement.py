"""Measure ACTUAL weekly turnover of the v6 strategy (vs the 0.43 assumption).

For each Friday t, compute weights[t]. Turnover = sum(|weights[t] - weights[t-1]|) / 2,
where missing positions (stocks not in top-30 at t but were at t-1) get weight 0.

This is the standard portfolio turnover metric — half the L1 distance between
consecutive weights, accounting for stocks dropping in/out of top-30.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.io import processed_dir, repo_root
from src.utils.rl_env import project_to_simplex

REPO_ROOT = repo_root()
PANEL_DIR = processed_dir() / "panel"
TRAIN_PANEL_DIR = processed_dir() / "training_panel"
TOP_K = 30
MAX_WEIGHT = 0.10
EPS = 1e-8


def mcap_weights(mcaps):
    safe = np.maximum(mcaps, EPS)
    return project_to_simplex(np.log(safe), max_weight=MAX_WEIGHT)


# Load all v6 scoreboards (test year = walk_id)
all_weeks = []
for walk_id in range(1, 18):
    sb = pd.read_parquet(REPO_ROOT / "artifacts" / "rl_factor_v6" / f"walk-{walk_id:03d}" / "scoreboard.parquet")
    sb["date"] = pd.to_datetime(sb["date"])
    sb["permno"] = sb["permno"].astype("int64")
    test_year = 2008 + walk_id
    sb = sb[(sb["date"] >= f"{test_year}-01-01") & (sb["date"] <= f"{test_year}-12-31")]
    all_weeks.append(sb)
sb_all = pd.concat(all_weeks, ignore_index=True).sort_values(["date", "score"], ascending=[True, False]).reset_index(drop=True)
sb_all = sb_all.groupby("date", sort=False, group_keys=False).head(TOP_K).reset_index(drop=True)
print(f"Loaded {len(sb_all)} rows across {sb_all['date'].nunique()} Fridays")

# Per-Friday weights as a dict
by_date = sb_all.groupby("date")
weights_per_date = {}
for d, g in by_date:
    g = g.reset_index(drop=True)
    mcaps = g["mcap"].to_numpy(dtype=np.float64)
    w = mcap_weights(mcaps)
    weights_per_date[d] = dict(zip(g["permno"].astype(int).tolist(), w.tolist()))

dates_sorted = sorted(weights_per_date.keys())
turnovers = []
for i in range(1, len(dates_sorted)):
    d_prev = dates_sorted[i - 1]
    d_cur = dates_sorted[i]
    prev_w = weights_per_date[d_prev]
    cur_w = weights_per_date[d_cur]
    all_permnos = set(prev_w.keys()) | set(cur_w.keys())
    diff = sum(abs(cur_w.get(p, 0.0) - prev_w.get(p, 0.0)) for p in all_permnos)
    turnover = diff / 2  # standard turnover = half-L1
    turnovers.append({"date": d_cur, "turnover": turnover})

t_df = pd.DataFrame(turnovers)
t_df["year"] = t_df["date"].dt.year
t_df_oos = t_df[t_df["year"] >= 2010]

print(f"\n=== v6 actual turnover (one-way per week, 2010-2025) ===")
print(f"  mean   : {t_df_oos['turnover'].mean():.4f}")
print(f"  median : {t_df_oos['turnover'].median():.4f}")
print(f"  std    : {t_df_oos['turnover'].std():.4f}")
print(f"  p10    : {t_df_oos['turnover'].quantile(0.10):.4f}")
print(f"  p90    : {t_df_oos['turnover'].quantile(0.90):.4f}")
print(f"  max    : {t_df_oos['turnover'].max():.4f}")

annual = t_df_oos.groupby("year")["turnover"].mean() * 52
print(f"\n=== Annualized turnover by year (one-way) ===")
for y, t in annual.items():
    print(f"  {y}: {t:.2%}")
print(f"\n  Mean annualized one-way turnover: {annual.mean():.2%}")
print(f"  → equivalent round-trip = 2x = {annual.mean()*2:.2%}")
print(f"  → Cost @ 5bps: {annual.mean() * 2 * 0.0005 * 100:.2f}% drag/year")
print(f"  → Cost @ 10bps: {annual.mean() * 2 * 0.001 * 100:.2f}% drag/year")

# Recompute the cost-sensitivity with REAL turnover
print(f"\n=== Cost sensitivity with REAL turnover ===")
real_turnover_per_week = t_df_oos["turnover"].mean()
print(f"  using actual weekly turnover = {real_turnover_per_week:.4f}")
v6_df = pd.read_parquet(REPO_ROOT / "artifacts" / "backtest_factor_v1" / "weekly_v6_value_quality.parquet")
v6_df["date"] = pd.to_datetime(v6_df["date"])
v6_df = v6_df.sort_values("date").reset_index(drop=True)
v6_rets = v6_df["weekly_ret"].to_numpy()

from src.utils.io import processed_dir
SPY_PATH = REPO_ROOT / "artifacts" / "benchmarks" / "spy_daily.parquet"
spy = pd.read_parquet(SPY_PATH).reset_index()[["Date", "close"]].rename(columns={"Date": "date"})
spy["date"] = pd.to_datetime(spy["date"])
spy = spy.sort_values("date").set_index("date")["close"]
dates = pd.DatetimeIndex(v6_df["date"])
closes = spy.reindex(spy.index.union(dates)).sort_index().ffill().reindex(dates)
spy_rets = closes.pct_change().fillna(0.0).to_numpy()
years_arr = dates.year
mask = years_arr >= 2010


def metrics(r):
    cum = float(np.prod(1.0 + r) - 1.0)
    ann = (1.0 + cum) ** (52.0 / len(r)) - 1.0
    vol = float(np.std(r, ddof=1) * np.sqrt(52.0))
    sh = ann / vol if vol > 0 else 0.0
    eq = np.cumprod(1.0 + r); peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    return {"ann": ann, "vol": vol, "sh": sh, "mdd": mdd}


print(f"  {'cost_bps':>10} {'v6_ann':>10} {'blend_ann':>11} {'blend_sharpe':>14} {'blend_mdd':>11}")
for c_bps in [0, 5, 10, 15, 20, 30]:
    cost_per_wk = (c_bps / 1e4) * real_turnover_per_week
    v6_net = v6_rets[mask] - cost_per_wk
    blend_net = 0.5 * v6_net + 0.5 * spy_rets[mask]
    m = metrics(blend_net); mv6 = metrics(v6_net)
    print(f"  {c_bps:>10} {mv6['ann']:>10.2%} {m['ann']:>11.2%} {m['sh']:>14.3f} {m['mdd']:>11.2%}")
