"""Out-of-sample robustness check: does v6_spy_50_50 also beat SPY on 2002-2009?

The strategy has no learned parameters (factor weights are 50/50 hardcoded;
mcap weighting is deterministic). So 2002-2009 should be a valid OOS test
even though our walk artifacts start at walk-1=2009. We just need to compute
the strategy on the 2002-2008 panel data directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.io import processed_dir, repo_root
from src.utils.ranker import friday_only
from src.utils.rl_env import project_to_simplex

REPO_ROOT = repo_root()
PANEL_DIR = processed_dir() / "panel"
TRAIN_PANEL_DIR = processed_dir() / "training_panel"
SPY_PATH = REPO_ROOT / "artifacts" / "benchmarks" / "spy_daily.parquet"
TOP_K = 30
MAX_WEIGHT = 0.10
EPS = 1e-8


def load_panel(years, cols):
    frames = []
    for y in years:
        for p in sorted((PANEL_DIR / f"year={y}").glob("*.parquet")):
            df = pd.read_parquet(p, columns=cols)
            df["date"] = pd.to_datetime(df["date"])
            df["permno"] = df["permno"].astype("int64")
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_train_panel(years):
    frames = []
    for y in years:
        for p in sorted((TRAIN_PANEL_DIR / f"year={y}").glob("*.parquet")):
            df = pd.read_parquet(p)[["permno", "date", "fwd_ret_5d"]]
            df["date"] = pd.to_datetime(df["date"])
            df["permno"] = df["permno"].astype("int64")
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# Build v6 picks for 2002-2008
daily = load_panel(range(2001, 2010), cols=[
    "permno", "date", "prc", "shrout", "marketcap", "in_universe",
    "netinc", "equity"])
fri_panel = load_train_panel(range(2002, 2010))

df = daily.merge(fri_panel, on=["permno", "date"], how="inner")
df = df.dropna(subset=["fwd_ret_5d"]).copy()
df = friday_only(df).reset_index(drop=True)
df = df[df["in_universe"]].copy()

df["mcap"] = df["marketcap"]
df.loc[df["mcap"].isna(), "mcap"] = (np.abs(df.loc[df["mcap"].isna(), "prc"]) *
                                      df.loc[df["mcap"].isna(), "shrout"])
df["ep"] = (df["netinc"] / df["mcap"]).clip(lower=0)
df["roe"] = (df["netinc"] / df["equity"]).clip(lower=-1.0, upper=2.0)
df.loc[df["equity"] <= 0, "roe"] = np.nan

# Per-Friday z-scores (transform-style, not apply)
for col_in, col_out in [("ep", "z_ep"), ("roe", "z_roe")]:
    g = df.groupby("date", sort=False)[col_in]
    df[col_out] = (df[col_in] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    df[col_out] = df[col_out].fillna(0.0)
df["score"] = 0.5 * df["z_ep"] + 0.5 * df["z_roe"]

# Top-30 per Friday
sb = (df.dropna(subset=["score"])
        .sort_values(["date", "score"], ascending=[True, False])
        .groupby("date", sort=False, group_keys=False)
        .head(TOP_K)
        .reset_index(drop=True))

# Per-Friday mcap-weighted backtest
by_date = {d: g.reset_index(drop=True) for d, g in sb.groupby("date")}
dates_sorted = sorted(by_date.keys())
rets = []
for d in dates_sorted:
    cur = by_date[d]
    mcaps = cur["mcap"].to_numpy(dtype=np.float64)[:TOP_K]
    mcaps = np.where(np.isnan(mcaps), 0.0, mcaps)
    if mcaps.sum() <= 0:
        w = np.full(TOP_K, 1.0 / TOP_K)
    else:
        w = project_to_simplex(np.log(np.maximum(mcaps, EPS)), max_weight=MAX_WEIGHT)
    fwd = cur["fwd_ret_5d"].to_numpy(dtype=np.float64)[:TOP_K]
    fwd = np.where(np.isnan(fwd), 0.0, fwd)
    rets.append(float(np.dot(w, fwd)))

v6_rets = np.array(rets)
dates = pd.DatetimeIndex(dates_sorted)

# Get SPY weekly returns over same dates
spy = pd.read_parquet(SPY_PATH).reset_index()[["Date", "close"]].rename(columns={"Date": "date"})
spy["date"] = pd.to_datetime(spy["date"])
spy = spy.sort_values("date").set_index("date")["close"]
closes = spy.reindex(spy.index.union(dates)).sort_index().ffill().reindex(dates)
spy_rets = closes.pct_change().fillna(0.0).to_numpy()

blend = 0.5 * v6_rets + 0.5 * spy_rets

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
    return {"n": len(r), "ann": ann, "vol": vol, "sh": sh, "mdd": mdd, "tot": cum}


print(f"\n=== 2002-2009 (PRE-OOS-WINDOW robustness check, includes GFC) ===")
years = dates.year
for label, mask in [("2002-2009 (full)", np.ones(len(dates), dtype=bool)),
                    ("2002-2007", (years >= 2002) & (years <= 2007)),
                    ("2008-2009 (GFC + recovery)", (years >= 2008) & (years <= 2009))]:
    print(f"\n--- {label} ---")
    vm = metrics(v6_rets, mask); spm = metrics(spy_rets, mask); blm = metrics(blend, mask)
    print(f"  {'strategy':<20} {'wks':>4} {'ann':>8} {'vol':>8} {'sharpe':>8} {'mdd':>8}")
    print(f"  {'v6 (pure)':<20} {vm['n']:>4} {vm['ann']:>8.2%} {vm['vol']:>8.2%} {vm['sh']:>8.3f} {vm['mdd']:>8.2%}")
    print(f"  {'SPY':<20} {spm['n']:>4} {spm['ann']:>8.2%} {spm['vol']:>8.2%} {spm['sh']:>8.3f} {spm['mdd']:>8.2%}")
    print(f"  {'blend 50/50':<20} {blm['n']:>4} {blm['ann']:>8.2%} {blm['vol']:>8.2%} {blm['sh']:>8.3f} {blm['mdd']:>8.2%}")
