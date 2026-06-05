"""Final K-sweep on sp_fcfa: does top-30 remain optimal?

Tests K = 20, 30, 40, 50, 75, 100 with mcap weighting + 10% cap.
For each K, also computes the SPY 50/50 blend.
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


def load_train(years):
    frames = []
    for y in years:
        for p in sorted((TRAIN_PANEL_DIR / f"year={y}").glob("*.parquet")):
            df = pd.read_parquet(p)[["permno", "date", "fwd_ret_5d"]]
            df["date"] = pd.to_datetime(df["date"])
            df["permno"] = df["permno"].astype("int64")
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


print("Loading panel + training_panel 2001-2025 ...")
daily = load_panel(range(2001, 2026), cols=[
    "permno", "date", "prc", "shrout", "marketcap", "in_universe",
    "revenue", "fcf", "assets"])
fri_panel = load_train(range(2002, 2026))
df = daily.merge(fri_panel, on=["permno", "date"], how="inner")
df = df.dropna(subset=["fwd_ret_5d"]).copy()
df = friday_only(df).reset_index(drop=True)
df = df[df["in_universe"]].copy()
df["mcap"] = df["marketcap"]
df.loc[df["mcap"].isna(), "mcap"] = (np.abs(df.loc[df["mcap"].isna(), "prc"]) *
                                      df.loc[df["mcap"].isna(), "shrout"])
df["sp"] = (df["revenue"] / df["mcap"]).clip(lower=0)
df["fcfa"] = (df["fcf"] / df["assets"]).clip(lower=-1.0, upper=2.0)
df.loc[df["assets"] <= 0, "fcfa"] = np.nan

for col in ["sp", "fcfa"]:
    g = df.groupby("date", sort=False)[col]
    df[f"z_{col}"] = (df[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    df[f"z_{col}"] = df[f"z_{col}"].fillna(0.0)
df["score"] = 0.5 * df["z_sp"] + 0.5 * df["z_fcfa"]
print(f"Done. {len(df)} Friday rows over {df['date'].nunique()} Fridays")


def backtest_K(K: int) -> np.ndarray:
    sb = (df.sort_values(["date", "score"], ascending=[True, False])
            .groupby("date", sort=False, group_keys=False)
            .head(K)
            .reset_index(drop=True))
    rets = []
    for d, g in sb.groupby("date"):
        g = g.reset_index(drop=True)
        mcaps = g["mcap"].to_numpy(dtype=np.float64)
        mcaps = np.where(np.isnan(mcaps), 0.0, mcaps)
        if mcaps.sum() <= 0:
            n = len(g); w = np.full(n, 1.0 / n)
        else:
            w = project_to_simplex(np.log(np.maximum(mcaps, EPS)), max_weight=MAX_WEIGHT)
        fwd = g["fwd_ret_5d"].to_numpy(dtype=np.float64)
        fwd = np.where(np.isnan(fwd), 0.0, fwd)
        rets.append({"date": d, "weekly_ret": float(np.dot(w, fwd))})
    out = pd.DataFrame(rets).sort_values("date").reset_index(drop=True)
    return out


dates_all = pd.DatetimeIndex(sorted(df["date"].unique()))
spy = pd.read_parquet(SPY_PATH).reset_index()[["Date", "close"]].rename(columns={"Date": "date"})
spy["date"] = pd.to_datetime(spy["date"])
spy = spy.sort_values("date").set_index("date")["close"]
closes = spy.reindex(spy.index.union(dates_all)).sort_index().ffill().reindex(dates_all)
spy_rets = closes.pct_change().fillna(0.0).to_numpy()
years = dates_all.year
mask = years >= 2010


def metrics(rets):
    r = np.asarray(rets, dtype=float)
    cum = float(np.prod(1.0 + r) - 1.0)
    ann = (1.0 + cum) ** (52.0 / len(r)) - 1.0
    vol = float(np.std(r, ddof=1) * np.sqrt(52.0))
    sh = ann / vol if vol > 0 else 0.0
    eq = np.cumprod(1.0 + r); peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    return {"ann": ann, "vol": vol, "sh": sh, "mdd": mdd}


print(f"\n=== K-sweep on sp_fcfa (2010-2025, with 50/50 SPY blend) ===")
print(f"  {'K':>4} {'alone_sh':>9} {'alone_ann':>10} {'blend_sh':>9} {'blend_ann':>10} {'blend_vol':>10} {'blend_mdd':>10}")
print("-" * 80)
for K in [10, 15, 20, 25, 30, 40, 50, 75, 100]:
    wk = backtest_K(K)
    sf_r = wk.set_index("date").reindex(dates_all)["weekly_ret"].to_numpy()
    blend_r = 0.5 * sf_r + 0.5 * spy_rets
    a = metrics(sf_r[mask]); b = metrics(blend_r[mask])
    print(f"  {K:>4} {a['sh']:>9.3f} {a['ann']:>10.2%} {b['sh']:>9.3f} {b['ann']:>10.2%} "
          f"{b['vol']:>10.2%} {b['mdd']:>10.2%}")
