"""Train-test split robustness on the (V × Q) factor pair selection.

Method:
  1. Compute weekly returns for ALL 36 (V × Q) factor pairs across 2002-2025.
  2. Use 2002-2014 to pick the best pair by blend Sharpe.
  3. VERIFY that pair on 2015-2025 (true holdout).
  4. Report whether the winner generalizes.

If sp_fcfa was the winner on training and ALSO wins on holdout, the
selection wasn't an OOS artifact. If something else wins on training,
report what it would have done on holdout.
"""
from __future__ import annotations

import itertools
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
    "netinc", "equity", "revenue", "gp", "fcf", "ebit", "invcap",
    "ebitda", "ev", "assets"])
fri_panel = load_train(range(2002, 2026))
df = daily.merge(fri_panel, on=["permno", "date"], how="inner")
df = df.dropna(subset=["fwd_ret_5d"]).copy()
df = friday_only(df).reset_index(drop=True)
df = df[df["in_universe"]].copy()
df["mcap"] = df["marketcap"]
df.loc[df["mcap"].isna(), "mcap"] = (np.abs(df.loc[df["mcap"].isna(), "prc"]) *
                                      df.loc[df["mcap"].isna(), "shrout"])

# Define all factors
df["ep"]        = (df["netinc"] / df["mcap"]).clip(lower=0)
df["bp"]        = (df["equity"] / df["mcap"]).clip(lower=0)
df["sp"]        = (df["revenue"] / df["mcap"]).clip(lower=0)
df["fcfp"]      = (df["fcf"] / df["mcap"]).clip(lower=0)
df["ebitda_ev"] = (df["ebitda"] / df["ev"]).clip(lower=0)
df.loc[df["ev"] <= 0, "ebitda_ev"] = np.nan
df["e_ev"]      = (df["netinc"] / df["ev"]).clip(lower=0)
df.loc[df["ev"] <= 0, "e_ev"] = np.nan
df["roe"]       = (df["netinc"] / df["equity"]).clip(lower=-1.0, upper=2.0)
df.loc[df["equity"] <= 0, "roe"] = np.nan
df["roa"]       = (df["netinc"] / df["assets"]).clip(lower=-1.0, upper=2.0)
df.loc[df["assets"] <= 0, "roa"] = np.nan
df["roic"]      = (df["ebit"] / df["invcap"]).clip(lower=-1.0, upper=2.0)
df.loc[df["invcap"] <= 0, "roic"] = np.nan
df["gpa"]       = (df["gp"] / df["assets"]).clip(lower=-1.0, upper=2.0)
df.loc[df["assets"] <= 0, "gpa"] = np.nan
df["gross_margin"] = (df["gp"] / df["revenue"]).clip(lower=-1.0, upper=2.0)
df.loc[df["revenue"] <= 0, "gross_margin"] = np.nan
df["fcfa"]      = (df["fcf"] / df["assets"]).clip(lower=-1.0, upper=2.0)
df.loc[df["assets"] <= 0, "fcfa"] = np.nan

VALUE = ["ep", "bp", "sp", "fcfp", "ebitda_ev", "e_ev"]
QUALITY = ["roe", "roa", "roic", "gpa", "gross_margin", "fcfa"]

# Compute z-scores per-Friday
print("Computing per-Friday z-scores ...")
for col in VALUE + QUALITY:
    g = df.groupby("date", sort=False)[col]
    df[f"z_{col}"] = (df[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    df[f"z_{col}"] = df[f"z_{col}"].fillna(0.0)


def backtest_pair(vf: str, qf: str) -> np.ndarray:
    """Returns per-date weekly returns for the (vf, qf) factor pair."""
    df2 = df.copy()
    df2["score"] = 0.5 * df2[f"z_{vf}"] + 0.5 * df2[f"z_{qf}"]
    sb = (df2.sort_values(["date", "score"], ascending=[True, False])
             .groupby("date", sort=False, group_keys=False)
             .head(TOP_K)
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
        rets.append({"date": d, "ret": float(np.dot(w, fwd))})
    return pd.DataFrame(rets).sort_values("date").reset_index(drop=True)


# Compute all 36 pairs
print(f"\nComputing 36 V × Q pair returns ...")
all_dates = pd.DatetimeIndex(sorted(df["date"].unique()))
pair_returns = {}
for vf in VALUE:
    for qf in QUALITY:
        wk = backtest_pair(vf, qf)
        r = wk.set_index("date").reindex(all_dates)["ret"].to_numpy()
        pair_returns[(vf, qf)] = r
print(f"Done. {len(pair_returns)} pairs over {len(all_dates)} weeks")

# SPY weekly
spy = pd.read_parquet(SPY_PATH).reset_index()[["Date", "close"]].rename(columns={"Date": "date"})
spy["date"] = pd.to_datetime(spy["date"])
spy = spy.sort_values("date").set_index("date")["close"]
closes = spy.reindex(spy.index.union(all_dates)).sort_index().ffill().reindex(all_dates)
spy_rets = closes.pct_change().fillna(0.0).to_numpy()

years = all_dates.year
train_mask = (years >= 2002) & (years <= 2014)
test_mask = (years >= 2015) & (years <= 2025)


def metrics(rets):
    cum = float(np.prod(1.0 + rets) - 1.0)
    ann = (1.0 + cum) ** (52.0 / len(rets)) - 1.0
    vol = float(np.std(rets, ddof=1) * np.sqrt(52.0))
    sh = ann / vol if vol > 0 else 0.0
    return {"ann": ann, "vol": vol, "sh": sh}


# RANK all pairs by TRAIN (2002-2014) blend Sharpe
print(f"\n=== TRAIN (2002-2014, {train_mask.sum()} weeks) ranking ===")
spy_train = metrics(spy_rets[train_mask])
print(f"  SPY                                  : Sharpe = {spy_train['sh']:.3f}  ann = {spy_train['ann']:.2%}")

train_results = []
for (vf, qf), r in pair_returns.items():
    blend = 0.5 * r + 0.5 * spy_rets
    m_tr = metrics(blend[train_mask])
    m_te = metrics(blend[test_mask])
    train_results.append({
        "value": vf, "quality": qf,
        "train_sh": m_tr["sh"], "train_ann": m_tr["ann"],
        "test_sh": m_te["sh"], "test_ann": m_te["ann"],
    })
tr_df = pd.DataFrame(train_results).sort_values("train_sh", ascending=False).reset_index(drop=True)
print(f"\n  Top 5 by TRAIN Sharpe:")
print(f"  {'V × Q':<22} {'train_sh':>10} {'train_ann':>10} {'test_sh':>10} {'test_ann':>10}")
for _, r in tr_df.head(5).iterrows():
    print(f"  {r['value']:>3} × {r['quality']:<14}  {r['train_sh']:>10.3f} {r['train_ann']:>10.2%} {r['test_sh']:>10.3f} {r['test_ann']:>10.2%}")

# Winner on TRAIN
winner = tr_df.iloc[0]
print(f"\n=== Pure OOS verification ===")
print(f"  Best on TRAIN (2002-2014): {winner['value']} × {winner['quality']}")
print(f"     train Sharpe: {winner['train_sh']:.3f}  ann: {winner['train_ann']:.2%}")
print(f"     test  Sharpe: {winner['test_sh']:.3f}  ann: {winner['test_ann']:.2%}")
spy_test = metrics(spy_rets[test_mask])
print(f"  SPY (2015-2025): Sharpe = {spy_test['sh']:.3f}  ann = {spy_test['ann']:.2%}")
print(f"\n  Verdict on holdout 2015-2025:")
if winner['test_sh'] > spy_test['sh']:
    print(f"  ✓ The TRAIN-selected pair BEATS SPY on the pure holdout by Δ = {winner['test_sh'] - spy_test['sh']:+.3f}")
else:
    print(f"  ✗ The TRAIN-selected pair FAILS on holdout (Δ = {winner['test_sh'] - spy_test['sh']:+.3f})")

# Also rank by TEST so we can see how much overfitting there is
print(f"\n=== TEST (2015-2025, {test_mask.sum()} weeks) ranking (for comparison) ===")
te_df = tr_df.sort_values("test_sh", ascending=False).reset_index(drop=True)
print(f"  Top 5 by TEST Sharpe:")
print(f"  {'V × Q':<22} {'train_sh':>10} {'test_sh':>10} {'train_rank':>11}")
for _, r in te_df.head(5).iterrows():
    train_rank = int(tr_df[(tr_df['value'] == r['value']) & (tr_df['quality'] == r['quality'])].index[0]) + 1
    print(f"  {r['value']:>3} × {r['quality']:<14}  {r['train_sh']:>10.3f} {r['test_sh']:>10.3f} {train_rank:>11}")
