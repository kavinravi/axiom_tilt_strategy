"""Regime-conditional dynamic factor pair selector.

Per Friday t, a LightGBM classifier (trained walk-forward) picks ONE of 5
candidate factor pairs (top-5 from the V×Q sweep) to use for that week's
top-30 selection. Picks are then mcap-weighted with 10% cap. No SPY in
portfolio.

Candidate pairs:
  sp × fcfa     (S/P × FCF/Assets)        [global winner]
  e_ev × fcfa   (E/EV × FCF/Assets)       [#2]
  sp × roic     (S/P × ROIC)              [#3]
  sp × roe      (S/P × ROE)               [#4]
  sp × gpa      (S/P × GP/Assets)         [#5]

Regime features (PIT, all derived from data ≤ t):
  - macro_vixcls (VIX level)
  - macro_dgs10  (10Y Treasury yield)
  - macro_t10y2y (term spread)
  - spy_ret_4w   (trailing 4-week SPY return)
  - spy_ret_12w  (trailing 12-week SPY return)
  - spy_vol_12w  (trailing 12-week SPY realized vol)
  - spy_vol_26w  (trailing 26-week SPY realized vol)

Target (for training only): index of best pair in NEXT week (t+5).

PIT discipline:
  At each test Friday t, the classifier was trained ONLY on Fridays
  s where s + 5 days <= train_end of walk N (i.e., we know the label).

Output:
  artifacts/backtest_regime/weekly_regime_lgbm.parquet
  reports/regime_conditional_vs_spy.md
"""
from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.utils.io import processed_dir, repo_root
from src.utils.logging_utils import configure_logging, get_logger
from src.utils.ranker import friday_only
from src.utils.rl_env import project_to_simplex

log = get_logger(__name__)
REPO_ROOT = repo_root()
PANEL_DIR = processed_dir() / "panel"
TRAIN_PANEL_DIR = processed_dir() / "training_panel"
SPY_PATH = REPO_ROOT / "artifacts" / "benchmarks" / "spy_daily.parquet"
TOP_K = 30
MAX_WEIGHT = 0.10
EPS = 1e-8

PAIRS = [
    ("sp", "fcfa"),
    ("e_ev", "fcfa"),
    ("sp", "roic"),
    ("sp", "roe"),
    ("sp", "gpa"),
]

REGIME_FEATURES = ["macro_vixcls", "macro_dgs10", "macro_t10y2y",
                   "spy_ret_4w", "spy_ret_12w", "spy_vol_12w", "spy_vol_26w"]


def load_panel():
    cols = ["permno", "date", "prc", "shrout", "marketcap", "in_universe",
            "netinc", "equity", "revenue", "fcf", "ebit", "invcap",
            "ev", "gp", "assets"]
    frames = []
    for y in range(2001, 2026):
        for p in sorted((PANEL_DIR / f"year={y}").glob("*.parquet")):
            d = pd.read_parquet(p, columns=cols)
            d["date"] = pd.to_datetime(d["date"])
            d["permno"] = d["permno"].astype("int64")
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


def load_train():
    frames = []
    for y in range(2002, 2026):
        for p in sorted((TRAIN_PANEL_DIR / f"year={y}").glob("*.parquet")):
            d = pd.read_parquet(p, columns=["permno", "date", "fwd_ret_5d",
                                            "macro_vixcls", "macro_dgs10", "macro_t10y2y"])
            d["date"] = pd.to_datetime(d["date"])
            d["permno"] = d["permno"].astype("int64")
            frames.append(d)
    return pd.concat(frames, ignore_index=True)


print("Loading panel + training_panel ...")
daily = load_panel()
fri = load_train()
df = daily.merge(fri, on=["permno", "date"], how="inner")
df = df.dropna(subset=["fwd_ret_5d"]).copy()
df = friday_only(df).reset_index(drop=True)
df = df[df["in_universe"]].copy()
df["mcap"] = df["marketcap"].where(df["marketcap"].notna(),
                                    np.abs(df["prc"]) * df["shrout"])

# Build all factor signals
df["sp"] = (df["revenue"] / df["mcap"]).clip(lower=0)
df["e_ev"] = (df["netinc"] / df["ev"]).clip(lower=0)
df.loc[df["ev"] <= 0, "e_ev"] = np.nan
df["fcfa"] = (df["fcf"] / df["assets"]).clip(lower=-1.0, upper=2.0)
df.loc[df["assets"] <= 0, "fcfa"] = np.nan
df["roic"] = (df["ebit"] / df["invcap"]).clip(lower=-1.0, upper=2.0)
df.loc[df["invcap"] <= 0, "roic"] = np.nan
df["roe"] = (df["netinc"] / df["equity"]).clip(lower=-1.0, upper=2.0)
df.loc[df["equity"] <= 0, "roe"] = np.nan
df["gpa"] = (df["gp"] / df["assets"]).clip(lower=-1.0, upper=2.0)
df.loc[df["assets"] <= 0, "gpa"] = np.nan

# z-scores per Friday
for col in ["sp", "e_ev", "fcfa", "roic", "roe", "gpa"]:
    g = df.groupby("date", sort=False)[col]
    df[f"z_{col}"] = (df[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    df[f"z_{col}"] = df[f"z_{col}"].fillna(0.0)


def pair_top30_returns(vf: str, qf: str) -> pd.Series:
    """For each Friday, top-30 by 0.5*z(v)+0.5*z(q), mcap-weighted weekly return."""
    df["composite"] = 0.5 * df[f"z_{vf}"] + 0.5 * df[f"z_{qf}"]
    sb = (df.sort_values(["date", "composite"], ascending=[True, False])
            .groupby("date", sort=False, group_keys=False)
            .head(TOP_K)
            .reset_index(drop=True))
    rets = []
    for d, g in sb.groupby("date"):
        mcaps = g["mcap"].to_numpy(dtype=np.float64)
        mcaps = np.where(np.isnan(mcaps), 0.0, mcaps)
        if mcaps.sum() <= 0:
            n = len(g); w = np.full(n, 1.0 / n)
        else:
            w = project_to_simplex(np.log(np.maximum(mcaps, EPS)), max_weight=MAX_WEIGHT)
        fwd = g["fwd_ret_5d"].to_numpy(dtype=np.float64)
        fwd = np.where(np.isnan(fwd), 0.0, fwd)
        rets.append({"date": d, "ret": float(np.dot(w, fwd))})
    return pd.DataFrame(rets).sort_values("date").set_index("date")["ret"]


print("Computing per-pair weekly returns ...")
pair_returns = {}
for vf, qf in PAIRS:
    pair_returns[(vf, qf)] = pair_top30_returns(vf, qf)
all_dates = sorted(set().union(*[set(s.index) for s in pair_returns.values()]))
all_dates = pd.DatetimeIndex(all_dates)
pair_mat = pd.DataFrame({f"{vf}_{qf}": pair_returns[(vf, qf)].reindex(all_dates).values
                          for vf, qf in PAIRS}, index=all_dates)
print(f"  {pair_mat.shape}: {pair_mat.notna().all(axis=1).sum()} fully observed dates")

# Build SPY-based regime features
print("Building regime features ...")
spy = pd.read_parquet(SPY_PATH).reset_index()[["Date", "close"]].rename(columns={"Date": "date"})
spy["date"] = pd.to_datetime(spy["date"])
spy = spy.sort_values("date").set_index("date")["close"]
spy_close_at = spy.reindex(spy.index.union(all_dates)).sort_index().ffill().reindex(all_dates)
spy_w_ret = spy_close_at.pct_change().fillna(0.0)
# Multi-week trailing windows
spy_ret_4w = (1 + spy_w_ret).rolling(4).apply(lambda x: x.prod() - 1, raw=False).shift(1)
spy_ret_12w = (1 + spy_w_ret).rolling(12).apply(lambda x: x.prod() - 1, raw=False).shift(1)
spy_vol_12w = spy_w_ret.rolling(12).std().shift(1) * np.sqrt(52)
spy_vol_26w = spy_w_ret.rolling(26).std().shift(1) * np.sqrt(52)

# Macro features per Friday: take any one stock's row's macro values (cross-sectionally identical)
macro_by_date = df.groupby("date", sort=False)[["macro_vixcls", "macro_dgs10", "macro_t10y2y"]].first()
macro_by_date = macro_by_date.reindex(all_dates)

regime_df = pd.DataFrame({
    "macro_vixcls": macro_by_date["macro_vixcls"].values,
    "macro_dgs10":  macro_by_date["macro_dgs10"].values,
    "macro_t10y2y": macro_by_date["macro_t10y2y"].values,
    "spy_ret_4w":   spy_ret_4w.values,
    "spy_ret_12w":  spy_ret_12w.values,
    "spy_vol_12w":  spy_vol_12w.values,
    "spy_vol_26w":  spy_vol_26w.values,
}, index=all_dates)
regime_df = regime_df.fillna(method="ffill").fillna(method="bfill").fillna(0.0)
print(f"  regime features ready: {regime_df.shape}")

# Label: which pair had highest return this week (for training the meta-classifier)
label_argmax = pair_mat.idxmax(axis=1)  # column name of best pair per Friday
pair_to_idx = {f"{vf}_{qf}": i for i, (vf, qf) in enumerate(PAIRS)}
labels = label_argmax.map(pair_to_idx)


# Walk-forward training
print("Walk-forward training of regime classifier ...")
all_test_rets = []
years_arr = all_dates.year
for walk_id in range(1, 18):
    train_end_year = 2007 + walk_id - 1  # walk 1 train ends 2007
    val_year = train_end_year + 1
    test_year = train_end_year + 2
    train_mask = years_arr <= train_end_year
    val_mask = years_arr == val_year
    test_mask = years_arr == test_year
    if test_mask.sum() < 10:
        continue
    Xtr = regime_df[train_mask].fillna(0.0)
    ytr = labels[train_mask]
    Xvl = regime_df[val_mask].fillna(0.0)
    yvl = labels[val_mask]
    Xte = regime_df[test_mask].fillna(0.0)
    # Drop train rows where label is NaN
    valid_tr = ytr.notna()
    Xtr = Xtr[valid_tr]; ytr = ytr[valid_tr].astype(int)
    valid_vl = yvl.notna()
    Xvl = Xvl[valid_vl]; yvl = yvl[valid_vl].astype(int)
    if len(Xtr) < 100 or len(Xvl) < 5:
        log.warning("walk %d: insufficient train/val (%d/%d), skipping", walk_id, len(Xtr), len(Xvl))
        continue

    clf = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=15,
        min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
        lambda_l2=2.0, verbose=-1,
        objective="multiclass", num_class=len(PAIRS),
    )
    clf.fit(Xtr.to_numpy(), ytr.to_numpy(),
            eval_set=[(Xvl.to_numpy(), yvl.to_numpy())],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])

    # Inference: which pair does the model pick each test Friday?
    test_dates = all_dates[test_mask]
    pred_idx = clf.predict(Xte.to_numpy()).astype(int)
    for d, idx in zip(test_dates, pred_idx):
        vf, qf = PAIRS[idx]
        weekly_ret = pair_returns[(vf, qf)].get(d, 0.0)
        all_test_rets.append({"date": d, "pair_chosen": f"{vf}_{qf}",
                              "weekly_ret": float(weekly_ret)})
    log.info("walk %2d (test %d): train=%d val=%d test=%d, pair freqs in test = %s",
             walk_id, test_year, len(Xtr), len(Xvl), len(Xte),
             pd.Series([PAIRS[i] for i in pred_idx]).value_counts().to_dict())

w_df = pd.DataFrame(all_test_rets).sort_values("date").reset_index(drop=True)
w_df.to_parquet(REPO_ROOT / "artifacts" / "backtest_factor_v1" / "weekly_regime_lgbm.parquet",
                compression="zstd", index=False)


def metrics(rets):
    r = np.asarray(rets, dtype=float)
    cum = float(np.prod(1.0 + r) - 1.0)
    ann = (1.0 + cum) ** (52.0 / len(r)) - 1.0
    vol = float(np.std(r, ddof=1) * np.sqrt(52.0))
    sh = ann / vol if vol > 0 else 0.0
    eq = np.cumprod(1.0 + r); peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    return {"ann": ann, "vol": vol, "sh": sh, "mdd": mdd}


# Final comparison
spy_aligned = spy_close_at.reindex(pd.DatetimeIndex(w_df["date"])).pct_change().fillna(0.0).to_numpy()
det_sp = pd.read_parquet(REPO_ROOT / "artifacts" / "backtest_factor_v1" / "weekly_sp_fcfa.parquet")
det_sp["date"] = pd.to_datetime(det_sp["date"])
det_aligned = det_sp.set_index("date").reindex(pd.DatetimeIndex(w_df["date"]))["weekly_ret"].to_numpy()
regime_rets = w_df["weekly_ret"].to_numpy()
years_w = pd.DatetimeIndex(w_df["date"]).year
mask = years_w >= 2010

print(f"\n=== Regime-conditional factor (top-5 pairs, LGBM classifier) ===")
print(f"  2010-2025 BAR ({mask.sum()} weeks):")
print(f"    {'strategy':<40} {'ann':>8} {'vol':>8} {'sharpe':>8} {'mdd':>8}")
ms = metrics(spy_aligned[mask]); md = metrics(det_aligned[mask]); mr = metrics(regime_rets[mask])
print(f"    {'SPY':<40} {ms['ann']:>8.2%} {ms['vol']:>8.2%} {ms['sh']:>8.3f} {ms['mdd']:>8.2%}")
print(f"    {'sp_fcfa det (best static pair)':<40} {md['ann']:>8.2%} {md['vol']:>8.2%} {md['sh']:>8.3f} {md['mdd']:>8.2%}")
print(f"    {'regime-LGBM (dynamic pair)':<40} {mr['ann']:>8.2%} {mr['vol']:>8.2%} {mr['sh']:>8.3f} {mr['mdd']:>8.2%}")

print(f"\n  Δ vs SPY:        Sharpe {mr['sh'] - ms['sh']:+.3f}  return {mr['ann'] - ms['ann']:+.2%}")
print(f"  Δ vs det sp_fcfa: Sharpe {mr['sh'] - md['sh']:+.3f}  return {mr['ann'] - md['ann']:+.2%}")

# Pair pick frequency
print(f"\n  Pair pick frequencies (test years 2009-2025):")
print(f"    {w_df['pair_chosen'].value_counts(normalize=True).to_string()}")
