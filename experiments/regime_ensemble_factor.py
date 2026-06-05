"""Regime-conditional ENSEMBLE factor mixer.

Same setup as regime_conditional_factor.py but instead of picking ARGMAX pair,
we use the LGBM's softmax probabilities to MIX the top-30 portfolios of all 5
candidate pairs. Each week:

  weighted_portfolio = sum_p P(pair_p | regime_t) * portfolio_p

where portfolio_p is the per-Friday top-30 mcap-weighted weights of pair p.

This produces a single combined portfolio (different from any single pair).
We then enforce the 10% per-stock cap on the combined portfolio.

Output: weekly returns of the regime-mixed portfolio.
"""
from __future__ import annotations

import time
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
    ("sp", "fcfa"), ("e_ev", "fcfa"), ("sp", "roic"),
    ("sp", "roe"), ("sp", "gpa"),
]
REGIME_FEATURES = ["macro_vixcls", "macro_dgs10", "macro_t10y2y",
                   "spy_ret_4w", "spy_ret_12w", "spy_vol_12w", "spy_vol_26w"]


def load_data():
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
    daily = pd.concat(frames, ignore_index=True)
    tframes = []
    for y in range(2002, 2026):
        for p in sorted((TRAIN_PANEL_DIR / f"year={y}").glob("*.parquet")):
            d = pd.read_parquet(p, columns=["permno", "date", "fwd_ret_5d",
                                            "macro_vixcls", "macro_dgs10", "macro_t10y2y"])
            d["date"] = pd.to_datetime(d["date"])
            d["permno"] = d["permno"].astype("int64")
            tframes.append(d)
    fri = pd.concat(tframes, ignore_index=True)
    df = daily.merge(fri, on=["permno", "date"], how="inner")
    df = df.dropna(subset=["fwd_ret_5d"]).copy()
    df = friday_only(df).reset_index(drop=True)
    df = df[df["in_universe"]].copy()
    df["mcap"] = df["marketcap"].where(df["marketcap"].notna(),
                                        np.abs(df["prc"]) * df["shrout"])

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
    for col in ["sp", "e_ev", "fcfa", "roic", "roe", "gpa"]:
        g = df.groupby("date", sort=False)[col]
        df[f"z_{col}"] = (df[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        df[f"z_{col}"] = df[f"z_{col}"].fillna(0.0)
    return df


print("Loading + computing factors ...")
df = load_data()


def pair_picks(vf, qf):
    df["composite"] = 0.5 * df[f"z_{vf}"] + 0.5 * df[f"z_{qf}"]
    sb = (df.sort_values(["date", "composite"], ascending=[True, False])
            .groupby("date", sort=False, group_keys=False)
            .head(TOP_K)
            .reset_index(drop=True))
    return sb[["permno", "date", "mcap", "fwd_ret_5d"]].copy()


print("Building per-pair picks ...")
pair_dfs = {(vf, qf): pair_picks(vf, qf) for vf, qf in PAIRS}

# Build regime features
print("Building regime features ...")
spy = pd.read_parquet(SPY_PATH).reset_index()[["Date", "close"]].rename(columns={"Date": "date"})
spy["date"] = pd.to_datetime(spy["date"])
spy = spy.sort_values("date").set_index("date")["close"]
all_dates = pd.DatetimeIndex(sorted(df["date"].unique()))
spy_at = spy.reindex(spy.index.union(all_dates)).sort_index().ffill().reindex(all_dates)
spy_w_ret = spy_at.pct_change().fillna(0.0)
macro_by_date = df.groupby("date", sort=False)[["macro_vixcls", "macro_dgs10", "macro_t10y2y"]].first().reindex(all_dates)
regime_df = pd.DataFrame({
    "macro_vixcls": macro_by_date["macro_vixcls"].values,
    "macro_dgs10":  macro_by_date["macro_dgs10"].values,
    "macro_t10y2y": macro_by_date["macro_t10y2y"].values,
    "spy_ret_4w":   (1 + spy_w_ret).rolling(4).apply(lambda x: x.prod() - 1, raw=False).shift(1).values,
    "spy_ret_12w":  (1 + spy_w_ret).rolling(12).apply(lambda x: x.prod() - 1, raw=False).shift(1).values,
    "spy_vol_12w":  (spy_w_ret.rolling(12).std() * np.sqrt(52)).shift(1).values,
    "spy_vol_26w":  (spy_w_ret.rolling(26).std() * np.sqrt(52)).shift(1).values,
}, index=all_dates).ffill().bfill().fillna(0.0)


# Per-Friday top-30 weights per pair
def pair_friday_weights(pair_sb):
    """Returns dict {date: dict{permno: weight}} for the pair."""
    out = {}
    for d, g in pair_sb.groupby("date"):
        g = g.reset_index(drop=True)
        mcaps = g["mcap"].to_numpy(dtype=np.float64)
        mcaps = np.where(np.isnan(mcaps), 0.0, mcaps)
        if mcaps.sum() <= 0:
            w = np.full(len(g), 1.0 / len(g))
        else:
            w = project_to_simplex(np.log(np.maximum(mcaps, EPS)), max_weight=MAX_WEIGHT)
        out[d] = dict(zip(g["permno"].astype(int).tolist(), w.tolist()))
    return out


print("Computing per-pair weights ...")
pair_weights = {(vf, qf): pair_friday_weights(pair_dfs[(vf, qf)]) for vf, qf in PAIRS}


def metrics(rets):
    r = np.asarray(rets, dtype=float)
    cum = float(np.prod(1.0 + r) - 1.0)
    ann = (1.0 + cum) ** (52.0 / len(r)) - 1.0
    vol = float(np.std(r, ddof=1) * np.sqrt(52.0))
    sh = ann / vol if vol > 0 else 0.0
    eq = np.cumprod(1.0 + r); peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    return {"ann": ann, "vol": vol, "sh": sh, "mdd": mdd}


# Compute per-pair return for labels
def pair_weekly_return(pair_w_dict, fwd_lookup):
    """Given a dict {date: {permno: weight}} and a dict {(permno, date): fwd_ret_5d},
    returns a pd.Series of weekly returns indexed by date."""
    out = []
    for d, w in pair_w_dict.items():
        ret = sum(weight * fwd_lookup.get((p, d), 0.0) for p, weight in w.items())
        out.append({"date": d, "ret": ret})
    return pd.DataFrame(out).sort_values("date").set_index("date")["ret"]


# Build fwd lookup
fwd_lookup = {}
for _, row in df.iterrows():
    fwd_lookup[(int(row["permno"]), row["date"])] = float(row["fwd_ret_5d"]) if pd.notna(row["fwd_ret_5d"]) else 0.0

pair_returns = {(vf, qf): pair_weekly_return(pair_weights[(vf, qf)], fwd_lookup)
                for vf, qf in PAIRS}

# Build label: argmax pair per Friday
pair_mat = pd.DataFrame({f"{vf}_{qf}": pair_returns[(vf, qf)].reindex(all_dates).values
                          for vf, qf in PAIRS}, index=all_dates)
labels = pair_mat.idxmax(axis=1).map({f"{vf}_{qf}": i for i, (vf, qf) in enumerate(PAIRS)})

# Walk-forward train + ensemble inference
print("Walk-forward training + ensemble inference ...")
years_arr = all_dates.year
ensemble_rows = []
for walk_id in range(1, 18):
    train_end_year = 2007 + walk_id - 1
    val_year = train_end_year + 1
    test_year = train_end_year + 2
    train_mask = years_arr <= train_end_year
    val_mask = years_arr == val_year
    test_mask = years_arr == test_year
    if test_mask.sum() < 10:
        continue
    Xtr = regime_df[train_mask]; ytr = labels[train_mask]
    Xvl = regime_df[val_mask]; yvl = labels[val_mask]
    Xte = regime_df[test_mask]
    valid_tr = ytr.notna(); Xtr = Xtr[valid_tr]; ytr = ytr[valid_tr].astype(int)
    valid_vl = yvl.notna(); Xvl = Xvl[valid_vl]; yvl = yvl[valid_vl].astype(int)
    if len(Xtr) < 100 or len(Xvl) < 5:
        continue
    clf = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=15,
        min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
        lambda_l2=2.0, verbose=-1, objective="multiclass", num_class=len(PAIRS),
    )
    clf.fit(Xtr.to_numpy(), ytr.to_numpy(),
            eval_set=[(Xvl.to_numpy(), yvl.to_numpy())],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
    test_dates = all_dates[test_mask]
    proba = clf.predict_proba(Xte.to_numpy())  # shape (n_test, n_pairs)
    for i, d in enumerate(test_dates):
        # Combine pair-portfolios weighted by probability
        combined_w: dict[int, float] = {}
        for j, (vf, qf) in enumerate(PAIRS):
            w_dict = pair_weights[(vf, qf)].get(d, {})
            for permno, w in w_dict.items():
                combined_w[permno] = combined_w.get(permno, 0.0) + proba[i, j] * w
        # The combined portfolio sums to 1 (each pair sums to 1, weighted by prob summing to 1)
        # Apply 10% cap via water-fill projection
        permnos = list(combined_w.keys())
        weights = np.array([combined_w[p] for p in permnos], dtype=np.float64)
        # Project to simplex with cap
        weights_capped = project_to_simplex(np.log(np.maximum(weights, EPS)), max_weight=MAX_WEIGHT)
        # Re-create dict
        final_w = dict(zip(permnos, weights_capped.tolist()))
        # Compute weekly return
        ret = sum(w * fwd_lookup.get((p, d), 0.0) for p, w in final_w.items())
        ensemble_rows.append({"date": d, "weekly_ret": float(ret)})
    log.info("walk %2d (test %d): ensemble done, mean proba on most-picked = %.3f",
             walk_id, test_year, proba.max(axis=1).mean())

w_df = pd.DataFrame(ensemble_rows).sort_values("date").reset_index(drop=True)
w_df["date"] = pd.to_datetime(w_df["date"])
out_dir = REPO_ROOT / "artifacts" / "backtest_factor_v1"
out_dir.mkdir(parents=True, exist_ok=True)
w_df.to_parquet(out_dir / "weekly_regime_ensemble.parquet", compression="zstd", index=False)

# Final metrics
test_dates_ens = pd.DatetimeIndex(w_df["date"])
spy_aligned = spy_at.reindex(test_dates_ens).pct_change().fillna(0.0).to_numpy()
det_sp = pd.read_parquet(REPO_ROOT / "artifacts" / "backtest_factor_v1" / "weekly_sp_fcfa.parquet")
det_sp["date"] = pd.to_datetime(det_sp["date"])
det_aligned = det_sp.set_index("date").reindex(test_dates_ens)["weekly_ret"].to_numpy()
years_w = test_dates_ens.year
mask = years_w >= 2010

print(f"\n=== Regime ENSEMBLE (probability-weighted pair mix) ===")
print(f"  2010-2025 BAR ({mask.sum()} weeks):")
print(f"    {'strategy':<40} {'ann':>8} {'vol':>8} {'sharpe':>8} {'mdd':>8}")
ms = metrics(spy_aligned[mask])
md = metrics(det_aligned[mask])
me = metrics(w_df["weekly_ret"].to_numpy()[mask])
print(f"    {'SPY':<40} {ms['ann']:>8.2%} {ms['vol']:>8.2%} {ms['sh']:>8.3f} {ms['mdd']:>8.2%}")
print(f"    {'sp_fcfa det (best static)':<40} {md['ann']:>8.2%} {md['vol']:>8.2%} {md['sh']:>8.3f} {md['mdd']:>8.2%}")
print(f"    {'regime-LGBM ENSEMBLE':<40} {me['ann']:>8.2%} {me['vol']:>8.2%} {me['sh']:>8.3f} {me['mdd']:>8.2%}")
print(f"\n  Δ ensemble vs SPY:        Sharpe {me['sh'] - ms['sh']:+.3f}  return {me['ann'] - ms['ann']:+.2%}")
print(f"  Δ ensemble vs det sp_fcfa: Sharpe {me['sh'] - md['sh']:+.3f}  return {me['ann'] - md['ann']:+.2%}")
