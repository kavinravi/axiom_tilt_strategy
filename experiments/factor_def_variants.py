"""Alternative VALUE and QUALITY factor definitions — sweep to see if any
combination beats v6 (E/P + ROE).

VALUE candidates:
  E/P = netinc / mcap                  [classic value]
  B/P = equity / mcap                  [book-to-price]
  S/P = revenue / mcap                 [sales-to-price]
  FCF/P = fcf / mcap                   [free-cash-flow yield]
  EBITDA/EV = ebitda / ev              [EV-based value]
  E/EV = netinc / ev                   [enterprise-value E/P]

QUALITY candidates:
  ROE = netinc / equity                [classic]
  ROA = netinc / assets                [asset-light bias]
  ROIC = ebit / invcap                 [returns on invested capital]
  GP/A = gp / assets                   [Novy-Marx gross-profitability]
  GrossMargin = gp / revenue           [structural margin]
  FCF/A = fcf / assets                 [free-cash-flow on assets]

For each (value, quality) pair → top-30 → mcap-weight 10% cap → blend 50/50 with SPY.
Report top 10 by Sharpe.
"""
from __future__ import annotations

import itertools
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


# Load entire 2002-2025 once
print("Loading panel + training_panel 2002-2025 ...")
daily = load_panel(range(2001, 2026), cols=[
    "permno", "date", "prc", "shrout", "marketcap", "in_universe",
    "netinc", "equity", "revenue", "gp", "fcf", "ebit", "invcap",
    "ebitda", "ev", "assets"])
fri_panel = load_train_panel(range(2002, 2026))
df_all = daily.merge(fri_panel, on=["permno", "date"], how="inner")
df_all = df_all.dropna(subset=["fwd_ret_5d"]).copy()
df_all = friday_only(df_all).reset_index(drop=True)
df_all = df_all[df_all["in_universe"]].copy()
df_all["mcap"] = df_all["marketcap"]
df_all.loc[df_all["mcap"].isna(), "mcap"] = (np.abs(df_all.loc[df_all["mcap"].isna(), "prc"]) *
                                              df_all.loc[df_all["mcap"].isna(), "shrout"])
print(f"  panel rows: {len(df_all)}, dates: {df_all['date'].nunique()}")


# VALUE signals (all "yield"-style, higher = cheaper = better)
df_all["ep"] = (df_all["netinc"] / df_all["mcap"]).clip(lower=0)
df_all["bp"] = (df_all["equity"] / df_all["mcap"]).clip(lower=0)
df_all["sp"] = (df_all["revenue"] / df_all["mcap"]).clip(lower=0)
df_all["fcfp"] = (df_all["fcf"] / df_all["mcap"]).clip(lower=0)
df_all["ebitda_ev"] = (df_all["ebitda"] / df_all["ev"]).clip(lower=0)
df_all.loc[df_all["ev"] <= 0, "ebitda_ev"] = np.nan
df_all["e_ev"] = (df_all["netinc"] / df_all["ev"]).clip(lower=0)
df_all.loc[df_all["ev"] <= 0, "e_ev"] = np.nan

# QUALITY signals (higher = better)
df_all["roe"] = (df_all["netinc"] / df_all["equity"]).clip(lower=-1.0, upper=2.0)
df_all.loc[df_all["equity"] <= 0, "roe"] = np.nan
df_all["roa"] = (df_all["netinc"] / df_all["assets"]).clip(lower=-1.0, upper=2.0)
df_all.loc[df_all["assets"] <= 0, "roa"] = np.nan
df_all["roic"] = (df_all["ebit"] / df_all["invcap"]).clip(lower=-1.0, upper=2.0)
df_all.loc[df_all["invcap"] <= 0, "roic"] = np.nan
df_all["gpa"] = (df_all["gp"] / df_all["assets"]).clip(lower=-1.0, upper=2.0)
df_all.loc[df_all["assets"] <= 0, "gpa"] = np.nan
df_all["gross_margin"] = (df_all["gp"] / df_all["revenue"]).clip(lower=-1.0, upper=2.0)
df_all.loc[df_all["revenue"] <= 0, "gross_margin"] = np.nan
df_all["fcfa"] = (df_all["fcf"] / df_all["assets"]).clip(lower=-1.0, upper=2.0)
df_all.loc[df_all["assets"] <= 0, "fcfa"] = np.nan

VALUE_FACTORS = ["ep", "bp", "sp", "fcfp", "ebitda_ev", "e_ev"]
QUALITY_FACTORS = ["roe", "roa", "roic", "gpa", "gross_margin", "fcfa"]

# Per-Friday z-scores
print("Computing per-Friday z-scores ...")
for col in VALUE_FACTORS + QUALITY_FACTORS:
    g = df_all.groupby("date", sort=False)[col]
    df_all[f"z_{col}"] = (df_all[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    df_all[f"z_{col}"] = df_all[f"z_{col}"].fillna(0.0)


def compute_strategy_returns(value_col: str, quality_col: str, df: pd.DataFrame):
    """Compute weekly returns of (value + quality 50/50) → top-30 → mcap-weighted."""
    df = df.copy()
    df["score"] = 0.5 * df[f"z_{value_col}"] + 0.5 * df[f"z_{quality_col}"]
    sb = (df.sort_values(["date", "score"], ascending=[True, False])
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
        rets.append({"date": d, "weekly_ret": float(np.dot(w, fwd))})
    return pd.DataFrame(rets).sort_values("date").reset_index(drop=True)


# SPY weekly
dates = pd.DatetimeIndex(sorted(df_all["date"].unique()))
spy = pd.read_parquet(SPY_PATH).reset_index()[["Date", "close"]].rename(columns={"Date": "date"})
spy["date"] = pd.to_datetime(spy["date"])
spy = spy.sort_values("date").set_index("date")["close"]
closes = spy.reindex(spy.index.union(dates)).sort_index().ffill().reindex(dates)
spy_rets = closes.pct_change().fillna(0.0).to_numpy()
years_arr = dates.year
mask_2010_2025 = years_arr >= 2010


def metrics(rets):
    r = np.asarray(rets, dtype=float)
    cum = float(np.prod(1.0 + r) - 1.0)
    ann = (1.0 + cum) ** (52.0 / len(r)) - 1.0
    vol = float(np.std(r, ddof=1) * np.sqrt(52.0))
    sh = ann / vol if vol > 0 else 0.0
    eq = np.cumprod(1.0 + r); peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    return {"ann": ann, "vol": vol, "sh": sh, "mdd": mdd}


# All combinations
print(f"\nSweeping {len(VALUE_FACTORS)} × {len(QUALITY_FACTORS)} = {len(VALUE_FACTORS)*len(QUALITY_FACTORS)} (V × Q) combinations ...")
results = []
for vf, qf in itertools.product(VALUE_FACTORS, QUALITY_FACTORS):
    wk = compute_strategy_returns(vf, qf, df_all)
    rets = wk.set_index("date").reindex(dates)["weekly_ret"].to_numpy()
    blend = 0.5 * rets + 0.5 * spy_rets
    rets_oos = rets[mask_2010_2025]
    blend_oos = blend[mask_2010_2025]
    m_v6 = metrics(rets_oos)
    m_bl = metrics(blend_oos)
    results.append({
        "value": vf, "quality": qf,
        "v6_sh": m_v6["sh"], "v6_ann": m_v6["ann"], "v6_vol": m_v6["vol"], "v6_mdd": m_v6["mdd"],
        "blend_sh": m_bl["sh"], "blend_ann": m_bl["ann"], "blend_vol": m_bl["vol"], "blend_mdd": m_bl["mdd"],
    })

res_df = pd.DataFrame(results).sort_values("blend_sh", ascending=False).reset_index(drop=True)
print(f"\n{'V × Q':<24} {'v6_sh':>7} {'blend_sh':>9} {'blend_ann':>10} {'blend_vol':>10} {'blend_mdd':>10}")
print("-" * 90)
for _, r in res_df.iterrows():
    key = f"{r['value']} × {r['quality']}"
    marker = " ✓" if r['blend_sh'] > 0.872 else ""
    print(f"{key:<24} {r['v6_sh']:>7.3f} {r['blend_sh']:>9.3f} {r['blend_ann']:>10.2%} {r['blend_vol']:>10.2%} {r['blend_mdd']:>10.2%}{marker}")

# Save top 5
res_df.head(5).to_csv(REPO_ROOT / "reports" / "factor_def_variants_top5.csv", index=False)
