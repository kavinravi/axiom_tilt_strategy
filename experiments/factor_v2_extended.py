"""Extended factor recipes + weight-scheme + K sweep on the v6 winning lane.

Builds on v6_value_quality (Sharpe 1.021 vs SPY 0.872). Tries:
- More value factors (B/P, S/P, CF/P alongside E/P)
- More quality factors (gross_margin, debt/equity, earnings stability)
- Different weighting schemes (mcap, equal, invvol)
- Different K values (top-20, 30, 50, 100)

usage: python experiments/factor_v2_extended.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

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
MAX_WEIGHT = 0.10
EPS = 1e-8


def zscore(s):
    s = s.astype(float)
    mu = s.mean(); sd = s.std(ddof=0)
    return s * 0 if sd == 0 or pd.isna(sd) else (s - mu) / sd


def load_panel_years(years, cols):
    frames = []
    for y in years:
        for p in sorted((PANEL_DIR / f"year={y}").glob("*.parquet")):
            df = pd.read_parquet(p, columns=cols)
            df["date"] = pd.to_datetime(df["date"])
            df["permno"] = df["permno"].astype("int64")
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_train_panel_year(year):
    files = sorted((TRAIN_PANEL_DIR / f"year={year}").glob("*.parquet"))
    if not files: return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["permno"] = df["permno"].astype("int64")
    return df


def build_extended_scored_panel(walk_id: int, test_year: int) -> pd.DataFrame:
    """Build Friday panel with ALL value + quality factor signals.

    Value : E/P, B/P, S/P, CF/P  (all clipped at 0 for non-positive)
    Quality : ROE, ROIC-proxy, GrossMargin, FCF-margin, Debt-to-Equity (negated)
    """
    sf1_cols = ["permno", "date", "prc", "shrout", "ret", "marketcap", "in_universe",
                "netinc", "equity", "revenue", "gp", "fcf", "ebit", "invcap",
                "debt", "ebitda", "opex"]
    panel_years = list(range(2001, test_year + 1))
    daily = load_panel_years(panel_years, cols=sf1_cols)
    # Rolling vol/momentum aren't needed here (we're not using them in factors) - skip
    fri_panel = pd.concat([
        load_train_panel_year(y)[["permno", "date", "fwd_ret_5d"]]
        for y in range(2002, test_year + 1)], ignore_index=True)
    df = daily.merge(fri_panel, on=["permno", "date"], how="inner")
    df = df.dropna(subset=["fwd_ret_5d"]).copy()
    df = friday_only(df).reset_index(drop=True)
    df = df[df["in_universe"]].copy()

    df["mcap"] = df["marketcap"]
    df.loc[df["mcap"].isna(), "mcap"] = (np.abs(df.loc[df["mcap"].isna(), "prc"]) *
                                          df.loc[df["mcap"].isna(), "shrout"])

    # VALUE signals — all "yield"-style (higher = cheaper = better)
    df["ep"] = (df["netinc"] / df["mcap"]).clip(lower=0)
    df["bp"] = (df["equity"] / df["mcap"]).clip(lower=0)
    df["sp"] = (df["revenue"] / df["mcap"]).clip(lower=0)
    df["cfp"] = (df["fcf"] / df["mcap"]).clip(lower=0)

    # QUALITY signals
    df["roe"] = (df["netinc"] / df["equity"]).clip(lower=-1.0, upper=2.0)
    df.loc[df["equity"] <= 0, "roe"] = np.nan
    df["roic"] = (df["ebit"] / df["invcap"]).clip(lower=-1.0, upper=2.0)
    df.loc[df["invcap"] <= 0, "roic"] = np.nan
    df["gross_margin"] = (df["gp"] / df["revenue"]).clip(lower=-1.0, upper=2.0)
    df.loc[df["revenue"] <= 0, "gross_margin"] = np.nan
    df["fcf_margin"] = (df["fcf"] / df["revenue"]).clip(lower=-1.0, upper=2.0)
    df.loc[df["revenue"] <= 0, "fcf_margin"] = np.nan
    # Debt-to-equity, lower is better → negate
    df["de_neg"] = -(df["debt"] / df["equity"]).clip(lower=-5.0, upper=10.0)
    df.loc[df["equity"] <= 0, "de_neg"] = np.nan

    # Per-Friday z-scores
    factor_cols = ["ep", "bp", "sp", "cfp", "roe", "roic",
                   "gross_margin", "fcf_margin", "de_neg"]
    for col in factor_cols:
        g = df.groupby("date", sort=False)[col]
        df[f"z_{col}"] = (df[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        df[f"z_{col}"] = df[f"z_{col}"].fillna(0.0)

    return df


def apply_composite(df_scored, recipe, top_k):
    cols = list(recipe.keys())
    weights = np.array([recipe[c] for c in cols], dtype=float)
    weights = weights / weights.sum()
    mat = df_scored[cols].to_numpy(dtype=float)
    df_scored = df_scored.copy()
    df_scored["score"] = (mat * weights[None, :]).sum(axis=1)
    df_scored = df_scored.dropna(subset=["score"])
    df_scored = (df_scored.sort_values(["date", "score"], ascending=[True, False])
                          .groupby("date", sort=False, group_keys=False)
                          .head(top_k)
                          .reset_index(drop=True))
    return df_scored[["permno", "date", "score", "fwd_ret_5d", "mcap"]]


def w_mcap(mcaps, k):
    safe = np.maximum(mcaps, EPS)
    return project_to_simplex(np.log(safe), max_weight=MAX_WEIGHT)


def w_equal(mcaps, k):
    # Equal-weight, but if K > 10 we hit the implicit 1/K natural; cap doesn't bind for K>=10.
    n = len(mcaps)
    return np.full(n, 1.0 / n)


def w_invvol(mcaps_unused, vols):
    # placeholder; we'd need to pass vols. Skip for now or compute approximately.
    return None  # not used in this script


def backtest_test_year(sb, test_year, w_fn, top_k=30):
    df = sb[(sb["date"] >= f"{test_year}-01-01") &
            (sb["date"] <= f"{test_year}-12-31")].copy().reset_index(drop=True)
    by_date = {d: g.reset_index(drop=True) for d, g in df.groupby("date")}
    out = []
    for d in sorted(by_date.keys()):
        cur = by_date[d]
        mcaps = cur["mcap"].to_numpy(dtype=np.float64)[:top_k]
        mcaps = np.where(np.isnan(mcaps), 0.0, mcaps)
        if mcaps.sum() <= 0:
            n = max(len(cur), 1)
            w = np.full(n, 1.0 / n)
        else:
            w = w_fn(mcaps, top_k)
        fwd = cur["fwd_ret_5d"].to_numpy(dtype=np.float64)[:top_k]
        fwd = np.where(np.isnan(fwd), 0.0, fwd)
        out.append({"date": d, "weekly_ret": float(np.dot(w, fwd))})
    return pd.DataFrame(out)


def metrics(rets):
    rets = np.asarray(rets, dtype=float)
    if len(rets) < 2: return {}
    cum = float(np.prod(1.0 + rets) - 1.0)
    ann = (1.0 + cum) ** (52.0 / len(rets)) - 1.0
    vol = float(np.std(rets, ddof=1) * np.sqrt(52.0))
    sh = ann / vol if vol > 0 else 0.0
    eq = np.cumprod(1.0 + rets); peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    cal = ann / abs(mdd) if mdd < 0 else 0.0
    return {"n": len(rets), "tot": cum, "ann": ann, "vol": vol, "sh": sh, "mdd": mdd, "cal": cal}


def build_spy(dates):
    spy = pd.read_parquet(SPY_PATH).reset_index()[["Date", "close"]].rename(columns={"Date": "date"})
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").set_index("date")["close"]
    closes = spy.reindex(spy.index.union(dates)).sort_index().ffill().reindex(dates)
    return closes.pct_change().fillna(0.0).to_numpy()


# Extended recipes
EXTENDED_RECIPES = {
    "x1_full_value_quality": {  # all value, all quality - equal weight within each
        "z_ep": 1, "z_bp": 1, "z_sp": 1, "z_cfp": 1,
        "z_roe": 1, "z_roic": 1, "z_gross_margin": 1, "z_fcf_margin": 1, "z_de_neg": 1,
    },
    "x2_v6_plus_bp": {"z_ep": 1, "z_bp": 1, "z_roe": 1},  # add B/P to v6
    "x3_v6_plus_cfp": {"z_ep": 1, "z_cfp": 1, "z_roe": 1},  # add CF/P to v6
    "x4_ep_roe_roic": {"z_ep": 1, "z_roe": 1, "z_roic": 1},  # double-quality
    "x5_value_only_4": {"z_ep": 1, "z_bp": 1, "z_sp": 1, "z_cfp": 1},  # pure value
    "x6_quality_only_5": {"z_roe": 1, "z_roic": 1, "z_gross_margin": 1, "z_fcf_margin": 1, "z_de_neg": 1},  # pure quality
    "x7_v6_plus_fcf_margin": {"z_ep": 1, "z_roe": 1, "z_fcf_margin": 1},  # v6 + FCF margin
    "x8_v6_60_40": {"z_ep": 0.60, "z_roe": 0.40},  # 60/40 value/quality
    "x9_v6_40_60": {"z_ep": 0.40, "z_roe": 0.60},  # 40/60
    "x10_v6": {"z_ep": 1, "z_roe": 1},  # baseline replication
    "x11_v6_plus_gm": {"z_ep": 1, "z_roe": 1, "z_gross_margin": 1},
    "x12_balanced_2v_2q": {"z_ep": 1, "z_bp": 1, "z_roe": 1, "z_roic": 1},  # 2 value 2 quality
}


def main():
    configure_logging()
    all_walk_scored = {}
    for walk_id in range(1, 18):
        t0 = time.time()
        sp = build_extended_scored_panel(walk_id, 2008 + walk_id)
        all_walk_scored[walk_id] = sp
        log.info("extended panel walk %2d: %d rows [%.1fs]", walk_id, len(sp), time.time() - t0)

    # For each recipe + K + weight scheme combination
    schemes_to_test = [(30, "mcap", w_mcap), (30, "equal", w_equal),
                       (50, "mcap", w_mcap), (100, "mcap", w_mcap)]
    results = {}
    for recipe_name, recipe in EXTENDED_RECIPES.items():
        for K, scheme_name, w_fn in schemes_to_test:
            key = f"{recipe_name}__K{K}__{scheme_name}"
            rows_all = []
            for walk_id in range(1, 18):
                sp = all_walk_scored[walk_id]
                sb = apply_composite(sp, recipe, top_k=K)
                wk = backtest_test_year(sb, 2008 + walk_id, w_fn, top_k=K)
                rows_all.append(wk)
            weekly = pd.concat(rows_all, ignore_index=True).sort_values("date").reset_index(drop=True)
            results[key] = weekly
            log.info("recipe %-30s K=%3d w=%-5s: %d weeks", recipe_name, K, scheme_name, len(weekly))

    dates = pd.DatetimeIndex(results[next(iter(results))]["date"])
    spy_rets = build_spy(dates)
    years = pd.DatetimeIndex(dates).year

    # Print sorted by 2010-2025 Sharpe descending
    mask_target = years >= 2010
    sm_target = metrics(spy_rets[mask_target])

    summary_rows = []
    for key, df in results.items():
        r = df.set_index("date").reindex(dates)["weekly_ret"].to_numpy()
        m = metrics(r[mask_target])
        summary_rows.append({"recipe": key, **m})
    summary = pd.DataFrame(summary_rows).sort_values("sh", ascending=False).reset_index(drop=True)
    print(f"\n=== Ranked by 2010-2025 Sharpe (n=806 weeks) ===")
    print(f"  SPY                                       sharpe={sm_target['sh']:.3f}  ann={sm_target['ann']:.2%}  vol={sm_target['vol']:.2%}  mdd={sm_target['mdd']:.2%}")
    print("-" * 110)
    for _, row in summary.iterrows():
        marker = " ✓" if row['sh'] > sm_target['sh'] else ""
        print(f"  {row['recipe']:<42} sharpe={row['sh']:.3f}  ann={row['ann']:.2%}  vol={row['vol']:.2%}  mdd={row['mdd']:.2%}  cal={row['cal']:.3f}{marker}")

    summary.to_csv(REPO_ROOT / "reports" / "factor_v2_extended_summary.csv", index=False)
    # Save weekly returns of top-3 by Sharpe
    out_root = REPO_ROOT / "artifacts" / "backtest_factor_v2"
    out_root.mkdir(parents=True, exist_ok=True)
    for _, row in summary.head(5).iterrows():
        key = row["recipe"]
        results[key].to_parquet(out_root / f"weekly_{key}.parquet", compression="zstd", index=False)


if __name__ == "__main__":
    main()
