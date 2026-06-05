"""sp_fcfa + LightGBM RE-RANKER (selection-stage ML).

Two-stage:
  Stage 1 — Wide-net factor screen (deterministic):
    sp_fcfa composite → take top-K_wide (default 100) per Friday.
  Stage 2 — LightGBM re-ranker (ML model, per-walk trained):
    Trained on the training-year top-K_wide rows. Features: 22-ish fundamentals
    + macro + sp_fcfa z-scores. Target: y_excess = fwd_ret_5d − cross-mean.
    At inference: predict y_excess for each Friday's top-K_wide, pick top-K_final
    (default 30) by predicted score. mcap-weight + 10% cap.

This puts ML in the SELECTION layer (where the parent project tried) but on a
much smaller, pre-filtered universe (100 candidates instead of 500). The hope
is that the ML can ADD signal vs the deterministic sp_fcfa cut by using
additional features.
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
K_WIDE = 100
K_FINAL = 30
MAX_WEIGHT = 0.10
EPS = 1e-8

ML_FEATURES = [
    "pe", "pb", "ps", "evebitda",
    "currentratio", "de", "netmargin",
    "ret", "prc_pct_change_1m", "prc_pct_change_3m",
    "macro_vixcls", "macro_dgs10", "macro_t10y2y",
    "z_sp", "z_fcfa",
    "marketcap", "fcfp_local", "roic_local", "gpa_local", "gross_margin_local",
]


def load_walk(test_year):
    pcols = ["permno", "date", "prc", "shrout", "marketcap", "in_universe", "ret",
             "netinc", "equity", "revenue", "fcf", "assets",
             "pe", "pb", "ps", "evebitda", "currentratio", "de", "netmargin",
             "ebit", "invcap", "gp"]
    frames = []
    for y in range(2001, test_year + 1):
        for p in sorted((PANEL_DIR / f"year={y}").glob("*.parquet")):
            d = pd.read_parquet(p, columns=pcols)
            d["date"] = pd.to_datetime(d["date"])
            d["permno"] = d["permno"].astype("int64")
            frames.append(d)
    daily = pd.concat(frames, ignore_index=True).sort_values(["permno", "date"]).reset_index(drop=True)
    grp = daily.groupby("permno", sort=False)
    daily["prc_pct_change_1m"] = grp["prc"].transform(lambda x: x.pct_change(21, fill_method=None))
    daily["prc_pct_change_3m"] = grp["prc"].transform(lambda x: x.pct_change(63, fill_method=None))
    daily["fcfp_local"] = (daily["fcf"] / np.maximum(daily["marketcap"], EPS)).clip(lower=0)
    daily["roic_local"] = (daily["ebit"] / daily["invcap"]).clip(lower=-1.0, upper=2.0)
    daily.loc[daily["invcap"] <= 0, "roic_local"] = np.nan
    daily["gpa_local"] = (daily["gp"] / daily["assets"]).clip(lower=-1.0, upper=2.0)
    daily.loc[daily["assets"] <= 0, "gpa_local"] = np.nan
    daily["gross_margin_local"] = (daily["gp"] / daily["revenue"]).clip(lower=-1.0, upper=2.0)
    daily.loc[daily["revenue"] <= 0, "gross_margin_local"] = np.nan

    tframes = []
    for y in range(2002, test_year + 1):
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
    df["fcfa"] = (df["fcf"] / df["assets"]).clip(lower=-1.0, upper=2.0)
    df.loc[df["assets"] <= 0, "fcfa"] = np.nan
    for col_in, col_out in [("sp", "z_sp"), ("fcfa", "z_fcfa")]:
        g = df.groupby("date", sort=False)[col_in]
        df[col_out] = (df[col_in] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        df[col_out] = df[col_out].fillna(0.0)
    df["composite"] = 0.5 * df["z_sp"] + 0.5 * df["z_fcfa"]
    df["y_excess"] = df["fwd_ret_5d"] - df.groupby("date", sort=False)["fwd_ret_5d"].transform("mean")

    # Top-K_WIDE per Friday by sp_fcfa composite
    wide = (df.sort_values(["date", "composite"], ascending=[True, False])
              .groupby("date", sort=False, group_keys=False)
              .head(K_WIDE)
              .reset_index(drop=True))
    wide["year"] = wide["date"].dt.year
    return wide


def metrics(rets):
    r = np.asarray(rets, dtype=float)
    cum = float(np.prod(1.0 + r) - 1.0)
    ann = (1.0 + cum) ** (52.0 / len(r)) - 1.0
    vol = float(np.std(r, ddof=1) * np.sqrt(52.0))
    sh = ann / vol if vol > 0 else 0.0
    eq = np.cumprod(1.0 + r); peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    return {"ann": ann, "vol": vol, "sh": sh, "mdd": mdd}


def main():
    configure_logging()
    all_rows = []
    for walk_id in range(1, 18):
        t0 = time.time()
        test_year = 2008 + walk_id
        wide = load_walk(test_year)
        train_end = test_year - 2
        val_year = test_year - 1
        tr = wide[(wide["year"] >= 2002) & (wide["year"] <= train_end)].copy()
        vl = wide[wide["year"] == val_year].copy()
        te = wide[wide["year"] == test_year].copy()
        if len(vl) < 50 or len(te) < 50:
            log.warning("walk %d insufficient data", walk_id); continue

        Xtr = tr[ML_FEATURES].astype(float).fillna(0.0).to_numpy()
        ytr = tr["y_excess"].astype(float).to_numpy()
        Xvl = vl[ML_FEATURES].astype(float).fillna(0.0).to_numpy()
        yvl = vl["y_excess"].astype(float).to_numpy()
        Xte = te[ML_FEATURES].astype(float).fillna(0.0).to_numpy()

        model = lgb.LGBMRegressor(
            n_estimators=2000, learning_rate=0.02,
            num_leaves=15, min_data_in_leaf=50,
            feature_fraction=0.8, bagging_fraction=0.8,
            lambda_l2=2.0, verbose=-1,
        )
        model.fit(Xtr, ytr, eval_set=[(Xvl, yvl)],
                  callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        te = te.assign(lgbm_score=model.predict(Xte))

        # Per Friday: take top-K_FINAL by lgbm_score from the K_WIDE pool, mcap-weight + 10% cap
        for d, g in te.groupby("date"):
            g = g.reset_index(drop=True)
            g_sorted = g.sort_values("lgbm_score", ascending=False).head(K_FINAL).reset_index(drop=True)
            mcaps = g_sorted["mcap"].to_numpy(dtype=np.float64)
            mcaps = np.where(np.isnan(mcaps), 0.0, mcaps)
            if mcaps.sum() <= 0:
                n = len(g_sorted); w = np.full(n, 1.0 / n)
            else:
                w = project_to_simplex(np.log(np.maximum(mcaps, EPS)), max_weight=MAX_WEIGHT)
            fwd = g_sorted["fwd_ret_5d"].to_numpy(dtype=np.float64)
            fwd = np.where(np.isnan(fwd), 0.0, fwd)
            all_rows.append({"date": d, "weekly_ret": float(np.dot(w, fwd))})

        # Also compute deterministic top-K_FINAL for comparison (no LGBM)
        log.info("walk %2d done [%.1fs, best_iter=%d, train=%d val=%d test=%d]",
                 walk_id, time.time() - t0, model.best_iteration_, len(tr), len(vl), len(te))

    w_df = pd.DataFrame(all_rows).sort_values("date").reset_index(drop=True)
    w_df["date"] = pd.to_datetime(w_df["date"])

    spy = pd.read_parquet(SPY_PATH).reset_index()[["Date", "close"]].rename(columns={"Date": "date"})
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").set_index("date")["close"]
    all_dates = pd.DatetimeIndex(sorted(w_df["date"].unique()))
    closes = spy.reindex(spy.index.union(all_dates)).sort_index().ffill().reindex(all_dates)
    spy_rets = closes.pct_change().fillna(0.0).to_numpy()
    years = all_dates.year
    mask = years >= 2010

    # Compare to deterministic sp_fcfa baseline
    det = pd.read_parquet(REPO_ROOT / "artifacts" / "backtest_factor_v1" / "weekly_sp_fcfa.parquet")
    det["date"] = pd.to_datetime(det["date"])
    det_rets = det.set_index("date").reindex(all_dates)["weekly_ret"].to_numpy()

    lgbm_rets = w_df.set_index("date").reindex(all_dates)["weekly_ret"].to_numpy()

    print(f"\n=== sp_fcfa wide(100) → LGBM re-rank → top-30 (STANDALONE, no SPY) ===")
    print(f"  2010-2025 BAR (806 wks):")
    print(f"    {'strategy':<40} {'ann':>8} {'vol':>8} {'sharpe':>8} {'mdd':>8}")
    m_spy = metrics(spy_rets[mask])
    m_det = metrics(det_rets[mask])
    m_lgbm = metrics(lgbm_rets[mask])
    print(f"    {'SPY (benchmark)':<40} {m_spy['ann']:>8.2%} {m_spy['vol']:>8.2%} {m_spy['sh']:>8.3f} {m_spy['mdd']:>8.2%}")
    print(f"    {'sp_fcfa det top-30 (mcap)':<40} {m_det['ann']:>8.2%} {m_det['vol']:>8.2%} {m_det['sh']:>8.3f} {m_det['mdd']:>8.2%}")
    print(f"    {'sp_fcfa→LGBM re-rank→top-30':<40} {m_lgbm['ann']:>8.2%} {m_lgbm['vol']:>8.2%} {m_lgbm['sh']:>8.3f} {m_lgbm['mdd']:>8.2%}")
    print(f"\n  ΔSharpe vs det baseline: {m_lgbm['sh'] - m_det['sh']:+.3f}")
    print(f"  ΔSharpe vs SPY:           {m_lgbm['sh'] - m_spy['sh']:+.3f}")

    out_dir = REPO_ROOT / "artifacts" / "backtest_spfcfa_lgbm"
    out_dir.mkdir(parents=True, exist_ok=True)
    w_df.to_parquet(out_dir / "weekly_lgbm_rerank.parquet", compression="zstd", index=False)


if __name__ == "__main__":
    main()
