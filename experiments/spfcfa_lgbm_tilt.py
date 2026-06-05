"""sp_fcfa + LightGBM weight-tilt strategy.

Two-stage standalone (no SPY, IPS-compliant):

  Stage 1 — Selection (deterministic factor screen):
    Per Friday t, score each S&P 500 PIT member by
       composite = 0.5 * z(revenue/mcap) + 0.5 * z(fcf/assets)
    Take top-30 by composite.

  Stage 2 — Tilt (LightGBM regression model, walk-forward retrained):
    Train a LightGBM regressor per walk on the training-period rows of
    sp_fcfa-selected picks (the same top-30 picks but during training years).
    Target: fwd_ret_5d - cross-sectional-mean (next-week excess return).
    Features: a broad set of Sharadar fundamentals + macro + sp_fcfa z-scores
              that the parent's permutation importance flagged as helpful.

    At inference (test year), score each Friday's sp_fcfa top-30. Convert
    raw predictions to within-30 z-scores. Tilt the mcap baseline:

       baseline_w = mcap-weighted top-30 with 10% cap (water-fill projection)
       log_w = log(baseline_w) + α * tilt_score
       new_w = softmax(log_w) → water-fill to 10% cap

    α = tilt scale (default 1.0; 0 = pure mcap baseline).

Outputs:
  artifacts/rl_factor_spfcfa/walk-NNN/lgbm_model.joblib
  artifacts/backtest_spfcfa_lgbm/weekly.parquet
  reports/spfcfa_lgbm_vs_spy.md
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
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
SPFCFA_SB_ROOT = REPO_ROOT / "artifacts" / "rl_factor_spfcfa"
SPY_PATH = REPO_ROOT / "artifacts" / "benchmarks" / "spy_daily.parquet"
OUT_BACKTEST = REPO_ROOT / "artifacts" / "backtest_spfcfa_lgbm"
TOP_K = 30
MAX_WEIGHT = 0.10
EPS = 1e-8

# Feature set for LightGBM tilt model. Trimmed list — fundamentals known to
# work as factor signals + macro + sp_fcfa z-scores themselves.
ML_FEATURES = [
    # Value-related (different from sp_fcfa to add complementary signal)
    "pe", "pb", "ps", "evebitda", "fcfp_local",
    # Quality (different from FCFA)
    "roic_local", "gpa_local", "gross_margin_local",
    "currentratio", "de", "netmargin",
    # Momentum / price-based
    "ret", "prc_pct_change_1m",  # we'll compute prc_pct_change_1m
    # Macro
    "macro_vixcls", "macro_dgs10", "macro_t10y2y",
    # The sp_fcfa z-scores themselves (model may learn to use)
    "z_sp", "z_fcfa",
    # Size proxy
    "marketcap",
]


def load_panel_full(years):
    """Load panel + training_panel + sp_fcfa scoreboards joined for ML training."""
    panel_cols = [
        "permno", "date", "prc", "shrout", "marketcap", "in_universe", "ret",
        "netinc", "equity", "revenue", "fcf", "assets",
        "pe", "pb", "ps", "evebitda", "currentratio", "de", "netmargin",
        "ebit", "invcap", "gp",
    ]
    frames = []
    for y in years:
        for p in sorted((PANEL_DIR / f"year={y}").glob("*.parquet")):
            df = pd.read_parquet(p, columns=panel_cols)
            df["date"] = pd.to_datetime(df["date"])
            df["permno"] = df["permno"].astype("int64")
            frames.append(df)
    daily = pd.concat(frames, ignore_index=True)

    train_cols = ["permno", "date", "fwd_ret_5d",
                  "macro_vixcls", "macro_dgs10", "macro_t10y2y"]
    train_frames = []
    for y in years:
        for p in sorted((TRAIN_PANEL_DIR / f"year={y}").glob("*.parquet")):
            df = pd.read_parquet(p, columns=train_cols)
            df["date"] = pd.to_datetime(df["date"])
            df["permno"] = df["permno"].astype("int64")
            train_frames.append(df)
    fri = pd.concat(train_frames, ignore_index=True)
    return daily, fri


def compute_extra_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute prc_pct_change_1m + safer ratios. Sorted within permno."""
    daily = daily.sort_values(["permno", "date"]).reset_index(drop=True)
    grp = daily.groupby("permno", sort=False)
    daily["prc_pct_change_1m"] = grp["prc"].transform(
        lambda x: x.pct_change(21))  # 21 trading days ≈ 1 month
    # Local versions of factors (clipped, NaN-safe) — different from sp_fcfa's
    daily["fcfp_local"] = (daily["fcf"] / np.maximum(daily["marketcap"], EPS)).clip(lower=0)
    daily["roic_local"] = (daily["ebit"] / daily["invcap"]).clip(lower=-1.0, upper=2.0)
    daily.loc[daily["invcap"] <= 0, "roic_local"] = np.nan
    daily["gpa_local"] = (daily["gp"] / daily["assets"]).clip(lower=-1.0, upper=2.0)
    daily.loc[daily["assets"] <= 0, "gpa_local"] = np.nan
    daily["gross_margin_local"] = (daily["gp"] / daily["revenue"]).clip(lower=-1.0, upper=2.0)
    daily.loc[daily["revenue"] <= 0, "gross_margin_local"] = np.nan
    return daily


def build_walk_dataset(walk_id: int, test_year_end: int):
    """Build the training and test dataframes for walk_id.

    train_df = sp_fcfa-selected rows from train years (2002..test_year_end-2)
    val_df   = sp_fcfa-selected rows from val year (test_year_end - 1)
    test_df  = sp_fcfa-selected rows from test year (test_year_end)
    All Friday-only, with ML_FEATURES + target = excess fwd_ret_5d.
    """
    panel_years = list(range(2001, test_year_end + 1))
    daily, fri = load_panel_full(panel_years)
    daily = compute_extra_features(daily)

    # join
    df = daily.merge(fri, on=["permno", "date"], how="inner")
    df = df.dropna(subset=["fwd_ret_5d"]).copy()
    df = friday_only(df).reset_index(drop=True)
    df = df[df["in_universe"]].copy()

    df["mcap"] = df["marketcap"]
    df.loc[df["mcap"].isna(), "mcap"] = (np.abs(df.loc[df["mcap"].isna(), "prc"]) *
                                          df.loc[df["mcap"].isna(), "shrout"])

    # Compute sp_fcfa signals + z-scores per Friday
    df["sp"] = (df["revenue"] / df["mcap"]).clip(lower=0)
    df["fcfa"] = (df["fcf"] / df["assets"]).clip(lower=-1.0, upper=2.0)
    df.loc[df["assets"] <= 0, "fcfa"] = np.nan
    for col_in, col_out in [("sp", "z_sp"), ("fcfa", "z_fcfa")]:
        g = df.groupby("date", sort=False)[col_in]
        df[col_out] = (df[col_in] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        df[col_out] = df[col_out].fillna(0.0)
    df["score"] = 0.5 * df["z_sp"] + 0.5 * df["z_fcfa"]

    # Cross-sectional excess label
    df["y_excess"] = df["fwd_ret_5d"] - df.groupby("date", sort=False)["fwd_ret_5d"].transform("mean")

    # sp_fcfa top-30 per Friday
    sb = (df.sort_values(["date", "score"], ascending=[True, False])
            .groupby("date", sort=False, group_keys=False)
            .head(TOP_K)
            .reset_index(drop=True))

    # Year splits
    train_year_end = test_year_end - 2
    val_year = test_year_end - 1
    test_year = test_year_end

    sb["year"] = sb["date"].dt.year
    train_df = sb[(sb["year"] >= 2002) & (sb["year"] <= train_year_end)].copy()
    val_df = sb[sb["year"] == val_year].copy()
    test_df = sb[sb["year"] == test_year].copy()

    return train_df, val_df, test_df


def fit_lgbm(train_df, val_df, features):
    """Fit a LightGBM regressor on (features → y_excess) with early stopping on val."""
    X_tr = train_df[features].astype(float).fillna(0.0).to_numpy()
    y_tr = train_df["y_excess"].astype(float).to_numpy()
    X_vl = val_df[features].astype(float).fillna(0.0).to_numpy()
    y_vl = val_df["y_excess"].astype(float).to_numpy()

    model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=15,
        min_data_in_leaf=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        lambda_l2=2.0,
        verbose=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
              callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
    return model


def predict_test(model, test_df, features):
    X = test_df[features].astype(float).fillna(0.0).to_numpy()
    return model.predict(X)


def tilt_weights(mcaps: np.ndarray, tilt_scores: np.ndarray, alpha: float) -> np.ndarray:
    """Combine mcap baseline with tilt scores. Then project to capped simplex.
    tilt_scores are predictions (NOT z-scored yet); we z-score them within the
    30 picks here so the alpha parameter has consistent interpretation."""
    safe_mcap = np.maximum(mcaps, EPS)
    log_baseline = np.log(safe_mcap / safe_mcap.sum())
    # z-score within the 30 picks
    s = np.asarray(tilt_scores, dtype=np.float64)
    if s.std() > 0:
        s = (s - s.mean()) / s.std()
    else:
        s = np.zeros_like(s)
    tilted = log_baseline + alpha * s
    return project_to_simplex(tilted, max_weight=MAX_WEIGHT)


def metrics(rets):
    r = np.asarray(rets, dtype=float)
    cum = float(np.prod(1.0 + r) - 1.0)
    ann = (1.0 + cum) ** (52.0 / len(r)) - 1.0
    vol = float(np.std(r, ddof=1) * np.sqrt(52.0))
    sh = ann / vol if vol > 0 else 0.0
    eq = np.cumprod(1.0 + r); peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    cal = ann / abs(mdd) if mdd < 0 else 0.0
    return {"ann": ann, "vol": vol, "sh": sh, "mdd": mdd, "cal": cal}


def main():
    configure_logging()
    OUT_BACKTEST.mkdir(parents=True, exist_ok=True)

    all_weekly = []
    feature_imps = {}
    for walk_id in range(1, 18):
        t0 = time.time()
        test_year = 2008 + walk_id
        train_df, val_df, test_df = build_walk_dataset(walk_id, test_year)
        if len(val_df) < 10 or len(test_df) < 10:
            log.info("walk %2d: insufficient val/test data (val=%d, test=%d), skipping",
                     walk_id, len(val_df), len(test_df))
            continue
        model = fit_lgbm(train_df, val_df, ML_FEATURES)
        preds = predict_test(model, test_df, ML_FEATURES)
        test_df = test_df.copy()
        test_df["lgbm_pred"] = preds

        # Compute weekly returns under several α values
        for alpha in [0.0, 0.5, 1.0, 2.0]:
            for d, g in test_df.groupby("date"):
                g = g.reset_index(drop=True)
                mcaps = g["mcap"].to_numpy(dtype=np.float64)
                mcaps = np.where(np.isnan(mcaps), 0.0, mcaps)
                if mcaps.sum() <= 0:
                    n = len(g); w = np.full(n, 1.0 / n)
                else:
                    if alpha == 0.0:
                        # pure mcap baseline
                        w = project_to_simplex(np.log(np.maximum(mcaps, EPS)), max_weight=MAX_WEIGHT)
                    else:
                        w = tilt_weights(mcaps, g["lgbm_pred"].to_numpy(), alpha)
                fwd = g["fwd_ret_5d"].to_numpy(dtype=np.float64)
                fwd = np.where(np.isnan(fwd), 0.0, fwd)
                all_weekly.append({
                    "walk_id": walk_id, "date": d, "alpha": alpha,
                    "weekly_ret": float(np.dot(w, fwd)),
                })

        feature_imps[walk_id] = dict(zip(ML_FEATURES, model.booster_.feature_importance(importance_type='gain')))
        log.info("walk %2d (test %d) done [%.1fs, train_rows=%d, val_rows=%d, test_rows=%d, best_iter=%d]",
                 walk_id, test_year, time.time() - t0, len(train_df), len(val_df), len(test_df),
                 model.best_iteration_)

    w_df = pd.DataFrame(all_weekly)
    w_df["date"] = pd.to_datetime(w_df["date"])

    # SPY benchmark aligned
    spy = pd.read_parquet(SPY_PATH).reset_index()[["Date", "close"]].rename(columns={"Date": "date"})
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").set_index("date")["close"]
    all_dates = pd.DatetimeIndex(sorted(w_df["date"].unique()))
    closes = spy.reindex(spy.index.union(all_dates)).sort_index().ffill().reindex(all_dates)
    spy_rets = pd.Series(closes.pct_change().fillna(0.0).values, index=all_dates)

    # Print metrics for each α + SPY across windows
    print()
    print(f"=== sp_fcfa + LightGBM tilt (STANDALONE, no SPY in portfolio) ===")
    years = all_dates.year
    masks = [
        ("2009-2025 (full)", np.ones(len(all_dates), dtype=bool)),
        ("2010-2024",        (years >= 2010) & (years <= 2024)),
        ("2010-2025 (BAR)",  years >= 2010),
    ]
    for label, mask in masks:
        spy_m = metrics(spy_rets[mask].to_numpy())
        print(f"\n--- {label} ---")
        print(f"  {'strategy':<32} {'ann':>8} {'vol':>8} {'sharpe':>8} {'mdd':>8} {'cal':>7}")
        print(f"  {'SPY (benchmark)':<32} {spy_m['ann']:>8.2%} {spy_m['vol']:>8.2%} {spy_m['sh']:>8.3f} {spy_m['mdd']:>8.2%} {spy_m['cal']:>7.3f}")
        for alpha in [0.0, 0.5, 1.0, 2.0]:
            sub = w_df[w_df["alpha"] == alpha].set_index("date").reindex(all_dates)["weekly_ret"].to_numpy()
            m = metrics(sub[mask])
            tag = "mcap-only" if alpha == 0.0 else f"+ LGBM tilt α={alpha}"
            marker = " ✓" if m['sh'] > spy_m['sh'] else ""
            print(f"  {'sp_fcfa ' + tag:<32} {m['ann']:>8.2%} {m['vol']:>8.2%} {m['sh']:>8.3f} {m['mdd']:>8.2%} {m['cal']:>7.3f}{marker}")

    # Save weekly returns
    w_df.to_parquet(OUT_BACKTEST / "weekly_all_alphas.parquet", compression="zstd", index=False)
    log.info("wrote -> %s", OUT_BACKTEST / "weekly_all_alphas.parquet")


if __name__ == "__main__":
    main()
