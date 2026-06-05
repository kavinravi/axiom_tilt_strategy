"""BACKTEST-ONLY data path: bulk historical panel loader + label builders.

Shared by experiments/regime_k_selector*.py and src/strategy/train.py to remove
the prior copy-paste. NOT for live trading — the live snapshot is built
separately (trading/data/snapshot.py, Plan 2). Not imported by the package
__init__ to keep the live core import light.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.allocate import topk_mcap_weights
from src.strategy.constants import K_CANDIDATES, MAX_WEIGHT
from src.strategy.factors import score_universe
from src.utils.io import processed_dir, repo_root
from src.utils.ranker import friday_only

REPO_ROOT = repo_root()
PANEL_DIR = processed_dir() / "panel"
TRAIN_PANEL_DIR = processed_dir() / "training_panel"
SPY_PATH = REPO_ROOT / "artifacts" / "benchmarks" / "spy_daily.parquet"


def load_data() -> pd.DataFrame:
    """Load + merge the historical panel, keep Friday in-universe rows, and
    attach factor scores. Mirrors the prior inline load_data, but scoring is
    delegated to score_universe (single source of truth)."""
    cols = ["permno", "date", "prc", "shrout", "marketcap", "in_universe",
            "revenue", "fcf", "assets"]
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
    df = score_universe(df, id_col="permno")
    return df


def per_k_weights_and_returns(df: pd.DataFrame, K: int,
                              max_weight: float = MAX_WEIGHT):
    """Per Friday: top-K mcap-weighted (cap10) weights + the portfolio's
    fwd_ret_5d. Returns (weight_df[date,permno,weight], return_series[date])."""
    weight_rows = []
    return_rows = []
    for d, g in df.groupby("date", sort=False):
        w = topk_mcap_weights(g, K, max_weight=max_weight, id_col="permno")
        gk = g.sort_values("score", ascending=False).head(K)
        fwd = dict(zip(gk["permno"].astype(int).to_numpy(),
                       np.nan_to_num(gk["fwd_ret_5d"].to_numpy(dtype=np.float64))))
        ret = float(sum(w[p] * fwd[int(p)] for p in w))
        return_rows.append({"date": d, "ret": ret})
        for p, wt in w.items():
            if wt > 0:
                weight_rows.append({"date": d, "permno": int(p), "weight": float(wt)})
    wdf = pd.DataFrame(weight_rows)
    rdf = pd.DataFrame(return_rows).sort_values("date").set_index("date")["ret"]
    return wdf, rdf


def build_k_labels(k_returns: dict, all_dates: pd.DatetimeIndex,
                   K_candidates: list | None = None):
    """Label = argmax-K of per-K weekly returns per Friday. Returns (labels, k_mat).
    labels are int class indices (0..len(K)-1), NaN where all K returns are NaN."""
    if K_candidates is None:
        K_candidates = K_CANDIDATES
    k_mat = pd.DataFrame(
        {f"K{K}": k_returns[K].reindex(all_dates).values for K in K_candidates},
        index=all_dates,
    )
    k_to_idx = {K: i for i, K in enumerate(K_candidates)}
    labels = k_mat.idxmax(axis=1).str[1:].astype("Int64").map(k_to_idx)
    return labels, k_mat


def load_spy_at(all_dates: pd.DatetimeIndex) -> pd.Series:
    """SPY close sampled at all_dates (ffilled)."""
    spy = pd.read_parquet(SPY_PATH).reset_index()[["Date", "close"]].rename(columns={"Date": "date"})
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").set_index("date")["close"]
    return spy.reindex(spy.index.union(all_dates)).sort_index().ffill().reindex(all_dates)


def macro_by_date(df: pd.DataFrame, all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """First macro reading per date, reindexed to all_dates."""
    return (df.groupby("date", sort=False)[["macro_vixcls", "macro_dgs10", "macro_t10y2y"]]
              .first().reindex(all_dates))
