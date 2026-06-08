"""Band allocator: mcap-tilted weights clamped to [floor, cap] summing to 1,
plus the per-K portfolio and the band-aware per-K returns used by the backtest.

Imports only unchanged src/strategy primitives; no edits to the live core.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# band_water_fill now lives in the live core (src/strategy/allocate.py); the
# backtest re-uses the exact same projection so research and production agree.
from src.strategy.allocate import band_water_fill


def band_topk(scored_df: pd.DataFrame, K: int, floor: float = 0.02,
              cap: float = 0.10, id_col: str = "id") -> dict[Any, float]:
    """Top-K by `score`, band-weighted by mcap. Single-date frame in,
    {id: weight} out. Exactly K holdings, each in [floor, cap], summing to 1."""
    g = scored_df.sort_values("score", ascending=False).head(K).reset_index(drop=True)
    w = band_water_fill(g["mcap"].to_numpy(dtype=np.float64), floor=floor, cap=cap)
    return {idv: float(min(max(wt, floor), cap))
            for idv, wt in zip(g[id_col].to_numpy(), w)}


def band_per_k_weights_and_returns(df: pd.DataFrame, K: int, floor: float = 0.02,
                                   cap: float = 0.10):
    """Per Friday: top-K band-weighted weights + the portfolio's fwd_ret_5d.

    Mirrors src/strategy/historical.per_k_weights_and_returns but uses band_topk
    instead of the cap-only topk_mcap_weights. Returns
    (weight_df[date,permno,weight], return_series indexed by date)."""
    weight_rows = []
    return_rows = []
    for d, g in df.groupby("date", sort=False):
        w = band_topk(g, K, floor=floor, cap=cap, id_col="permno")
        gk = g.sort_values("score", ascending=False).head(K)
        fwd = dict(zip(gk["permno"].astype(int).to_numpy(),
                       np.nan_to_num(gk["fwd_ret_5d"].to_numpy(dtype=np.float64))))
        ret = float(sum(w[p] * fwd[int(p)] for p in w))
        return_rows.append({"date": d, "ret": ret})
        for p, wt in w.items():
            weight_rows.append({"date": d, "permno": int(p), "weight": float(wt)})
    wdf = pd.DataFrame(weight_rows)
    rdf = pd.DataFrame(return_rows).sort_values("date").set_index("date")["ret"]
    return wdf, rdf
