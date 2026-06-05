"""Allocation: top-K mcap weighting with 10% water-fill cap, and the
probability-weighted ensemble blend.

Wraps src/utils/rl_env.py:project_to_simplex. Because each per-K portfolio
already satisfies w_K(i) <= max_weight and the probabilities sum to 1, the
convex blend sum_K p_K * w_K(i) also satisfies the cap and sums to 1 — no
re-cap needed (matches the backtest).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.strategy.constants import EPS, K_CANDIDATES, MAX_WEIGHT
from src.utils.rl_env import project_to_simplex


def topk_mcap_weights(scored_df: pd.DataFrame, K: int,
                      max_weight: float = MAX_WEIGHT, id_col: str = "id") -> dict[Any, float]:
    """Top-K by score, mcap-weighted via water-fill cap. Single-date frame in,
    {id: weight} out. Requires columns: `id_col`, score, mcap."""
    g = scored_df.sort_values("score", ascending=False).head(K).reset_index(drop=True)
    mcaps = g["mcap"].to_numpy(dtype=np.float64)
    mcaps = np.where(np.isnan(mcaps), 0.0, mcaps)
    if mcaps.sum() <= 0:
        n = len(g)
        w = np.full(n, 1.0 / n)
    else:
        w = project_to_simplex(np.log(np.maximum(mcaps, EPS)), max_weight=max_weight)
    return {idv: min(float(wt), max_weight) for idv, wt in zip(g[id_col].to_numpy(), w)}  # min clamp: project_to_simplex returns float32; guard the float32->float64 rounding so the 10% cap stays a hard bound


def ensemble_weights(scored_df: pd.DataFrame, k_probs: dict,
                     K_candidates: list | None = None,
                     max_weight: float = MAX_WEIGHT, id_col: str = "id") -> dict[Any, float]:
    """Convex combination w(i) = sum_K p(K) * w_K(i). Single-date frame in,
    {id: weight} out. `k_probs` maps K -> probability."""
    if K_candidates is None:
        K_candidates = K_CANDIDATES
    total_p = sum(k_probs.values())
    if abs(total_p - 1.0) > 1e-6:
        raise ValueError(f"k_probs must sum to 1, got {total_p:.6f}")
    combined: dict = {}
    for K in K_candidates:
        p = float(k_probs[K])
        wK = topk_mcap_weights(scored_df, K, max_weight=max_weight, id_col=id_col)
        for idv, wt in wK.items():
            combined[idv] = combined.get(idv, 0.0) + p * wt
    return {idv: wt for idv, wt in combined.items() if wt > EPS}
