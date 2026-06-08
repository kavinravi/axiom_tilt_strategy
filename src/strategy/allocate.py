"""Allocation: top-K mcap weighting with 10% water-fill cap, and the
probability-weighted ensemble blend.

Wraps src/utils/rl_env.py:project_to_simplex. Because each per-K portfolio
already satisfies w_K(i) <= max_weight and the probabilities sum to 1, the
convex blend sum_K p_K * w_K(i) also satisfies the cap and sums to 1 — no
re-cap needed (matches the backtest).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.strategy.constants import EPS, K_CANDIDATES, MAX_WEIGHT, MIN_ALLOCATION
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


def band_water_fill(base, floor: float = MIN_ALLOCATION,
                    cap: float = MAX_WEIGHT) -> np.ndarray:
    """Project a positive-tilt target onto {w : floor<=w<=cap, sum w = 1}.

    `base` is any non-negative tilt (mcaps, or already-blended weights). It is
    normalised, then out-of-band names are iteratively clamped (sticky pins) and
    the residual redistributed across still-free names proportional to their
    base, preserving the tilt. A final slack-based finalizer repairs float
    residual without violating the band. Raises if the band is infeasible for
    n names (needs n*floor <= 1 <= n*cap)."""
    base = np.asarray(base, dtype=np.float64)
    n = len(base)
    if n * cap < 1.0 - 1e-12 or n * floor > 1.0 + 1e-12:
        raise ValueError(
            f"infeasible band: n={n}, floor={floor}, cap={cap} "
            f"(need n*cap>=1>=n*floor)"
        )

    clean = np.where(np.isnan(base) | (base <= 0.0), 0.0, base)
    tilt = np.full(n, 1.0 / n) if clean.sum() <= 0 else clean / clean.sum()

    w = tilt.copy()
    pinned = np.zeros(n, dtype=bool)
    for _ in range(2 * n + 5):
        over = (w > cap + 1e-15) & ~pinned
        under = (w < floor - 1e-15) & ~pinned
        if not over.any() and not under.any():
            break
        w[over] = cap
        w[under] = floor
        pinned |= over | under
        free = ~pinned
        if not free.any():
            break
        residual = 1.0 - w[pinned].sum()
        fb = tilt[free]
        w[free] = (residual / free.sum() if fb.sum() <= 0
                   else residual * fb / fb.sum())

    # Finalizer: repair float residual by moving only into available slack.
    for _ in range(n + 5):  # converges in <=2 iterations analytically; budget is defensive
        w = np.clip(w, floor, cap)
        residual = 1.0 - w.sum()
        if abs(residual) < 1e-12:
            break
        slack = (cap - w) if residual > 0 else (w - floor)
        s = slack.sum()
        if s <= 1e-15:
            break
        w = w + residual * slack / s
    return w


def apply_min_allocation(weights: dict, floor: float = MIN_ALLOCATION,
                         cap: float = MAX_WEIGHT) -> dict[Any, float]:
    """Impose a minimum allocation on a blended book: drop the sub-`floor` dust,
    then band-project the survivors onto [floor, cap] summing to 1 (their blend
    weights stay the tilt). Single dict in, {id: weight} out.

    Keeps every name whose blended weight clears `floor`; never holds fewer than
    ceil(1/cap) names so the band stays feasible (a guard that does not bind in
    practice — the live blend always leaves well above that many names)."""
    if not weights:
        return {}
    items = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    n_above = sum(1 for _, wt in items if wt >= floor)
    min_feasible = math.ceil(1.0 / cap)            # need >= 1/cap names to fill to 1 under the cap
    n_hold = min(max(n_above, min_feasible), len(items))
    held = items[:n_hold]
    w = band_water_fill(np.asarray([wt for _, wt in held], dtype=np.float64),
                        floor=floor, cap=cap)
    return {idv: float(min(max(wt, floor), cap)) for (idv, _), wt in zip(held, w)}
