"""Band allocator: mcap-tilted weights clamped to [floor, cap] summing to 1,
plus the per-K portfolio and the band-aware per-K returns used by the backtest.

Imports only unchanged src/strategy primitives; no edits to the live core.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def band_water_fill(mcaps, floor: float = 0.02, cap: float = 0.10) -> np.ndarray:
    """Project an mcap-proportional target onto {w : floor<=w<=cap, sum w = 1}.

    mcap-proportional base, then iteratively clamp out-of-band names (sticky
    pins) and redistribute the residual across still-free names proportional to
    their mcap base, preserving the tilt. A final slack-based finalizer repairs
    any float residual without violating the band. Raises if the band is
    infeasible for K names (needs K*floor <= 1 <= K*cap).
    """
    mcaps = np.asarray(mcaps, dtype=np.float64)
    K = len(mcaps)
    if K * cap < 1.0 - 1e-12 or K * floor > 1.0 + 1e-12:
        raise ValueError(
            f"infeasible band: K={K}, floor={floor}, cap={cap} "
            f"(need K*cap>=1>=K*floor)"
        )

    clean = np.where(np.isnan(mcaps) | (mcaps <= 0.0), 0.0, mcaps)
    base = np.full(K, 1.0 / K) if clean.sum() <= 0 else clean / clean.sum()

    w = base.copy()
    pinned = np.zeros(K, dtype=bool)
    for _ in range(2 * K + 5):
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
        fb = base[free]
        w[free] = (residual / free.sum() if fb.sum() <= 0
                   else residual * fb / fb.sum())

    # Finalizer: repair float residual by moving only into available slack.
    for _ in range(K + 5):  # converges in <=2 iterations analytically; budget is defensive
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
