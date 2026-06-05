"""Shared constants for the strategy core (single source of truth)."""
from __future__ import annotations

K_CANDIDATES = [10, 20, 30, 50]
MAX_WEIGHT = 0.10
EPS = 1e-8
REGIME_FEATURES = [
    "macro_vixcls", "macro_dgs10", "macro_t10y2y",
    "spy_ret_4w", "spy_ret_12w", "spy_vol_12w", "spy_vol_26w",
]
