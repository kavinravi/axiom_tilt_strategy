"""Strategy core: single source of truth for the k-ensemble math.

Imported by both the backtest (experiments/) and the live trading system.
This package exposes the LIVE core; the backtest-only bulk-panel loader lives
in src/strategy/historical.py and is imported explicitly by its consumers.
"""
from __future__ import annotations

from src.strategy.allocate import ensemble_weights, topk_mcap_weights
from src.strategy.constants import EPS, K_CANDIDATES, MAX_WEIGHT, REGIME_FEATURES
from src.strategy.factors import score_universe
from src.strategy.k_selector import (
    build_regime_features, load_model, make_k_classifier,
    predict_k_probs, save_model, train_model,
)

__all__ = [
    "K_CANDIDATES", "MAX_WEIGHT", "EPS", "REGIME_FEATURES",
    "score_universe", "topk_mcap_weights", "ensemble_weights",
    "build_regime_features", "make_k_classifier", "train_model",
    "save_model", "load_model", "predict_k_probs",
]
