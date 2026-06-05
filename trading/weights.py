"""Full weights pipeline: snapshot → scores → regime → k_probs → weights → freeze.

Public API
----------
compute_target_weights(asof, snapshot, regime_row, model) -> dict
    All four parameters are injectable for unit tests; when None each is built
    from the appropriate live data source (network).

freeze_weights(weights, k_probs, asof) -> None
    Persist the result to trading/audit/weights/<asof>.json.

validate_weights(result) -> list[str]
    Return a list of human-readable problems (empty == OK).
"""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from src.strategy.allocate import ensemble_weights
from src.strategy.factors import score_universe
from src.strategy.k_selector import load_model, predict_k_probs
from trading.config import (
    MAX_HOLDINGS,
    MAX_WEIGHT,
    MIN_HOLDINGS,
    MODEL_PATH,
    WEIGHT_SUM_TOL,
    WEIGHTS_DIR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# freeze_weights
# ---------------------------------------------------------------------------

def freeze_weights(
    weights: dict[Any, float],
    k_probs: dict[int, float],
    asof: pd.Timestamp,
) -> None:
    """Persist weights to ``trading/audit/weights/<asof>.json``.

    Creates ``WEIGHTS_DIR`` if it does not exist.  The JSON has three keys:
    ``asof`` (ISO date string), ``k_probs`` (dict with string keys), and
    ``weights`` (dict with ticker keys).
    """
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    audit_path = WEIGHTS_DIR / f"{asof.date()}.json"
    payload = {
        "asof": str(asof.date()),
        "k_probs": {str(k): float(v) for k, v in k_probs.items()},
        "weights": {str(ticker): float(w) for ticker, w in weights.items()},
    }
    with audit_path.open("w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Froze weights → %s  (%d holdings)", audit_path, len(weights))


# ---------------------------------------------------------------------------
# validate_weights
# ---------------------------------------------------------------------------

def validate_weights(result: dict) -> list[str]:
    """Return a list of human-readable problems with the result dict.

    Returns an empty list when all sanity checks pass.

    Checks
    ------
    - ``weight_sum`` is within ``WEIGHT_SUM_TOL`` of 1.0.
    - ``max_weight`` does not exceed ``MAX_WEIGHT + 1e-9``.
    - ``n_holdings`` is between ``MIN_HOLDINGS`` and ``MAX_HOLDINGS`` (inclusive).
    """
    problems: list[str] = []

    weight_sum = result["weight_sum"]
    if abs(weight_sum - 1.0) > WEIGHT_SUM_TOL:
        problems.append(
            f"weight_sum {weight_sum:.8f} deviates from 1.0 by more than {WEIGHT_SUM_TOL}"
        )

    max_w = result["max_weight"]
    if max_w > MAX_WEIGHT + 1e-9:
        problems.append(
            f"max_weight {max_w:.6f} exceeds cap {MAX_WEIGHT:.6f}"
        )

    n = result["n_holdings"]
    if not (MIN_HOLDINGS <= n <= MAX_HOLDINGS):
        problems.append(
            f"n_holdings {n} outside [{MIN_HOLDINGS}, {MAX_HOLDINGS}]"
        )

    return problems


# ---------------------------------------------------------------------------
# compute_target_weights
# ---------------------------------------------------------------------------

def compute_target_weights(
    asof: pd.Timestamp | None = None,
    snapshot: pd.DataFrame | None = None,
    regime_row: np.ndarray | None = None,
    model=None,
) -> dict:
    """Run the full pipeline and return the target weight dict.

    All parameters can be injected (useful for unit tests that avoid network
    calls).  When None, each is sourced from the appropriate live data layer.

    Steps
    -----
    1.  asof  → most recent Friday ≤ today (if not supplied).
    2.  snapshot → build_snapshot(asof) (fetches Sharadar).
    3.  scored  → score_universe(snapshot, id_col="ticker").
    4.  regime_row → build_current_regime_row(asof) (fetches FRED).
    5.  model → load_model(MODEL_PATH).
    6.  k_probs → predict_k_probs(model, regime_row).
    7.  weights → ensemble_weights(scored, k_probs, id_col="ticker").
    8.  freeze_weights(weights, k_probs, asof) → audit JSON.
    9.  return result dict.

    Returns
    -------
    dict with keys: asof, weights, k_probs, n_holdings, weight_sum, max_weight.
    """
    # Step 1 — rebalance date
    if asof is None:
        from trading.data.snapshot import most_recent_friday  # noqa: PLC0415
        asof = most_recent_friday()
    asof = pd.Timestamp(asof).normalize()

    # Step 2 — snapshot (one cross-section row per S&P 500 ticker)
    if snapshot is None:
        from trading.data.snapshot import build_snapshot  # noqa: PLC0415
        snapshot = build_snapshot(asof)

    # Step 3 — factor scores
    scored = score_universe(snapshot, id_col="ticker")

    # Step 4 — current regime feature row
    if regime_row is None:
        from trading.regime import build_current_regime_row  # noqa: PLC0415
        regime_row = build_current_regime_row(asof)
    regime_row = np.asarray(regime_row, dtype=float)

    # Step 5 — K-selector model
    if model is None:
        model = load_model(MODEL_PATH)

    # Step 6 — regime-conditioned K-probabilities
    k_probs: dict[int, float] = predict_k_probs(model, regime_row)

    # Step 7 — probability-weighted ensemble of top-K portfolios
    weights: dict[Any, float] = ensemble_weights(scored, k_probs, id_col="ticker")

    # Step 8 — persist to audit directory
    freeze_weights(weights, k_probs, asof)

    # Step 9 — build and return result dict
    w_vals = list(weights.values())
    result: dict = {
        "asof": asof,
        "weights": weights,
        "k_probs": k_probs,
        "n_holdings": len(weights),
        "weight_sum": float(np.sum(w_vals)) if w_vals else 0.0,
        "max_weight": float(np.max(w_vals)) if w_vals else 0.0,
    }
    logger.info(
        "compute_target_weights: asof=%s  n=%d  sum=%.6f  max=%.4f",
        asof.date(), result["n_holdings"], result["weight_sum"], result["max_weight"],
    )
    return result
