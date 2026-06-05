"""Live end-to-end smoke test for the weights pipeline.

Marked ``slow`` — hits real external APIs (Sharadar + FRED).  Skip when the
Sharadar key or the persisted model is absent so CI stays fast.

Run manually:
    python -m pytest tests/trading/test_live_smoke.py -v -m slow
"""
import pytest

from trading.config import MODEL_PATH
from src.utils.env import get_env


def _has_api_key() -> bool:
    """Return True iff the Sharadar API key is reachable (loads .env if needed)."""
    return bool(get_env("NASDAQ_DATA_LINK_API_KEY"))


@pytest.mark.slow
@pytest.mark.skipif(
    not _has_api_key() or not MODEL_PATH.exists(),
    reason="needs Sharadar API key + persisted model (trading/models/k_selector.txt)",
)
def test_live_weights_end_to_end():
    """Fetch current S&P 500 data, run the full pipeline, validate the result."""
    from trading.weights import compute_target_weights, validate_weights

    result = compute_target_weights()  # hits Sharadar + FRED for the current Friday

    problems = validate_weights(result)
    assert problems == [], f"validate_weights found problems: {problems}\nresult={result}"

    assert 10 <= result["n_holdings"] <= 503, (
        f"n_holdings={result['n_holdings']} out of [10, 503]"
    )
    assert abs(result["weight_sum"] - 1.0) < 1e-6, (
        f"weight_sum={result['weight_sum']} deviates from 1.0"
    )
    assert result["max_weight"] <= 0.10 + 1e-9, (
        f"max_weight={result['max_weight']} exceeds 0.10 cap"
    )
