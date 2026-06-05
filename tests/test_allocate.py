import numpy as np
import pandas as pd

from src.strategy.allocate import ensemble_weights, topk_mcap_weights


def _scored(n):
    return pd.DataFrame({
        "id": list(range(n)),
        "score": np.linspace(1.0, 0.0, n),   # descending
        "mcap": np.linspace(100.0, 10.0, n),  # descending
    })


def test_topk_picks_k_names_sums_to_one_respects_cap():
    w = topk_mcap_weights(_scored(20), K=10, max_weight=0.10, id_col="id")
    assert len(w) == 10
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert max(w.values()) <= 0.10 + 1e-9
    # picks the top-10 by score (ids 0..9)
    assert set(w.keys()) == set(range(10))


def test_topk_equal_weight_fallback_when_all_mcap_zero():
    df = _scored(10)
    df["mcap"] = 0.0
    w = topk_mcap_weights(df, K=10, max_weight=0.10, id_col="id")
    np.testing.assert_allclose(sorted(w.values()), [0.1] * 10)


def test_ensemble_is_convex_sums_to_one_and_preserves_cap():
    df = _scored(60)
    k_probs = {10: 0.25, 20: 0.25, 30: 0.25, 50: 0.25}
    w = ensemble_weights(df, k_probs, id_col="id")
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert max(w.values()) <= 0.10 + 1e-9


def test_ensemble_concentrated_prob_matches_single_k():
    df = _scored(60)
    only10 = ensemble_weights(df, {10: 1.0, 20: 0.0, 30: 0.0, 50: 0.0}, id_col="id")
    direct10 = topk_mcap_weights(df, K=10, max_weight=0.10, id_col="id")
    assert set(only10.keys()) == set(direct10.keys())
    for k in direct10:
        assert abs(only10[k] - direct10[k]) < 1e-9


def test_ensemble_rejects_non_normalized_probs():
    import pytest
    df = _scored(60)
    with pytest.raises(ValueError):
        ensemble_weights(df, {10: 0.3, 20: 0.3, 30: 0.3, 50: 0.3}, id_col="id")
