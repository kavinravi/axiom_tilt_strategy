import numpy as np
import pandas as pd
import pytest

from src.strategy.allocate import (
    apply_min_allocation, band_water_fill, ensemble_weights, topk_mcap_weights,
)
from src.strategy.constants import MIN_ALLOCATION


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
    df = _scored(60)
    with pytest.raises(ValueError):
        ensemble_weights(df, {10: 0.3, 20: 0.3, 30: 0.3, 50: 0.3}, id_col="id")


# ---------------------------------------------------------------------------
# band_water_fill — mcap/weight tilt projected onto [floor, cap] summing to 1
# ---------------------------------------------------------------------------

def test_band_water_fill_sums_to_one_and_respects_band():
    w = band_water_fill(np.linspace(100.0, 10.0, 25), floor=0.01, cap=0.10)
    assert len(w) == 25
    assert abs(w.sum() - 1.0) < 1e-9
    assert w.min() >= 0.01 - 1e-9
    assert w.max() <= 0.10 + 1e-9


def test_band_water_fill_monotone_in_base():
    # descending base -> non-increasing weights (heavier name never weighs less)
    w = band_water_fill(np.linspace(100.0, 10.0, 20), floor=0.01, cap=0.10)
    assert np.all(np.diff(w) <= 1e-9)


def test_band_water_fill_infeasible_band_raises():
    with pytest.raises(ValueError):
        band_water_fill(np.ones(5), floor=0.01, cap=0.10)   # 5*0.10 = 0.5 < 1


# ---------------------------------------------------------------------------
# apply_min_allocation — post-blend floor: drop dust, band-project survivors
# ---------------------------------------------------------------------------

def _blend_with_dust():
    """A blended book: 15 names clearing 1%, plus 36 sub-1% dust names; sums to 1."""
    big = [0.10, 0.09, 0.08, 0.08, 0.07, 0.06, 0.06, 0.05,
           0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02]      # 15 names, sum 0.82
    w = {f"B{i}": v for i, v in enumerate(big)}
    w.update({f"D{i}": 0.005 for i in range(36)})          # 36 * 0.005 = 0.18 dust
    return w


def test_apply_min_allocation_drops_dust_and_respects_band():
    w = apply_min_allocation(_blend_with_dust(), floor=0.01, cap=0.10)
    assert all(not k.startswith("D") for k in w)           # every sub-1% dust name removed
    assert min(w.values()) >= 0.01 - 1e-9
    assert max(w.values()) <= 0.10 + 1e-9
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_apply_min_allocation_reduces_holdings_to_names_above_floor():
    blend = _blend_with_dust()
    w = apply_min_allocation(blend, floor=0.01, cap=0.10)
    assert len(w) < len(blend)
    assert len(w) == 15                                    # exactly the names clearing the floor


def test_apply_min_allocation_preserves_tilt_order():
    w = apply_min_allocation(_blend_with_dust(), floor=0.01, cap=0.10)
    assert w["B0"] >= w["B14"]                             # heavier blend weight -> heavier (or equal)


def test_apply_min_allocation_idempotent():
    once = apply_min_allocation(_blend_with_dust(), floor=0.01, cap=0.10)
    twice = apply_min_allocation(once, floor=0.01, cap=0.10)
    assert set(once) == set(twice)
    for k in once:
        assert abs(once[k] - twice[k]) < 1e-9


def test_apply_min_allocation_keeps_all_names_when_no_dust():
    clean = {f"C{i}": v for i, v in enumerate(
        [0.10, 0.09, 0.08, 0.08, 0.07, 0.06, 0.06, 0.05,
         0.05, 0.04, 0.04, 0.03, 0.03, 0.02, 0.02])}        # all >= 1%
    w = apply_min_allocation(clean, floor=0.01, cap=0.10)
    assert set(w) == set(clean)                            # nothing dropped


def test_apply_min_allocation_default_floor_is_min_allocation():
    assert MIN_ALLOCATION == 0.01
    w = apply_min_allocation(_blend_with_dust())           # default floor
    assert min(w.values()) >= MIN_ALLOCATION - 1e-9
