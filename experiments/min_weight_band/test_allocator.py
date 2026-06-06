import numpy as np
import pandas as pd
import pytest

from experiments.min_weight_band.allocator import band_water_fill


def test_sums_to_one_and_respects_band():
    w = band_water_fill(np.linspace(100.0, 10.0, 25), floor=0.02, cap=0.10)
    assert len(w) == 25
    assert abs(w.sum() - 1.0) < 1e-9
    assert w.min() >= 0.02 - 1e-9
    assert w.max() <= 0.10 + 1e-9


def test_k10_forces_all_at_cap():
    w = band_water_fill(np.linspace(100.0, 10.0, 10), floor=0.02, cap=0.10)
    np.testing.assert_allclose(w, np.full(10, 0.10), atol=1e-9)


def test_k50_forces_all_at_floor():
    w = band_water_fill(np.linspace(100.0, 10.0, 50), floor=0.02, cap=0.10)
    np.testing.assert_allclose(w, np.full(50, 0.02), atol=1e-9)


def test_tilt_is_monotone_in_mcap_at_intermediate_k():
    # mcaps descending -> weights non-increasing (bigger mcap never weighs less)
    w = band_water_fill(np.linspace(100.0, 10.0, 20), floor=0.02, cap=0.10)
    assert np.all(np.diff(w) <= 1e-9)


def test_equal_weight_when_all_mcap_nonpositive():
    w = band_water_fill(np.zeros(20), floor=0.02, cap=0.10)
    np.testing.assert_allclose(w, np.full(20, 0.05), atol=1e-9)


def test_infeasible_band_raises():
    with pytest.raises(ValueError):
        band_water_fill(np.ones(5), floor=0.02, cap=0.10)   # 5*0.10 = 0.5 < 1
    with pytest.raises(ValueError):
        band_water_fill(np.ones(60), floor=0.02, cap=0.10)  # 60*0.02 = 1.2 > 1
