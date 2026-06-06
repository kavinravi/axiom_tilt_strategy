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


def test_finalizer_fires_when_all_pinned_simultaneously():
    # K=3, floor=0.2, cap=0.7: iteration 0 pins all three simultaneously,
    # leaving sum=1.1; finalizer must repair to [0.6, 0.2, 0.2].
    w = band_water_fill([10.0, 1.0, 1.0], floor=0.2, cap=0.7)
    assert abs(w.sum() - 1.0) < 1e-9
    assert w.min() >= 0.2 - 1e-9
    assert w.max() <= 0.7 + 1e-9
    assert w[0] > w[1]  # mcap order preserved: heaviest name above floor


def test_nan_and_negative_mcaps_get_floor():
    # indices 1 and 3 have bad mcaps -> should receive floor; others get
    # mcap-proportional above floor.
    w = band_water_fill([100.0, np.nan, 50.0, -5.0, 10.0], floor=0.02, cap=0.50)
    assert abs(w.sum() - 1.0) < 1e-9
    assert w.min() >= 0.02 - 1e-9
    assert w.max() <= 0.50 + 1e-9
    np.testing.assert_allclose(w[1], 0.02, atol=1e-9)
    np.testing.assert_allclose(w[3], 0.02, atol=1e-9)


from experiments.min_weight_band.allocator import band_topk


def _scored(n):
    return pd.DataFrame({
        "id": list(range(n)),
        "score": np.linspace(1.0, 0.0, n),    # descending
        "mcap": np.linspace(100.0, 10.0, n),  # descending
    })


def test_band_topk_picks_k_names_in_band_summing_to_one():
    w = band_topk(_scored(60), K=25, floor=0.02, cap=0.10, id_col="id")
    assert len(w) == 25
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert min(w.values()) >= 0.02 - 1e-9
    assert max(w.values()) <= 0.10 + 1e-9
    assert set(w.keys()) == set(range(25))  # top-25 by score


def test_band_topk_k10_all_at_cap():
    w = band_topk(_scored(60), K=10, floor=0.02, cap=0.10, id_col="id")
    np.testing.assert_allclose(sorted(w.values()), [0.10] * 10, atol=1e-9)


from experiments.min_weight_band.allocator import band_per_k_weights_and_returns


def _panel_two_dates():
    rows = []
    for d in (pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-10")):
        for i in range(30):
            rows.append({"date": d, "permno": i, "score": 30 - i,
                         "mcap": float(100 - i), "fwd_ret_5d": 0.01 * (i % 5 - 2)})
    return pd.DataFrame(rows)


def test_band_per_k_returns_shape_and_band():
    wdf, rdf = band_per_k_weights_and_returns(_panel_two_dates(), K=20)
    # weights: each date has exactly 20 names, all in band, summing to 1
    for d, g in wdf.groupby("date"):
        assert len(g) == 20
        assert g["weight"].min() >= 0.02 - 1e-9
        assert g["weight"].max() <= 0.10 + 1e-9
        assert abs(g["weight"].sum() - 1.0) < 1e-9
    # returns: one finite value per date
    assert len(rdf) == 2
    assert rdf.notna().all()
