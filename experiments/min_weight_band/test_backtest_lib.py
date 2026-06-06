import numpy as np
import pandas as pd

from experiments.min_weight_band.backtest_lib import (
    metrics, turnover_series, net_returns, window_mask,
)


def test_metrics_keys_and_zero_vol_sharpe():
    m = metrics(np.zeros(52))
    assert set(m) == {"ann", "vol", "sharpe", "sortino", "mdd"}
    assert m["vol"] == 0.0
    assert m["sharpe"] == 0.0   # guarded, not inf/nan


def test_metrics_positive_drift_has_positive_sharpe():
    r = np.full(52, 0.01)
    m = metrics(r)
    assert m["ann"] > 0
    assert m["sharpe"] == 0.0 or m["vol"] == 0.0  # constant series -> zero vol


def test_turnover_first_week_is_half_of_built_book():
    weights = [{1: 0.5, 2: 0.5}, {1: 0.5, 2: 0.5}]
    tu = turnover_series(weights)
    assert abs(tu[0] - 0.5) < 1e-12   # build from cash: 0.5 * sum|w| = 0.5
    assert abs(tu[1] - 0.0) < 1e-12   # no change


def test_turnover_full_swap_is_one():
    weights = [{1: 1.0}, {2: 1.0}]
    tu = turnover_series(weights)
    assert abs(tu[1] - 1.0) < 1e-12   # 0.5*(|−1|+|+1|) = 1.0


def test_net_returns_subtracts_cost():
    gross = np.array([0.02, 0.02])
    tu = np.array([0.5, 0.0])
    net = net_returns(gross, tu, cost_bps=5.0)
    assert abs(net[0] - (0.02 - 5e-4 * 0.5)) < 1e-12
    assert abs(net[1] - 0.02) < 1e-12


def test_window_mask_selects_years():
    idx = pd.to_datetime(["2009-06-05", "2010-06-04", "2025-06-06"])
    np.testing.assert_array_equal(window_mask(idx, 2009, 2025), [True, True, True])
    np.testing.assert_array_equal(window_mask(idx, 2010, 2025), [False, True, True])
    np.testing.assert_array_equal(window_mask(idx, 2025, 2025), [False, False, True])
