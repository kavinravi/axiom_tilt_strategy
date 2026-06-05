"""Tests for trading/regime.py.

All tests inject synthetic spy_weekly and macro frames — no network calls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.strategy.constants import REGIME_FEATURES
from src.strategy.k_selector import build_regime_features
from trading.config import REGIME_HISTORY_WEEKS
from trading.regime import build_current_regime_row


ASOF = pd.Timestamp("2026-05-29")  # a known Friday

# Build the same index that build_current_regime_row will use internally
_INDEX = pd.date_range(end=ASOF, periods=REGIME_HISTORY_WEEKS, freq="W-FRI")


def _make_spy(index: pd.DatetimeIndex) -> pd.Series:
    """Synthetic rising SPY series with no NaNs."""
    levels = np.linspace(3000.0, 5000.0, len(index))
    return pd.Series(levels, index=index, name="close")


def _make_macro(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Constant macro frame — one row per index entry."""
    return pd.DataFrame({
        "macro_vixcls": 20.0,
        "macro_dgs10":  4.0,
        "macro_t10y2y": -0.5,
    }, index=index)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_numpy_array():
    spy = _make_spy(_INDEX)
    macro = _make_macro(_INDEX)
    row = build_current_regime_row(ASOF, spy_weekly=spy, macro=macro)
    assert isinstance(row, np.ndarray)


def test_row_length_is_7():
    spy = _make_spy(_INDEX)
    macro = _make_macro(_INDEX)
    row = build_current_regime_row(ASOF, spy_weekly=spy, macro=macro)
    assert len(row) == 7


def test_row_is_finite():
    spy = _make_spy(_INDEX)
    macro = _make_macro(_INDEX)
    row = build_current_regime_row(ASOF, spy_weekly=spy, macro=macro)
    assert np.all(np.isfinite(row)), f"Non-finite values in regime row: {row}"


def test_column_order_matches_REGIME_FEATURES():
    """The regime row order must exactly match REGIME_FEATURES."""
    spy = _make_spy(_INDEX)
    macro = _make_macro(_INDEX)
    row = build_current_regime_row(ASOF, spy_weekly=spy, macro=macro)
    # Compare against core function's last row
    regime_df = build_regime_features(_INDEX, spy, macro)
    expected = regime_df.iloc[-1].to_numpy(dtype=float)
    np.testing.assert_array_almost_equal(row, expected)


def test_consistency_with_core_build_regime_features():
    """build_current_regime_row must equal build_regime_features(...).iloc[-1]."""
    spy = _make_spy(_INDEX)
    macro = _make_macro(_INDEX)

    row = build_current_regime_row(ASOF, spy_weekly=spy, macro=macro)
    regime_df = build_regime_features(_INDEX, spy, macro)
    expected = regime_df.iloc[-1].to_numpy(dtype=float)

    assert len(row) == len(REGIME_FEATURES)
    np.testing.assert_array_equal(row, expected)


def test_asof_determines_index_end():
    """Different asof dates produce the same row when SPY/macro inputs are identical.

    This confirms the function correctly uses its index and that injected series
    are passed straight through to build_regime_features without modification.
    """
    spy = _make_spy(_INDEX)
    macro = _make_macro(_INDEX)

    row = build_current_regime_row(ASOF, spy_weekly=spy, macro=macro)
    regime_df = build_regime_features(_INDEX, spy, macro)
    expected = regime_df.iloc[-1].to_numpy(dtype=float)

    np.testing.assert_array_equal(row, expected)
    assert len(row) == len(REGIME_FEATURES)


def test_different_macro_produces_different_row():
    """Different macro inputs (e.g. different VIX) produce different regime rows."""
    macro_high_vix = pd.DataFrame({
        "macro_vixcls": 40.0,
        "macro_dgs10":  4.0,
        "macro_t10y2y": -0.5,
    }, index=_INDEX)
    macro_low_vix = pd.DataFrame({
        "macro_vixcls": 10.0,
        "macro_dgs10":  4.0,
        "macro_t10y2y": -0.5,
    }, index=_INDEX)

    spy = _make_spy(_INDEX)

    row_high = build_current_regime_row(ASOF, spy_weekly=spy, macro=macro_high_vix)
    row_low = build_current_regime_row(ASOF, spy_weekly=spy, macro=macro_low_vix)

    assert not np.allclose(row_high, row_low), "Expected different rows for different macro inputs"
