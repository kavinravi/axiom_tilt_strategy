"""Tests for the FRED REST-API observation parser (api.stlouisfed.org path).

The live trading box can't reach the pandas_datareader CSV host
(fred.stlouisfed.org) but CAN reach the REST API host (api.stlouisfed.org) with
a key. _parse_fred_observations turns that JSON into a clean float series.
"""
from __future__ import annotations

import pandas as pd
import pytest

from trading.data.sources import _parse_fred_observations


def test_parse_fred_observations_skips_placeholders_and_sorts():
    payload = {"observations": [
        {"date": "2026-06-02", "value": "4.41"},
        {"date": "2026-06-01", "value": "."},        # FRED placeholder for missing -> skipped
        {"date": "2026-06-03", "value": "4.45"},
    ]}
    s = _parse_fred_observations(payload)
    assert list(s.index) == [pd.Timestamp("2026-06-02"), pd.Timestamp("2026-06-03")]
    assert s.loc[pd.Timestamp("2026-06-02")] == 4.41
    assert s.loc[pd.Timestamp("2026-06-03")] == 4.45
    assert s.index.is_monotonic_increasing


def test_parse_fred_observations_all_placeholders_raises():
    # fail loud rather than silently returning an empty series that zeroes a feature
    with pytest.raises(Exception):
        _parse_fred_observations({"observations": [{"date": "2026-06-01", "value": "."}]})


def test_parse_fred_observations_no_observations_key_raises():
    with pytest.raises(Exception):
        _parse_fred_observations({"error_message": "Bad Request. Invalid api_key."})
