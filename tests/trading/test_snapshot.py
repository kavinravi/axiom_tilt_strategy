"""Tests for trading/data/snapshot.py.

Uses small in-memory frames so no network calls are made.
Tests the PURE assemble_snapshot helper and most_recent_friday utility.
"""
from __future__ import annotations

import pandas as pd
import pytest

from trading.data.snapshot import assemble_snapshot, most_recent_friday


# ---------------------------------------------------------------------------
# most_recent_friday
# ---------------------------------------------------------------------------

def test_most_recent_friday_on_wednesday_returns_prev_friday():
    # 2026-06-03 is a Wednesday; previous Friday is 2026-05-29
    result = most_recent_friday(pd.Timestamp("2026-06-03"))
    assert result == pd.Timestamp("2026-05-29")


def test_most_recent_friday_on_friday_returns_same_day():
    # 2026-05-29 is itself a Friday
    result = most_recent_friday(pd.Timestamp("2026-05-29"))
    assert result == pd.Timestamp("2026-05-29")


def test_most_recent_friday_on_saturday_returns_prev_friday():
    # 2026-05-30 is Saturday; previous Friday is 2026-05-29
    result = most_recent_friday(pd.Timestamp("2026-05-30"))
    assert result == pd.Timestamp("2026-05-29")


def test_most_recent_friday_on_monday_returns_prev_friday():
    # 2026-06-01 is a Monday; previous Friday is 2026-05-29
    result = most_recent_friday(pd.Timestamp("2026-06-01"))
    assert result == pd.Timestamp("2026-05-29")


# ---------------------------------------------------------------------------
# assemble_snapshot — pure merge helper
# ---------------------------------------------------------------------------

ASOF = pd.Timestamp("2026-05-29")

_UNIVERSE = ["AAA", "BBB", "CCC"]

_FUNDAMENTALS = pd.DataFrame({
    "ticker":    ["AAA", "BBB"],
    "datekey":   pd.to_datetime(["2026-02-01", "2026-01-15"]),
    "revenue":   [100.0, 50.0],
    "fcf":       [10.0, 5.0],
    "assets":    [500.0, 200.0],
    "price":     [25.0, 10.0],
    "sharesbas": [1000.0, 500.0],
})

_MARKETCAPS = pd.DataFrame({
    "ticker":    ["AAA", "BBB"],
    "date":      pd.to_datetime(["2026-05-28", "2026-05-27"]),
    "marketcap": [30000.0, 6000.0],
})


def test_assemble_snapshot_required_columns():
    snap = assemble_snapshot(_UNIVERSE, _FUNDAMENTALS, _MARKETCAPS, ASOF)
    required = {"ticker", "date", "prc", "shrout", "marketcap", "revenue", "fcf", "assets"}
    assert set(snap.columns) == required


def test_assemble_snapshot_marketcap_from_daily():
    snap = assemble_snapshot(_UNIVERSE, _FUNDAMENTALS, _MARKETCAPS, ASOF)
    row = snap.set_index("ticker").loc["AAA"]
    assert row["marketcap"] == 30000.0


def test_assemble_snapshot_prc_and_shrout_from_sf1():
    snap = assemble_snapshot(_UNIVERSE, _FUNDAMENTALS, _MARKETCAPS, ASOF)
    row = snap.set_index("ticker").loc["BBB"]
    assert row["prc"] == 10.0
    assert row["shrout"] == 500.0


def test_assemble_snapshot_date_equals_asof_for_all_rows():
    snap = assemble_snapshot(_UNIVERSE, _FUNDAMENTALS, _MARKETCAPS, ASOF)
    assert (snap["date"] == ASOF).all()


def test_assemble_snapshot_drops_ticker_missing_fundamentals():
    # CCC is in universe but has no fundamentals row → should be dropped
    snap = assemble_snapshot(_UNIVERSE, _FUNDAMENTALS, _MARKETCAPS, ASOF)
    assert "CCC" not in snap["ticker"].values
    assert len(snap) == 2


def test_assemble_snapshot_missing_marketcap_is_nan():
    # Only AAA has a marketcap; BBB's marketcap should be NaN (score_universe fallback)
    mktcaps_partial = _MARKETCAPS[_MARKETCAPS["ticker"] == "AAA"].copy()
    snap = assemble_snapshot(_UNIVERSE, _FUNDAMENTALS, mktcaps_partial, ASOF)
    row = snap.set_index("ticker").loc["BBB"]
    assert pd.isna(row["marketcap"])
    # But prc and shrout must still be present
    assert row["prc"] == 10.0
    assert row["shrout"] == 500.0


def test_assemble_snapshot_column_order():
    snap = assemble_snapshot(_UNIVERSE, _FUNDAMENTALS, _MARKETCAPS, ASOF)
    expected = ["ticker", "date", "prc", "shrout", "marketcap", "revenue", "fcf", "assets"]
    assert list(snap.columns) == expected
