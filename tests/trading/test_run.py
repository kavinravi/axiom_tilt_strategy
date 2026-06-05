"""Tests for trading/run.py — Task 7.

format_report is tested purely on a canned result dict: no network, no model.
"""
from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_canned_result() -> dict:
    """Build a deterministic result dict to feed to format_report."""
    weights = {
        "AAPL": 0.09,
        "MSFT": 0.08,
        "NVDA": 0.07,
        "GOOG": 0.06,
        "AMZN": 0.05,
        "META": 0.05,
        "TSLA": 0.04,
        "AVGO": 0.04,
        "BRK.B": 0.04,
        "JPMC": 0.03,
        "JPM": 0.03,
        "UNH": 0.03,
        "XOM": 0.03,
        "V": 0.02,
        "LLY": 0.02,
        "PG": 0.02,
        "MA": 0.02,
        "HD": 0.02,
        "COST": 0.02,
        "ABBV": 0.02,
    }
    total = sum(weights.values())
    return {
        "asof": pd.Timestamp("2026-05-29"),
        "weights": weights,
        "k_probs": {10: 0.4, 20: 0.3, 30: 0.2, 50: 0.1},
        "n_holdings": len(weights),
        "weight_sum": total,
        "max_weight": max(weights.values()),
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_format_report_contains_holdings_count():
    from trading.run import format_report
    result = _make_canned_result()
    text = format_report(result)
    assert "20" in text, "Expected holdings count (20) in report"


def test_format_report_contains_weight_sum():
    from trading.run import format_report
    result = _make_canned_result()
    text = format_report(result)
    # Weight sum line must appear somewhere (either "sum" or the value)
    assert "sum" in text.lower() or "weight_sum" in text.lower(), (
        "Expected 'sum' label in report"
    )


def test_format_report_contains_top_ticker():
    from trading.run import format_report
    result = _make_canned_result()
    text = format_report(result)
    # The top-weighted ticker (AAPL, 9%) must appear in the table
    assert "AAPL" in text, "Expected top ticker AAPL in report"


def test_format_report_contains_asof():
    from trading.run import format_report
    result = _make_canned_result()
    text = format_report(result)
    assert "2026-05-29" in text, "Expected asof date in report"


def test_format_report_returns_string():
    from trading.run import format_report
    result = _make_canned_result()
    text = format_report(result)
    assert isinstance(text, str)
    assert len(text) > 0


def test_format_report_weights_sorted_descending():
    """The ticker with the highest weight must appear before the lowest in the text."""
    from trading.run import format_report
    result = _make_canned_result()
    text = format_report(result)
    aapl_pos = text.find("AAPL")
    abbv_pos = text.find("ABBV")   # lowest weight (0.02)
    assert aapl_pos != -1 and abbv_pos != -1
    assert aapl_pos < abbv_pos, "Expected descending weight order (AAPL before ABBV)"


def test_format_report_contains_k_probs():
    from trading.run import format_report
    result = _make_canned_result()
    text = format_report(result)
    # k_probs must appear — at minimum the numeric keys
    assert "k_probs" in text.lower() or "k=" in text.lower() or "k10" in text.lower() or "k=10" in text.lower() or "10:" in text, (
        "Expected k_probs info in report"
    )


def test_format_report_is_pure():
    """Calling format_report twice with the same input must return the same string."""
    from trading.run import format_report
    result = _make_canned_result()
    assert format_report(result) == format_report(result)
