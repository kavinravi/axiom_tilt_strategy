"""Tests for trading/run.py — format_rebalance_report (pure helper) + rebalance CLI."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from trading.broker.base import Fill


# ---------------------------------------------------------------------------
# Canned summary dict
# ---------------------------------------------------------------------------

def _make_canned_summary() -> dict:
    fills = [
        Fill(ticker="AAPL", side="BUY", quantity=100.0, avg_price=150.25, status="filled"),
        Fill(ticker="MSFT", side="BUY", quantity=33.0, avg_price=300.50, status="partial"),
        Fill(ticker="GOOG", side="SELL", quantity=10.0, avg_price=180.00, status="filled"),
    ]
    return {
        "asof": pd.Timestamp("2026-06-02"),
        "mode": "dryrun",
        "n_orders": 3,
        "n_filled": 3,
        "fills": fills,
        "audit": {},
        "orders_path": Path("/tmp/2026-06-02.json"),
        "first_build": False,
    }


# ---------------------------------------------------------------------------
# format_rebalance_report — pure unit tests
# ---------------------------------------------------------------------------

def test_format_rebalance_report_returns_string():
    from trading.run import format_rebalance_report
    summary = _make_canned_summary()
    report = format_rebalance_report(summary)
    assert isinstance(report, str)
    assert len(report) > 0


def test_format_rebalance_report_contains_asof():
    from trading.run import format_rebalance_report
    summary = _make_canned_summary()
    report = format_rebalance_report(summary)
    assert "2026-06-02" in report


def test_format_rebalance_report_contains_mode():
    from trading.run import format_rebalance_report
    summary = _make_canned_summary()
    report = format_rebalance_report(summary)
    assert "dryrun" in report


def test_format_rebalance_report_contains_tickers():
    from trading.run import format_rebalance_report
    summary = _make_canned_summary()
    report = format_rebalance_report(summary)
    assert "AAPL" in report
    assert "MSFT" in report
    assert "GOOG" in report


def test_format_rebalance_report_contains_order_count():
    from trading.run import format_rebalance_report
    summary = _make_canned_summary()
    report = format_rebalance_report(summary)
    assert "3" in report


def test_format_rebalance_report_contains_audit_path():
    from trading.run import format_rebalance_report
    summary = _make_canned_summary()
    report = format_rebalance_report(summary)
    assert "2026-06-02.json" in report


def test_format_rebalance_report_is_pure():
    """Calling twice with same input returns same string."""
    from trading.run import format_rebalance_report
    summary = _make_canned_summary()
    assert format_rebalance_report(summary) == format_rebalance_report(summary)


def test_format_rebalance_report_first_build_flag():
    """When first_build=True, the report mentions it."""
    from trading.run import format_rebalance_report
    summary = _make_canned_summary()
    summary["first_build"] = True
    report = format_rebalance_report(summary)
    assert "cash" in report.lower() or "first" in report.lower()


# ---------------------------------------------------------------------------
# Safety: skip_turnover_check in pre_trade_checks
# ---------------------------------------------------------------------------

def test_skip_turnover_check_allows_high_turnover():
    """skip_turnover_check=True bypasses the turnover cap."""
    from trading.broker.base import Order
    from trading.execution.safety import pre_trade_checks
    from trading import config

    nav = 100_000.0
    n = 15
    tickers = [f"T{i:03d}" for i in range(n)]
    target_weights = {t: 1.0 / n for t in tickers}
    prices = {t: 100.0 for t in tickers}
    # Build orders with 100% turnover (each order is the full target allocation)
    orders = [Order(t, "BUY", quantity=(1.0/n) * nav / 100.0) for t in tickers]
    # Total notional = 100_000 = 100% of NAV >> 60%

    # Without skip: should flag turnover
    problems_without = pre_trade_checks(
        target_weights, orders, {}, nav, prices, config=config
    )
    assert any("turnover" in p.lower() for p in problems_without)

    # With skip: turnover check bypassed
    problems_with = pre_trade_checks(
        target_weights, orders, {}, nav, prices,
        config=config, skip_turnover_check=True,
    )
    assert not any("turnover" in p.lower() for p in problems_with)


def test_skip_turnover_check_false_still_flags():
    """skip_turnover_check=False (default) still flags excessive turnover."""
    from trading.broker.base import Order
    from trading.execution.safety import pre_trade_checks
    from trading import config

    nav = 100_000.0
    n = 15
    tickers = [f"T{i:03d}" for i in range(n)]
    target_weights = {t: 1.0 / n for t in tickers}
    prices = {t: 100.0 for t in tickers}
    orders = [Order(t, "BUY", quantity=(1.0/n) * nav / 100.0) for t in tickers]

    problems = pre_trade_checks(
        target_weights, orders, {}, nav, prices,
        config=config, skip_turnover_check=False,
    )
    assert any("turnover" in p.lower() for p in problems)
