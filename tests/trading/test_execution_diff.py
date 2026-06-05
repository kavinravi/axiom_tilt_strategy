"""Tests for trading/execution/diff.py — Part A3."""
from __future__ import annotations

import logging
import pytest


# ---------------------------------------------------------------------------
# target_shares tests
# ---------------------------------------------------------------------------

def test_target_shares_basic():
    """shares_i = (w_i * nav) / price_i."""
    from trading.execution.diff import target_shares
    weights = {"AAPL": 0.40, "MSFT": 0.60}
    nav = 10_000.0
    prices = {"AAPL": 100.0, "MSFT": 200.0}
    result = target_shares(weights, nav, prices)
    assert result["AAPL"] == pytest.approx(40.0)   # 0.40 * 10000 / 100
    assert result["MSFT"] == pytest.approx(30.0)   # 0.60 * 10000 / 200


def test_target_shares_fractional():
    """Fractional shares are allowed by default."""
    from trading.execution.diff import target_shares
    weights = {"AAPL": 0.333}
    nav = 10_000.0
    prices = {"AAPL": 150.0}
    result = target_shares(weights, nav, prices)
    # 0.333 * 10000 / 150 = 22.2
    assert result["AAPL"] == pytest.approx(22.2)


def test_target_shares_missing_price_skipped(caplog):
    """Ticker with no price entry is skipped and logged."""
    from trading.execution.diff import target_shares
    weights = {"AAPL": 0.50, "MSFT": 0.50}
    prices = {"AAPL": 100.0}  # MSFT has no price
    with caplog.at_level(logging.WARNING):
        result = target_shares(weights, nav=10_000.0, prices=prices)
    assert "AAPL" in result
    assert "MSFT" not in result
    assert any("MSFT" in r.message for r in caplog.records)


def test_target_shares_zero_price_skipped(caplog):
    """Ticker with price=0 is skipped and logged."""
    from trading.execution.diff import target_shares
    weights = {"AAPL": 0.50, "ZERO": 0.50}
    prices = {"AAPL": 100.0, "ZERO": 0.0}
    with caplog.at_level(logging.WARNING):
        result = target_shares(weights, nav=10_000.0, prices=prices)
    assert "AAPL" in result
    assert "ZERO" not in result


# ---------------------------------------------------------------------------
# diff_to_orders tests
# ---------------------------------------------------------------------------

def test_diff_to_orders_buy_and_sell():
    """2-name portfolio: one needs buying, one selling."""
    from trading.execution.diff import diff_to_orders
    target_weights = {"AAPL": 0.60, "MSFT": 0.40}
    current_positions = {"AAPL": 10.0, "MSFT": 30.0}  # MSFT is over-held
    nav = 10_000.0
    prices = {"AAPL": 100.0, "MSFT": 100.0}
    # target AAPL = 0.60*10000/100 = 60.0 → delta = +50
    # target MSFT = 0.40*10000/100 = 40.0 → delta = +10 (needs to buy too? no: 30 held)
    # Actually MSFT: target=40, current=30 → delta=+10 BUY
    # Let's set MSFT current to 50 → delta = -10 SELL
    current_positions = {"AAPL": 10.0, "MSFT": 50.0}
    orders = diff_to_orders(target_weights, current_positions, nav, prices)
    order_map = {o.ticker: o for o in orders}
    assert "AAPL" in order_map
    assert "MSFT" in order_map
    assert order_map["AAPL"].side == "BUY"
    assert order_map["AAPL"].quantity == pytest.approx(50.0)
    assert order_map["MSFT"].side == "SELL"
    assert order_map["MSFT"].quantity == pytest.approx(10.0)


def test_diff_to_orders_full_liquidation_of_held_ticker():
    """Ticker held but not in target → full liquidation order."""
    from trading.execution.diff import diff_to_orders
    target_weights = {"AAPL": 1.0}
    current_positions = {"AAPL": 20.0, "GOOG": 5.0}  # GOOG not in target
    nav = 10_000.0
    prices = {"AAPL": 100.0, "GOOG": 200.0}
    orders = diff_to_orders(target_weights, current_positions, nav, prices)
    order_map = {o.ticker: o for o in orders}
    assert "GOOG" in order_map
    assert order_map["GOOG"].side == "SELL"
    assert order_map["GOOG"].quantity == pytest.approx(5.0)


def test_diff_to_orders_dust_skipped():
    """Orders below min_order_notional are skipped."""
    from trading.execution.diff import diff_to_orders
    # MSFT: target=100.01 shares, current=100.00 → delta=0.01 → notional=0.01*100=1.0 (boundary)
    # Use delta such that notional < 1.0 (default min)
    target_weights = {"AAPL": 1.0 - 1e-4, "MSFT": 1e-4}
    nav = 10_000.0
    prices = {"AAPL": 100.0, "MSFT": 100.0}
    # MSFT target = 1e-4 * 10000 / 100 = 0.01 shares → notional = 0.01 * 100 = 1.0
    # Use an even smaller fraction
    target_weights = {"AAPL": 1.0 - 5e-6, "MSFT": 5e-6}
    # MSFT target = 5e-6 * 10000 / 100 = 0.0005 shares → notional = 0.0005*100 = 0.05 < 1.0
    current_positions = {}  # no current holdings
    orders = diff_to_orders(target_weights, current_positions, nav, prices,
                            min_order_notional=1.0)
    order_map = {o.ticker: o for o in orders}
    assert "MSFT" not in order_map  # dust, skipped
    assert "AAPL" in order_map


def test_diff_to_orders_no_price_skipped(caplog):
    """Target ticker with no price is skipped."""
    from trading.execution.diff import diff_to_orders
    target_weights = {"AAPL": 0.60, "NOPR": 0.40}
    current_positions = {}
    nav = 10_000.0
    prices = {"AAPL": 100.0}  # NOPR has no price
    with caplog.at_level(logging.WARNING):
        orders = diff_to_orders(target_weights, current_positions, nav, prices)
    tickers = {o.ticker for o in orders}
    assert "NOPR" not in tickers
    assert "AAPL" in tickers


def test_diff_to_orders_no_delta_no_order():
    """No change needed → empty order list."""
    from trading.execution.diff import diff_to_orders
    target_weights = {"AAPL": 1.0}
    nav = 10_000.0
    prices = {"AAPL": 100.0}
    # target = 100 shares; current = 100 shares → delta = 0
    current_positions = {"AAPL": 100.0}
    orders = diff_to_orders(target_weights, current_positions, nav, prices)
    assert orders == []


def test_diff_to_orders_returns_list_of_orders():
    """Return type is a list of Order objects."""
    from trading.broker.base import Order
    from trading.execution.diff import diff_to_orders
    target_weights = {"AAPL": 1.0}
    current_positions = {}
    nav = 5_000.0
    prices = {"AAPL": 100.0}
    orders = diff_to_orders(target_weights, current_positions, nav, prices)
    assert len(orders) == 1
    assert isinstance(orders[0], Order)
    assert orders[0].side == "BUY"
    assert orders[0].quantity == pytest.approx(50.0)
