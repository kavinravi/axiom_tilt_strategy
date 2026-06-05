"""Tests for trading/broker/dryrun.py — Part A2.

Updated for the non-blocking submit / get_fill / cancel interface.
"""
from __future__ import annotations

import logging

import pytest

from trading.broker.base import Order, OrderHandle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

POSITIONS = {"AAPL": 10.0, "MSFT": 5.0}
NAV = 50_000.0
QUOTES = {
    "AAPL": (149.0, 151.0),   # midpoint = 150.0
    "MSFT": (299.0, 301.0),   # midpoint = 300.0
}


def make_broker(**kwargs):
    from trading.broker.dryrun import DryRunBroker
    defaults = dict(positions=POSITIONS, nav=NAV, quotes=QUOTES)
    defaults.update(kwargs)
    return DryRunBroker(**defaults)


# ---------------------------------------------------------------------------
# Accessor tests
# ---------------------------------------------------------------------------

def test_get_positions():
    b = make_broker()
    assert b.get_positions() == POSITIONS


def test_get_nav():
    b = make_broker()
    assert b.get_nav() == NAV


def test_get_quote_known_ticker():
    b = make_broker()
    bid, ask = b.get_quote("AAPL")
    assert bid == 149.0
    assert ask == 151.0


def test_get_quote_unknown_ticker_synthesized():
    """Unknown ticker gets synthesized quote (price ± half-spread defaults)."""
    b = make_broker()
    bid, ask = b.get_quote("UNKNOWN")
    assert bid > 0
    assert ask > bid


# ---------------------------------------------------------------------------
# connect / disconnect are no-ops (no exception)
# ---------------------------------------------------------------------------

def test_connect_disconnect_noop():
    b = make_broker()
    b.connect()
    b.disconnect()


# ---------------------------------------------------------------------------
# submit_limit tests
# ---------------------------------------------------------------------------

def test_submit_limit_returns_order_handle():
    b = make_broker()
    order = Order("AAPL", "BUY", 2.0)
    handle = b.submit_limit(order, limit_price=152.0)
    assert isinstance(handle, OrderHandle)
    assert handle.ticker == "AAPL"
    assert handle.side == "BUY"
    assert handle.quantity == 2.0
    assert handle.order_type == "LMT"


def test_submit_limit_buy_marketable_fills_at_limit():
    """BUY limit above ask → marketable → get_fill returns filled status."""
    b = make_broker()
    order = Order("AAPL", "BUY", 2.0)
    handle = b.submit_limit(order, limit_price=152.0)  # above ask 151 → marketable
    fill = b.get_fill(handle)
    assert fill.status == "filled"
    assert fill.quantity == 2.0
    assert fill.avg_price == 152.0
    assert fill.side == "BUY"


def test_submit_limit_buy_non_marketable_unfilled():
    """BUY limit below bid → non-marketable → get_fill returns unfilled."""
    b = make_broker()
    order = Order("AAPL", "BUY", 2.0)
    handle = b.submit_limit(order, limit_price=148.0)  # below bid 149 → not marketable
    fill = b.get_fill(handle)
    assert fill.status == "unfilled"
    assert fill.quantity == 0.0


def test_submit_limit_sell_marketable_fills_at_limit():
    """SELL limit below bid → marketable → filled."""
    b = make_broker()
    order = Order("AAPL", "SELL", 3.0)
    handle = b.submit_limit(order, limit_price=148.0)  # below bid 149 → marketable
    fill = b.get_fill(handle)
    assert fill.status == "filled"
    assert fill.quantity == 3.0
    assert fill.avg_price == 148.0


def test_submit_limit_sell_non_marketable_unfilled():
    """SELL limit above ask → non-marketable → unfilled."""
    b = make_broker()
    order = Order("AAPL", "SELL", 3.0)
    handle = b.submit_limit(order, limit_price=152.0)  # above ask → not marketable
    fill = b.get_fill(handle)
    assert fill.status == "unfilled"
    assert fill.quantity == 0.0


# ---------------------------------------------------------------------------
# submit_midprice tests
# ---------------------------------------------------------------------------

def test_submit_midprice_buy_fills_at_midpoint():
    b = make_broker()
    order = Order("AAPL", "BUY", 4.0)
    handle = b.submit_midprice(order)
    assert handle.order_type == "MIDPRICE"
    fill = b.get_fill(handle)
    assert fill.status == "filled"
    assert fill.avg_price == pytest.approx(150.0)  # (149+151)/2
    assert fill.quantity == 4.0


def test_submit_midprice_sell_fills_at_midpoint():
    b = make_broker()
    order = Order("MSFT", "SELL", 2.0)
    handle = b.submit_midprice(order)
    fill = b.get_fill(handle)
    assert fill.status == "filled"
    assert fill.avg_price == pytest.approx(300.0)  # (299+301)/2
    assert fill.quantity == 2.0


# ---------------------------------------------------------------------------
# submit_market tests
# ---------------------------------------------------------------------------

def test_submit_market_buy_fills_at_ask():
    b = make_broker()
    order = Order("AAPL", "BUY", 1.0)
    handle = b.submit_market(order)
    assert handle.order_type == "MKT"
    fill = b.get_fill(handle)
    assert fill.status == "filled"
    assert fill.avg_price == 151.0  # ask


def test_submit_market_sell_fills_at_bid():
    b = make_broker()
    order = Order("AAPL", "SELL", 1.0)
    handle = b.submit_market(order)
    fill = b.get_fill(handle)
    assert fill.status == "filled"
    assert fill.avg_price == 149.0  # bid


# ---------------------------------------------------------------------------
# cancel tests
# ---------------------------------------------------------------------------

def test_cancel_is_idempotent():
    """cancel() can be called multiple times without raising."""
    b = make_broker()
    order = Order("AAPL", "BUY", 2.0)
    handle = b.submit_limit(order, limit_price=152.0)
    b.cancel(handle)
    b.cancel(handle)  # second call must not raise


def test_cancel_does_not_reverse_fill():
    """cancel() marks the handle cancelled but does not un-fill already-filled qty."""
    b = make_broker()
    order = Order("AAPL", "BUY", 2.0)
    handle = b.submit_limit(order, limit_price=152.0)  # marketable → filled
    fill_before = b.get_fill(handle)
    assert fill_before.quantity == 2.0
    b.cancel(handle)
    fill_after = b.get_fill(handle)
    assert fill_after.quantity == 2.0  # unchanged


def test_cancel_on_unknown_handle_is_harmless():
    """cancel() on an unregistered handle does not raise."""
    from trading.broker.base import OrderHandle
    b = make_broker()
    fake_handle = OrderHandle(ref=9999, ticker="AAPL", side="BUY",
                              quantity=1.0, order_type="LMT")
    b.cancel(fake_handle)  # should not raise


# ---------------------------------------------------------------------------
# fill_ratio tests
# ---------------------------------------------------------------------------

def test_fill_ratio_partial_midprice():
    b = make_broker(fill_ratio=0.5)
    order = Order("AAPL", "BUY", 4.0)
    handle = b.submit_midprice(order)
    fill = b.get_fill(handle)
    assert fill.status == "partial"
    assert fill.quantity == pytest.approx(2.0)


def test_fill_ratio_zero_midprice_unfilled():
    b = make_broker(fill_ratio=0.0)
    order = Order("AAPL", "BUY", 4.0)
    handle = b.submit_midprice(order)
    fill = b.get_fill(handle)
    assert fill.status == "unfilled"
    assert fill.quantity == 0.0


def test_fill_ratio_partial_market():
    b = make_broker(fill_ratio=0.25)
    order = Order("MSFT", "SELL", 4.0)
    handle = b.submit_market(order)
    fill = b.get_fill(handle)
    assert fill.status == "partial"
    assert fill.quantity == pytest.approx(1.0)


def test_fill_ratio_partial_limit_marketable():
    b = make_broker(fill_ratio=0.5)
    order = Order("AAPL", "BUY", 4.0)
    handle = b.submit_limit(order, limit_price=152.0)  # marketable
    fill = b.get_fill(handle)
    assert fill.status == "partial"
    assert fill.quantity == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Multiple handles are independent
# ---------------------------------------------------------------------------

def test_multiple_handles_independent():
    """Each submit call gets a distinct handle; get_fill returns the right fill each time."""
    b = make_broker()
    order_a = Order("AAPL", "BUY", 2.0)
    order_b = Order("MSFT", "SELL", 3.0)
    h_a = b.submit_market(order_a)
    h_b = b.submit_market(order_b)
    fill_a = b.get_fill(h_a)
    fill_b = b.get_fill(h_b)
    assert fill_a.ticker == "AAPL"
    assert fill_a.avg_price == 151.0   # AAPL ask
    assert fill_b.ticker == "MSFT"
    assert fill_b.avg_price == 299.0   # MSFT bid


# ---------------------------------------------------------------------------
# Logging: submit_* should log the intended order
# ---------------------------------------------------------------------------

def test_submit_market_logs_order(caplog):
    b = make_broker()
    order = Order("AAPL", "BUY", 1.0)
    with caplog.at_level(logging.INFO):
        b.submit_market(order)
    assert any("AAPL" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Fix 3 — cancel() + get_fill() for unfilled-then-cancelled handles
# ---------------------------------------------------------------------------

def test_cancel_unfilled_handle_reports_cancelled_status():
    """An unfilled order that is then cancelled must report status='cancelled'."""
    b = make_broker()
    order = Order("AAPL", "BUY", 2.0)
    # Non-marketable limit → unfilled (limit below bid 149)
    handle = b.submit_limit(order, limit_price=148.0)
    fill_before = b.get_fill(handle)
    assert fill_before.status == "unfilled"
    assert fill_before.quantity == 0.0

    b.cancel(handle)
    fill_after = b.get_fill(handle)
    assert fill_after.status == "cancelled", (
        f"Expected 'cancelled' after cancel() of unfilled order, got {fill_after.status!r}"
    )
    assert fill_after.quantity == 0.0  # qty unchanged (still 0)


def test_cancel_filled_handle_does_not_change_status():
    """Cancelling an already-filled order must NOT change its status to 'cancelled'."""
    b = make_broker()
    order = Order("AAPL", "BUY", 2.0)
    handle = b.submit_limit(order, limit_price=152.0)  # marketable → filled
    b.cancel(handle)
    fill = b.get_fill(handle)
    # qty > 0, so should remain "filled" not "cancelled"
    assert fill.status == "filled"
    assert fill.quantity == pytest.approx(2.0)
