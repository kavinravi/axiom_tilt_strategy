"""Mock unit tests for IBKRBroker — no network required.

Strategy: construct an IBKRBroker without calling connect(), then inject a
fake ``ib`` object via direct attribute assignment (IBKRBroker.ib = ...).
All ib_async interactions are replaced with in-memory fakes.

Updated for the non-blocking submit_* / get_fill / cancel interface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from trading.broker.base import Fill, Order, OrderHandle
from trading.broker.ibkr import IBKRBroker, _map_status


# ---------------------------------------------------------------------------
# Fake ib_async helpers
# ---------------------------------------------------------------------------

def _fake_position(symbol: str, pos: float, sec_type: str = "STK"):
    """Build a fake ib_async Position-like object."""
    contract = SimpleNamespace(symbol=symbol, secType=sec_type)
    return SimpleNamespace(contract=contract, position=pos)


def _fake_account_value(tag: str, value: str, currency: str = "USD"):
    """Build a fake AccountValue-like object."""
    return SimpleNamespace(tag=tag, value=value, currency=currency)


def _fake_trade(
    status: str = "Filled",
    filled: float = 10.0,
    avg_fill_price: float = 150.0,
) -> SimpleNamespace:
    """Build a fake Trade-like object."""
    order_status = SimpleNamespace(
        status=status,
        filled=filled,
        avgFillPrice=avg_fill_price,
    )
    trade = SimpleNamespace(
        order=SimpleNamespace(orderId=42),
        orderStatus=order_status,
    )
    # isDone() returns True when status is Filled or Cancelled
    done_states = {"Filled", "ApiCancelled", "Cancelled", "Inactive"}
    trade.isDone = lambda: status in done_states
    return trade


def _fake_ticker(bid: float = 150.0, ask: float = 151.0):
    """Build a fake Ticker-like object returned by reqMktData."""
    return SimpleNamespace(bid=bid, ask=ask)


def _make_broker(host="127.0.0.1", port=4002, client_id=11, readonly=False) -> IBKRBroker:
    """Construct a broker without connecting."""
    broker = IBKRBroker.__new__(IBKRBroker)
    broker._host = host
    broker._port = port
    broker._client_id = client_id
    broker._readonly = readonly
    return broker


def _inject_ib(broker: IBKRBroker, ib=None) -> MagicMock:
    """Attach a MagicMock (or provided fake) as broker.ib."""
    if ib is None:
        ib = MagicMock()
    broker.ib = ib
    return ib


# ---------------------------------------------------------------------------
# _map_status
# ---------------------------------------------------------------------------

class TestMapStatus:
    def test_filled(self):
        assert _map_status("Filled") == "filled"

    def test_partial_fill(self):
        assert _map_status("PartialFill") == "partial"

    def test_cancelled(self):
        assert _map_status("Cancelled") == "cancelled"

    def test_api_cancelled(self):
        assert _map_status("ApiCancelled") == "cancelled"

    def test_inactive(self):
        assert _map_status("Inactive") == "cancelled"

    def test_submitted(self):
        assert _map_status("Submitted") == "unfilled"

    def test_presubmitted(self):
        assert _map_status("PreSubmitted") == "unfilled"


# ---------------------------------------------------------------------------
# get_positions
# ---------------------------------------------------------------------------

class TestGetPositions:
    def test_empty(self):
        broker = _make_broker()
        ib = _inject_ib(broker)
        ib.positions.return_value = []
        assert broker.get_positions() == {}

    def test_single_stk(self):
        broker = _make_broker()
        ib = _inject_ib(broker)
        ib.positions.return_value = [_fake_position("AAPL", 100.0)]
        result = broker.get_positions()
        assert result == {"AAPL": 100.0}

    def test_multiple_tickers(self):
        broker = _make_broker()
        ib = _inject_ib(broker)
        ib.positions.return_value = [
            _fake_position("AAPL", 50.0),
            _fake_position("MSFT", 30.0),
        ]
        result = broker.get_positions()
        assert result == {"AAPL": 50.0, "MSFT": 30.0}

    def test_non_stk_excluded(self):
        """Non-STK contracts (options, futures) are filtered out."""
        broker = _make_broker()
        ib = _inject_ib(broker)
        ib.positions.return_value = [
            _fake_position("AAPL", 100.0, sec_type="STK"),
            _fake_position("AAPL", 1.0, sec_type="OPT"),
        ]
        result = broker.get_positions()
        assert result == {"AAPL": 100.0}

    def test_duplicate_symbol_summed(self):
        """Two STK entries for the same symbol (different accounts) are summed."""
        broker = _make_broker()
        ib = _inject_ib(broker)
        ib.positions.return_value = [
            _fake_position("AAPL", 60.0),
            _fake_position("AAPL", 40.0),
        ]
        result = broker.get_positions()
        assert result == {"AAPL": 100.0}


# ---------------------------------------------------------------------------
# get_nav
# ---------------------------------------------------------------------------

class TestGetNav:
    def test_returns_net_liquidation(self):
        broker = _make_broker()
        ib = _inject_ib(broker)
        ib.accountSummary.return_value = [
            _fake_account_value("TotalCashValue", "50000.00"),
            _fake_account_value("NetLiquidation", "123456.78"),
            _fake_account_value("EquityWithLoanValue", "100000.00"),
        ]
        assert broker.get_nav() == pytest.approx(123456.78)

    def test_filters_non_usd(self):
        """Only USD NetLiquidation counts."""
        broker = _make_broker()
        ib = _inject_ib(broker)
        ib.accountSummary.return_value = [
            _fake_account_value("NetLiquidation", "999.00", currency="EUR"),
            _fake_account_value("NetLiquidation", "123456.78", currency="USD"),
        ]
        assert broker.get_nav() == pytest.approx(123456.78)

    def test_missing_tag_raises(self):
        broker = _make_broker()
        ib = _inject_ib(broker)
        ib.accountSummary.return_value = [
            _fake_account_value("TotalCashValue", "50000.00"),
        ]
        with pytest.raises(RuntimeError, match="NetLiquidation"):
            broker.get_nav()


# ---------------------------------------------------------------------------
# get_quote
# ---------------------------------------------------------------------------

class TestGetQuote:
    def _setup_qualify(self, ib, symbol="AAPL"):
        """Make qualifyContracts return a simple contract."""
        contract = SimpleNamespace(symbol=symbol, secType="STK")
        ib.qualifyContracts.return_value = [contract]
        return contract

    def test_real_time_bid_ask(self):
        broker = _make_broker()
        ib = _inject_ib(broker)
        self._setup_qualify(ib)
        ib.reqMktData.return_value = _fake_ticker(bid=149.5, ask=150.0)
        ib.sleep = MagicMock()

        bid, ask = broker.get_quote("AAPL")
        assert bid == pytest.approx(149.5)
        assert ask == pytest.approx(150.0)
        # Should cancel the subscription
        ib.cancelMktData.assert_called()
        # Should NOT have switched to delayed data
        ib.reqMarketDataType.assert_not_called()

    def test_fallback_to_delayed(self):
        """When real-time returns NaN, fall back to delayed data type 3."""
        broker = _make_broker()
        ib = _inject_ib(broker)
        self._setup_qualify(ib)
        # First call: NaN (no real-time sub); second call: valid delayed quote
        ib.reqMktData.side_effect = [
            _fake_ticker(bid=float("nan"), ask=float("nan")),
            _fake_ticker(bid=149.0, ask=150.5),
        ]
        ib.sleep = MagicMock()

        bid, ask = broker.get_quote("AAPL")
        assert bid == pytest.approx(149.0)
        assert ask == pytest.approx(150.5)
        ib.reqMarketDataType.assert_any_call(3)

    def test_still_nan_after_delayed_raises(self):
        """If still NaN after delayed data, raise RuntimeError."""
        broker = _make_broker()
        ib = _inject_ib(broker)
        self._setup_qualify(ib)
        ib.reqMktData.return_value = _fake_ticker(bid=float("nan"), ask=float("nan"))
        ib.sleep = MagicMock()

        with pytest.raises(RuntimeError, match="get_quote"):
            broker.get_quote("AAPL")


# ---------------------------------------------------------------------------
# Non-blocking submit_limit — contract + order construction + tif
# ---------------------------------------------------------------------------

class TestSubmitLimit:
    def _setup(self, status="Filled", filled=100.0, avg_price=150.0):
        broker = _make_broker(readonly=False)
        ib = _inject_ib(broker)
        contract = SimpleNamespace(symbol="AAPL", secType="STK")
        ib.qualifyContracts.return_value = [contract]
        trade = _fake_trade(status=status, filled=filled, avg_fill_price=avg_price)
        ib.placeOrder.return_value = trade
        ib.sleep = MagicMock()
        return broker, ib, trade

    def test_returns_order_handle_immediately(self):
        """submit_limit must return an OrderHandle without waiting."""
        broker, ib, trade = self._setup()
        order = Order(ticker="AAPL", side="BUY", quantity=100.0)
        handle = broker.submit_limit(order, limit_price=150.0)

        assert isinstance(handle, OrderHandle)
        assert handle.ticker == "AAPL"
        assert handle.side == "BUY"
        assert handle.quantity == pytest.approx(100.0)
        assert handle.order_type == "LMT"
        assert handle.ref is trade  # ib_async Trade is stored as ref

    def test_constructs_limit_order_with_tif_day(self):
        """submit_limit must build a LMT order with tif='DAY'."""
        broker, ib, _ = self._setup()
        order = Order(ticker="AAPL", side="BUY", quantity=100.0)
        broker.submit_limit(order, limit_price=150.0)

        placed_ib_order = ib.placeOrder.call_args[0][1]
        assert placed_ib_order.orderType == "LMT"
        assert placed_ib_order.action == "BUY"
        assert placed_ib_order.totalQuantity == 100.0
        assert placed_ib_order.lmtPrice == pytest.approx(150.0)
        assert placed_ib_order.tif == "DAY"

    def test_get_fill_returns_filled_fill(self):
        """get_fill reads trade.orderStatus snapshot after submit."""
        broker, ib, trade = self._setup(status="Filled", filled=100.0, avg_price=150.25)
        ib.sleep = MagicMock()
        order = Order(ticker="AAPL", side="BUY", quantity=100.0)
        handle = broker.submit_limit(order, limit_price=150.0)

        fill = broker.get_fill(handle)
        assert fill.ticker == "AAPL"
        assert fill.side == "BUY"
        assert fill.quantity == pytest.approx(100.0)
        assert fill.avg_price == pytest.approx(150.25)
        assert fill.status == "filled"

    def test_get_fill_partial(self):
        broker, ib, _ = self._setup(status="PartialFill", filled=50.0, avg_price=149.5)
        order = Order(ticker="AAPL", side="BUY", quantity=100.0)
        handle = broker.submit_limit(order, limit_price=149.0)
        fill = broker.get_fill(handle)
        assert fill.status == "partial"
        assert fill.quantity == pytest.approx(50.0)

    def test_get_fill_cancelled(self):
        broker, ib, _ = self._setup(status="Cancelled", filled=0.0, avg_price=0.0)
        order = Order(ticker="AAPL", side="SELL", quantity=50.0)
        handle = broker.submit_limit(order, limit_price=200.0)
        fill = broker.get_fill(handle)
        assert fill.status == "cancelled"
        assert fill.quantity == pytest.approx(0.0)

    def test_readonly_raises(self):
        broker = _make_broker(readonly=True)
        _inject_ib(broker)
        with pytest.raises(RuntimeError, match="readonly"):
            broker.submit_limit(Order("AAPL", "BUY", 10.0), 150.0)

    def test_sell_side(self):
        broker, ib, _ = self._setup()
        order = Order(ticker="MSFT", side="SELL", quantity=25.5)
        broker.submit_limit(order, limit_price=300.0)
        placed = ib.placeOrder.call_args[0][1]
        assert placed.action == "SELL"
        assert placed.totalQuantity == pytest.approx(25.5)

    def test_fractional_quantity(self):
        broker, ib, _ = self._setup(filled=0.75, avg_price=150.0)
        order = Order(ticker="AAPL", side="BUY", quantity=0.75)
        handle = broker.submit_limit(order, limit_price=150.0)
        placed = ib.placeOrder.call_args[0][1]
        assert placed.totalQuantity == pytest.approx(0.75)
        fill = broker.get_fill(handle)
        assert fill.quantity == pytest.approx(0.75)

    def test_get_fill_calls_ib_sleep_zero(self):
        """get_fill must call ib.sleep(0) to flush IB events."""
        broker, ib, _ = self._setup()
        order = Order(ticker="AAPL", side="BUY", quantity=10.0)
        handle = broker.submit_limit(order, limit_price=150.0)
        ib.sleep.reset_mock()
        broker.get_fill(handle)
        ib.sleep.assert_called_once_with(0)


class TestSubmitMidprice:
    def _setup(self, status="Filled", filled=50.0, avg_price=149.75):
        broker = _make_broker(readonly=False)
        ib = _inject_ib(broker)
        contract = SimpleNamespace(symbol="AAPL", secType="STK")
        ib.qualifyContracts.return_value = [contract]
        trade = _fake_trade(status=status, filled=filled, avg_fill_price=avg_price)
        ib.placeOrder.return_value = trade
        ib.sleep = MagicMock()
        return broker, ib, trade

    def test_constructs_midprice_order_with_tif_day(self):
        broker, ib, _ = self._setup()
        order = Order(ticker="AAPL", side="BUY", quantity=50.0)
        handle = broker.submit_midprice(order)

        placed = ib.placeOrder.call_args[0][1]
        assert placed.orderType == "MIDPRICE"
        assert placed.action == "BUY"
        assert placed.totalQuantity == pytest.approx(50.0)
        assert placed.tif == "DAY"

    def test_returns_order_handle(self):
        broker, ib, trade = self._setup()
        order = Order(ticker="AAPL", side="BUY", quantity=50.0)
        handle = broker.submit_midprice(order)
        assert isinstance(handle, OrderHandle)
        assert handle.order_type == "MIDPRICE"
        assert handle.ref is trade

    def test_get_fill_returns_fill(self):
        broker, ib, _ = self._setup(status="Filled", filled=50.0, avg_price=149.75)
        order = Order(ticker="AAPL", side="BUY", quantity=50.0)
        handle = broker.submit_midprice(order)
        fill = broker.get_fill(handle)
        assert fill.status == "filled"
        assert fill.avg_price == pytest.approx(149.75)

    def test_sell_midprice(self):
        broker, ib, _ = self._setup()
        order = Order(ticker="AAPL", side="SELL", quantity=20.0)
        broker.submit_midprice(order)
        placed = ib.placeOrder.call_args[0][1]
        assert placed.action == "SELL"

    def test_readonly_raises(self):
        broker = _make_broker(readonly=True)
        _inject_ib(broker)
        with pytest.raises(RuntimeError, match="readonly"):
            broker.submit_midprice(Order("AAPL", "BUY", 10.0))


class TestSubmitMarket:
    def _setup(self, status="Filled", filled=100.0, avg_price=150.5):
        broker = _make_broker(readonly=False)
        ib = _inject_ib(broker)
        contract = SimpleNamespace(symbol="GOOG", secType="STK")
        ib.qualifyContracts.return_value = [contract]
        trade = _fake_trade(status=status, filled=filled, avg_fill_price=avg_price)
        ib.placeOrder.return_value = trade
        ib.sleep = MagicMock()
        return broker, ib, trade

    def test_constructs_market_order_with_tif_day(self):
        broker, ib, _ = self._setup()
        order = Order(ticker="GOOG", side="BUY", quantity=10.0)
        handle = broker.submit_market(order)

        placed = ib.placeOrder.call_args[0][1]
        assert placed.orderType == "MKT"
        assert placed.action == "BUY"
        assert placed.totalQuantity == pytest.approx(10.0)
        assert placed.tif == "DAY"

    def test_returns_order_handle(self):
        broker, ib, trade = self._setup()
        order = Order(ticker="GOOG", side="BUY", quantity=10.0)
        handle = broker.submit_market(order)
        assert isinstance(handle, OrderHandle)
        assert handle.order_type == "MKT"
        assert handle.ref is trade

    def test_get_fill_returns_fill(self):
        broker, ib, _ = self._setup(filled=10.0, avg_price=2800.0)
        order = Order(ticker="GOOG", side="BUY", quantity=10.0)
        handle = broker.submit_market(order)
        fill = broker.get_fill(handle)
        assert fill.status == "filled"
        assert fill.avg_price == pytest.approx(2800.0)

    def test_sell_market(self):
        broker, ib, _ = self._setup()
        order = Order(ticker="GOOG", side="SELL", quantity=5.0)
        broker.submit_market(order)
        placed = ib.placeOrder.call_args[0][1]
        assert placed.action == "SELL"
        assert placed.orderType == "MKT"

    def test_readonly_raises(self):
        broker = _make_broker(readonly=True)
        _inject_ib(broker)
        with pytest.raises(RuntimeError, match="readonly"):
            broker.submit_market(Order("AAPL", "BUY", 10.0))


# ---------------------------------------------------------------------------
# cancel() — calls ib.cancelOrder when trade is not done
# ---------------------------------------------------------------------------

class TestCancel:
    def _setup_with_trade(self, is_done: bool):
        broker = _make_broker(readonly=False)
        ib = _inject_ib(broker)
        contract = SimpleNamespace(symbol="AAPL", secType="STK")
        ib.qualifyContracts.return_value = [contract]
        status = "Filled" if is_done else "Submitted"
        trade = _fake_trade(status=status, filled=0.0, avg_fill_price=0.0)
        ib.placeOrder.return_value = trade
        ib.sleep = MagicMock()
        return broker, ib, trade

    def test_cancel_calls_ib_cancel_order_when_not_done(self):
        """cancel() calls ib.cancelOrder when the trade is not done."""
        broker, ib, trade = self._setup_with_trade(is_done=False)
        order = Order(ticker="AAPL", side="BUY", quantity=10.0)
        handle = broker.submit_limit(order, limit_price=150.0)
        broker.cancel(handle)
        ib.cancelOrder.assert_called_once_with(trade.order)

    def test_cancel_skips_ib_cancel_order_when_already_done(self):
        """cancel() does NOT call ib.cancelOrder when trade is already done."""
        broker, ib, trade = self._setup_with_trade(is_done=True)
        order = Order(ticker="AAPL", side="BUY", quantity=10.0)
        handle = broker.submit_limit(order, limit_price=150.0)
        broker.cancel(handle)
        ib.cancelOrder.assert_not_called()

    def test_cancel_is_idempotent_via_exception_guard(self):
        """If cancelOrder raises, cancel() swallows the exception (idempotent)."""
        broker, ib, trade = self._setup_with_trade(is_done=False)
        ib.cancelOrder.side_effect = RuntimeError("already cancelled")
        order = Order(ticker="AAPL", side="BUY", quantity=10.0)
        handle = broker.submit_limit(order, limit_price=150.0)
        # Should not raise
        broker.cancel(handle)


# ---------------------------------------------------------------------------
# get_fill maps status correctly
# ---------------------------------------------------------------------------

class TestGetFillStatusMapping:
    def _make_handle_with_trade(self, status: str, filled: float, avg_price: float):
        broker = _make_broker(readonly=False)
        ib = _inject_ib(broker)
        contract = SimpleNamespace(symbol="AAPL", secType="STK")
        ib.qualifyContracts.return_value = [contract]
        trade = _fake_trade(status=status, filled=filled, avg_fill_price=avg_price)
        ib.placeOrder.return_value = trade
        ib.sleep = MagicMock()
        order = Order(ticker="AAPL", side="BUY", quantity=100.0)
        handle = broker.submit_limit(order, limit_price=150.0)
        return broker, handle

    def test_filled_status(self):
        broker, handle = self._make_handle_with_trade("Filled", 100.0, 150.0)
        fill = broker.get_fill(handle)
        assert fill.status == "filled"
        assert fill.quantity == pytest.approx(100.0)

    def test_partial_fill_status(self):
        broker, handle = self._make_handle_with_trade("PartialFill", 40.0, 149.0)
        fill = broker.get_fill(handle)
        assert fill.status == "partial"
        assert fill.quantity == pytest.approx(40.0)

    def test_cancelled_status(self):
        broker, handle = self._make_handle_with_trade("Cancelled", 0.0, 0.0)
        fill = broker.get_fill(handle)
        assert fill.status == "cancelled"

    def test_submitted_status_maps_to_unfilled(self):
        broker, handle = self._make_handle_with_trade("Submitted", 0.0, 0.0)
        fill = broker.get_fill(handle)
        assert fill.status == "unfilled"

    def test_avg_price_zero_when_no_fill(self):
        """IB returns avgFillPrice=0.0 when nothing has filled; we pass it through."""
        broker, handle = self._make_handle_with_trade("Submitted", 0.0, 0.0)
        fill = broker.get_fill(handle)
        assert fill.avg_price == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Side validation
# ---------------------------------------------------------------------------

class TestSideValidation:
    """Invalid order side must raise ValueError before hitting the IB wire."""

    def _setup(self):
        broker = _make_broker(readonly=False)
        ib = _inject_ib(broker)
        contract = SimpleNamespace(symbol="AAPL", secType="STK")
        ib.qualifyContracts.return_value = [contract]
        ib.sleep = MagicMock()
        return broker, ib

    def test_submit_limit_invalid_side(self):
        broker, _ = self._setup()
        with pytest.raises(ValueError, match="invalid order side"):
            broker.submit_limit(Order("AAPL", "buy", 10.0), 150.0)

    def test_submit_midprice_invalid_side(self):
        broker, _ = self._setup()
        with pytest.raises(ValueError, match="invalid order side"):
            broker.submit_midprice(Order("AAPL", "sell", 10.0))

    def test_submit_market_invalid_side(self):
        broker, _ = self._setup()
        with pytest.raises(ValueError, match="invalid order side"):
            broker.submit_market(Order("AAPL", "BuY", 10.0))

    def test_submit_limit_valid_buy(self):
        """BUY (exact uppercase) must not raise."""
        broker, ib = self._setup()
        trade = _fake_trade(status="Filled", filled=10.0, avg_fill_price=150.0)
        ib.placeOrder.return_value = trade
        broker.submit_limit(Order("AAPL", "BUY", 10.0), 150.0)

    def test_submit_limit_valid_sell(self):
        """SELL (exact uppercase) must not raise."""
        broker, ib = self._setup()
        trade = _fake_trade(status="Filled", filled=10.0, avg_fill_price=150.0)
        ib.placeOrder.return_value = trade
        broker.submit_limit(Order("AAPL", "SELL", 10.0), 150.0)
