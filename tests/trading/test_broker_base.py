"""Tests for trading/broker/base.py — Part A1."""
from __future__ import annotations

import pytest


def test_config_has_new_constants():
    from trading import config
    # IBKR_HOST/PORT/CLIENT_ID are env-overridable (.env sets the WSL host), so assert
    # type/shape rather than a machine-specific value.
    assert isinstance(config.IBKR_HOST, str) and config.IBKR_HOST
    assert isinstance(config.IBKR_PORT, int) and config.IBKR_PORT > 0
    assert isinstance(config.IBKR_CLIENT_ID, int)
    assert config.KILL_SWITCH_FILE.name == "KILL_SWITCH"
    assert config.MAX_ORDER_FRAC_NAV == 0.12
    assert config.MAX_TURNOVER_FRAC == 0.60
    assert config.LADDER_PASSIVE_WAIT_SEC == 180
    assert config.LADDER_MIDPRICE_WAIT_SEC == 120
    assert config.LADDER_CANCEL_GRACE_SEC == 3
    assert config.LADDER_TERMINAL_CROSS is True
    assert config.ORDERS_DIR.name == "orders"


def test_order_dataclass():
    from trading.broker.base import Order
    o = Order(ticker="AAPL", side="BUY", quantity=10.5)
    assert o.ticker == "AAPL"
    assert o.side == "BUY"
    assert o.quantity == 10.5


def test_fill_dataclass():
    from trading.broker.base import Fill
    f = Fill(ticker="AAPL", side="BUY", quantity=10.5, avg_price=150.0, status="filled")
    assert f.ticker == "AAPL"
    assert f.status == "filled"
    assert f.avg_price == 150.0


def test_broker_abc_cannot_be_instantiated():
    from trading.broker.base import Broker
    with pytest.raises(TypeError):
        Broker()


def test_broker_concrete_subclass_works():
    from trading.broker.base import Broker, Order, Fill, OrderHandle

    class FakeBroker(Broker):
        _next_id = 0

        def connect(self) -> None:
            pass

        def disconnect(self) -> None:
            pass

        def get_positions(self) -> dict[str, float]:
            return {"AAPL": 5.0}

        def get_nav(self) -> float:
            return 10_000.0

        def get_quote(self, ticker: str) -> tuple[float, float]:
            return (149.0, 151.0)

        def _make_handle(self, order: Order, order_type: str) -> OrderHandle:
            hid = FakeBroker._next_id
            FakeBroker._next_id += 1
            return OrderHandle(ref=hid, ticker=order.ticker, side=order.side,
                               quantity=order.quantity, order_type=order_type)

        def submit_limit(self, order: Order, limit_price: float) -> OrderHandle:
            h = self._make_handle(order, "LMT")
            self._fill = Fill(ticker=order.ticker, side=order.side, quantity=order.quantity,
                              avg_price=limit_price, status="filled")
            return h

        def submit_midprice(self, order: Order) -> OrderHandle:
            h = self._make_handle(order, "MIDPRICE")
            self._fill = Fill(ticker=order.ticker, side=order.side, quantity=order.quantity,
                              avg_price=150.0, status="filled")
            return h

        def submit_market(self, order: Order) -> OrderHandle:
            h = self._make_handle(order, "MKT")
            self._fill = Fill(ticker=order.ticker, side=order.side, quantity=order.quantity,
                              avg_price=151.0, status="filled")
            return h

        def get_fill(self, handle: OrderHandle) -> Fill:
            return self._fill

        def cancel(self, handle: OrderHandle) -> None:
            pass

    b = FakeBroker()
    b.connect()
    assert b.get_nav() == 10_000.0
    assert b.get_positions() == {"AAPL": 5.0}
    assert b.get_quote("AAPL") == (149.0, 151.0)
    o = Order("AAPL", "BUY", 3.0)
    handle = b.submit_limit(o, 149.5)
    fill = b.get_fill(handle)
    assert fill.status == "filled"
    assert fill.avg_price == 149.5
    b.cancel(handle)
    b.disconnect()
