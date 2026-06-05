"""DryRunBroker — fully synthetic broker for testing + dry-run mode.

No network, no IB Gateway. Constructed with injected positions/NAV/quotes.
submit_* methods log the intended order, compute a simulated fill immediately,
and store it keyed by handle id.  get_fill/cancel complete the non-blocking API.
"""
from __future__ import annotations

import logging
from typing import Optional

from trading.broker.base import Broker, Fill, Order, OrderHandle

logger = logging.getLogger(__name__)

# Default spread used when synthesizing quotes for unknown tickers
_DEFAULT_PRICE = 100.0
_DEFAULT_HALF_SPREAD = 0.50


class DryRunBroker(Broker):
    """Fake broker for unit tests and dry-run rebalance execution.

    Parameters
    ----------
    positions : dict[str, float]
        Synthetic current positions {ticker: shares}.
    nav : float
        Synthetic NAV (total portfolio value).
    quotes : dict[str, tuple[float, float]]
        Synthetic market quotes {ticker: (bid, ask)}.
    fill_ratio : float
        Fraction of the order quantity that is filled (default 1.0 = full fill).
        Values < 1 simulate partial fills for testing ladder escalation.
    """

    def __init__(
        self,
        positions: Optional[dict[str, float]] = None,
        nav: float = 100_000.0,
        quotes: Optional[dict[str, tuple[float, float]]] = None,
        fill_ratio: float = 1.0,
    ) -> None:
        self._positions: dict[str, float] = dict(positions) if positions else {}
        self._nav = nav
        self._quotes: dict[str, tuple[float, float]] = dict(quotes) if quotes else {}
        self._fill_ratio = fill_ratio
        # Internal state: auto-incrementing id → (Fill, cancelled)
        self._next_id: int = 0
        self._fills: dict[int, Fill] = {}
        self._cancelled: dict[int, bool] = {}

    # ------------------------------------------------------------------
    # Broker interface: connection (no-ops)
    # ------------------------------------------------------------------

    def connect(self) -> None:
        logger.info("DryRunBroker: connect() — no-op")

    def disconnect(self) -> None:
        logger.info("DryRunBroker: disconnect() — no-op")

    # ------------------------------------------------------------------
    # Broker interface: account state
    # ------------------------------------------------------------------

    def get_positions(self) -> dict[str, float]:
        return dict(self._positions)

    def get_nav(self) -> float:
        return self._nav

    def get_quote(self, ticker: str) -> tuple[float, float]:
        if ticker in self._quotes:
            return self._quotes[ticker]
        # Synthesize a quote for unknown tickers using default price ± half-spread
        bid = _DEFAULT_PRICE - _DEFAULT_HALF_SPREAD
        ask = _DEFAULT_PRICE + _DEFAULT_HALF_SPREAD
        logger.warning("DryRunBroker: no quote for %s, synthesizing (%s, %s)", ticker, bid, ask)
        return (bid, ask)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_fill(self, order: Order, price: float, filled_qty: float) -> Fill:
        """Build a Fill with the correct status given quantity and filled_qty."""
        if filled_qty <= 0.0:
            status = "unfilled"
            filled_qty = 0.0
        elif filled_qty < order.quantity:
            status = "partial"
        else:
            status = "filled"
        return Fill(
            ticker=order.ticker,
            side=order.side,
            quantity=filled_qty,
            avg_price=price,
            status=status,
        )

    def _apply_fill_ratio(self, quantity: float) -> float:
        return quantity * self._fill_ratio

    def _next_handle_id(self) -> int:
        hid = self._next_id
        self._next_id += 1
        return hid

    # ------------------------------------------------------------------
    # Broker interface: non-blocking submit / poll / cancel
    # ------------------------------------------------------------------

    def submit_limit(self, order: Order, limit_price: float) -> OrderHandle:
        """Simulate a limit order submission.

        A BUY is marketable if limit_price >= ask (would execute immediately).
        A SELL is marketable if limit_price <= bid.
        Non-marketable → unfilled; marketable → filled at limit_price * fill_ratio.
        The simulated fill is computed immediately and stored; get_fill returns it.
        """
        bid, ask = self.get_quote(order.ticker)
        logger.info(
            "DryRunBroker: submit_limit(%s %s qty=%.4f @ limit=%.4f) quote=(%s,%s)",
            order.side, order.ticker, order.quantity, limit_price, bid, ask,
        )
        if order.side == "BUY":
            marketable = limit_price >= ask
        else:
            marketable = limit_price <= bid

        if not marketable:
            fill = Fill(ticker=order.ticker, side=order.side, quantity=0.0,
                        avg_price=limit_price, status="unfilled")
        else:
            filled_qty = self._apply_fill_ratio(order.quantity)
            fill = self._make_fill(order, limit_price, filled_qty)

        hid = self._next_handle_id()
        self._fills[hid] = fill
        self._cancelled[hid] = False
        return OrderHandle(ref=hid, ticker=order.ticker, side=order.side,
                           quantity=order.quantity, order_type="LMT")

    def submit_midprice(self, order: Order) -> OrderHandle:
        """Simulate a MIDPRICE order — fills at (bid+ask)/2 * fill_ratio."""
        bid, ask = self.get_quote(order.ticker)
        mid = (bid + ask) / 2.0
        logger.info(
            "DryRunBroker: submit_midprice(%s %s qty=%.4f) mid=%.4f",
            order.side, order.ticker, order.quantity, mid,
        )
        filled_qty = self._apply_fill_ratio(order.quantity)
        fill = self._make_fill(order, mid, filled_qty)

        hid = self._next_handle_id()
        self._fills[hid] = fill
        self._cancelled[hid] = False
        return OrderHandle(ref=hid, ticker=order.ticker, side=order.side,
                           quantity=order.quantity, order_type="MIDPRICE")

    def submit_market(self, order: Order) -> OrderHandle:
        """Simulate a market order — BUY fills at ask, SELL fills at bid."""
        bid, ask = self.get_quote(order.ticker)
        price = ask if order.side == "BUY" else bid
        logger.info(
            "DryRunBroker: submit_market(%s %s qty=%.4f) price=%.4f",
            order.side, order.ticker, order.quantity, price,
        )
        filled_qty = self._apply_fill_ratio(order.quantity)
        fill = self._make_fill(order, price, filled_qty)

        hid = self._next_handle_id()
        self._fills[hid] = fill
        self._cancelled[hid] = False
        return OrderHandle(ref=hid, ticker=order.ticker, side=order.side,
                           quantity=order.quantity, order_type="MKT")

    def get_fill(self, handle: OrderHandle) -> Fill:
        """Return the stored simulated Fill for this handle.

        If the handle was cancelled AND its simulated filled quantity is 0,
        the returned Fill has status="cancelled" to represent an unfilled order
        that was explicitly cancelled before any shares traded.  If the order
        was (partially or fully) filled before cancellation the stored fill is
        returned unchanged, preserving the already-filled qty and avg_price.
        """
        hid = handle.ref
        if hid not in self._fills:
            raise KeyError(f"DryRunBroker: unknown handle id {hid}")
        stored = self._fills[hid]
        if self._cancelled.get(hid, False) and stored.quantity == 0.0:
            return Fill(
                ticker=stored.ticker,
                side=stored.side,
                quantity=0.0,
                avg_price=stored.avg_price,
                status="cancelled",
            )
        return stored

    def cancel(self, handle: OrderHandle) -> None:
        """Mark the handle cancelled.  Already-filled qty is NOT reversed."""
        hid = handle.ref
        if hid in self._cancelled:
            self._cancelled[hid] = True
            logger.info(
                "DryRunBroker: cancel(handle id=%d ticker=%s) — marked cancelled",
                hid, handle.ticker,
            )
