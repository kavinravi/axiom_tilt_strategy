"""IBKRBroker — live broker implementation using ib_async.

Connects to IB Gateway (paper or live) and implements the full Broker interface.
Uses synchronous-style ib_async API (IB.sleep, not asyncio).

Order placement is NON-BLOCKING: submit_* places the order and returns an
OrderHandle immediately.  The caller is responsible for sleeping, then calling
cancel() followed by get_fill() to read the cumulative fill state.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

from ib_async import IB, LimitOrder, MarketOrder, Order as IBOrder, Stock

import trading.config as config
from trading.broker.base import Broker, Fill, Order, OrderHandle

logger = logging.getLogger(__name__)

# Timeout constants
_QUOTE_WAIT_SEC = 3.0          # seconds to wait for real-time bid/ask
_QUOTE_DELAYED_WAIT_SEC = 5.0  # seconds to wait after switching to delayed data


def _clean_num(v) -> float | None:
    """None for missing/NaN broker figures, float otherwise."""
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)


def _map_status(ib_status: str) -> str:
    """Map ib_async order status string to our Fill.status vocab."""
    s = ib_status.lower()
    if s == "filled":
        return "filled"
    if "partial" in s:
        return "partial"
    if s in ("cancelled", "apicancelled", "inactive"):
        return "cancelled"
    return "unfilled"


class IBKRBroker(Broker):
    """Live IBKR broker backed by ib_async.

    Parameters
    ----------
    host : str | None
        IB Gateway host.  Defaults to ``trading.config.IBKR_HOST``.
    port : int | None
        IB Gateway port.  Defaults to ``trading.config.IBKR_PORT``.
    client_id : int | None
        Client ID.  Defaults to ``trading.config.IBKR_CLIENT_ID``.
    readonly : bool
        If True, connect in read-only mode (no order placement).
        Tests use ``readonly=True`` to prevent accidental orders.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        client_id: Optional[int] = None,
        readonly: bool = False,
    ) -> None:
        self._host = host if host is not None else config.IBKR_HOST
        self._port = port if port is not None else config.IBKR_PORT
        self._client_id = client_id if client_id is not None else config.IBKR_CLIENT_ID
        self._readonly = readonly
        self.ib: IB  # set by connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to IB Gateway."""
        self.ib = IB()
        self.ib.connect(
            self._host,
            self._port,
            clientId=self._client_id,
            timeout=15,
            readonly=self._readonly,
        )
        logger.info(
            "IBKRBroker: connected to %s:%s (clientId=%s, readonly=%s)",
            self._host, self._port, self._client_id, self._readonly,
        )

    def disconnect(self) -> None:
        """Disconnect from IB Gateway."""
        self.ib.disconnect()
        logger.info("IBKRBroker: disconnected")

    # ------------------------------------------------------------------
    # Account state
    # ------------------------------------------------------------------

    def get_positions(self) -> dict[str, float]:
        """Return current positions as {ticker: shares}.

        Only US equity (STK) contracts are included; duplicates are summed.
        """
        positions: dict[str, float] = {}
        for p in self.ib.positions():
            if p.contract.secType == "STK":
                sym = p.contract.symbol
                positions[sym] = positions.get(sym, 0.0) + p.position
        logger.info("IBKRBroker: get_positions() → %d holdings", len(positions))
        return positions

    def get_nav(self) -> float:
        """Return account Net Liquidation Value in USD."""
        for av in self.ib.accountSummary():
            if av.tag == "NetLiquidation" and av.currency == "USD":
                nav = float(av.value)
                logger.info("IBKRBroker: get_nav() → %.2f", nav)
                return nav
        raise RuntimeError("IBKRBroker: NetLiquidation tag not found in accountSummary()")

    def get_portfolio(self) -> list[dict]:
        """Per-position snapshot via the account-update channel (no market data
        subscription): ib.portfolio() supplies position/market_price/market_value/
        avg_cost/unrealized_pnl; reqPnLSingle supplies the day P&L — the same
        figures the IBKR mobile portfolio tab shows.
        """
        items = [p for p in self.ib.portfolio() if p.contract.secType == "STK"]
        accounts = self.ib.managedAccounts()
        account = accounts[0] if accounts else ""

        # Batch-request day P&L for every position, give the account channel a
        # moment to populate, then read + cancel. PnL subscriptions are account
        # data — free, no quote entitlement involved.
        pnl_reqs: dict[int, object] = {}
        for it in items:
            try:
                pnl_reqs[it.contract.conId] = self.ib.reqPnLSingle(account, "", it.contract.conId)
            except Exception as exc:  # noqa: BLE001
                logger.warning("IBKRBroker: reqPnLSingle(%s) failed: %s", it.contract.symbol, exc)
        if pnl_reqs:
            self.ib.sleep(2.0)

        rows: list[dict] = []
        for it in items:
            ps = pnl_reqs.get(it.contract.conId)
            rows.append(
                {
                    "ticker": it.contract.symbol,
                    "position": float(it.position),
                    "market_price": _clean_num(it.marketPrice),
                    "market_value": _clean_num(it.marketValue),
                    "avg_cost": _clean_num(it.averageCost),
                    "unrealized_pnl": _clean_num(it.unrealizedPNL),
                    "daily_pnl": _clean_num(getattr(ps, "dailyPnL", None)),
                }
            )
        for con_id in pnl_reqs:
            try:
                self.ib.cancelPnLSingle(account, "", con_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("IBKRBroker: cancelPnLSingle(%s) raised (harmless): %s", con_id, exc)
        logger.info("IBKRBroker: get_portfolio() → %d positions", len(rows))
        return rows

    def get_account_pnl(self) -> dict:
        """Account-level day/unrealized/realized P&L via reqPnL.

        Account-channel data (free, no quote entitlement). Includes the realized
        P&L of positions fully closed today, which a per-position sum misses.
        """
        accounts = self.ib.managedAccounts()
        account = accounts[0] if accounts else ""
        pnl = self.ib.reqPnL(account, "")
        try:
            self.ib.sleep(2.0)  # let the account channel populate
            out = {
                "daily_pnl": _clean_num(getattr(pnl, "dailyPnL", None)),
                "unrealized_pnl": _clean_num(getattr(pnl, "unrealizedPnL", None)),
                "realized_pnl": _clean_num(getattr(pnl, "realizedPnL", None)),
            }
        finally:
            try:
                self.ib.cancelPnL(account, "")
            except Exception as exc:  # noqa: BLE001
                logger.warning("IBKRBroker: cancelPnL raised (harmless): %s", exc)
        logger.info("IBKRBroker: get_account_pnl() → %s", out)
        return out

    def get_quote(self, ticker: str) -> tuple[float, float]:
        """Return (bid, ask) for *ticker*.

        Tries real-time data first; falls back to delayed (type 3) if bid/ask
        are NaN.  Raises RuntimeError if data is still unavailable after the
        fallback.
        """
        contract = Stock(ticker, "SMART", "USD")
        [contract] = self.ib.qualifyContracts(contract)

        # --- try real-time first ---
        ticker_obj = self.ib.reqMktData(contract, "", False, False)
        try:
            self.ib.sleep(_QUOTE_WAIT_SEC)
            bid, ask = ticker_obj.bid, ticker_obj.ask
        finally:
            self.ib.cancelMktData(contract)

        if not (math.isnan(bid) or bid <= 0 or math.isnan(ask) or ask <= 0):
            logger.info("IBKRBroker: get_quote(%s) real-time → bid=%.4f ask=%.4f", ticker, bid, ask)
            return (bid, ask)

        # --- fallback: delayed data (type 3) ---
        logger.info(
            "IBKRBroker: get_quote(%s) real-time bid/ask unavailable (bid=%s, ask=%s); "
            "switching to delayed data (type 3)",
            ticker, bid, ask,
        )
        self.ib.reqMarketDataType(3)
        try:
            ticker_obj = self.ib.reqMktData(contract, "", False, False)
            self.ib.sleep(_QUOTE_DELAYED_WAIT_SEC)
            bid, ask = ticker_obj.bid, ticker_obj.ask
        finally:
            self.ib.cancelMktData(contract)
            # Always reset to real-time data type, even if an exception was raised
            # above — leaving the session stuck on delayed would silently degrade
            # all subsequent quote requests.
            self.ib.reqMarketDataType(1)

        if math.isnan(bid) or bid <= 0 or math.isnan(ask) or ask <= 0:
            raise RuntimeError(
                f"IBKRBroker: get_quote({ticker!r}) failed — bid={bid}, ask={ask} "
                "even after switching to delayed data (type 3). "
                "Check market data subscriptions."
            )

        logger.info(
            "IBKRBroker: get_quote(%s) delayed → bid=%.4f ask=%.4f", ticker, bid, ask
        )
        return (bid, ask)

    # ------------------------------------------------------------------
    # Order placement helpers
    # ------------------------------------------------------------------

    def _qualify_stock(self, ticker: str) -> object:
        """Qualify and return a Stock contract for *ticker*."""
        contract = Stock(ticker, "SMART", "USD")
        [contract] = self.ib.qualifyContracts(contract)
        return contract

    def _build_fill(self, trade, order: Order) -> Fill:
        """Convert an ib_async Trade to our Fill dataclass (snapshot, non-blocking).

        Note: ``avg_price == 0.0`` means no shares were filled (IB returns 0
        when ``avgFillPrice`` is undefined).  Callers should check
        ``fill.quantity`` rather than ``fill.avg_price`` to detect an unfilled
        order.
        """
        status_str = trade.orderStatus.status
        filled_qty = trade.orderStatus.filled
        avg_price = trade.orderStatus.avgFillPrice

        mapped_status = _map_status(status_str)
        return Fill(
            ticker=order.ticker,
            side=order.side,
            quantity=filled_qty,
            avg_price=avg_price,
            status=mapped_status,
        )

    # ------------------------------------------------------------------
    # Non-blocking submit / poll / cancel
    # ------------------------------------------------------------------

    def submit_limit(self, order: Order, limit_price: float) -> OrderHandle:
        """Submit a limit order to IB Gateway; return immediately (non-blocking).

        ``tif="DAY"`` is set explicitly so that Gateway order presets cannot
        override the time-in-force to GTC or any other value.
        """
        if self._readonly:
            raise RuntimeError("IBKRBroker is in readonly mode — order placement not allowed")
        if order.side not in ("BUY", "SELL"):
            raise ValueError(
                f"IBKRBroker: invalid order side {order.side!r} — must be 'BUY' or 'SELL'"
            )
        contract = self._qualify_stock(order.ticker)
        ib_order = LimitOrder(order.side, order.quantity, limit_price)
        ib_order.tif = "DAY"
        logger.info(
            "IBKRBroker: submit_limit(%s %s qty=%.4f @ %.4f)",
            order.side, order.ticker, order.quantity, limit_price,
        )
        trade = self.ib.placeOrder(contract, ib_order)
        return OrderHandle(ref=trade, ticker=order.ticker, side=order.side,
                           quantity=order.quantity, order_type="LMT")

    def submit_midprice(self, order: Order) -> OrderHandle:
        """Submit a MIDPRICE order to IB Gateway; return immediately (non-blocking).

        ``tif="DAY"`` is set explicitly so that Gateway order presets cannot
        override the time-in-force.
        """
        if self._readonly:
            raise RuntimeError("IBKRBroker is in readonly mode — order placement not allowed")
        if order.side not in ("BUY", "SELL"):
            raise ValueError(
                f"IBKRBroker: invalid order side {order.side!r} — must be 'BUY' or 'SELL'"
            )
        contract = self._qualify_stock(order.ticker)
        ib_order = IBOrder(
            orderType="MIDPRICE",
            action=order.side,
            totalQuantity=order.quantity,
        )
        ib_order.tif = "DAY"
        logger.info(
            "IBKRBroker: submit_midprice(%s %s qty=%.4f)",
            order.side, order.ticker, order.quantity,
        )
        trade = self.ib.placeOrder(contract, ib_order)
        return OrderHandle(ref=trade, ticker=order.ticker, side=order.side,
                           quantity=order.quantity, order_type="MIDPRICE")

    def submit_market(self, order: Order) -> OrderHandle:
        """Submit a market order to IB Gateway; return immediately (non-blocking).

        ``tif="DAY"`` is set explicitly so that Gateway order presets cannot
        override the time-in-force.
        """
        if self._readonly:
            raise RuntimeError("IBKRBroker is in readonly mode — order placement not allowed")
        if order.side not in ("BUY", "SELL"):
            raise ValueError(
                f"IBKRBroker: invalid order side {order.side!r} — must be 'BUY' or 'SELL'"
            )
        contract = self._qualify_stock(order.ticker)
        ib_order = MarketOrder(order.side, order.quantity)
        ib_order.tif = "DAY"
        logger.info(
            "IBKRBroker: submit_market(%s %s qty=%.4f)",
            order.side, order.ticker, order.quantity,
        )
        trade = self.ib.placeOrder(contract, ib_order)
        return OrderHandle(ref=trade, ticker=order.ticker, side=order.side,
                           quantity=order.quantity, order_type="MKT")

    def get_fill(self, handle: OrderHandle) -> Fill:
        """Return the current cumulative fill state for *handle* (non-blocking).

        Calls ``self.ib.sleep(0)`` to process any pending IB event loop
        callbacks, then delegates to ``_build_fill`` for the status snapshot.
        """
        trade = handle.ref
        self.ib.sleep(0)  # flush pending IB events (non-blocking 0 s tick)
        # Reconstruct a minimal Order so _build_fill has ticker/side.
        order = Order(ticker=handle.ticker, side=handle.side, quantity=handle.quantity)
        return self._build_fill(trade, order)

    def cancel(self, handle: OrderHandle) -> None:
        """Cancel the order referenced by *handle*.

        Idempotent: safe to call even if the order is already done (filled,
        cancelled, or inactive).  IB ignores cancel requests for completed
        orders, so no guard is strictly needed, but we check isDone() to avoid
        noisy warnings in the IB logs.
        """
        trade = handle.ref
        try:
            if not trade.isDone():
                logger.info(
                    "IBKRBroker: cancel(orderId=%s ticker=%s)",
                    trade.order.orderId, handle.ticker,
                )
                self.ib.cancelOrder(trade.order)
        except Exception as exc:  # noqa: BLE001
            # Harmless if the order was already done or the cancel races with a fill
            logger.warning("IBKRBroker: cancel() raised (harmless): %s", exc)
