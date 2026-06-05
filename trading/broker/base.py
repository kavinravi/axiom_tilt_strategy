"""Broker interface (ABC) + core dataclasses."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class Order:
    ticker: str
    side: str        # "BUY" | "SELL"
    quantity: float  # shares (fractional allowed)


@dataclass
class Fill:
    ticker: str
    side: str
    quantity: float       # filled qty
    avg_price: float
    status: str           # "filled" | "partial" | "unfilled" | "cancelled"


@dataclass
class OrderHandle:
    """Broker-internal handle returned by submit_*.

    ``ref`` is broker-specific (an ib_async Trade for IBKRBroker, an integer
    id for DryRunBroker).  Callers treat it as opaque.
    """
    ref: Any                 # broker-internal reference
    ticker: str
    side: str                # "BUY" | "SELL"
    quantity: float          # requested quantity
    order_type: str          # "LMT" | "MIDPRICE" | "MKT"


class Broker(ABC):
    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def get_positions(self) -> dict[str, float]: ...    # {ticker: shares}

    @abstractmethod
    def get_nav(self) -> float: ...

    @abstractmethod
    def get_quote(self, ticker: str) -> tuple[float, float]: ...   # (bid, ask)

    # ------------------------------------------------------------------
    # Non-blocking order submission
    # ------------------------------------------------------------------

    @abstractmethod
    def submit_limit(self, order: Order, limit_price: float) -> OrderHandle:
        """Submit a limit order; return immediately with a handle."""
        ...

    @abstractmethod
    def submit_midprice(self, order: Order) -> OrderHandle:
        """Submit a MIDPRICE order; return immediately with a handle."""
        ...

    @abstractmethod
    def submit_market(self, order: Order) -> OrderHandle:
        """Submit a market order; return immediately with a handle."""
        ...

    @abstractmethod
    def get_fill(self, handle: OrderHandle) -> Fill:
        """Return current cumulative fill state for the handle (non-blocking)."""
        ...

    @abstractmethod
    def cancel(self, handle: OrderHandle) -> None:
        """Cancel the order referenced by handle.  Idempotent; safe if already done."""
        ...
