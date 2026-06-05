"""Execution diff: convert target weights + current positions + NAV → orders."""
from __future__ import annotations

import logging

from trading.broker.base import Order

logger = logging.getLogger(__name__)


def target_shares(
    target_weights: dict[str, float],
    nav: float,
    prices: dict[str, float],
) -> dict[str, float]:
    """Compute target share quantities: shares_i = (w_i * nav) / price_i.

    Fractional shares are used by default (no rounding).
    Tickers with missing or zero prices are skipped and logged.

    Returns
    -------
    dict[str, float]
        {ticker: target_shares} — only tickers with valid prices included.
    """
    result: dict[str, float] = {}
    for ticker, weight in target_weights.items():
        price = prices.get(ticker)
        if price is None:
            logger.warning("target_shares: no price for %s — skipping", ticker)
            continue
        if price == 0.0:
            logger.warning("target_shares: zero price for %s — skipping", ticker)
            continue
        result[ticker] = (weight * nav) / price
    return result


def diff_to_orders(
    target_weights: dict[str, float],
    current_positions: dict[str, float],
    nav: float,
    prices: dict[str, float],
    min_order_notional: float = 1.0,
) -> list[Order]:
    """Compare target shares vs current positions and produce BUY/SELL orders.

    Algorithm
    ---------
    1. Compute target_shares from target_weights/nav/prices.
    2. For each ticker in the *union* of target and current:
       - If ticker is in target but not current → current shares = 0 → full buy.
       - If ticker is in current but not target → target shares = 0 → full sell.
       - delta_i = target_shares_i - current_shares_i
    3. Skip |delta * price| < min_order_notional (dust).
    4. BUY for delta > 0, SELL for delta < 0.

    Tickers not in ``prices`` are logged and skipped.

    Parameters
    ----------
    target_weights : dict[str, float]
        Desired portfolio weights (should sum ≈ 1).
    current_positions : dict[str, float]
        Current share holdings {ticker: shares}.
    nav : float
        Current net asset value (portfolio total value).
    prices : dict[str, float]
        Current market prices {ticker: price}.
    min_order_notional : float
        Orders whose |delta_shares * price| < this value are dropped as dust.

    Returns
    -------
    list[Order]
        List of BUY/SELL orders to bring portfolio to target.
    """
    tgt_shares = target_shares(target_weights, nav, prices)

    # Union of all tickers across target and current positions
    all_tickers = set(tgt_shares.keys()) | set(current_positions.keys())

    orders: list[Order] = []
    for ticker in all_tickers:
        price = prices.get(ticker)
        if price is None:
            logger.warning("diff_to_orders: no price for %s — skipping", ticker)
            continue

        t_shares = tgt_shares.get(ticker, 0.0)
        c_shares = current_positions.get(ticker, 0.0)
        delta = t_shares - c_shares

        # Skip dust
        notional = abs(delta * price)
        if notional < min_order_notional:
            continue

        side = "BUY" if delta > 0 else "SELL"
        orders.append(Order(ticker=ticker, side=side, quantity=abs(delta)))

    return orders
