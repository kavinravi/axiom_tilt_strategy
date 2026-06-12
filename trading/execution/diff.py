"""Execution diff: convert target weights + current positions + NAV → orders."""
from __future__ import annotations

import logging
import math

from trading.broker.base import Order
from trading.config import MAX_WEIGHT

logger = logging.getLogger(__name__)


def _fill_residual_cash(
    shares: dict[str, float],
    target_weights: dict[str, float],
    nav: float,
    prices: dict[str, float],
    max_weight: float,
) -> dict[str, float]:
    """Redeploy the cash stranded by whole-share flooring back into the book.

    Greedy water-fill on dollar deficits: repeatedly buy ONE share of the most
    dollar-underweight name whose price fits the remaining budget and whose new
    value stays within ``max_weight * nav``. The budget is exactly the sum of
    flooring losses, so invested never exceeds sum(w_i)*nav. Stops when no name
    is both affordable and under the cap, leaving a residual smaller than the
    cheapest eligible share — the floor for what integer shares can deploy.

    Trade-off (intentional): once only the cheapest names remain affordable,
    they absorb the tail of the budget and drift slightly overweight. The
    strategy is fully-invested by design, so a small overweight in a top-scored
    name beats stranded cash (pure tracking error vs the backtest). The cap
    bounds the worst case. Mutates and returns ``shares``.
    """
    values = {t: shares[t] * prices[t] for t in shares}
    budget = sum(target_weights[t] * nav - values[t] for t in shares)
    cap_value = max_weight * nav
    while budget > 0:
        best = None
        best_deficit = float("-inf")
        for t in sorted(shares):  # sorted → deterministic tie-break (first wins)
            price = prices[t]
            if price > budget + 1e-9 or values[t] + price > cap_value + 1e-9:
                continue
            deficit = target_weights[t] * nav - values[t]
            if deficit > best_deficit + 1e-12:
                best, best_deficit = t, deficit
        if best is None:
            break
        shares[best] += 1.0
        values[best] += prices[best]
        budget -= prices[best]
    return shares


def target_shares(
    target_weights: dict[str, float],
    nav: float,
    prices: dict[str, float],
    whole_shares: bool = False,
    max_weight: float = MAX_WEIGHT,
) -> dict[str, float]:
    """Compute target share quantities: shares_i = (w_i * nav) / price_i.

    Fractional shares by default. With ``whole_shares=True`` each target is
    floored to an integer (IBKR rejects fractional orders via API — error
    10243), then the cash stranded by flooring (~half a share per name) is
    redeployed by ``_fill_residual_cash``: one share at a time into the most
    dollar-underweight name, never lifting any name above ``max_weight`` and
    never spending more than the flooring losses — so the book still can't
    exceed NAV (no buying-power reject). Tickers with missing or zero prices
    are skipped and logged.

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
        shares = (weight * nav) / price
        result[ticker] = float(math.floor(shares)) if whole_shares else shares
    if whole_shares and result:
        before = sum(result[t] * prices[t] for t in result)
        _fill_residual_cash(result, target_weights, nav, prices, max_weight)
        after = sum(result[t] * prices[t] for t in result)
        if after > before:
            logger.info(
                "target_shares: residual top-up deployed %.2f of stranded cash "
                "(invested %.2f -> %.2f, %.2f%% of nav)",
                after - before, before, after, after / nav * 100.0,
            )
    return result


def diff_to_orders(
    target_weights: dict[str, float],
    current_positions: dict[str, float],
    nav: float,
    prices: dict[str, float],
    min_order_notional: float = 1.0,
    whole_shares: bool = False,
    max_weight: float = MAX_WEIGHT,
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
    max_weight : float
        Per-name cap honored by the whole-share residual top-up (see
        ``target_shares``). Defaults to the strategy-wide MAX_WEIGHT.

    Returns
    -------
    list[Order]
        List of BUY/SELL orders to bring portfolio to target.
    """
    tgt_shares = target_shares(
        target_weights, nav, prices, whole_shares=whole_shares, max_weight=max_weight
    )

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
        if whole_shares:
            # Keep the traded quantity an integer even if the current position
            # carries a fractional remainder from a legacy fill.
            delta = float(round(delta))
        if delta == 0:
            continue

        # Skip dust
        notional = abs(delta * price)
        if notional < min_order_notional:
            continue

        side = "BUY" if delta > 0 else "SELL"
        orders.append(Order(ticker=ticker, side=side, quantity=abs(delta)))

    return orders
