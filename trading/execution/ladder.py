"""Execution ladder: passive limit → MIDPRICE → terminal market.

Each order is attempted first passively (bid for BUY, ask for SELL) to
capture the spread.  After a configurable wait, unfilled remainder is
escalated to MIDPRICE, then optionally to a terminal market cross.

All three stages use the same BATCH pattern:
    1. Submit ALL remaining orders in one loop (non-blocking).
    2. Sleep once (the stage rest period) — all orders rest concurrently.
    3. Cancel ALL submitted handles (idempotent; safe for already-filled).
    4. Sleep once (LADDER_CANCEL_GRACE_SEC) — one settle wait for ACKs.
    5. Poll ALL handles (get_fill) to accumulate filled qty / cost.

This means 50 orders in stage 1 all rest simultaneously rather than each
blocking for up to the full timeout in sequence.

Public API
----------
execute_ladder(broker, orders, *, config, sleep_fn=None) -> (list[Fill], LadderAuditRecord)
    Execute ``orders`` against ``broker`` using a three-stage ladder.
    Returns one aggregated ``Fill`` per ticker.
    ``sleep_fn`` is injectable (default: time.sleep) so tests run instantly.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from trading.broker.base import Broker, Fill, Order, OrderHandle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-stage audit record
# ---------------------------------------------------------------------------

@dataclass
class StageAuditRecord:
    """Realized price vs contemporaneous midpoint for a single stage/ticker."""
    ticker: str
    stage: str           # "passive" | "midprice" | "terminal"
    qty_attempted: float
    qty_filled: float
    realized_price: float | None   # None if unfilled
    midpoint_at_fill: float        # (bid+ask)/2 at the time of placement


@dataclass
class LadderAuditRecord:
    """Aggregated audit for one ``execute_ladder`` call."""
    stages: list[StageAuditRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# execute_ladder
# ---------------------------------------------------------------------------

def execute_ladder(
    broker: Broker,
    orders: list[Order],
    *,
    config: Any,
    sleep_fn: Callable[[float], None] | None = None,
) -> tuple[list[Fill], LadderAuditRecord]:
    """Execute ``orders`` via a three-stage passive→midprice→market ladder.

    Stage 1 — PASSIVE LIMIT
        BUY: submit_limit at the bid (passive; non-marketable by design).
        SELL: submit_limit at the ask (passive; non-marketable by design).
        ALL orders are submitted first; then we sleep once; then cancel all;
        then sleep LADDER_CANCEL_GRACE_SEC; then poll all fills.

    Stage 2 — MIDPRICE
        For any remaining unfilled quantity, submit_midprice (batched).
        Same submit→rest→cancel→grace→poll pattern.

    Stage 3 — TERMINAL MARKET (if ``config.LADDER_TERMINAL_CROSS``)
        For any still-unfilled quantity, submit_market (batched).
        Same submit→rest(grace)→cancel→grace→poll pattern.

    Aggregation
        One ``Fill`` is returned per ticker, combining all stage fills:
        - avg_price is quantity-weighted average of realized prices.
        - status is "filled" if total qty = original qty, "partial" otherwise.

    Parameters
    ----------
    broker : Broker
        Connected broker instance.
    orders : list[Order]
        Orders to execute (must not be empty; empty list returns []).
    config : module/object
        Must expose: LADDER_PASSIVE_WAIT_SEC, LADDER_MIDPRICE_WAIT_SEC,
        LADDER_CANCEL_GRACE_SEC, LADDER_TERMINAL_CROSS.
    sleep_fn : callable(seconds) | None
        Injectable sleep. Defaults to ``time.sleep``.

    Returns
    -------
    (fills, audit)
        fills : list[Fill] — one per ticker.
        audit : LadderAuditRecord — per-stage realized vs midpoint data.
    """
    if sleep_fn is None:
        sleep_fn = time.sleep

    if not orders:
        return [], LadderAuditRecord()

    audit = LadderAuditRecord()

    # Track per-ticker state: total filled qty and cost basis
    filled_qty: dict[str, float] = {o.ticker: 0.0 for o in orders}
    cost_basis: dict[str, float] = {o.ticker: 0.0 for o in orders}
    remaining_qty: dict[str, float] = {o.ticker: o.quantity for o in orders}
    original_order: dict[str, Order] = {o.ticker: o for o in orders}

    # ------------------------------------------------------------------
    # Helper: run one batch stage
    # ------------------------------------------------------------------

    def _run_batch_stage(
        stage_name: str,
        stage_orders: list[Order],
        submit_fn: Callable,          # submit_fn(order) -> OrderHandle (price injected via closure)
        rest_sec: float,
        quotes_at_submit: dict[str, tuple[float, float]],
    ) -> None:
        """Submit all → rest → cancel all → grace → poll all."""
        if not stage_orders:
            return

        logger.info(
            "Ladder %s: submitting %d orders (batch)", stage_name, len(stage_orders)
        )

        # 1. Submit ALL (non-blocking)
        # If any submit raises, cancel already-submitted handles (best-effort)
        # before re-raising so no live orders are orphaned at the broker.
        handles: dict[str, OrderHandle] = {}
        for order in stage_orders:
            try:
                handles[order.ticker] = submit_fn(order)
            except Exception:
                for h in handles.values():
                    try:
                        broker.cancel(h)
                    except Exception:
                        pass
                raise

        # 2. Rest — all orders rest concurrently
        sleep_fn(rest_sec)

        # 3. Cancel ALL (idempotent)
        for h in handles.values():
            broker.cancel(h)

        # 4. Cancel grace — one settle wait
        sleep_fn(config.LADDER_CANCEL_GRACE_SEC)

        # 5. Poll ALL fills and accumulate
        for order in stage_orders:
            ticker = order.ticker
            handle = handles[ticker]
            fill = broker.get_fill(handle)

            bid, ask = quotes_at_submit[ticker]
            mid = (bid + ask) / 2.0

            qty_f = fill.quantity
            filled_qty[ticker] += qty_f
            if qty_f > 0 and fill.avg_price is not None:
                cost_basis[ticker] += qty_f * fill.avg_price
            remaining_qty[ticker] = original_order[ticker].quantity - filled_qty[ticker]

            audit.stages.append(StageAuditRecord(
                ticker=ticker,
                stage=stage_name,
                qty_attempted=order.quantity,
                qty_filled=qty_f,
                realized_price=fill.avg_price if qty_f > 0 else None,
                midpoint_at_fill=mid,
            ))
            logger.info(
                "Ladder %s [%s]: filled %.4f / %.4f (mid=%.4f)",
                stage_name, ticker, qty_f, order.quantity, mid,
            )

    # ------------------------------------------------------------------
    # Stage 1 — Passive limit
    # ------------------------------------------------------------------

    # Fetch all quotes first (before submitting) so we have mid for the audit
    stage1_quotes: dict[str, tuple[float, float]] = {}
    for order in orders:
        stage1_quotes[order.ticker] = broker.get_quote(order.ticker)

    def _submit_limit_passive(order: Order) -> OrderHandle:
        bid, ask = stage1_quotes[order.ticker]
        passive_price = bid if order.side == "BUY" else ask
        logger.info(
            "Ladder passive [%s]: submit_limit %s qty=%.4f @ %.4f (bid=%.4f ask=%.4f)",
            order.ticker, order.side, order.quantity, passive_price, bid, ask,
        )
        return broker.submit_limit(order, passive_price)

    _run_batch_stage(
        stage_name="passive",
        stage_orders=orders,
        submit_fn=_submit_limit_passive,
        rest_sec=config.LADDER_PASSIVE_WAIT_SEC,
        quotes_at_submit=stage1_quotes,
    )

    # ------------------------------------------------------------------
    # Stage 2 — MIDPRICE for unfilled remainder
    # ------------------------------------------------------------------

    stage2_orders = [
        Order(ticker=t, side=original_order[t].side, quantity=remaining_qty[t])
        for t in original_order
        if remaining_qty[t] > 1e-9
    ]
    if stage2_orders:
        logger.info("Ladder stage 2: midprice for %d remainders", len(stage2_orders))
        # Fetch quotes for remaining tickers
        stage2_quotes: dict[str, tuple[float, float]] = {
            o.ticker: broker.get_quote(o.ticker) for o in stage2_orders
        }
        _run_batch_stage(
            stage_name="midprice",
            stage_orders=stage2_orders,
            submit_fn=broker.submit_midprice,
            rest_sec=config.LADDER_MIDPRICE_WAIT_SEC,
            quotes_at_submit=stage2_quotes,
        )

    # ------------------------------------------------------------------
    # Stage 3 — Terminal market (if configured)
    # ------------------------------------------------------------------

    if config.LADDER_TERMINAL_CROSS:
        stage3_orders = [
            Order(ticker=t, side=original_order[t].side, quantity=remaining_qty[t])
            for t in original_order
            if remaining_qty[t] > 1e-9
        ]
        if stage3_orders:
            logger.info("Ladder stage 3: terminal market for %d remainders", len(stage3_orders))
            stage3_quotes: dict[str, tuple[float, float]] = {
                o.ticker: broker.get_quote(o.ticker) for o in stage3_orders
            }
            _run_batch_stage(
                stage_name="terminal",
                stage_orders=stage3_orders,
                submit_fn=broker.submit_market,
                rest_sec=config.LADDER_CANCEL_GRACE_SEC,  # short rest for terminal cross
                quotes_at_submit=stage3_quotes,
            )

    # ------------------------------------------------------------------
    # Aggregate one Fill per ticker
    # ------------------------------------------------------------------
    fills: list[Fill] = []
    for ticker, orig_order in original_order.items():
        total_filled = filled_qty[ticker]
        total_qty = orig_order.quantity

        if total_filled <= 0.0:
            status = "unfilled"
            avg_price = 0.0
        elif total_filled < total_qty - 1e-9:
            status = "partial"
            avg_price = cost_basis[ticker] / total_filled
        else:
            status = "filled"
            avg_price = cost_basis[ticker] / total_filled if total_filled > 0 else 0.0

        fills.append(Fill(
            ticker=ticker,
            side=orig_order.side,
            quantity=total_filled,
            avg_price=avg_price,
            status=status,
        ))

    return fills, audit
