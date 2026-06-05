"""Rebalance orchestrator — the Monday execution pipeline.

Public API
----------
run_rebalance(asof, *, mode, confirm, broker, config) -> dict
    End-to-end rebalance: load frozen weights, reconcile positions, compute
    orders, run safety checks, optionally request confirmation, execute via
    the ladder, write audit, return summary.

All heavy logic is injectable (broker, config, weights_dir, orders_dir) so
tests can run end-to-end against DryRunBroker without any network or real
files.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def run_rebalance(
    asof: str | pd.Timestamp | None = None,
    *,
    mode: str | None = None,
    confirm: bool = False,
    broker=None,
    config: Any = None,
    # Injectable overrides (primarily for testing)
    weights_dir: Path | None = None,
    orders_dir: Path | None = None,
    input_fn=None,   # injectable for 'yes' prompt in tests
) -> dict:
    """Orchestrate the Monday rebalance.

    Steps
    -----
    1.  Resolve mode (mode arg > config.EXECUTION_MODE).
    2.  Pick broker: dryrun if mode=="dryrun" or broker injected; for
        paper/live raises NotImplementedError (IBKRBroker not yet implemented).
    3.  broker.connect(); reconcile current_positions + nav.
    4.  Load frozen weights from weights_dir/<asof>.json (fail clearly if missing).
    5.  Fetch quotes for union(held tickers, target tickers).
    6.  diff_to_orders; pre_trade_checks; raise SafetyError if problems.
        (skip_turnover_check=True when current_positions is empty — first build.)
    7.  If confirm=True: print order table, require typed 'yes' to proceed.
    8.  execute_ladder; write audit to orders_dir/<asof>.json.
    9.  broker.disconnect(); return summary dict.

    Parameters
    ----------
    asof : str | pd.Timestamp | None
        Rebalance date. Defaults to the most recent Friday.
    mode : str | None
        One of "dryrun", "paper", "live". Defaults to config.EXECUTION_MODE.
    confirm : bool
        If True, print the order table and require explicit "yes" before placing.
    broker : Broker | None
        Injectable broker (bypasses mode selection; used in tests).
    config : module/object | None
        Injectable config (defaults to trading.config).
    weights_dir : Path | None
        Override for trading/audit/weights/ (used in tests).
    orders_dir : Path | None
        Override for trading/audit/orders/ (used in tests).
    input_fn : callable | None
        Injectable replacement for builtins.input (used in tests).

    Returns
    -------
    dict with keys:
        asof, mode, n_orders, fills, audit, orders_path
    """
    # ------------------------------------------------------------------
    # 0. Default config
    # ------------------------------------------------------------------
    if config is None:
        import trading.config as config  # noqa: PLC0415

    # ------------------------------------------------------------------
    # 1. Resolve mode
    # ------------------------------------------------------------------
    if mode is None:
        mode = config.EXECUTION_MODE
    mode = mode.lower()

    # ------------------------------------------------------------------
    # 2. Pick broker
    # ------------------------------------------------------------------
    if broker is None:
        if mode == "dryrun":
            from trading.broker.dryrun import DryRunBroker  # noqa: PLC0415
            broker = DryRunBroker()
            logger.info("run_rebalance: using DryRunBroker (mode=dryrun)")
        else:
            # IBKRBroker is Phase C — not yet implemented.
            raise NotImplementedError(
                f"mode='{mode}' requires IBKRBroker (Phase C, not yet implemented). "
                "Use mode='dryrun' or inject a broker directly."
            )

    # ------------------------------------------------------------------
    # 3. Connect + reconcile
    # ------------------------------------------------------------------
    broker.connect()
    current_positions: dict[str, float] = broker.get_positions()
    nav: float = broker.get_nav()
    logger.info(
        "run_rebalance: reconciled nav=%.2f, %d positions", nav, len(current_positions)
    )

    # ------------------------------------------------------------------
    # 4. Load frozen weights
    # ------------------------------------------------------------------
    if asof is None:
        from trading.data.snapshot import most_recent_friday  # noqa: PLC0415
        asof_ts = most_recent_friday()
    else:
        asof_ts = pd.Timestamp(asof).normalize()

    _weights_dir = weights_dir if weights_dir is not None else config.WEIGHTS_DIR
    weights_path = Path(_weights_dir) / f"{asof_ts.date()}.json"
    if not weights_path.exists():
        broker.disconnect()
        raise FileNotFoundError(
            f"Frozen weights not found: {weights_path}\n"
            "Run 'python -m trading.run weights' first to generate and freeze weights."
        )

    with weights_path.open() as f:
        weights_payload = json.load(f)
    target_weights: dict[str, float] = {
        str(k): float(v) for k, v in weights_payload["weights"].items()
    }
    logger.info(
        "run_rebalance: loaded %d target weights from %s", len(target_weights), weights_path
    )

    # ------------------------------------------------------------------
    # 5. Fetch quotes for union of held + target tickers
    # ------------------------------------------------------------------
    all_tickers = set(target_weights.keys()) | set(current_positions.keys())
    prices: dict[str, float] = {}
    for ticker in all_tickers:
        try:
            bid, ask = broker.get_quote(ticker)
            prices[ticker] = (bid + ask) / 2.0
        except Exception as exc:  # noqa: BLE001
            logger.warning("run_rebalance: could not get quote for %s: %s", ticker, exc)

    # ------------------------------------------------------------------
    # 6. Compute orders + safety checks
    # ------------------------------------------------------------------
    from trading.execution.diff import diff_to_orders  # noqa: PLC0415
    from trading.execution.safety import SafetyError, assert_safe  # noqa: PLC0415

    orders = diff_to_orders(
        target_weights=target_weights,
        current_positions=current_positions,
        nav=nav,
        prices=prices,
    )

    # First build from cash: skip turnover check (100% turnover is expected)
    first_build = not bool(current_positions)
    if first_build:
        logger.info(
            "run_rebalance: current_positions is empty — first-build from cash; "
            "turnover cap check will be skipped"
        )

    try:
        assert_safe(
            target_weights,
            orders,
            current_positions,
            nav,
            prices,
            config=config,
            skip_turnover_check=first_build,
        )
    except SafetyError:
        broker.disconnect()
        raise

    # ------------------------------------------------------------------
    # 7. Optional confirmation prompt
    # ------------------------------------------------------------------
    if confirm:
        _print_order_table(orders, prices, nav)
        _input = input_fn if input_fn is not None else input
        answer = _input("Type 'yes' to place orders, anything else to abort: ").strip().lower()
        if answer != "yes":
            broker.disconnect()
            raise SafetyError("Rebalance aborted by user (confirmation declined).")

    # ------------------------------------------------------------------
    # 8. Execute via ladder + write audit
    # ------------------------------------------------------------------
    from trading.execution.ladder import execute_ladder  # noqa: PLC0415

    fills, ladder_audit = execute_ladder(broker, orders, config=config)

    # Post-trade positions = pre-positions + signed fills (no extra broker round-trip).
    # This is the reconciliation trail: what the system believes it holds after execution.
    post_positions = dict(current_positions)
    for f in fills:
        signed = f.quantity * (1.0 if f.side == "BUY" else -1.0)
        post_positions[f.ticker] = post_positions.get(f.ticker, 0.0) + signed

    # Build audit record
    audit_record = {
        "asof": str(asof_ts.date()),
        "mode": mode,
        "nav": nav,
        "first_build": first_build,
        "orders": [
            {"ticker": o.ticker, "side": o.side, "quantity": o.quantity}
            for o in orders
        ],
        "fills": [
            {
                "ticker": f.ticker,
                "side": f.side,
                "quantity": f.quantity,
                "avg_price": f.avg_price,
                "status": f.status,
            }
            for f in fills
        ],
        "ladder_stages": [
            {
                "ticker": s.ticker,
                "stage": s.stage,
                "qty_attempted": s.qty_attempted,
                "qty_filled": s.qty_filled,
                "realized_price": s.realized_price,
                "midpoint_at_fill": s.midpoint_at_fill,
            }
            for s in ladder_audit.stages
        ],
        "pre_positions": {t: float(v) for t, v in current_positions.items()},
        "post_positions": {t: float(v) for t, v in post_positions.items()},
    }

    _orders_dir = orders_dir if orders_dir is not None else config.ORDERS_DIR
    orders_dir_path = Path(_orders_dir)
    orders_dir_path.mkdir(parents=True, exist_ok=True)
    orders_path = orders_dir_path / f"{asof_ts.date()}.json"
    with orders_path.open("w") as f:
        json.dump(audit_record, f, indent=2)
    logger.info("run_rebalance: audit written → %s", orders_path)

    # ------------------------------------------------------------------
    # 9. Disconnect + return summary
    # ------------------------------------------------------------------
    broker.disconnect()

    filled_count = sum(1 for f in fills if f.status in ("filled", "partial"))
    summary = {
        "asof": asof_ts,
        "mode": mode,
        "n_orders": len(orders),
        "n_filled": filled_count,
        "fills": fills,
        "audit": audit_record,
        "orders_path": orders_path,
        "first_build": first_build,
    }
    logger.info(
        "run_rebalance: done — %d orders, %d fills, audit=%s",
        len(orders), filled_count, orders_path,
    )
    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _print_order_table(orders, prices, nav: float) -> None:
    """Print a human-readable order table before the confirmation prompt."""
    print()
    print("=" * 60)
    print("  PENDING ORDERS — CONFIRM BEFORE PLACEMENT")
    print("=" * 60)
    print(f"{'Ticker':<12}  {'Side':<6}  {'Qty':>12}  {'Price':>10}  {'Notional':>12}")
    print("-" * 60)
    total_notional = 0.0
    for order in sorted(orders, key=lambda o: o.ticker):
        price = prices.get(order.ticker, 0.0)
        notional = abs(order.quantity * price)
        total_notional += notional
        print(
            f"{order.ticker:<12}  {order.side:<6}  {order.quantity:>12.4f}"
            f"  {price:>10.2f}  {notional:>12.2f}"
        )
    print("-" * 60)
    print(f"  Total notional:  {total_notional:>12.2f}  ({total_notional/nav*100:.1f}% of NAV)")
    print(f"  NAV:             {nav:>12.2f}")
    print("=" * 60)
    print()
