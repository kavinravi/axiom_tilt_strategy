"""Pre-trade safety checks + kill switch for the execution layer."""
from __future__ import annotations

import logging
from typing import Any

from trading.broker.base import Order

logger = logging.getLogger(__name__)


class SafetyError(RuntimeError):
    """Raised by assert_safe when pre_trade_checks finds problems."""


def pre_trade_checks(
    target_weights: dict[str, float],
    orders: list[Order],
    current_positions: dict[str, float],
    nav: float,
    prices: dict[str, float],
    *,
    config: Any,
    skip_turnover_check: bool = False,
) -> list[str]:
    """Return a list of problem strings (empty == safe to trade).

    Checks (in order)
    -----------------
    1. Kill switch: config.KILL_SWITCH_FILE exists → 'KILL SWITCH ENGAGED'
    2. nav > 0
    3. weights sum ≈ 1 (within config.WEIGHT_SUM_TOL)
    4. max target weight ≤ config.MAX_WEIGHT + 1e-9
    5. holdings count in [config.MIN_HOLDINGS, config.MAX_HOLDINGS]
    6. every order ticker has a price in prices
    7. per-order notional ≤ config.MAX_ORDER_FRAC_NAV * nav
    8. total traded notional ≤ config.MAX_TURNOVER_FRAC * nav
       (skipped when skip_turnover_check=True — used for first-build from cash
       where ~100% turnover is expected and should not trip the cap)

    Parameters
    ----------
    target_weights : dict[str, float]
    orders : list[Order]
    current_positions : dict[str, float]
    nav : float
    prices : dict[str, float]
    config : module / object with safety constants
    skip_turnover_check : bool
        When True, check 8 (total turnover cap) is skipped. Set by
        ``run_rebalance`` when current_positions is empty (first build from cash).
    """
    problems: list[str] = []

    # 1. Kill switch
    if config.KILL_SWITCH_FILE.exists():
        problems.append(
            f"KILL SWITCH ENGAGED: {config.KILL_SWITCH_FILE} exists — all order placement aborted"
        )

    # 2. NAV > 0
    if nav <= 0:
        problems.append(f"NAV must be positive, got nav={nav}")

    # 3. Weight sum ≈ 1
    weight_sum = sum(target_weights.values())
    if abs(weight_sum - 1.0) > config.WEIGHT_SUM_TOL:
        problems.append(
            f"Weight sum {weight_sum:.8f} deviates from 1.0 by more than {config.WEIGHT_SUM_TOL}"
        )

    # 4. Max weight cap
    if target_weights:
        max_weight = max(target_weights.values())
        if max_weight > config.MAX_WEIGHT + 1e-9:
            problems.append(
                f"Max target weight {max_weight:.6f} exceeds cap {config.MAX_WEIGHT}"
            )

    # 5. Holdings count
    n_holdings = len(target_weights)
    if n_holdings < config.MIN_HOLDINGS or n_holdings > config.MAX_HOLDINGS:
        problems.append(
            f"Holdings count {n_holdings} outside [{config.MIN_HOLDINGS}, {config.MAX_HOLDINGS}]"
        )

    # 6. Every order ticker has a price + per-order notional cap
    total_notional = 0.0
    max_order_notional = (config.MAX_ORDER_FRAC_NAV * nav) if nav > 0 else 0.0

    for order in orders:
        price = prices.get(order.ticker)
        if price is None:
            problems.append(
                f"No price available for order ticker {order.ticker}"
            )
            continue
        notional = abs(order.quantity * price)
        total_notional += notional

        # 7. Per-order notional cap
        if nav > 0 and notional > max_order_notional:
            problems.append(
                f"Order {order.ticker} notional {notional:.2f} exceeds"
                f" MAX_ORDER_FRAC_NAV cap {max_order_notional:.2f}"
                f" ({config.MAX_ORDER_FRAC_NAV * 100:.0f}% of NAV)"
            )

    # 8. Total turnover cap (exempted for first-build from cash)
    if not skip_turnover_check:
        max_turnover_notional = (config.MAX_TURNOVER_FRAC * nav) if nav > 0 else 0.0
        if nav > 0 and total_notional > max_turnover_notional:
            problems.append(
                f"Total turnover notional {total_notional:.2f} exceeds"
                f" MAX_TURNOVER_FRAC cap {max_turnover_notional:.2f}"
                f" ({config.MAX_TURNOVER_FRAC * 100:.0f}% of NAV)"
            )
    else:
        logger.info(
            "pre_trade_checks: turnover check SKIPPED (first-build from cash;"
            " total_notional=%.2f)", total_notional
        )

    if problems:
        for p in problems:
            logger.warning("pre_trade_checks: %s", p)
    else:
        logger.info("pre_trade_checks: all checks passed")

    return problems


def assert_safe(
    target_weights: dict[str, float],
    orders: list[Order],
    current_positions: dict[str, float],
    nav: float,
    prices: dict[str, float],
    *,
    config: Any,
    skip_turnover_check: bool = False,
) -> None:
    """Run pre_trade_checks and raise SafetyError if any problems found.

    Parameters
    ----------
    See pre_trade_checks.

    Raises
    ------
    SafetyError
        If pre_trade_checks returns a non-empty list.
    """
    problems = pre_trade_checks(
        target_weights, orders, current_positions, nav, prices,
        config=config, skip_turnover_check=skip_turnover_check,
    )
    if problems:
        raise SafetyError(
            f"{len(problems)} pre-trade check(s) failed:\n"
            + "\n".join(f"  - {p}" for p in problems)
        )
