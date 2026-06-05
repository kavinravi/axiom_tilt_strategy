"""Tests for trading/execution/safety.py — Part A4."""
from __future__ import annotations

import pytest

from trading.broker.base import Order


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clean_inputs(nav: float = 100_000.0, n: int = 15):
    """Build valid target_weights, orders, positions, prices for a clean check.

    Simulates a small rebalance (not a first-build) so turnover stays well under 60%.
    Each ticker's current position is close to (but slightly below) its target,
    so each BUY order represents a small top-up (~5% of target notional).
    """
    tickers = [f"T{i:03d}" for i in range(n)]
    weight = 1.0 / n
    target_weights = {t: weight for t in tickers}
    price = 100.0
    prices = {t: price for t in tickers}
    # target_shares_i = weight * nav / price
    target_s = weight * nav / price
    # Current positions: already at 95% of target → only 5% adjustment needed
    current_positions: dict[str, float] = {t: target_s * 0.95 for t in tickers}
    # Small top-up orders: delta = 5% of target_s per ticker
    delta = target_s * 0.05
    orders = [Order(ticker=t, side="BUY", quantity=delta) for t in tickers]
    # Total turnover = n * delta * price = 15 * (target_s * 0.05) * 100
    # = 15 * (100_000/15/100 * 0.05) * 100 = 15 * (6.67 * 0.05) * 100 = 15 * 33.3 ≈ 500 << 60_000
    return target_weights, orders, current_positions, nav, prices


# ---------------------------------------------------------------------------
# Clean inputs → no problems
# ---------------------------------------------------------------------------

def test_clean_inputs_no_problems():
    from trading.execution.safety import pre_trade_checks
    from trading import config
    target_weights, orders, current_positions, nav, prices = _make_clean_inputs()
    problems = pre_trade_checks(target_weights, orders, current_positions, nav, prices, config=config)
    assert problems == []


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------

def test_kill_switch_engaged(tmp_path, monkeypatch):
    """Creating the kill-switch file triggers 'KILL SWITCH ENGAGED'."""
    from trading.execution.safety import pre_trade_checks
    import trading.config as cfg
    kill_file = tmp_path / "KILL_SWITCH"
    kill_file.touch()
    monkeypatch.setattr(cfg, "KILL_SWITCH_FILE", kill_file)
    target_weights, orders, current_positions, nav, prices = _make_clean_inputs()
    problems = pre_trade_checks(target_weights, orders, current_positions, nav, prices, config=cfg)
    assert any("KILL SWITCH" in p.upper() for p in problems)


def test_kill_switch_not_engaged(tmp_path, monkeypatch):
    """Kill-switch file absent → no kill-switch problem."""
    from trading.execution.safety import pre_trade_checks
    import trading.config as cfg
    kill_file = tmp_path / "KILL_SWITCH_ABSENT"
    monkeypatch.setattr(cfg, "KILL_SWITCH_FILE", kill_file)
    target_weights, orders, current_positions, nav, prices = _make_clean_inputs()
    problems = pre_trade_checks(target_weights, orders, current_positions, nav, prices, config=cfg)
    assert not any("KILL SWITCH" in p.upper() for p in problems)


# ---------------------------------------------------------------------------
# NAV
# ---------------------------------------------------------------------------

def test_zero_nav_flagged():
    from trading.execution.safety import pre_trade_checks
    from trading import config
    target_weights, orders, _, _, prices = _make_clean_inputs()
    problems = pre_trade_checks(target_weights, orders, {}, nav=0.0, prices=prices, config=config)
    assert any("nav" in p.lower() for p in problems)


def test_negative_nav_flagged():
    from trading.execution.safety import pre_trade_checks
    from trading import config
    target_weights, orders, _, _, prices = _make_clean_inputs()
    problems = pre_trade_checks(target_weights, orders, {}, nav=-1000.0, prices=prices, config=config)
    assert any("nav" in p.lower() for p in problems)


# ---------------------------------------------------------------------------
# Weight sum check
# ---------------------------------------------------------------------------

def test_weight_sum_too_low_flagged():
    from trading.execution.safety import pre_trade_checks
    from trading import config
    # Weights that sum to 0.8 (not ≈ 1)
    target_weights = {f"T{i}": 0.05 for i in range(16)}   # 16 * 0.05 = 0.80
    _, orders, current_positions, nav, prices = _make_clean_inputs()
    prices = {t: 100.0 for t in target_weights}
    problems = pre_trade_checks(target_weights, orders, {}, nav=nav, prices=prices, config=config)
    assert any("sum" in p.lower() or "weight" in p.lower() for p in problems)


def test_weight_sum_near_one_ok():
    from trading.execution.safety import pre_trade_checks
    from trading import config
    target_weights, orders, current_positions, nav, prices = _make_clean_inputs()
    problems = pre_trade_checks(target_weights, orders, current_positions, nav, prices, config=config)
    assert not any("sum" in p.lower() for p in problems)


# ---------------------------------------------------------------------------
# Max weight cap
# ---------------------------------------------------------------------------

def test_over_cap_weight_flagged():
    from trading.execution.safety import pre_trade_checks
    from trading import config
    # One ticker at 0.20 > MAX_WEIGHT (0.10); use 15 clean for rest
    target_weights, orders, current_positions, nav, prices = _make_clean_inputs(n=15)
    # Override to put 20% in first ticker
    tickers = list(target_weights.keys())
    target_weights[tickers[0]] = 0.20
    # Renormalize the rest
    remainder = 0.80
    for t in tickers[1:]:
        target_weights[t] = remainder / (len(tickers) - 1)
    problems = pre_trade_checks(target_weights, orders, current_positions, nav, prices, config=config)
    assert any("max" in p.lower() or "weight" in p.lower() or "cap" in p.lower() for p in problems)


# ---------------------------------------------------------------------------
# Holdings count
# ---------------------------------------------------------------------------

def test_too_few_holdings_flagged():
    from trading.execution.safety import pre_trade_checks
    from trading import config
    # MIN_HOLDINGS = 10; use only 5
    target_weights = {f"T{i}": 0.20 for i in range(5)}
    prices = {t: 100.0 for t in target_weights}
    orders = [Order(t, "BUY", 1.0) for t in target_weights]
    problems = pre_trade_checks(target_weights, orders, {}, nav=100_000.0, prices=prices, config=config)
    assert any("hold" in p.lower() or "holdings" in p.lower() or "count" in p.lower() for p in problems)


# ---------------------------------------------------------------------------
# Per-order notional cap
# ---------------------------------------------------------------------------

def test_oversized_single_order_flagged():
    from trading.execution.safety import pre_trade_checks
    from trading import config
    # MAX_ORDER_FRAC_NAV = 0.12 → max order notional = 0.12 * 100_000 = 12_000
    # Make one order for 13_000 notional
    nav = 100_000.0
    target_weights, _, current_positions, _, prices = _make_clean_inputs(nav=nav)
    big_order = Order("T000", "BUY", quantity=130.0)  # 130 shares * 100 = 13_000 > 12_000
    small_orders = [Order(f"T{i:03d}", "BUY", 1.0) for i in range(1, 15)]
    problems = pre_trade_checks(
        target_weights, [big_order] + small_orders,
        current_positions, nav, prices, config=config
    )
    assert any("order" in p.lower() or "notional" in p.lower() or "max" in p.lower() for p in problems)


# ---------------------------------------------------------------------------
# Turnover cap
# ---------------------------------------------------------------------------

def test_excessive_turnover_flagged():
    from trading.execution.safety import pre_trade_checks
    from trading import config
    # MAX_TURNOVER_FRAC = 0.60 → max turnover notional = 60_000 on 100_000 NAV
    # Build orders totalling > 60_000
    nav = 100_000.0
    target_weights, _, current_positions, _, prices = _make_clean_inputs(nav=nav)
    # 15 orders of 10_000 each = 150_000 total → way over 60%
    big_orders = [Order(f"T{i:03d}", "BUY", quantity=100.0) for i in range(15)]  # 100*100=10000 each
    problems = pre_trade_checks(
        target_weights, big_orders,
        current_positions, nav, prices, config=config
    )
    assert any("turnover" in p.lower() for p in problems)


# ---------------------------------------------------------------------------
# Missing price for order ticker
# ---------------------------------------------------------------------------

def test_order_ticker_missing_price_flagged():
    from trading.execution.safety import pre_trade_checks
    from trading import config
    target_weights, orders, current_positions, nav, prices = _make_clean_inputs()
    # Add an order for a ticker not in prices
    ghost_order = Order("GHOST", "BUY", 1.0)
    problems = pre_trade_checks(
        target_weights, orders + [ghost_order],
        current_positions, nav, prices, config=config
    )
    assert any("price" in p.lower() or "ghost" in p.lower() for p in problems)


# ---------------------------------------------------------------------------
# assert_safe
# ---------------------------------------------------------------------------

def test_assert_safe_raises_on_problems():
    from trading.execution.safety import assert_safe, SafetyError
    from trading import config
    target_weights = {f"T{i}": 0.20 for i in range(5)}  # too few holdings
    prices = {t: 100.0 for t in target_weights}
    orders = [Order(t, "BUY", 1.0) for t in target_weights]
    with pytest.raises(SafetyError):
        assert_safe(target_weights, orders, {}, nav=100_000.0, prices=prices, config=config)


def test_assert_safe_passes_when_clean():
    from trading.execution.safety import assert_safe
    from trading import config
    target_weights, orders, current_positions, nav, prices = _make_clean_inputs()
    # Should not raise
    assert_safe(target_weights, orders, current_positions, nav, prices, config=config)
