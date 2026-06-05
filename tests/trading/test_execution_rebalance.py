"""Tests for trading/execution/rebalance.py — Part B2.

All tests run against DryRunBroker. No network, no IB Gateway.
Frozen weights JSON files are written to tmp dirs and paths are monkeypatched.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from trading.broker.dryrun import DryRunBroker
from trading.execution.rebalance import run_rebalance
from trading.execution.safety import SafetyError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_weights_json(weights_dir: Path, asof: str, weights: dict, k_probs: dict | None = None):
    """Write a minimal frozen-weights JSON file to weights_dir."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    if k_probs is None:
        k_probs = {"10": 0.5, "20": 0.5}
    payload = {
        "asof": asof,
        "k_probs": k_probs,
        "weights": weights,
    }
    path = weights_dir / f"{asof}.json"
    with path.open("w") as f:
        json.dump(payload, f)
    return path


def _make_valid_weights(n: int = 15) -> dict[str, float]:
    """Return n equal-weight tickers summing to 1.0."""
    tickers = [f"T{i:03d}" for i in range(n)]
    w = 1.0 / n
    return {t: w for t in tickers}


def _make_broker(weights: dict[str, float], nav: float = 100_000.0, fill_ratio: float = 1.0,
                 positions: dict | None = None) -> DryRunBroker:
    """Build a DryRunBroker whose quotes cover all tickers in weights."""
    quotes = {t: (99.50, 100.50) for t in weights}
    pos = positions if positions is not None else {}
    return DryRunBroker(positions=pos, nav=nav, quotes=quotes, fill_ratio=fill_ratio)


_ASOF = "2026-06-02"


# ---------------------------------------------------------------------------
# Config stub that redirects paths to tmp
# ---------------------------------------------------------------------------

class _Cfg:
    """Config stub for tests — pointing ORDERS/KILL_SWITCH to tmp."""
    EXECUTION_MODE = "dryrun"
    KILL_SWITCH_FILE = Path("/nonexistent/KILL_SWITCH")

    # Execution ladder
    LADDER_PASSIVE_WAIT_SEC = 0
    LADDER_MIDPRICE_WAIT_SEC = 0
    LADDER_CANCEL_GRACE_SEC = 0
    LADDER_TERMINAL_CROSS = True

    # Safety rails — must match the production values to pass safety
    MAX_ORDER_FRAC_NAV = 0.12
    MAX_TURNOVER_FRAC = 0.60
    MIN_HOLDINGS = 10
    MAX_HOLDINGS = 503
    WEIGHT_SUM_TOL = 1e-6
    MAX_WEIGHT = 0.10   # from src.strategy.constants — matches backtest cap


# ---------------------------------------------------------------------------
# End-to-end success: first-build from cash
# ---------------------------------------------------------------------------

def test_end_to_end_first_build(tmp_path):
    """First-build from cash (empty positions): full pipeline passes safety + produces fills."""
    weights = _make_valid_weights(n=15)
    weights_dir = tmp_path / "weights"
    orders_dir = tmp_path / "orders"
    _write_weights_json(weights_dir, _ASOF, weights)

    broker = _make_broker(weights, nav=150_000.0, fill_ratio=1.0, positions={})
    cfg = _Cfg()
    cfg.KILL_SWITCH_FILE = tmp_path / "KILL_SWITCH_ABSENT"

    summary = run_rebalance(
        asof=_ASOF,
        broker=broker,
        config=cfg,
        weights_dir=weights_dir,
        orders_dir=orders_dir,
        mode="dryrun",
    )

    assert summary["n_orders"] == 15
    assert summary["first_build"] is True
    assert summary["n_filled"] == 15
    # Audit file written
    orders_path = summary["orders_path"]
    assert Path(orders_path).exists()
    # Audit content
    with open(orders_path) as f:
        audit = json.load(f)
    assert audit["asof"] == _ASOF
    assert len(audit["orders"]) == 15
    assert len(audit["fills"]) == 15
    assert audit["first_build"] is True


# ---------------------------------------------------------------------------
# End-to-end success: incremental rebalance (positions already exist)
# ---------------------------------------------------------------------------

def test_end_to_end_incremental(tmp_path):
    """Incremental rebalance (existing positions): small top-ups pass safety."""
    weights = _make_valid_weights(n=15)
    weights_dir = tmp_path / "weights"
    orders_dir = tmp_path / "orders"
    _write_weights_json(weights_dir, _ASOF, weights)

    # Current positions: already 95% of target → small incremental BUYs
    nav = 150_000.0
    price = 100.0
    target_shares_per = (1.0 / 15) * nav / price  # ~1000
    current_positions = {t: target_shares_per * 0.95 for t in weights}

    broker = DryRunBroker(
        positions=current_positions,
        nav=nav,
        quotes={t: (99.50, 100.50) for t in weights},
        fill_ratio=1.0,
    )
    cfg = _Cfg()
    cfg.KILL_SWITCH_FILE = tmp_path / "KILL_SWITCH_ABSENT"

    summary = run_rebalance(
        asof=_ASOF,
        broker=broker,
        config=cfg,
        weights_dir=weights_dir,
        orders_dir=orders_dir,
        mode="dryrun",
    )

    assert summary["first_build"] is False
    assert summary["n_orders"] == 15  # all 15 small top-ups
    assert Path(summary["orders_path"]).exists()


# ---------------------------------------------------------------------------
# Kill switch aborts before any placement
# ---------------------------------------------------------------------------

def test_kill_switch_aborts(tmp_path):
    """Kill-switch file present → SafetyError raised, no fills written."""
    weights = _make_valid_weights(n=15)
    weights_dir = tmp_path / "weights"
    orders_dir = tmp_path / "orders"
    _write_weights_json(weights_dir, _ASOF, weights)

    kill_file = tmp_path / "KILL_SWITCH"
    kill_file.touch()

    broker = _make_broker(weights, nav=150_000.0)
    cfg = _Cfg()
    cfg.KILL_SWITCH_FILE = kill_file

    with pytest.raises(SafetyError, match="KILL SWITCH"):
        run_rebalance(
            asof=_ASOF,
            broker=broker,
            config=cfg,
            weights_dir=weights_dir,
            orders_dir=orders_dir,
            mode="dryrun",
        )

    # No audit file should have been written
    assert not any(orders_dir.rglob("*.json")) if orders_dir.exists() else True


# ---------------------------------------------------------------------------
# Missing frozen-weights file gives a clear error
# ---------------------------------------------------------------------------

def test_missing_weights_file_raises_file_not_found(tmp_path):
    """If frozen weights don't exist, a clear FileNotFoundError is raised."""
    weights = _make_valid_weights(n=15)
    weights_dir = tmp_path / "weights"  # dir exists but no JSON inside
    weights_dir.mkdir(parents=True)
    orders_dir = tmp_path / "orders"

    broker = _make_broker(weights, nav=150_000.0)
    cfg = _Cfg()
    cfg.KILL_SWITCH_FILE = tmp_path / "KILL_SWITCH_ABSENT"

    with pytest.raises(FileNotFoundError, match="Frozen weights not found"):
        run_rebalance(
            asof=_ASOF,
            broker=broker,
            config=cfg,
            weights_dir=weights_dir,
            orders_dir=orders_dir,
            mode="dryrun",
        )


# ---------------------------------------------------------------------------
# paper/live mode without injected broker raises NotImplementedError
# ---------------------------------------------------------------------------

def test_paper_mode_without_broker_raises_not_implemented(tmp_path):
    """mode='paper' without injected broker raises NotImplementedError (Phase C)."""
    weights = _make_valid_weights(n=15)
    weights_dir = tmp_path / "weights"
    orders_dir = tmp_path / "orders"
    _write_weights_json(weights_dir, _ASOF, weights)
    cfg = _Cfg()
    cfg.KILL_SWITCH_FILE = tmp_path / "KILL_SWITCH_ABSENT"

    with pytest.raises(NotImplementedError):
        run_rebalance(
            asof=_ASOF,
            mode="paper",
            config=cfg,
            weights_dir=weights_dir,
            orders_dir=orders_dir,
        )


# ---------------------------------------------------------------------------
# confirm=True: user says 'yes' → proceeds
# ---------------------------------------------------------------------------

def test_confirm_yes_proceeds(tmp_path, capsys):
    """confirm=True with input_fn returning 'yes' proceeds normally."""
    weights = _make_valid_weights(n=15)
    weights_dir = tmp_path / "weights"
    orders_dir = tmp_path / "orders"
    _write_weights_json(weights_dir, _ASOF, weights)

    broker = _make_broker(weights, nav=150_000.0)
    cfg = _Cfg()
    cfg.KILL_SWITCH_FILE = tmp_path / "KILL_SWITCH_ABSENT"

    summary = run_rebalance(
        asof=_ASOF,
        broker=broker,
        config=cfg,
        weights_dir=weights_dir,
        orders_dir=orders_dir,
        mode="dryrun",
        confirm=True,
        input_fn=lambda _: "yes",
    )

    assert summary["n_orders"] == 15
    # The order table should have been printed
    captured = capsys.readouterr()
    assert "PENDING ORDERS" in captured.out


# ---------------------------------------------------------------------------
# confirm=True: user says 'no' → aborts
# ---------------------------------------------------------------------------

def test_confirm_no_aborts(tmp_path):
    """confirm=True with input_fn returning 'no' raises SafetyError."""
    weights = _make_valid_weights(n=15)
    weights_dir = tmp_path / "weights"
    orders_dir = tmp_path / "orders"
    _write_weights_json(weights_dir, _ASOF, weights)

    broker = _make_broker(weights, nav=150_000.0)
    cfg = _Cfg()
    cfg.KILL_SWITCH_FILE = tmp_path / "KILL_SWITCH_ABSENT"

    with pytest.raises(SafetyError, match="aborted by user"):
        run_rebalance(
            asof=_ASOF,
            broker=broker,
            config=cfg,
            weights_dir=weights_dir,
            orders_dir=orders_dir,
            mode="dryrun",
            confirm=True,
            input_fn=lambda _: "no",
        )


# ---------------------------------------------------------------------------
# Audit file content: ladder stages recorded
# ---------------------------------------------------------------------------

def test_audit_records_ladder_stages(tmp_path):
    """Audit JSON must contain ladder_stages with stage names."""
    weights = _make_valid_weights(n=15)
    weights_dir = tmp_path / "weights"
    orders_dir = tmp_path / "orders"
    _write_weights_json(weights_dir, _ASOF, weights)

    broker = _make_broker(weights, nav=150_000.0, fill_ratio=1.0)
    cfg = _Cfg()
    cfg.KILL_SWITCH_FILE = tmp_path / "KILL_SWITCH_ABSENT"

    summary = run_rebalance(
        asof=_ASOF,
        broker=broker,
        config=cfg,
        weights_dir=weights_dir,
        orders_dir=orders_dir,
        mode="dryrun",
    )

    with open(summary["orders_path"]) as f:
        audit = json.load(f)

    assert "ladder_stages" in audit
    stages = audit["ladder_stages"]
    assert len(stages) > 0
    stage_names = {s["stage"] for s in stages}
    # At minimum the passive stage should appear
    assert "passive" in stage_names


# ---------------------------------------------------------------------------
# First-build turnover exemption: no SafetyError despite ~100% turnover
# ---------------------------------------------------------------------------

def test_first_build_turnover_not_blocked(tmp_path):
    """Empty positions → first build → turnover cap NOT triggered even at 100% turnover."""
    weights = _make_valid_weights(n=15)
    weights_dir = tmp_path / "weights"
    orders_dir = tmp_path / "orders"
    _write_weights_json(weights_dir, _ASOF, weights)

    # Empty positions (first build from cash)
    broker = _make_broker(weights, nav=150_000.0, fill_ratio=1.0, positions={})
    cfg = _Cfg()
    cfg.KILL_SWITCH_FILE = tmp_path / "KILL_SWITCH_ABSENT"

    # Should NOT raise SafetyError despite 100% turnover
    summary = run_rebalance(
        asof=_ASOF,
        broker=broker,
        config=cfg,
        weights_dir=weights_dir,
        orders_dir=orders_dir,
        mode="dryrun",
    )
    assert summary["first_build"] is True
    assert summary["n_orders"] == 15


# ---------------------------------------------------------------------------
# Incremental rebalance: excessive turnover IS blocked
# ---------------------------------------------------------------------------

def test_incremental_excessive_turnover_blocked(tmp_path):
    """Incremental rebalance with huge orders triggers turnover SafetyError."""
    # Create weights that are uniformly distributed
    n = 15
    weights = _make_valid_weights(n=n)
    weights_dir = tmp_path / "weights"
    orders_dir = tmp_path / "orders"
    _write_weights_json(weights_dir, _ASOF, weights)

    nav = 100_000.0
    # Give existing positions that are very far from target (huge turnover)
    # Target: ~1000 shares each @ 100. Current: 0 shares (non-empty dict so not first-build)
    current_positions = {"T000": 0.01}  # one tiny position so it's not "first build"

    broker = DryRunBroker(
        positions=current_positions,
        nav=nav,
        quotes={t: (99.50, 100.50) for t in weights},
        fill_ratio=1.0,
    )
    cfg = _Cfg()
    cfg.KILL_SWITCH_FILE = tmp_path / "KILL_SWITCH_ABSENT"

    # Full build from a non-empty portfolio: total notional >> 60% of NAV
    with pytest.raises(SafetyError, match="[Tt]urnover"):
        run_rebalance(
            asof=_ASOF,
            broker=broker,
            config=cfg,
            weights_dir=weights_dir,
            orders_dir=orders_dir,
            mode="dryrun",
        )
