"""Safety-critical: broker selection for run_rebalance.

The cardinal rule — mode='paper' must connect to the PAPER gateway port, never
the live one — is what keeps a paper validation run from placing real orders.
These tests construct brokers WITHOUT connecting (no network).
"""
from __future__ import annotations

import types

import pytest

from trading.execution.rebalance import _select_broker


def _cfg(**kw):
    base = dict(IBKR_HOST="h", IBKR_LIVE_PORT=4001, IBKR_PAPER_PORT=4002,
                IBKR_CLIENT_ID=11, EXECUTION_MODE="dryrun")
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_paper_uses_paper_port_never_live():
    b = _select_broker("paper", _cfg())
    assert type(b).__name__ == "IBKRBroker"
    assert b._port == 4002          # SAFETY: paper must NEVER hit the live port


def test_live_uses_live_port():
    b = _select_broker("live", _cfg())
    assert type(b).__name__ == "IBKRBroker"
    assert b._port == 4001


def test_paper_and_live_ports_are_distinct():
    assert _select_broker("paper", _cfg())._port != _select_broker("live", _cfg())._port


def test_live_broker_is_not_readonly_so_it_can_place_orders():
    assert _select_broker("live", _cfg())._readonly is False


def test_dryrun_returns_dryrun_broker():
    assert type(_select_broker("dryrun", _cfg())).__name__ == "DryRunBroker"


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        _select_broker("yolo", _cfg())
