"""Tests for trading/execution/ladder.py — Part B1.

All tests use DryRunBroker with an injectable sleep_fn so no real time elapses.

The new batch-per-stage design means sleep_fn is called TWICE per stage that
runs: once for the rest period and once for the cancel-grace period.
"""
from __future__ import annotations

import pytest

from trading.broker.base import Fill, Order, OrderHandle
from trading.broker.dryrun import DryRunBroker
from trading.execution.ladder import LadderAuditRecord, execute_ladder


# ---------------------------------------------------------------------------
# Minimal config stub
# ---------------------------------------------------------------------------

class _Cfg:
    """Minimal config object with ladder settings."""
    LADDER_PASSIVE_WAIT_SEC = 180
    LADDER_MIDPRICE_WAIT_SEC = 120
    LADDER_CANCEL_GRACE_SEC = 3
    LADDER_TERMINAL_CROSS = True


_cfg = _Cfg()
_cfg_no_terminal = type("Cfg", (), {
    "LADDER_PASSIVE_WAIT_SEC": 180,
    "LADDER_MIDPRICE_WAIT_SEC": 120,
    "LADDER_CANCEL_GRACE_SEC": 3,
    "LADDER_TERMINAL_CROSS": False,
})()

_no_sleep = lambda seconds: None  # noqa: E731


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_broker_and_order(fill_ratio: float = 1.0):
    """Return a DryRunBroker + a single BUY order for AAPL."""
    broker = DryRunBroker(
        positions={},
        nav=100_000.0,
        quotes={"AAPL": (149.50, 150.50)},  # bid=149.50, ask=150.50, mid=150.0
        fill_ratio=fill_ratio,
    )
    order = Order(ticker="AAPL", side="BUY", quantity=10.0)
    return broker, order


# ---------------------------------------------------------------------------
# Empty orders list
# ---------------------------------------------------------------------------

def test_empty_orders_returns_empty():
    broker = DryRunBroker()
    fills, audit = execute_ladder(broker, [], config=_cfg, sleep_fn=_no_sleep)
    assert fills == []
    assert isinstance(audit, LadderAuditRecord)
    assert audit.stages == []


# ---------------------------------------------------------------------------
# Full fill at stage 1 — no escalation
# ---------------------------------------------------------------------------

def test_full_fill_at_stage1_no_escalation():
    """fill_ratio=1.0, zero-spread: passive BUY at bid=ask=150 is marketable.

    Stage 2 and 3 are skipped (remainder == 0 after stage 1).
    """
    broker = DryRunBroker(
        positions={},
        nav=100_000.0,
        quotes={"AAPL": (150.0, 150.0)},  # zero spread → BUY @ bid=150 is marketable
        fill_ratio=1.0,
    )
    order = Order(ticker="AAPL", side="BUY", quantity=10.0)
    fills, audit = execute_ladder(broker, [order], config=_cfg, sleep_fn=_no_sleep)

    assert len(fills) == 1
    fill = fills[0]
    assert fill.ticker == "AAPL"
    assert fill.quantity == pytest.approx(10.0)
    assert fill.status == "filled"

    # Only stage 1 should have recorded a fill; stages 2+3 should not appear
    stage_names = [s.stage for s in audit.stages if s.ticker == "AAPL"]
    assert "passive" in stage_names
    assert "midprice" not in stage_names
    assert "terminal" not in stage_names


# ---------------------------------------------------------------------------
# Partial fill at stage 1 → midprice at stage 2
# ---------------------------------------------------------------------------

def test_partial_stage1_escalates_to_midprice():
    """fill_ratio=0.5, zero-spread: stage 1 fills half; stage 2+3 consume the rest."""
    broker = DryRunBroker(
        positions={},
        nav=100_000.0,
        quotes={"AAPL": (150.0, 150.0)},  # zero spread → passive BUY @ bid=150 is marketable
        fill_ratio=0.5,
    )
    order = Order(ticker="AAPL", side="BUY", quantity=10.0)
    fills, audit = execute_ladder(broker, [order], config=_cfg, sleep_fn=_no_sleep)

    assert len(fills) == 1
    fill = fills[0]
    assert fill.ticker == "AAPL"
    # Stage1: 10*0.5=5.0 filled, remainder=5.0
    # Stage2: 5*0.5=2.5 filled, remainder=2.5
    # Stage3 (terminal): 2.5*0.5=1.25 filled
    # Total = 5 + 2.5 + 1.25 = 8.75 → partial
    assert fill.quantity == pytest.approx(8.75)
    assert fill.status == "partial"

    stage_names = [s.stage for s in audit.stages if s.ticker == "AAPL"]
    assert "passive" in stage_names
    assert "midprice" in stage_names


def test_partial_stage1_passive_limit_unfilled_escalates():
    """Standard spread (bid < ask): passive BUY at bid is non-marketable → unfilled.
    Stage 2 (midprice) fills fill_ratio of the original qty.
    Stage 3 (terminal) fills the rest.
    """
    broker = DryRunBroker(
        positions={},
        nav=100_000.0,
        quotes={"AAPL": (149.50, 150.50)},  # spread: passive BUY @ 149.50 < ask → unfilled
        fill_ratio=0.5,
    )
    order = Order(ticker="AAPL", side="BUY", quantity=10.0)
    fills, audit = execute_ladder(broker, [order], config=_cfg, sleep_fn=_no_sleep)

    assert len(fills) == 1
    fill = fills[0]
    # Stage1: unfilled (0). Stage2: 10*0.5=5. Stage3: 5*0.5=2.5. Total=7.5 → partial
    assert fill.quantity == pytest.approx(7.5)
    assert fill.status == "partial"

    stage_names = [s.stage for s in audit.stages if s.ticker == "AAPL"]
    assert "passive" in stage_names
    assert "midprice" in stage_names
    assert "terminal" in stage_names


# ---------------------------------------------------------------------------
# Terminal market fill
# ---------------------------------------------------------------------------

def test_partial_escalates_to_terminal_market():
    """fill_ratio=0.0 throughout: all three stages attempted, all unfilled."""
    broker = DryRunBroker(
        positions={},
        nav=100_000.0,
        quotes={"AAPL": (149.50, 150.50)},
        fill_ratio=0.0,
    )
    order = Order(ticker="AAPL", side="BUY", quantity=10.0)
    fills, audit = execute_ladder(broker, [order], config=_cfg, sleep_fn=_no_sleep)

    fill = fills[0]
    assert fill.quantity == pytest.approx(0.0)
    assert fill.status == "unfilled"

    stage_names = [s.stage for s in audit.stages if s.ticker == "AAPL"]
    # All three stages attempted
    assert "passive" in stage_names
    assert "midprice" in stage_names
    assert "terminal" in stage_names


def test_terminal_cross_disabled():
    """LADDER_TERMINAL_CROSS=False: stage 3 is skipped entirely."""
    broker = DryRunBroker(
        positions={},
        nav=100_000.0,
        quotes={"AAPL": (149.50, 150.50)},
        fill_ratio=0.0,
    )
    order = Order(ticker="AAPL", side="BUY", quantity=10.0)
    fills, audit = execute_ladder(broker, [order], config=_cfg_no_terminal, sleep_fn=_no_sleep)

    stage_names = [s.stage for s in audit.stages if s.ticker == "AAPL"]
    assert "terminal" not in stage_names


# ---------------------------------------------------------------------------
# fill_ratio=0 all three stages — audit should list all three stages
# ---------------------------------------------------------------------------

def test_fill_ratio_zero_all_three_stages_in_audit():
    """With fill_ratio=0 and LADDER_TERMINAL_CROSS=True, all 3 stage records appear."""
    broker = DryRunBroker(
        positions={},
        nav=100_000.0,
        quotes={"AAPL": (149.50, 150.50)},
        fill_ratio=0.0,
    )
    order = Order(ticker="AAPL", side="BUY", quantity=10.0)
    fills, audit = execute_ladder(broker, [order], config=_cfg, sleep_fn=_no_sleep)

    stage_names = [s.stage for s in audit.stages if s.ticker == "AAPL"]
    assert "passive" in stage_names
    assert "midprice" in stage_names
    assert "terminal" in stage_names
    assert fills[0].status == "unfilled"


# ---------------------------------------------------------------------------
# Audit record content
# ---------------------------------------------------------------------------

def test_audit_records_midpoint():
    """Each stage audit record captures the contemporaneous midpoint."""
    broker = DryRunBroker(
        positions={},
        nav=100_000.0,
        quotes={"AAPL": (149.50, 150.50)},  # mid = 150.0
        fill_ratio=0.5,
    )
    order = Order(ticker="AAPL", side="BUY", quantity=10.0)
    fills, audit = execute_ladder(broker, [order], config=_cfg, sleep_fn=_no_sleep)

    for record in audit.stages:
        if record.ticker == "AAPL":
            assert record.midpoint_at_fill == pytest.approx(150.0)


def test_audit_stage1_realized_price_at_bid_for_buy():
    """Stage 1 passive BUY should have realized_price = bid (when marketable)."""
    broker = DryRunBroker(
        positions={},
        nav=100_000.0,
        quotes={"AAPL": (150.0, 150.0)},  # zero spread, bid=150
        fill_ratio=1.0,
    )
    order = Order(ticker="AAPL", side="BUY", quantity=5.0)
    fills, audit = execute_ladder(broker, [order], config=_cfg, sleep_fn=_no_sleep)

    passive_records = [s for s in audit.stages if s.ticker == "AAPL" and s.stage == "passive"]
    assert passive_records, "Expected a passive stage record"
    rec = passive_records[0]
    assert rec.realized_price == pytest.approx(150.0)
    assert rec.qty_filled == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Multiple orders
# ---------------------------------------------------------------------------

def test_multiple_orders_one_fill_per_ticker():
    """With 3 orders (AAPL, MSFT, GOOG), ladder returns 3 fills."""
    quotes = {
        "AAPL": (150.0, 150.0),
        "MSFT": (300.0, 300.0),
        "GOOG": (180.0, 180.0),
    }
    broker = DryRunBroker(positions={}, nav=200_000.0, quotes=quotes, fill_ratio=1.0)
    orders = [
        Order("AAPL", "BUY", 10.0),
        Order("MSFT", "SELL", 5.0),
        Order("GOOG", "BUY", 20.0),
    ]
    fills, audit = execute_ladder(broker, orders, config=_cfg, sleep_fn=_no_sleep)

    assert len(fills) == 3
    fill_tickers = {f.ticker for f in fills}
    assert fill_tickers == {"AAPL", "MSFT", "GOOG"}


# ---------------------------------------------------------------------------
# sleep_fn call count — batch design: 2 calls per stage (rest + grace)
# ---------------------------------------------------------------------------

def test_sleep_fn_called_twice_per_stage_stage1_only():
    """Stage 1 completes fully (fill_ratio=1, zero spread): 2 sleep calls.

    Pattern: rest(LADDER_PASSIVE_WAIT_SEC) + grace(LADDER_CANCEL_GRACE_SEC).
    Stages 2 and 3 are skipped (no remainder), so total = 2.
    """
    sleep_calls = []

    def _track_sleep(secs: float) -> None:
        sleep_calls.append(secs)

    broker = DryRunBroker(
        positions={},
        nav=100_000.0,
        quotes={"AAPL": (150.0, 150.0)},
        fill_ratio=1.0,
    )
    order = Order("AAPL", "BUY", 5.0)
    execute_ladder(broker, [order], config=_cfg, sleep_fn=_track_sleep)

    # Stage 1 only: rest + grace = 2 calls
    assert len(sleep_calls) == 2
    assert sleep_calls[0] == _cfg.LADDER_PASSIVE_WAIT_SEC
    assert sleep_calls[1] == _cfg.LADDER_CANCEL_GRACE_SEC


def test_sleep_fn_called_four_times_for_two_stages():
    """Stage 1 unfilled, stage 2 fills: 4 sleep calls total (2 per stage).

    Standard spread → passive BUY at bid is non-marketable → all goes to stage 2.
    Stage 2 (midprice) with fill_ratio=1 fills completely → no stage 3.
    """
    sleep_calls = []

    def _track_sleep(secs: float) -> None:
        sleep_calls.append(secs)

    broker = DryRunBroker(
        positions={},
        nav=100_000.0,
        quotes={"AAPL": (149.50, 150.50)},  # spread: passive BUY @ 149.50 → unfilled
        fill_ratio=1.0,
    )
    order = Order("AAPL", "BUY", 5.0)
    execute_ladder(broker, [order], config=_cfg, sleep_fn=_track_sleep)

    # Stage 1: rest(180) + grace(3) = 2 calls
    # Stage 2: rest(120) + grace(3) = 2 calls
    # Stage 3: skipped (fill_ratio=1 means stage 2 filled it all)
    assert len(sleep_calls) == 4
    assert sleep_calls[0] == _cfg.LADDER_PASSIVE_WAIT_SEC
    assert sleep_calls[1] == _cfg.LADDER_CANCEL_GRACE_SEC
    assert sleep_calls[2] == _cfg.LADDER_MIDPRICE_WAIT_SEC
    assert sleep_calls[3] == _cfg.LADDER_CANCEL_GRACE_SEC


def test_sleep_fn_called_six_times_for_all_three_stages():
    """All 3 stages run: 6 sleep calls (2 per stage).

    fill_ratio=0 ensures nothing fills at stages 1 and 2, forcing escalation.
    Stage 3 still runs even though fill_ratio=0 (unfilled terminal).
    """
    sleep_calls = []

    def _track_sleep(secs: float) -> None:
        sleep_calls.append(secs)

    broker = DryRunBroker(
        positions={},
        nav=100_000.0,
        quotes={"AAPL": (149.50, 150.50)},
        fill_ratio=0.0,
    )
    order = Order("AAPL", "BUY", 5.0)
    execute_ladder(broker, [order], config=_cfg, sleep_fn=_track_sleep)

    # Stage 1: rest(180) + grace(3) = 2 calls
    # Stage 2: rest(120) + grace(3) = 2 calls
    # Stage 3: rest(grace=3) + grace(3) = 2 calls → total = 6
    assert len(sleep_calls) == 6
    # Stage 1
    assert sleep_calls[0] == _cfg.LADDER_PASSIVE_WAIT_SEC
    assert sleep_calls[1] == _cfg.LADDER_CANCEL_GRACE_SEC
    # Stage 2
    assert sleep_calls[2] == _cfg.LADDER_MIDPRICE_WAIT_SEC
    assert sleep_calls[3] == _cfg.LADDER_CANCEL_GRACE_SEC
    # Stage 3 (terminal rest = LADDER_CANCEL_GRACE_SEC)
    assert sleep_calls[4] == _cfg.LADDER_CANCEL_GRACE_SEC
    assert sleep_calls[5] == _cfg.LADDER_CANCEL_GRACE_SEC


# ---------------------------------------------------------------------------
# No over-fill: cancel before escalate
# ---------------------------------------------------------------------------

def test_no_overfill_cancel_before_escalate():
    """After stage 1, cancel is called for all handles before stage 2 submits.

    Verify: stage 2 only attempts the remainder after stage 1's fill, not the
    full original quantity.
    """
    # Zero-spread so stage 1 is marketable; fill_ratio=0.5 so it's partial.
    broker = DryRunBroker(
        positions={},
        nav=100_000.0,
        quotes={"AAPL": (150.0, 150.0)},
        fill_ratio=0.5,
    )
    order = Order(ticker="AAPL", side="BUY", quantity=10.0)
    fills, audit = execute_ladder(broker, [order], config=_cfg, sleep_fn=_no_sleep)

    stage2_records = [s for s in audit.stages if s.ticker == "AAPL" and s.stage == "midprice"]
    assert stage2_records, "Expected a midprice stage record"
    # stage 2 should only attempt the remainder (5.0), not the original 10.0
    assert stage2_records[0].qty_attempted == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Weighted average price
# ---------------------------------------------------------------------------

def test_aggregated_fill_avg_price_is_quantity_weighted():
    """avg_price in the aggregated Fill is weighted by qty at each stage."""
    # Use zero spread so stage 1 is marketable at 150.0, fill_ratio=0.5
    # Stage 1: fill 5 @ 150.0. Stage 2: fill 2.5 @ mid=150.0. Stage 3: fill 1.25 @ ask=150.0
    broker = DryRunBroker(
        positions={},
        nav=100_000.0,
        quotes={"AAPL": (150.0, 150.0)},
        fill_ratio=0.5,
    )
    order = Order("AAPL", "BUY", 10.0)
    fills, _ = execute_ladder(broker, [order], config=_cfg, sleep_fn=_no_sleep)

    fill = fills[0]
    # All fills at 150.0 → avg = 150.0
    assert fill.avg_price == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# Fix 1 — BLOCKING: orphaned orders on mid-batch submit failure
# ---------------------------------------------------------------------------

class _FailOnThirdSubmitBroker(DryRunBroker):
    """DryRunBroker that raises on the 3rd call to submit_limit, tracking
    cancel calls so we can assert orphan-cancel behaviour."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._submit_calls = 0
        self.cancelled_handle_ids: list[int] = []

    def submit_limit(self, order: Order, limit_price: float) -> OrderHandle:
        self._submit_calls += 1
        if self._submit_calls == 3:
            raise RuntimeError("simulated broker submit failure on 3rd order")
        return super().submit_limit(order, limit_price)

    def cancel(self, handle: OrderHandle) -> None:
        self.cancelled_handle_ids.append(handle.ref)
        super().cancel(handle)


def test_orphan_cancel_on_mid_batch_submit_failure():
    """If submit raises on the 3rd order, the first 2 already-submitted handles
    must be cancelled before re-raising — no orphaned live orders."""
    quotes = {
        "AAPL": (150.0, 150.0),
        "MSFT": (300.0, 300.0),
        "GOOG": (180.0, 180.0),
    }
    broker = _FailOnThirdSubmitBroker(
        positions={},
        nav=500_000.0,
        quotes=quotes,
        fill_ratio=1.0,
    )
    orders = [
        Order("AAPL", "BUY", 10.0),
        Order("MSFT", "BUY", 5.0),
        Order("GOOG", "BUY", 8.0),  # this is the 3rd — triggers the raise
    ]

    with pytest.raises(RuntimeError, match="simulated broker submit failure"):
        execute_ladder(broker, orders, config=_cfg, sleep_fn=_no_sleep)

    # The first two orders were submitted successfully (handle ids 0 and 1).
    # The cancel loop must have cancelled exactly those two handles.
    assert len(broker.cancelled_handle_ids) == 2, (
        f"Expected 2 cancels for orphaned orders, got {broker.cancelled_handle_ids}"
    )
    assert set(broker.cancelled_handle_ids) == {0, 1}
