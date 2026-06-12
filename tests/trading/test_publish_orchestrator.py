"""End-to-end publisher test against DryRunBroker + fixture audit files + fake store."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading.broker.dryrun import DryRunBroker
from trading.publish.publish import publish_live


class _RecordingStore:
    """Captures every write; serves a canned equity curve for read."""

    def __init__(self, equity=None):
        self.snapshot = None
        self.equity_point = None
        self.holdings = None
        self.weekly = None
        self.executions = None
        self._equity = equity or []

    def read_equity_curve(self):
        return list(self._equity)

    def upsert_equity_point(self, date, nav, spy_close, flow=0.0):
        self.equity_point = {"date": date, "nav": nav, "spy_close": spy_close, "flow": flow}

    def upsert_snapshot(self, row):
        self.snapshot = row

    def replace_holdings(self, rows):
        self.holdings = rows

    def insert_weekly_portfolio(self, asof_friday, rows):
        self.weekly = {"asof_friday": asof_friday, "rows": rows}

    def insert_executions(self, asof, rows):
        self.executions = {"asof": asof, "rows": rows}


def _write_weights(weights_dir: Path, asof: str):
    weights_dir.mkdir(parents=True, exist_ok=True)
    payload = {"asof": asof, "k_probs": {"10": 0.6, "20": 0.4},
               "weights": {"AAA": 0.6, "BBB": 0.4}}
    (weights_dir / f"{asof}.json").write_text(json.dumps(payload))


def test_publish_live_writes_all_products(tmp_path):
    asof = "2026-05-29"
    today = pd.Timestamp("2026-06-01")
    _write_weights(tmp_path / "weights", asof)

    broker = DryRunBroker(
        positions={"AAA": 60.0, "BBB": 40.0},
        nav=10_000.0,
        quotes={"AAA": (99.5, 100.5), "BBB": (99.5, 100.5)},
    )
    # Prior equity point so day P&L + inception return are computed.
    store = _RecordingStore(equity=[{"date": "2026-05-29", "nav": 9_000.0, "spy_close": 5000.0}])

    summary = publish_live(
        broker, store,
        weights_dir=tmp_path / "weights",
        orders_dir=tmp_path / "orders",   # no orders file → executions skipped
        asof=asof, today=today, spy_last=5100.0,
    )

    assert summary["asof"] == asof
    # DryRunBroker reports 0.0 account day P&L, so the 9000→10000 ΔNAV is an
    # unexplained +1000 ≥ threshold → detected as an external flow, and every
    # growth metric excludes it.
    assert store.equity_point == {"date": "2026-06-01", "nav": 10_000.0,
                                  "spy_close": 5100.0, "flow": 1000.0}
    assert store.snapshot["nav"] == 10_000.0
    assert store.snapshot["day_pnl"] == 0.0             # broker truth, not ΔNAV
    assert store.snapshot["total_return"] == 0.0        # deposit ≠ growth
    assert store.snapshot["k_probs"] == {"10": 0.6, "20": 0.4}
    assert {h["ticker"] for h in store.holdings} == {"AAA", "BBB"}
    assert all(h["asof"] == "2026-06-01" for h in store.holdings)
    assert store.weekly["asof_friday"] == asof
    assert len(store.weekly["rows"]) == 2
    assert store.executions is None                      # no orders file


def test_publish_live_includes_executions_when_orders_file_present(tmp_path):
    asof = "2026-05-29"
    _write_weights(tmp_path / "weights", asof)
    orders_dir = tmp_path / "orders"
    orders_dir.mkdir(parents=True)
    (orders_dir / f"{asof}.json").write_text(json.dumps({
        "fills": [{"ticker": "AAA", "side": "BUY"}],
        "ladder_stages": [{"ticker": "AAA", "qty_filled": 10.0,
                           "realized_price": 100.5, "midpoint_at_fill": 100.0}],
    }))
    broker = DryRunBroker(positions={"AAA": 60.0}, nav=10_000.0,
                          quotes={"AAA": (99.5, 100.5)})
    store = _RecordingStore()

    publish_live(broker, store, weights_dir=tmp_path / "weights",
                 orders_dir=orders_dir, asof=asof,
                 today=pd.Timestamp("2026-06-01"), spy_last=5100.0)

    assert store.executions["asof"] == asof
    assert store.executions["rows"][0]["ticker"] == "AAA"


from trading.publish.publish import is_market_hours


def test_is_market_hours_weekday_midday_true():
    # Monday 2026-06-01 12:00 ET
    assert is_market_hours(pd.Timestamp("2026-06-01 12:00", tz="America/New_York")) is True


def test_is_market_hours_weekend_false():
    # Saturday 2026-05-30 12:00 ET
    assert is_market_hours(pd.Timestamp("2026-05-30 12:00", tz="America/New_York")) is False


def test_is_market_hours_after_close_false():
    assert is_market_hours(pd.Timestamp("2026-06-01 16:30", tz="America/New_York")) is False


def test_publish_live_same_day_rerun_excludes_stale_point(tmp_path):
    # A prior run today already wrote an equity point for `today`. On re-run it must
    # NOT be treated as the "prior" NAV (idempotency — the systemd timer fires repeatedly).
    asof = "2026-05-29"
    today = pd.Timestamp("2026-06-01")
    _write_weights(tmp_path / "weights", asof)
    broker = DryRunBroker(positions={"AAA": 60.0, "BBB": 40.0}, nav=10_000.0,
                          quotes={"AAA": (99.5, 100.5), "BBB": (99.5, 100.5)})
    store = _RecordingStore(equity=[{"date": "2026-06-01", "nav": 9_500.0, "spy_close": 5050.0}])

    publish_live(broker, store, weights_dir=tmp_path / "weights",
                 orders_dir=tmp_path / "orders", asof=asof, today=today, spy_last=5100.0)

    # No point STRICTLY before today → no prior NAV → day P&L % has no base
    # (day P&L itself is the broker's figure, prior point or not).
    assert store.snapshot["day_pnl"] == 0.0
    assert store.snapshot["day_pnl_pct"] is None


def test_publish_live_inception_day_empty_curve(tmp_path):
    asof = "2026-05-29"
    today = pd.Timestamp("2026-06-01")
    _write_weights(tmp_path / "weights", asof)
    broker = DryRunBroker(positions={"AAA": 60.0}, nav=10_000.0, quotes={"AAA": (99.5, 100.5)})
    store = _RecordingStore(equity=[])  # nothing yet — first ever publish

    publish_live(broker, store, weights_dir=tmp_path / "weights",
                 orders_dir=tmp_path / "orders", asof=asof, today=today, spy_last=5100.0)

    assert store.snapshot["day_pnl"] == 0.0           # broker's figure (DryRun: flat 0)
    assert store.snapshot["day_pnl_pct"] is None      # no prior point → no base
    assert store.snapshot["total_return"] == 0.0      # index starts at 1.0 today
    assert store.snapshot["spy_return"] == 0.0        # spy_close == inception_spy
    assert store.snapshot["risk"]["sharpe"] is None   # single-point series → no risk stats


def test_is_market_hours_naive_input_is_localized_to_et():
    # tz-naive Monday midday is treated as America/New_York → within hours.
    assert is_market_hours(pd.Timestamp("2026-06-01 12:00")) is True


import math as _math


def test_publish_live_computes_turnover_vs_prior_friday(tmp_path):
    wdir = tmp_path / "weights"
    wdir.mkdir(parents=True)
    # prior Friday and current Friday weights both present
    (wdir / "2026-05-22.json").write_text(json.dumps(
        {"asof": "2026-05-22", "k_probs": {"10": 1.0}, "weights": {"AAA": 0.5, "BBB": 0.5}}))
    (wdir / "2026-05-29.json").write_text(json.dumps(
        {"asof": "2026-05-29", "k_probs": {"10": 1.0}, "weights": {"AAA": 0.4, "CCC": 0.6}}))
    broker = DryRunBroker(positions={"AAA": 40.0, "CCC": 60.0}, nav=10_000.0,
                          quotes={"AAA": (99.5, 100.5), "CCC": (99.5, 100.5)})
    store = _RecordingStore()

    publish_live(broker, store, weights_dir=wdir, orders_dir=tmp_path / "orders",
                 asof="2026-05-29", today=pd.Timestamp("2026-06-01"), spy_last=5100.0)

    t = store.snapshot["turnover"]
    assert t["added"] == ["CCC"]
    assert t["dropped"] == ["BBB"]
    # 0.5*(|0.4-0.5| + |0-0.5| + |0.6-0|) = 0.6
    assert _math.isclose(t["turnover_frac"], 0.6)


def test_publish_live_turnover_none_without_prior_week(tmp_path):
    asof = "2026-05-29"
    _write_weights(tmp_path / "weights", asof)  # only this week's file exists
    broker = DryRunBroker(positions={"AAA": 60.0, "BBB": 40.0}, nav=10_000.0,
                          quotes={"AAA": (99.5, 100.5), "BBB": (99.5, 100.5)})
    store = _RecordingStore()
    publish_live(broker, store, weights_dir=tmp_path / "weights",
                 orders_dir=tmp_path / "orders", asof=asof,
                 today=pd.Timestamp("2026-06-01"), spy_last=5100.0)
    assert store.snapshot["turnover"] is None


# ---------------------------------------------------------------------------
# publish_live — new broker-account-channel behavior
# ---------------------------------------------------------------------------

def test_publish_live_week_vs_spy_in_snapshot(tmp_path):
    asof = "2026-06-05"
    today = pd.Timestamp("2026-06-10")  # Wednesday
    _write_weights(tmp_path / "weights", asof)
    broker = DryRunBroker(positions={"AAA": 60.0, "BBB": 40.0}, nav=10_300.0,
                          quotes={"AAA": (99.5, 100.5), "BBB": (99.5, 100.5)})
    # Prior Friday close is the baseline; Monday/Tuesday points exist too.
    store = _RecordingStore(equity=[
        {"date": "2026-06-05", "nav": 10_000.0, "spy_close": 5000.0},
        {"date": "2026-06-08", "nav": 10_100.0, "spy_close": 5020.0},
        {"date": "2026-06-09", "nav": 10_200.0, "spy_close": 5010.0},
    ])
    publish_live(broker, store, weights_dir=tmp_path / "weights",
                 orders_dir=tmp_path / "orders", asof=asof, today=today,
                 spy_last=5100.0)
    w = store.snapshot["week_vs_spy"]
    # "From Monday until now": baseline is Monday's close, not prior Friday's.
    assert w["baseline_date"] == "2026-06-08"
    assert _math.isclose(w["portfolio_return"], 10_300.0 / 10_100.0 - 1.0)
    assert _math.isclose(w["spy_return"], 5100.0 / 5020.0 - 1.0)
    assert _math.isclose(w["excess_return"], w["portfolio_return"] - w["spy_return"])


def test_publish_live_holdings_carry_pnl_fields(tmp_path):
    asof = "2026-06-05"
    _write_weights(tmp_path / "weights", asof)
    broker = DryRunBroker(positions={"AAA": 60.0}, nav=10_000.0,
                          quotes={"AAA": (99.5, 100.5)})
    store = _RecordingStore()
    publish_live(broker, store, weights_dir=tmp_path / "weights",
                 orders_dir=tmp_path / "orders", asof=asof,
                 today=pd.Timestamp("2026-06-10"), spy_last=5100.0)
    h = store.holdings[0]
    # DryRunBroker marks at midpoint with flat P&L — keys must be present.
    assert h["price"] == 100.0
    assert h["avg_cost"] == 100.0
    assert h["unrealized_pnl"] == 0.0
    assert h["daily_pnl"] == 0.0


def test_publish_live_first_week_week_vs_spy_none(tmp_path):
    asof = "2026-06-05"
    _write_weights(tmp_path / "weights", asof)
    broker = DryRunBroker(positions={"AAA": 60.0}, nav=10_000.0,
                          quotes={"AAA": (99.5, 100.5)})
    store = _RecordingStore(equity=[])   # no baseline before this week's Monday
    publish_live(broker, store, weights_dir=tmp_path / "weights",
                 orders_dir=tmp_path / "orders", asof=asof,
                 today=pd.Timestamp("2026-06-10"), spy_last=5100.0)
    assert store.snapshot["week_vs_spy"] is None


# ---------------------------------------------------------------------------
# publish_live — external-flow detection + capital-flows ledger
# ---------------------------------------------------------------------------

def _deposit_setup(tmp_path):
    """Prior close 100k; today's NAV 176k with 0.0 broker day P&L → 76k deposit."""
    asof = "2026-06-05"
    _write_weights(tmp_path / "weights", asof)
    broker = DryRunBroker(positions={"AAA": 1000.0}, nav=176_000.0,
                          quotes={"AAA": (99.5, 100.5)})
    store = _RecordingStore(equity=[
        {"date": "2026-06-05", "nav": 99_000.0, "spy_close": 5000.0},
        {"date": "2026-06-11", "nav": 100_000.0, "spy_close": 5050.0},
    ])
    return asof, broker, store


def test_publish_live_detected_flow_recorded_and_stripped(tmp_path):
    asof, broker, store = _deposit_setup(tmp_path)
    flows_path = tmp_path / "capital_flows.json"

    publish_live(broker, store, weights_dir=tmp_path / "weights",
                 orders_dir=tmp_path / "orders", asof=asof,
                 today=pd.Timestamp("2026-06-12"), spy_last=5100.0,
                 flows_path=flows_path)

    assert store.equity_point["flow"] == 76_000.0
    assert json.loads(flows_path.read_text()) == {"2026-06-12": 76_000.0}
    # Growth excludes the deposit: 100/99 - 1, not 176/99 - 1.
    assert _math.isclose(store.snapshot["total_return"], 100_000.0 / 99_000.0 - 1.0)
    w = store.snapshot["week_vs_spy"]
    # Week baseline = first close of this week (06-11); since then the only
    # ΔNAV is the deposit → flat.
    assert w["baseline_date"] == "2026-06-11"
    assert _math.isclose(w["portfolio_return"], 0.0, abs_tol=1e-12)
    assert store.snapshot["day_pnl"] == 0.0


def test_publish_live_manual_ledger_entry_beats_detection(tmp_path):
    asof, broker, store = _deposit_setup(tmp_path)
    flows_path = tmp_path / "capital_flows.json"
    flows_path.write_text(json.dumps({"2026-06-12": 75_242.19}))

    publish_live(broker, store, weights_dir=tmp_path / "weights",
                 orders_dir=tmp_path / "orders", asof=asof,
                 today=pd.Timestamp("2026-06-12"), spy_last=5100.0,
                 flows_path=flows_path)

    assert store.equity_point["flow"] == 75_242.19
    assert json.loads(flows_path.read_text()) == {"2026-06-12": 75_242.19}  # untouched


def test_publish_live_ledger_overrides_stored_flow_column(tmp_path):
    # A hand-corrected PAST date in the ledger wins over the curve's flow column.
    asof = "2026-06-05"
    _write_weights(tmp_path / "weights", asof)
    broker = DryRunBroker(positions={"AAA": 1000.0}, nav=176_500.0,
                          quotes={"AAA": (99.5, 100.5)})
    store = _RecordingStore(equity=[
        {"date": "2026-06-05", "nav": 100_000.0, "spy_close": 5000.0},
        {"date": "2026-06-11", "nav": 176_000.0, "spy_close": 5050.0, "flow": 70_000.0},
    ])
    flows_path = tmp_path / "capital_flows.json"
    flows_path.write_text(json.dumps({"2026-06-11": 75_000.0}))

    publish_live(broker, store, weights_dir=tmp_path / "weights",
                 orders_dir=tmp_path / "orders", asof=asof,
                 today=pd.Timestamp("2026-06-12"), spy_last=5100.0,
                 flows_path=flows_path)

    # 06-11: (176000-75000)/100000 = 1.01; 06-12: 176500/176000 → small gain on top.
    expected = 1.01 * (176_500.0 / 176_000.0) - 1.0
    assert _math.isclose(store.snapshot["total_return"], expected)


def test_publish_live_no_ledger_path_still_detects(tmp_path):
    asof, broker, store = _deposit_setup(tmp_path)

    publish_live(broker, store, weights_dir=tmp_path / "weights",
                 orders_dir=tmp_path / "orders", asof=asof,
                 today=pd.Timestamp("2026-06-12"), spy_last=5100.0)

    assert store.equity_point["flow"] == 76_000.0
    assert _math.isclose(store.snapshot["total_return"], 100_000.0 / 99_000.0 - 1.0)


def test_publish_live_account_pnl_failure_falls_back_to_holdings_sum(tmp_path):
    asof = "2026-06-05"
    _write_weights(tmp_path / "weights", asof)

    class _NoAccountPnl(DryRunBroker):
        def get_account_pnl(self):
            raise RuntimeError("reqPnL unavailable")

    broker = _NoAccountPnl(positions={"AAA": 100.0}, nav=10_050.0,
                           quotes={"AAA": (99.5, 100.5)})
    store = _RecordingStore(equity=[{"date": "2026-06-11", "nav": 10_000.0,
                                     "spy_close": 5000.0}])

    publish_live(broker, store, weights_dir=tmp_path / "weights",
                 orders_dir=tmp_path / "orders", asof=asof,
                 today=pd.Timestamp("2026-06-12"), spy_last=5100.0)

    # Per-holding sum (DryRun: 0.0 each) is the fallback day P&L.
    assert store.snapshot["day_pnl"] == 0.0
    # ΔNAV +50 unexplained but below threshold → yield, not flow.
    assert store.equity_point["flow"] == 0.0


# ---------------------------------------------------------------------------
# main() — intraday flag behavior (no network: outside-hours path only)
# ---------------------------------------------------------------------------

def test_main_intraday_skips_outside_market_hours(monkeypatch):
    import trading.publish.publish as pub
    # Freeze "now" to a Saturday so is_market_hours() is False.
    monkeypatch.setattr(pub, "is_market_hours", lambda: False)
    assert pub.main(["--intraday"]) == 0
