"""End-to-end publisher test against DryRunBroker + fixture audit files + fake store."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading.broker.dryrun import DryRunBroker
from trading.publish.publish import publish_once


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

    def upsert_equity_point(self, date, nav, spy_close):
        self.equity_point = {"date": date, "nav": nav, "spy_close": spy_close}

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


def test_publish_once_writes_all_products(tmp_path):
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

    summary = publish_once(
        broker, store,
        weights_dir=tmp_path / "weights",
        orders_dir=tmp_path / "orders",   # no orders file → executions skipped
        asof=asof, today=today, spy_close=5100.0,
    )

    assert summary["asof"] == asof
    assert store.equity_point == {"date": "2026-06-01", "nav": 10_000.0, "spy_close": 5100.0}
    assert store.snapshot["nav"] == 10_000.0
    assert store.snapshot["day_pnl"] == 1000.0          # 10000 - 9000
    assert store.snapshot["k_probs"] == {"10": 0.6, "20": 0.4}
    assert {h["ticker"] for h in store.holdings} == {"AAA", "BBB"}
    assert all(h["asof"] == "2026-06-01" for h in store.holdings)
    assert store.weekly["asof_friday"] == asof
    assert len(store.weekly["rows"]) == 2
    assert store.executions is None                      # no orders file


def test_publish_once_includes_executions_when_orders_file_present(tmp_path):
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

    publish_once(broker, store, weights_dir=tmp_path / "weights",
                 orders_dir=orders_dir, asof=asof,
                 today=pd.Timestamp("2026-06-01"), spy_close=5100.0)

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


def test_publish_once_same_day_rerun_excludes_stale_point(tmp_path):
    # A prior run today already wrote an equity point for `today`. On re-run it must
    # NOT be treated as the "prior" NAV (idempotency — the systemd timer fires repeatedly).
    asof = "2026-05-29"
    today = pd.Timestamp("2026-06-01")
    _write_weights(tmp_path / "weights", asof)
    broker = DryRunBroker(positions={"AAA": 60.0, "BBB": 40.0}, nav=10_000.0,
                          quotes={"AAA": (99.5, 100.5), "BBB": (99.5, 100.5)})
    store = _RecordingStore(equity=[{"date": "2026-06-01", "nav": 9_500.0, "spy_close": 5050.0}])

    publish_once(broker, store, weights_dir=tmp_path / "weights",
                 orders_dir=tmp_path / "orders", asof=asof, today=today, spy_close=5100.0)

    # No point STRICTLY before today → no prior NAV → day P&L is None (not 10000-9500).
    assert store.snapshot["day_pnl"] is None
    assert store.snapshot["day_pnl_pct"] is None


def test_publish_once_inception_day_empty_curve(tmp_path):
    asof = "2026-05-29"
    today = pd.Timestamp("2026-06-01")
    _write_weights(tmp_path / "weights", asof)
    broker = DryRunBroker(positions={"AAA": 60.0}, nav=10_000.0, quotes={"AAA": (99.5, 100.5)})
    store = _RecordingStore(equity=[])  # nothing yet — first ever publish

    publish_once(broker, store, weights_dir=tmp_path / "weights",
                 orders_dir=tmp_path / "orders", asof=asof, today=today, spy_close=5100.0)

    assert store.snapshot["day_pnl"] is None          # no prior point
    assert store.snapshot["total_return"] == 0.0      # nav == inception_nav
    assert store.snapshot["spy_return"] == 0.0        # spy_close == inception_spy
    assert store.snapshot["risk"]["sharpe"] is None   # single-point series → no risk stats


def test_is_market_hours_naive_input_is_localized_to_et():
    # tz-naive Monday midday is treated as America/New_York → within hours.
    assert is_market_hours(pd.Timestamp("2026-06-01 12:00")) is True


import math as _math


def test_publish_once_computes_turnover_vs_prior_friday(tmp_path):
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

    publish_once(broker, store, weights_dir=wdir, orders_dir=tmp_path / "orders",
                 asof="2026-05-29", today=pd.Timestamp("2026-06-01"), spy_close=5100.0)

    t = store.snapshot["turnover"]
    assert t["added"] == ["CCC"]
    assert t["dropped"] == ["BBB"]
    # 0.5*(|0.4-0.5| + |0-0.5| + |0.6-0|) = 0.6
    assert _math.isclose(t["turnover_frac"], 0.6)


def test_publish_once_turnover_none_without_prior_week(tmp_path):
    asof = "2026-05-29"
    _write_weights(tmp_path / "weights", asof)  # only this week's file exists
    broker = DryRunBroker(positions={"AAA": 60.0, "BBB": 40.0}, nav=10_000.0,
                          quotes={"AAA": (99.5, 100.5), "BBB": (99.5, 100.5)})
    store = _RecordingStore()
    publish_once(broker, store, weights_dir=tmp_path / "weights",
                 orders_dir=tmp_path / "orders", asof=asof,
                 today=pd.Timestamp("2026-06-01"), spy_close=5100.0)
    assert store.snapshot["turnover"] is None
