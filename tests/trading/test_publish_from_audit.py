"""publish_from_audit against fixture audit files + injected price frame + fake store."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from trading.publish.publish import publish_from_audit


class _RecordingStore:
    def __init__(self):
        self.snapshot = self.holdings = self.weekly = self.executions = None
        self.equity_curve = None

    def replace_equity_curve(self, rows): self.equity_curve = rows
    def upsert_snapshot(self, row): self.snapshot = row
    def replace_holdings(self, rows): self.holdings = rows
    def insert_weekly_portfolio(self, asof_friday, rows):
        self.weekly = {"asof_friday": asof_friday, "rows": rows}
    def insert_executions(self, asof, rows):
        self.executions = {"asof": asof, "rows": rows}


def _setup(tmp_path, asof="2026-06-05"):
    wdir = tmp_path / "weights"; wdir.mkdir(parents=True)
    (wdir / f"{asof}.json").write_text(json.dumps(
        {"asof": asof, "k_probs": {"10": 1.0}, "weights": {"AAA": 1.0}}))
    odir = tmp_path / "orders"; odir.mkdir(parents=True)
    (odir / f"{asof}.json").write_text(json.dumps({
        "asof": asof, "mode": "live", "nav": 1000.0, "first_build": True,
        "post_positions": {"AAA": 5.0},
        "fills": [{"ticker": "AAA", "side": "BUY", "quantity": 5.0, "avg_price": 180.0}],
        "ladder_stages": [{"ticker": "AAA", "qty_filled": 5.0,
                           "realized_price": 180.0, "midpoint_at_fill": 179.0}],
    }))
    return wdir, odir, asof


def test_publish_from_audit_writes_curve_holdings_snapshot(tmp_path):
    wdir, odir, asof = _setup(tmp_path)
    idx = pd.to_datetime(["2026-06-05", "2026-06-08"]).normalize()
    frame = pd.DataFrame({"AAA": [180.0, 200.0], "SPY": [500.0, 510.0]}, index=idx)

    def price_fetch(tickers, start, end):
        return frame.reindex(columns=sorted(set(tickers))).ffill()

    store = _RecordingStore()
    summary = publish_from_audit(
        store, weights_dir=wdir, orders_dir=odir, asof=asof,
        today=pd.Timestamp("2026-06-08"), price_fetch=price_fetch,
    )

    assert summary["n_holdings"] == 1
    assert [p["date"] for p in store.equity_curve] == ["2026-06-05", "2026-06-08"]
    assert store.equity_curve[-1]["nav"] == 1100.0      # 100 cash + 5*200
    assert store.snapshot["nav"] == 1100.0
    assert store.snapshot["n_positions"] == 1
    assert store.snapshot["total_return"] == pytest.approx(0.1)  # 1100/1000 - 1
    assert store.holdings[0]["ticker"] == "AAA"
    assert store.holdings[0]["price"] == 200.0          # latest close
    assert all(h["asof"] == "2026-06-08" for h in store.holdings)
    assert store.weekly["asof_friday"] == asof
    assert store.executions["asof"] == asof             # orders file present


def test_publish_from_audit_no_broker_import(tmp_path, monkeypatch):
    # Guard: the audit path must never touch the IBKR broker.
    wdir, odir, asof = _setup(tmp_path)
    idx = pd.to_datetime(["2026-06-05"]).normalize()
    frame = pd.DataFrame({"AAA": [180.0], "SPY": [500.0]}, index=idx)
    import trading.broker.ibkr as ibkr

    def _boom(*a, **k):
        raise AssertionError("publish_from_audit must not construct IBKRBroker")

    monkeypatch.setattr(ibkr, "IBKRBroker", _boom)
    store = _RecordingStore()
    publish_from_audit(store, weights_dir=wdir, orders_dir=odir, asof=asof,
                       today=pd.Timestamp("2026-06-05"),
                       price_fetch=lambda t, s, e: frame.reindex(columns=sorted(set(t))).ffill())
    assert store.snapshot["nav"] == 1000.0              # 100 cash + 5*180
