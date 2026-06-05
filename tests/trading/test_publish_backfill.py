"""Backfill reads existing audit files into weekly_portfolio + executions."""
from __future__ import annotations

import json
from pathlib import Path

from trading.publish.backfill import backfill


class _RecordingStore:
    def __init__(self):
        self.weekly = []
        self.executions = []

    def insert_weekly_portfolio(self, asof_friday, rows):
        self.weekly.append((asof_friday, rows))

    def insert_executions(self, asof, rows):
        self.executions.append((asof, rows))


def test_backfill_loads_all_weeks_and_orders(tmp_path):
    wdir = tmp_path / "weights"
    odir = tmp_path / "orders"
    wdir.mkdir()
    odir.mkdir()
    for asof in ("2026-05-22", "2026-05-29"):
        (wdir / f"{asof}.json").write_text(json.dumps(
            {"asof": asof, "k_probs": {"10": 1.0}, "weights": {"AAA": 1.0}}))
    (odir / "2026-05-29.json").write_text(json.dumps({
        "fills": [{"ticker": "AAA", "side": "BUY"}],
        "ladder_stages": [{"ticker": "AAA", "qty_filled": 5.0,
                           "realized_price": 100.0, "midpoint_at_fill": 100.0}],
    }))

    store = _RecordingStore()
    n = backfill(store, weights_dir=wdir, orders_dir=odir)

    assert [w[0] for w in store.weekly] == ["2026-05-22", "2026-05-29"]
    assert store.executions[0][0] == "2026-05-29"
    assert store.executions[0][1][0]["asof"] == "2026-05-29"
    assert n == {"weeks": 2, "order_files": 1}
