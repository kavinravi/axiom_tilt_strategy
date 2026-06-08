"""Pure reconstruction of holdings/cash/NAV from the order audit."""
from __future__ import annotations

import json

import pandas as pd

from trading.publish import reconstruct


def _write(orders_dir, asof, record):
    orders_dir.mkdir(parents=True, exist_ok=True)
    (orders_dir / f"{asof}.json").write_text(json.dumps({"asof": asof, **record}))


def test_load_history_sorted_by_asof(tmp_path):
    od = tmp_path / "orders"
    _write(od, "2026-06-12", {"post_positions": {"AAA": 1.0}})
    _write(od, "2026-06-05", {"post_positions": {"AAA": 2.0}})
    hist = reconstruct.load_history(od)
    assert [r["asof"] for r in hist] == ["2026-06-05", "2026-06-12"]


def test_current_holdings_uses_latest_post_positions(tmp_path):
    od = tmp_path / "orders"
    _write(od, "2026-06-05", {"post_positions": {"AAA": 10.0, "BBB": 5.0}})
    _write(od, "2026-06-12", {"post_positions": {"AAA": 8.0, "CCC": 3.0}})
    hist = reconstruct.load_history(od)
    assert reconstruct.current_holdings(hist) == {"AAA": 8.0, "CCC": 3.0}


def test_current_holdings_drops_zero_shares(tmp_path):
    od = tmp_path / "orders"
    _write(od, "2026-06-05", {"post_positions": {"AAA": 10.0, "BBB": 0.0}})
    hist = reconstruct.load_history(od)
    assert reconstruct.current_holdings(hist) == {"AAA": 10.0}


def test_current_holdings_empty_history():
    assert reconstruct.current_holdings([]) == {}
