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


def test_cash_after_first_build_residual(tmp_path):
    od = tmp_path / "orders"
    _write(od, "2026-06-05", {
        "first_build": True, "nav": 1000.0,
        "post_positions": {"AAA": 5.0},
        "fills": [{"ticker": "AAA", "side": "BUY", "quantity": 5.0, "avg_price": 180.0}],
    })
    hist = reconstruct.load_history(od)
    # 1000 - 5*180 = 100
    assert reconstruct.cash_after(hist) == 100.0


def test_cash_after_accumulates_sells_and_buys(tmp_path):
    od = tmp_path / "orders"
    _write(od, "2026-06-05", {
        "first_build": True, "nav": 1000.0, "post_positions": {"AAA": 5.0},
        "fills": [{"ticker": "AAA", "side": "BUY", "quantity": 5.0, "avg_price": 180.0}],
    })
    _write(od, "2026-06-12", {
        "first_build": False, "nav": 0.0, "post_positions": {"AAA": 3.0},
        "fills": [{"ticker": "AAA", "side": "SELL", "quantity": 2.0, "avg_price": 200.0}],
    })
    hist = reconstruct.load_history(od)
    # 100 + (sell 2*200 adds cash) = 500
    assert reconstruct.cash_after(hist) == 500.0


def test_cash_after_no_first_build_raises(tmp_path):
    od = tmp_path / "orders"
    _write(od, "2026-06-05", {"first_build": False, "nav": 1.0,
                              "post_positions": {}, "fills": []})
    hist = reconstruct.load_history(od)
    import pytest
    with pytest.raises(ValueError):
        reconstruct.cash_after(hist)


def test_inception_date_is_first_build_asof(tmp_path):
    od = tmp_path / "orders"
    _write(od, "2026-06-05", {"first_build": True, "nav": 1000.0,
                              "post_positions": {"AAA": 5.0}, "fills": []})
    _write(od, "2026-06-12", {"first_build": False, "nav": 0.0,
                              "post_positions": {"AAA": 5.0}, "fills": []})
    hist = reconstruct.load_history(od)
    assert reconstruct.inception_date(hist) == pd.Timestamp("2026-06-05")
