"""Pure reconstruction of holdings/cash/NAV from the order audit."""
from __future__ import annotations

import json

import pandas as pd
import pytest

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


def _curve_fixture(tmp_path):
    od = tmp_path / "orders"
    _write(od, "2026-06-05", {
        "first_build": True, "nav": 1000.0, "post_positions": {"AAA": 5.0},
        "fills": [{"ticker": "AAA", "side": "BUY", "quantity": 5.0, "avg_price": 180.0}],
    })
    return reconstruct.load_history(od)


def test_reconstruct_curve_single_rebalance(tmp_path):
    hist = _curve_fixture(tmp_path)          # cash residual = 100, holds 5 AAA
    idx = pd.to_datetime(["2026-06-05", "2026-06-08"]).normalize()
    closes = pd.DataFrame({"AAA": [180.0, 200.0], "SPY": [500.0, 510.0]}, index=idx)
    curve = reconstruct.reconstruct_curve(hist, closes, closes["SPY"])
    assert [p["date"] for p in curve] == ["2026-06-05", "2026-06-08"]
    # day 1: 100 + 5*180 = 1000 ; day 2: 100 + 5*200 = 1100
    assert curve[0]["nav"] == 1000.0
    assert curve[1]["nav"] == 1100.0
    assert curve[0]["spy_close"] == 500.0


def test_reconstruct_curve_starts_at_inception(tmp_path):
    hist = _curve_fixture(tmp_path)
    idx = pd.to_datetime(["2026-06-01", "2026-06-05"]).normalize()  # one day pre-inception
    closes = pd.DataFrame({"AAA": [170.0, 180.0], "SPY": [490.0, 500.0]}, index=idx)
    curve = reconstruct.reconstruct_curve(hist, closes, closes["SPY"])
    assert [p["date"] for p in curve] == ["2026-06-05"]  # pre-inception day dropped


def test_reconstruct_curve_skips_missing_close(tmp_path):
    hist = _curve_fixture(tmp_path)  # first_build nav 1000, 5 AAA, cash residual 100
    idx = pd.to_datetime(["2026-06-05", "2026-06-08"]).normalize()
    closes = pd.DataFrame({"AAA": [180.0, float("nan")], "SPY": [500.0, 510.0]}, index=idx)
    curve = reconstruct.reconstruct_curve(hist, closes, closes["SPY"])
    # 06-08 AAA price missing -> contributes 0 market value -> nav == cash residual (100)
    assert curve[1]["nav"] == 100.0


def test_reconstruct_curve_empty_history():
    assert reconstruct.reconstruct_curve([], pd.DataFrame(), pd.Series(dtype=float)) == []


def test_reconstruct_curve_ticker_absent_from_columns(tmp_path):
    hist = _curve_fixture(tmp_path)  # holds 5 AAA, cash residual 100
    idx = pd.to_datetime(["2026-06-05", "2026-06-08"]).normalize()
    closes = pd.DataFrame({"SPY": [500.0, 510.0]}, index=idx)  # no AAA column at all
    curve = reconstruct.reconstruct_curve(hist, closes, closes["SPY"])
    # 06-08 AAA absent from columns -> contributes 0 -> nav == cash residual
    assert curve[1]["nav"] == 100.0


def test_reconstruct_curve_no_flows_arg_rows_carry_zero_flow(tmp_path):
    hist = _curve_fixture(tmp_path)
    idx = pd.to_datetime(["2026-06-05", "2026-06-08"]).normalize()
    closes = pd.DataFrame({"AAA": [180.0, 200.0], "SPY": [500.0, 510.0]}, index=idx)
    curve = reconstruct.reconstruct_curve(hist, closes, closes["SPY"])
    assert all(p["flow"] == 0.0 for p in curve)


def test_reconstruct_curve_flows_add_to_cash_and_stamp_rows(tmp_path):
    hist = _curve_fixture(tmp_path)  # cash residual 100, holds 5 AAA
    idx = pd.to_datetime(["2026-06-05", "2026-06-08", "2026-06-09"]).normalize()
    closes = pd.DataFrame({"AAA": [180.0, 200.0, 200.0],
                           "SPY": [500.0, 510.0, 512.0]}, index=idx)
    flows = {"2026-06-08": 750.0}
    curve = reconstruct.reconstruct_curve(hist, closes, closes["SPY"], flows=flows)
    # 06-05 anchored to first_build capital (no flow that day).
    assert curve[0]["nav"] == 1000.0 and curve[0]["flow"] == 0.0
    # 06-08: 100 cash + 750 deposit + 5*200 = 1850; row carries the flow.
    assert curve[1]["nav"] == 1850.0 and curve[1]["flow"] == 750.0
    # 06-09: deposit stays in cash until deployed.
    assert curve[2]["nav"] == 1850.0 and curve[2]["flow"] == 0.0


def test_reconstruct_curve_inception_uses_starting_capital_not_close(tmp_path):
    # Bought 5 AAA at avg 190 (notional 950) out of 1000 -> cash residual 50.
    # The Friday close (170) is BELOW cost, so a mark-to-close baseline would invent
    # a loss-then-gain; the inception point must be the starting capital instead.
    od = tmp_path / "orders"
    _write(od, "2026-06-05", {
        "first_build": True, "nav": 1000.0, "post_positions": {"AAA": 5.0},
        "fills": [{"ticker": "AAA", "side": "BUY", "quantity": 5.0, "avg_price": 190.0}],
    })
    hist = reconstruct.load_history(od)
    idx = pd.to_datetime(["2026-06-05", "2026-06-08"]).normalize()
    closes = pd.DataFrame({"AAA": [170.0, 200.0], "SPY": [500.0, 510.0]}, index=idx)
    curve = reconstruct.reconstruct_curve(hist, closes, closes["SPY"])
    assert curve[0]["nav"] == 1000.0          # starting capital, NOT 50 + 5*170 = 900
    assert curve[1]["nav"] == 1050.0          # 06-08 marks to close: 50 + 5*200


def test_reconstruct_curve_switches_holdings_on_rebalance(tmp_path):
    od = tmp_path / "orders"
    _write(od, "2026-06-05", {
        "first_build": True, "nav": 1000.0, "post_positions": {"AAA": 5.0},
        "fills": [{"ticker": "AAA", "side": "BUY", "quantity": 5.0, "avg_price": 180.0}],
    })  # cash after = 1000 - 5*180 = 100
    _write(od, "2026-06-12", {
        "first_build": False, "nav": 0.0, "post_positions": {"AAA": 2.0, "BBB": 4.0},
        "fills": [
            {"ticker": "AAA", "side": "SELL", "quantity": 3.0, "avg_price": 200.0},
            {"ticker": "BBB", "side": "BUY", "quantity": 4.0, "avg_price": 100.0},
        ],
    })  # cash after both = 100 + 3*200 - 4*100 = 300
    hist = reconstruct.load_history(od)
    idx = pd.to_datetime(["2026-06-08", "2026-06-12"]).normalize()
    closes = pd.DataFrame(
        {"AAA": [200.0, 210.0], "BBB": [100.0, 110.0], "SPY": [500.0, 510.0]}, index=idx)
    curve = reconstruct.reconstruct_curve(hist, closes, closes["SPY"])
    # 06-08: only first rebalance applies -> 5 AAA, cash 100 -> 100 + 5*200 = 1100
    # 06-12: both apply -> 2 AAA + 4 BBB, cash 300 -> 300 + 2*210 + 4*110 = 1160
    assert curve[0]["nav"] == 1100.0
    assert curve[1]["nav"] == 1160.0
