"""Tests for trading/publish/metrics.py — pure functions, no network."""
from __future__ import annotations

import math

from trading.publish.metrics import compute_holdings


def test_compute_holdings_sorts_by_actual_weight_and_skips_zero_shares():
    positions = {"AAA": 100.0, "BBB": 50.0, "CCC": 0.0}
    prices = {"AAA": 10.0, "BBB": 40.0, "CCC": 99.0}
    target = {"AAA": 0.30, "BBB": 0.70}
    nav = 3000.0

    rows = compute_holdings(positions, prices, target, nav)

    # CCC dropped (0 shares); BBB (mv 2000) before AAA (mv 1000)
    assert [r["ticker"] for r in rows] == ["BBB", "AAA"]
    assert math.isclose(rows[0]["market_value"], 2000.0)
    assert math.isclose(rows[0]["weight_actual"], 2000.0 / 3000.0)
    assert math.isclose(rows[0]["weight_target"], 0.70)
    assert math.isclose(rows[1]["weight_actual"], 1000.0 / 3000.0)


def test_compute_holdings_missing_price_is_zero_value():
    rows = compute_holdings({"AAA": 10.0}, {}, {}, nav=1000.0)
    assert rows[0]["price"] == 0.0
    assert rows[0]["market_value"] == 0.0
    assert rows[0]["weight_target"] == 0.0


from trading.publish.metrics import compute_day_pnl, pct_change


def test_compute_day_pnl_normal():
    pnl, pct = compute_day_pnl(nav=101_000.0, prev_nav=100_000.0)
    assert math.isclose(pnl, 1000.0)
    assert math.isclose(pct, 0.01)


def test_compute_day_pnl_no_prior_returns_none():
    assert compute_day_pnl(100_000.0, None) == (None, None)
    assert compute_day_pnl(100_000.0, 0.0) == (None, None)


def test_pct_change():
    assert math.isclose(pct_change(110.0, 100.0), 0.10)
    assert pct_change(110.0, None) is None
    assert pct_change(110.0, 0.0) is None


from trading.publish.metrics import compute_risk


def test_compute_risk_too_short_is_all_none():
    out = compute_risk([100.0])
    assert out == {"current_drawdown": None, "max_drawdown": None,
                   "sharpe": None, "ann_vol": None}


def test_compute_risk_drawdown():
    # peak 120 then down to 90 → max dd = 90/120 - 1 = -0.25; current dd vs all-time peak 120
    navs = [100.0, 120.0, 90.0, 108.0]
    out = compute_risk(navs)
    assert math.isclose(out["max_drawdown"], 90.0 / 120.0 - 1.0)
    assert math.isclose(out["current_drawdown"], 108.0 / 120.0 - 1.0)
    assert out["ann_vol"] is not None and out["ann_vol"] > 0
    assert out["sharpe"] is not None


from trading.publish.metrics import compute_turnover, compute_execution_quality


def test_compute_turnover():
    last = {"AAA": 0.5, "BBB": 0.5}
    this = {"AAA": 0.4, "CCC": 0.6}
    out = compute_turnover(this, last)
    assert out["added"] == ["CCC"]
    assert out["dropped"] == ["BBB"]
    # 0.5*(|0.4-0.5| + |0-0.5| + |0.6-0|) = 0.5*(0.1+0.5+0.6) = 0.6
    assert math.isclose(out["turnover_frac"], 0.6)


def test_compute_execution_quality_buy_above_mid_is_positive_slippage():
    audit = {
        "fills": [{"ticker": "AAA", "side": "BUY"}],
        "ladder_stages": [
            {"ticker": "AAA", "qty_filled": 100.0, "realized_price": 101.0,
             "midpoint_at_fill": 100.0},
            {"ticker": "AAA", "qty_filled": 0.0, "realized_price": None,
             "midpoint_at_fill": None},
        ],
    }
    rows = compute_execution_quality(audit)
    assert len(rows) == 1
    r = rows[0]
    assert r["ticker"] == "AAA" and r["side"] == "BUY"
    assert math.isclose(r["realized_price"], 101.0)
    assert math.isclose(r["midpoint"], 100.0)
    # BUY paid 1.0 above a 100.0 mid → +100 bps cost
    assert math.isclose(r["slippage_bps"], 100.0)


def test_compute_execution_quality_sell_below_mid_is_positive_slippage():
    # A SELL filled BELOW the midpoint is a cost → positive slippage_bps.
    audit = {
        "fills": [{"ticker": "BBB", "side": "SELL"}],
        "ladder_stages": [
            {"ticker": "BBB", "qty_filled": 50.0, "realized_price": 99.0,
             "midpoint_at_fill": 100.0},
        ],
    }
    rows = compute_execution_quality(audit)
    assert rows[0]["side"] == "SELL"
    # sold 1.0 below a 100 mid → +100 bps cost (sign inverted vs BUY)
    assert math.isclose(rows[0]["slippage_bps"], 100.0)


def test_compute_execution_quality_multi_stage_qty_weighted():
    audit = {
        "fills": [{"ticker": "AAA", "side": "BUY"}],
        "ladder_stages": [
            {"ticker": "AAA", "qty_filled": 100.0, "realized_price": 100.0,
             "midpoint_at_fill": 100.0},
            {"ticker": "AAA", "qty_filled": 300.0, "realized_price": 102.0,
             "midpoint_at_fill": 101.0},
        ],
    }
    rows = compute_execution_quality(audit)
    assert len(rows) == 1
    r = rows[0]
    assert math.isclose(r["qty"], 400.0)
    # qty-weighted realized = (100*100 + 300*102)/400 = 101.5
    assert math.isclose(r["realized_price"], 101.5)
    # qty-weighted mid = (100*100 + 300*101)/400 = 100.75
    assert math.isclose(r["midpoint"], 100.75)


def test_compute_execution_quality_sorted_by_ticker():
    audit = {
        "fills": [{"ticker": "ZZZ", "side": "BUY"}, {"ticker": "AAA", "side": "BUY"}],
        "ladder_stages": [
            {"ticker": "ZZZ", "qty_filled": 1.0, "realized_price": 100.0, "midpoint_at_fill": 100.0},
            {"ticker": "AAA", "qty_filled": 1.0, "realized_price": 100.0, "midpoint_at_fill": 100.0},
        ],
    }
    rows = compute_execution_quality(audit)
    assert [r["ticker"] for r in rows] == ["AAA", "ZZZ"]


def test_pct_change_none_now_is_none():
    assert pct_change(None, 100.0) is None
