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


def test_compute_holdings_attaches_company_name_and_sector():
    meta = {"NVDA": {"company_name": "NVIDIA CORP", "sector": "Technology"}}
    rows = compute_holdings({"NVDA": 10.0}, {"NVDA": 100.0}, {"NVDA": 0.1},
                            nav=1000.0, metadata=meta)
    assert rows[0]["company_name"] == "NVIDIA CORP"
    assert rows[0]["sector"] == "Technology"


def test_compute_holdings_metadata_absent_is_none_not_missing():
    # keys are always present (None when no metadata) so the schema stays uniform
    rows = compute_holdings({"AAA": 1.0}, {"AAA": 10.0}, {}, nav=100.0)
    assert rows[0]["company_name"] is None
    assert rows[0]["sector"] is None


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

# ---------------------------------------------------------------------------
# compute_week_to_date — trailing-trading-week portfolio vs SPY
# ---------------------------------------------------------------------------

from trading.publish.metrics import compute_week_to_date
import datetime as dt


_CURVE = [
    {"date": "2026-06-04", "nav": 98_000.0, "spy_close": 5000.0},  # Thu
    {"date": "2026-06-05", "nav": 100_000.0, "spy_close": 5050.0},  # Fri (baseline)
    {"date": "2026-06-08", "nav": 101_000.0, "spy_close": 5060.0},  # Mon
    {"date": "2026-06-09", "nav": 102_000.0, "spy_close": 5040.0},  # Tue
]


def test_week_to_date_baselines_on_prior_friday():
    # Wednesday 2026-06-10: the week covers Mon-Wed, so the baseline is the
    # prior Friday's close — Monday's move must be included.
    out = compute_week_to_date(_CURVE, dt.date(2026, 6, 10), nav_now=103_000.0, spy_now=5101.0)
    assert out["baseline_date"] == "2026-06-05"
    assert math.isclose(out["portfolio_return"], 0.03)
    assert math.isclose(out["spy_return"], 5101.0 / 5050.0 - 1.0)
    assert math.isclose(out["excess_return"],
                        out["portfolio_return"] - out["spy_return"])


def test_week_to_date_on_monday_uses_prior_friday():
    out = compute_week_to_date(_CURVE, dt.date(2026, 6, 8), nav_now=101_000.0, spy_now=5060.0)
    assert out["baseline_date"] == "2026-06-05"
    assert math.isclose(out["portfolio_return"], 0.01)


def test_week_to_date_no_baseline_returns_none():
    # Curve starts this week → nothing strictly before Monday.
    curve = [{"date": "2026-06-08", "nav": 100_000.0, "spy_close": 5000.0}]
    assert compute_week_to_date(curve, dt.date(2026, 6, 10), 101_000.0, 5050.0) is None


def test_week_to_date_missing_spy_returns_none_spy_fields():
    curve = [{"date": "2026-06-05", "nav": 100_000.0, "spy_close": None}]
    out = compute_week_to_date(curve, dt.date(2026, 6, 10), 102_000.0, 5050.0)
    assert math.isclose(out["portfolio_return"], 0.02)
    assert out["spy_return"] is None
    assert out["excess_return"] is None


def test_week_to_date_empty_curve_returns_none():
    assert compute_week_to_date([], dt.date(2026, 6, 10), 100_000.0, 5000.0) is None


def test_week_to_date_mid_week_deposit_contributes_zero():
    # $75k landed Wednesday: nav_now jumped but the week's return must chain
    # through (nav - flow), not compare raw NAVs across the deposit.
    curve = [
        {"date": "2026-06-05", "nav": 100_000.0, "spy_close": 5000.0},        # Fri baseline
        {"date": "2026-06-08", "nav": 101_000.0, "spy_close": 5060.0},        # Mon
        {"date": "2026-06-09", "nav": 102_000.0, "spy_close": 5040.0},        # Tue
    ]
    out = compute_week_to_date(curve, dt.date(2026, 6, 10),
                               nav_now=177_000.0, spy_now=5100.0, flow_today=75_000.0)
    # (101/100) * (102/101) * ((177-75)/102) = 1.02
    assert math.isclose(out["portfolio_return"], 0.02)


def test_week_to_date_deposit_on_earlier_curve_row_is_stripped():
    # The deposit landed Monday and is recorded on that row's flow column.
    curve = [
        {"date": "2026-06-05", "nav": 100_000.0, "spy_close": 5000.0},
        {"date": "2026-06-08", "nav": 176_000.0, "spy_close": 5060.0, "flow": 75_000.0},
    ]
    out = compute_week_to_date(curve, dt.date(2026, 6, 10),
                               nav_now=177_760.0, spy_now=5100.0)
    # Mon: (176000-75000)/100000 = 1.01; Wed: 177760/176000 = 1.01 → 2.01% total
    assert math.isclose(out["portfolio_return"], 1.01 * 1.01 - 1.0)


# ---------------------------------------------------------------------------
# twr_index / detect_flow / holdings_day_pnl — flow-adjusted return plumbing
# ---------------------------------------------------------------------------

from trading.publish.metrics import detect_flow, holdings_day_pnl, twr_index


def test_twr_index_no_flows_matches_nav_ratio():
    curve = [{"nav": 100.0}, {"nav": 110.0}, {"nav": 99.0}]
    idx = twr_index(curve)
    assert idx[0] == 1.0
    assert math.isclose(idx[1], 1.1)
    assert math.isclose(idx[2], 0.99)


def test_twr_index_deposit_day_contributes_zero_growth():
    # NAV jumps 100k → 176k on a 75k deposit day with 1k of real P&L.
    curve = [{"nav": 100_000.0}, {"nav": 176_000.0, "flow": 75_000.0}]
    idx = twr_index(curve)
    assert math.isclose(idx[1], 1.01)


def test_twr_index_withdrawal_is_added_back():
    # 10k withdrawn, market flat: no fake loss.
    curve = [{"nav": 100_000.0}, {"nav": 90_000.0, "flow": -10_000.0}]
    assert math.isclose(twr_index(curve)[1], 1.0)


def test_twr_index_carries_flat_across_non_positive_nav():
    curve = [{"nav": 100.0}, {"nav": 0.0}, {"nav": 50.0}]
    idx = twr_index(curve)
    assert idx[1] == 0.0          # 0/100 — measurable, a total loss that day
    assert idx[2] == 0.0          # prior nav 0 → unmeasurable → flat carry


def test_twr_index_none_flow_treated_as_zero():
    curve = [{"nav": 100.0}, {"nav": 105.0, "flow": None}]
    assert math.isclose(twr_index(curve)[1], 1.05)


def test_detect_flow_deposit_above_threshold():
    # ΔNAV +75,600 with only +358 of day P&L → 75,242 implied deposit.
    assert math.isclose(
        detect_flow(176_435.05, 100_835.0, 357.86), 176_435.05 - 100_835.0 - 357.86
    )


def test_detect_flow_small_residual_is_yield_not_flow():
    # $400 of ΔNAV unexplained by P&L (dividends/interest) stays a return.
    assert detect_flow(100_500.0, 100_000.0, 100.0) == 0.0


def test_detect_flow_threshold_scales_with_nav():
    # 0.5% of a $1M book is $5,000 — a $3k residual is below it.
    assert detect_flow(1_003_000.0, 1_000_000.0, 0.0, min_abs=1000.0) == 0.0


def test_detect_flow_unavailable_inputs_return_zero():
    assert detect_flow(100_000.0, None, 500.0) == 0.0
    assert detect_flow(100_000.0, 99_000.0, None) == 0.0


def test_detect_flow_withdrawal_is_negative():
    assert math.isclose(detect_flow(90_000.0, 100_000.0, 0.0), -10_000.0)


def test_holdings_day_pnl_sums_present_values():
    portfolio = [{"daily_pnl": 10.0}, {"daily_pnl": -2.5}, {"daily_pnl": None}]
    assert math.isclose(holdings_day_pnl(portfolio), 7.5)


def test_holdings_day_pnl_none_when_no_figures():
    assert holdings_day_pnl([{"daily_pnl": None}, {}]) is None


# ---------------------------------------------------------------------------
# compute_holdings_live — broker portfolio rows (IB-mobile style P&L)
# ---------------------------------------------------------------------------

from trading.publish.metrics import compute_holdings_live


def test_holdings_live_maps_portfolio_rows():
    portfolio = [
        {"ticker": "AAA", "position": 100.0, "market_price": 10.0, "market_value": 1000.0,
         "avg_cost": 9.0, "unrealized_pnl": 100.0, "daily_pnl": 12.5},
        {"ticker": "BBB", "position": 50.0, "market_price": 40.0, "market_value": 2000.0,
         "avg_cost": 41.0, "unrealized_pnl": -50.0, "daily_pnl": None},
    ]
    rows = compute_holdings_live(portfolio, {"AAA": 0.30, "BBB": 0.70}, nav=3000.0)
    # sorted by actual weight desc: BBB first
    assert [r["ticker"] for r in rows] == ["BBB", "AAA"]
    assert math.isclose(rows[0]["weight_actual"], 2000.0 / 3000.0)
    assert math.isclose(rows[0]["weight_target"], 0.70)
    assert rows[0]["daily_pnl"] is None          # absent from broker → None, key present
    assert math.isclose(rows[1]["shares"], 100.0)
    assert math.isclose(rows[1]["avg_cost"], 9.0)
    assert math.isclose(rows[1]["unrealized_pnl"], 100.0)
    assert math.isclose(rows[1]["daily_pnl"], 12.5)


def test_holdings_live_skips_zero_positions_and_attaches_metadata():
    portfolio = [
        {"ticker": "AAA", "position": 0.0, "market_price": 10.0, "market_value": 0.0,
         "avg_cost": None, "unrealized_pnl": None, "daily_pnl": None},
        {"ticker": "NVDA", "position": 10.0, "market_price": 100.0, "market_value": 1000.0,
         "avg_cost": 90.0, "unrealized_pnl": 100.0, "daily_pnl": 5.0},
    ]
    meta = {"NVDA": {"company_name": "NVIDIA CORP", "sector": "Technology"}}
    rows = compute_holdings_live(portfolio, {}, nav=1000.0, metadata=meta)
    assert len(rows) == 1
    assert rows[0]["ticker"] == "NVDA"
    assert rows[0]["company_name"] == "NVIDIA CORP"
    assert rows[0]["sector"] == "Technology"
    assert rows[0]["weight_target"] == 0.0
