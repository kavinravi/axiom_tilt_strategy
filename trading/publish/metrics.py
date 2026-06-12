"""Pure metric functions for the dashboard publisher.

No network, no Supabase, no broker — every function takes plain dicts/lists and
returns plain dicts/lists, so they are trivially unit-testable.
"""
from __future__ import annotations

import datetime as dt
import math


def compute_holdings(
    positions: dict[str, float],
    prices: dict[str, float],
    target_weights: dict[str, float],
    nav: float,
    metadata: dict[str, dict] | None = None,
) -> list[dict]:
    """Build the per-holding table for currently-held names, sorted by actual weight.

    Skips zero-share positions. weight_actual = shares*price / nav. ``metadata``
    maps ticker -> {"company_name", "sector"}; absent names get None for both so
    the row schema stays uniform.
    """
    metadata = metadata or {}
    rows: list[dict] = []
    for ticker, shares in positions.items():
        shares = float(shares)
        if shares == 0.0:
            continue
        price = float(prices.get(ticker, 0.0))
        market_value = shares * price
        meta = metadata.get(ticker, {})
        rows.append(
            {
                "ticker": ticker,
                "company_name": meta.get("company_name"),
                "sector": meta.get("sector"),
                "shares": shares,
                "price": price,
                "market_value": market_value,
                "weight_actual": (market_value / nav) if nav > 0 else 0.0,
                "weight_target": float(target_weights.get(ticker, 0.0)),
            }
        )
    rows.sort(key=lambda r: r["weight_actual"], reverse=True)
    return rows


def compute_holdings_live(
    portfolio: list[dict],
    target_weights: dict[str, float],
    nav: float,
    metadata: dict[str, dict] | None = None,
) -> list[dict]:
    """Per-holding table from broker portfolio rows (IB-mobile style).

    ``portfolio`` rows come from ``Broker.get_portfolio()``: ticker, position,
    market_price, market_value, avg_cost, unrealized_pnl, daily_pnl. Skips
    zero-share rows; sorts by actual weight. P&L fields pass through as-is
    (None when the broker had no figure) so the row schema stays uniform.
    """
    metadata = metadata or {}
    rows: list[dict] = []
    for p in portfolio:
        shares = float(p.get("position") or 0.0)
        if shares == 0.0:
            continue
        ticker = str(p["ticker"])
        market_value = float(p.get("market_value") or 0.0)
        meta = metadata.get(ticker, {})
        rows.append(
            {
                "ticker": ticker,
                "company_name": meta.get("company_name"),
                "sector": meta.get("sector"),
                "shares": shares,
                "price": float(p.get("market_price") or 0.0),
                "market_value": market_value,
                "weight_actual": (market_value / nav) if nav > 0 else 0.0,
                "weight_target": float(target_weights.get(ticker, 0.0)),
                "avg_cost": p.get("avg_cost"),
                "unrealized_pnl": p.get("unrealized_pnl"),
                "daily_pnl": p.get("daily_pnl"),
            }
        )
    rows.sort(key=lambda r: r["weight_actual"], reverse=True)
    return rows


def compute_week_to_date(
    curve: list[dict],
    today: dt.date,
    nav_now: float,
    spy_now: float | None,
) -> dict | None:
    """Trading-week-to-date: portfolio vs SPY since the prior week's last close.

    The baseline is the most recent equity point STRICTLY before this week's
    Monday, so Monday's move is always included: on Wednesday the comparison
    covers Mon-Wed; on Friday the full Mon-Fri week. Returns None until a
    baseline exists (first week of history). spy fields are None when either
    end of the SPY pair is missing.
    """
    monday = today - dt.timedelta(days=today.weekday())
    cutoff = monday.isoformat()
    baseline = None
    for p in curve:  # curve is chronological; keep the last qualifying point
        if str(p["date"]) < cutoff and p.get("nav") and float(p["nav"]) > 0:
            baseline = p
    if baseline is None:
        return None
    port = pct_change(nav_now, float(baseline["nav"]))
    base_spy = baseline.get("spy_close")
    spy = pct_change(spy_now, float(base_spy)) if (spy_now and base_spy) else None
    return {
        "baseline_date": str(baseline["date"]),
        "portfolio_return": port,
        "spy_return": spy,
        "excess_return": (port - spy) if (port is not None and spy is not None) else None,
    }


def pct_change(now: float | None, base: float | None) -> float | None:
    """Return now/base - 1, or None if base is missing/non-positive."""
    if now is None or base is None or base <= 0:
        return None
    return now / base - 1.0


def compute_day_pnl(nav: float, prev_nav: float | None) -> tuple[float | None, float | None]:
    """Portfolio-level P&L vs the prior NAV point. (None, None) when no prior."""
    if prev_nav is None or prev_nav <= 0:
        return None, None
    pnl = nav - prev_nav
    return pnl, pnl / prev_nav


_TRADING_DAYS = 252


def compute_risk(navs: list[float]) -> dict:
    """Drawdown / Sharpe / annualized vol from a chronological daily NAV series."""
    out: dict = {"current_drawdown": None, "max_drawdown": None,
                 "sharpe": None, "ann_vol": None}
    if len(navs) < 2:
        return out

    # Drawdowns
    peak = navs[0]
    max_dd = 0.0
    for v in navs:
        peak = max(peak, v)
        max_dd = min(max_dd, v / peak - 1.0)
    out["max_drawdown"] = max_dd
    out["current_drawdown"] = navs[-1] / max(navs) - 1.0

    # Daily simple returns
    rets = [navs[i] / navs[i - 1] - 1.0 for i in range(1, len(navs))]
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1) if len(rets) > 1 else 0.0
    sd = math.sqrt(var)
    out["ann_vol"] = sd * math.sqrt(_TRADING_DAYS)
    if sd > 0:
        out["sharpe"] = (mean / sd) * math.sqrt(_TRADING_DAYS)
    return out


def compute_turnover(this_weights: dict[str, float], last_weights: dict[str, float]) -> dict:
    """Names added/dropped and one-way turnover fraction (0.5 * sum|Δw|)."""
    this_set, last_set = set(this_weights), set(last_weights)
    tickers = this_set | last_set
    turnover = 0.5 * sum(
        abs(this_weights.get(t, 0.0) - last_weights.get(t, 0.0)) for t in tickers
    )
    return {
        "added": sorted(this_set - last_set),
        "dropped": sorted(last_set - this_set),
        "turnover_frac": turnover,
    }


def compute_execution_quality(orders_audit: dict) -> list[dict]:
    """Per-ticker realized fill vs NBBO midpoint, from a rebalance orders-audit dict.

    slippage_bps is signed so that POSITIVE = worse than midpoint (a cost): for a
    BUY, paying above the mid is positive; for a SELL, selling below the mid is positive.
    """
    side_by_ticker = {f["ticker"]: f.get("side", "BUY") for f in orders_audit.get("fills", [])}
    agg: dict[str, dict] = {}
    for s in orders_audit.get("ladder_stages", []):
        qty = float(s.get("qty_filled") or 0.0)
        rp = s.get("realized_price")
        mp = s.get("midpoint_at_fill")
        if qty <= 0.0 or rp is None or mp is None:
            continue
        d = agg.setdefault(s["ticker"], {"qty": 0.0, "rp_q": 0.0, "mp_q": 0.0})
        d["qty"] += qty
        d["rp_q"] += float(rp) * qty
        d["mp_q"] += float(mp) * qty

    rows: list[dict] = []
    for ticker, d in agg.items():
        realized = d["rp_q"] / d["qty"]
        mid = d["mp_q"] / d["qty"]
        side = side_by_ticker.get(ticker, "BUY")
        raw = (realized - mid) / mid if mid else 0.0
        slippage_bps = (raw if side == "BUY" else -raw) * 1e4
        rows.append(
            {
                "ticker": ticker,
                "side": side,
                "qty": d["qty"],
                "realized_price": realized,
                "midpoint": mid,
                "slippage_bps": slippage_bps,
            }
        )
    rows.sort(key=lambda r: r["ticker"])
    return rows
