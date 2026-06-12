"""Orchestrate one publish: broker account data (or audit files) -> metrics -> Supabase.

publish_live is the production path: NAV is the broker's own NetLiquidation and
per-holding P&L comes off the account-update channel (reqPnLSingle), so the
dashboard matches the IBKR app to the penny. SPY stays on yfinance — it is a
benchmark reference line, not a brokerage figure. main() runs it two ways:
`--intraday` (15-min timer; skips silently outside market hours or when the
Gateway is unreachable) and default EOD (16:30 timer; hard-fails into the
OnFailure alert). publish_from_audit is the legacy reconstruction from the order
audit + yfinance closes, kept for backfill/repair via `--from-audit`. All paths
are fully injectable so tests run against fakes with no network.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from trading.publish.flows import load_flows, save_flows
from trading.publish.metrics import (
    compute_day_pnl,
    compute_execution_quality,
    compute_holdings,
    compute_holdings_live,
    compute_risk,
    compute_turnover,
    compute_week_to_date,
    detect_flow,
    holdings_day_pnl,
    pct_change,
    twr_index,
)
from trading.publish.reconstruct import (
    current_holdings,
    inception_date,
    load_history,
    reconstruct_curve,
)

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _prev_nav(curve: list[dict], today: pd.Timestamp) -> float | None:
    """Most recent NAV strictly before `today` (the prior close)."""
    today_str = str(today.date())
    prior = [p for p in curve if str(p["date"]) < today_str]
    return float(prior[-1]["nav"]) if prior else None


def _prior_weights(weights_dir, asof: str) -> dict[str, float]:
    """Target weights from the most recent weights file strictly before `asof`.

    Used for week-over-week turnover. Returns {} when there is no earlier file.
    """
    prior = sorted(p for p in Path(weights_dir).glob("*.json") if p.stem < asof)
    if not prior:
        return {}
    payload = _load_json(prior[-1])
    return {str(k): float(v) for k, v in (payload.get("weights") or {}).items()}


def publish_live(broker, store, *, weights_dir, orders_dir, asof, today, spy_last,
                 fetch_metadata=None, flows_path=None):
    """Compute and write one snapshot from live broker account data.

    NAV is the broker's NetLiquidation and the per-holding rows (incl. day /
    unrealized P&L) come from ``broker.get_portfolio()`` — account-channel
    data, so no market-data subscription and the numbers match the broker's
    own app. ``spy_last`` is the latest SPY print (yfinance; benchmark only).

    All return metrics are time-weighted: external cash flows (deposits land
    weekly) are auto-detected as the ΔNAV the broker's account-level day P&L
    can't explain, recorded in the ``flows_path`` ledger + the equity point's
    ``flow`` column, and stripped from every growth figure via the chained TWR
    index. An existing ledger entry for today (manual stamp) beats detection.

    ``fetch_metadata`` is an optional callable ``tickers -> {ticker:
    {company_name, sector}}`` (injected so tests stay network-free; main()
    passes the real Sharadar fetcher). Returns a small summary dict.
    """
    asof = str(pd.Timestamp(asof).date())
    today = pd.Timestamp(today).normalize()
    if today.tz is not None:  # make the contract explicit: operate on a tz-naive date
        today = today.tz_localize(None)

    # 1. Live account state (connect/disconnect bracket).
    broker.connect()
    try:
        portfolio = broker.get_portfolio()
        nav = float(broker.get_nav())
        try:
            account_pnl = broker.get_account_pnl()
        except Exception as exc:  # noqa: BLE001
            logger.warning("publish: account-level P&L unavailable (%s) — "
                           "falling back to per-holding sum", exc)
            account_pnl = None
    finally:
        broker.disconnect()
    positions = {p["ticker"]: float(p["position"]) for p in portfolio}

    # 2. Frozen weights for this Friday.
    weights_payload = _load_json(Path(weights_dir) / f"{asof}.json")
    target_weights = {str(k): float(v) for k, v in (weights_payload.get("weights") or {}).items()}
    k_probs = weights_payload.get("k_probs") or {}
    regime_features = weights_payload.get("regime_features")  # None until weights pipeline adds it

    last_weights = _prior_weights(weights_dir, asof)
    turnover = compute_turnover(target_weights, last_weights) if last_weights else None

    # Ticker metadata (company name + sector) for everything we hold or target.
    metadata: dict[str, dict] = {}
    if fetch_metadata is not None:
        tickers = sorted(set(positions) | set(target_weights))
        try:
            metadata = fetch_metadata(tickers) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("publish: ticker metadata unavailable (%s)", exc)

    # 3. Equity history (for prior NAV, inception baselines, risk series).
    # Ledger flows override the stored flow column so hand edits to past dates
    # take effect immediately, not only after a --from-audit rebuild.
    curve = store.read_equity_curve()
    flows = load_flows(flows_path)
    for p in curve:
        if str(p["date"]) in flows:
            p["flow"] = flows[str(p["date"])]
    prev_nav = _prev_nav(curve, today)
    inception_spy = next(
        (p["spy_close"] for p in curve if p.get("spy_close") is not None), spy_last
    )
    today_str = str(today.date())

    # 4. Day P&L (broker truth — deposit-immune) and today's external flow.
    day_pnl = account_pnl.get("daily_pnl") if account_pnl else None
    if day_pnl is None:
        day_pnl = holdings_day_pnl(portfolio)
    if today_str in flows:  # manual stamp beats detection
        flow_today = flows[today_str]
    else:
        flow_today = detect_flow(nav, prev_nav, day_pnl)
        if flow_today != 0.0 and flows_path is not None:
            flows[today_str] = flow_today
            save_flows(flows_path, flows)
            logger.info("publish: external flow detected on %s: %+.2f (recorded to %s)",
                        today_str, flow_today, flows_path)
    if day_pnl is None and prev_nav is not None:
        day_pnl = nav - prev_nav - flow_today  # last resort: flow-adjusted ΔNAV
    day_pnl_pct = (day_pnl / prev_nav) if (day_pnl is not None and prev_nav) else None

    # 5. Metrics off the TWR index — deposits compound at zero everywhere.
    holdings = compute_holdings_live(portfolio, target_weights, nav, metadata=metadata)
    prior_rows = [p for p in curve if str(p["date"]) < today_str]
    index = twr_index(prior_rows + [{"date": today_str, "nav": nav, "flow": flow_today}])
    risk = compute_risk(index)
    week_vs_spy = compute_week_to_date(curve, today.date(), nav, spy_last,
                                       flow_today=flow_today)
    invested = sum(h["market_value"] for h in holdings)

    # 6. Writes (equity point first so it is present for the next run).
    store.upsert_equity_point(today_str, nav, spy_last, flow=flow_today)
    store.upsert_snapshot(
        {
            "asof": today_str,
            "nav": nav,
            "day_pnl": day_pnl,
            "day_pnl_pct": day_pnl_pct,
            "total_return": (index[-1] - 1.0) if index else 0.0,
            "spy_return": pct_change(spy_last, inception_spy),
            "week_vs_spy": week_vs_spy,
            "n_positions": len(holdings),
            "invested_pct": (invested / nav) if nav > 0 else None,
            "k_probs": k_probs,
            "regime_features": regime_features,
            "risk": risk,
            "turnover": turnover,
        }
    )
    store.replace_holdings([{**h, "asof": today_str} for h in holdings])
    store.insert_weekly_portfolio(
        asof,
        [
            {"asof_friday": asof, "ticker": t, "target_weight": w, "k_probs": k_probs,
             "company_name": metadata.get(t, {}).get("company_name"),
             "sector": metadata.get(t, {}).get("sector")}
            for t, w in target_weights.items()
        ],
    )

    orders_path = Path(orders_dir) / f"{asof}.json"
    if orders_path.exists():
        exec_rows = compute_execution_quality(_load_json(orders_path))
        store.insert_executions(asof, [{**r, "asof": asof} for r in exec_rows])

    return {"asof": asof, "nav": nav, "n_holdings": len(holdings), "flow": flow_today}


def publish_from_audit(store, *, weights_dir, orders_dir, asof, today, price_fetch,
                       fetch_metadata=None, flows_path=None):
    """Compute and write one snapshot from the order audit + injected prices.

    `price_fetch(tickers, start, end) -> DataFrame` returns forward-filled daily
    closes indexed by normalized date, one column per ticker (see
    sources.fetch_close_history). No broker is contacted. External flows come
    from the ``flows_path`` ledger so a rebuild reproduces deposit-aware NAVs
    and TWR metrics. Returns a summary dict.
    """
    asof = str(pd.Timestamp(asof).date())
    today = pd.Timestamp(today).normalize()
    if today.tz is not None:
        today = today.tz_localize(None)

    history = load_history(orders_dir)
    holdings_shares = current_holdings(history)

    weights_payload = _load_json(Path(weights_dir) / f"{asof}.json")
    target_weights = {str(k): float(v) for k, v in (weights_payload.get("weights") or {}).items()}
    k_probs = weights_payload.get("k_probs") or {}
    regime_features = weights_payload.get("regime_features")
    last_weights = _prior_weights(weights_dir, asof)
    turnover = compute_turnover(target_weights, last_weights) if last_weights else None

    # Every ticker ever held (for the historical curve) + currently held + SPY.
    ever: set[str] = set()
    for rec in history:
        ever |= {str(t) for t in (rec.get("post_positions") or {})}
    tickers = sorted(ever | set(holdings_shares) | set(target_weights))
    start = inception_date(history) if history else today
    closes = price_fetch(tickers + ["SPY"], start, today + pd.Timedelta(days=1))
    spy_history = closes["SPY"] if "SPY" in closes.columns else pd.Series(dtype=float)

    flows = load_flows(flows_path)
    curve = reconstruct_curve(history, closes, spy_history, flows=flows)
    navs = [p["nav"] for p in curve]
    nav = navs[-1] if navs else 0.0
    prev_nav = navs[-2] if len(navs) >= 2 else None
    flow_last = float(curve[-1].get("flow") or 0.0) if curve else 0.0

    latest_closes = {
        t: float(closes[t].iloc[-1]) for t in holdings_shares
        if t in closes.columns and not pd.isna(closes[t].iloc[-1])
    }

    metadata: dict[str, dict] = {}
    if fetch_metadata is not None:
        try:
            metadata = fetch_metadata(sorted(set(holdings_shares) | set(target_weights))) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("publish: ticker metadata unavailable (%s)", exc)

    holdings = compute_holdings(holdings_shares, latest_closes, target_weights, nav, metadata=metadata)
    # Day P&L net of any flow that landed on the last curve date (deposit ≠ gain).
    day_pnl, day_pnl_pct = compute_day_pnl(nav - flow_last, prev_nav)
    index = twr_index(curve)
    risk = compute_risk(index)
    invested = sum(h["market_value"] for h in holdings)
    inception_spy = next((p["spy_close"] for p in curve if p["spy_close"] is not None), None)
    spy_now = curve[-1]["spy_close"] if curve else None
    today_str = str(today.date())
    # Week-to-date needs a baseline STRICTLY before this week's Monday, so feed
    # it the curve minus its own last point (today's reconstruction).
    week_vs_spy = compute_week_to_date(curve[:-1], today.date(), nav, spy_now,
                                       flow_today=flow_last)

    store.replace_equity_curve(curve)
    store.upsert_snapshot(
        {
            "asof": today_str,
            "nav": nav,
            "day_pnl": day_pnl,
            "day_pnl_pct": day_pnl_pct,
            "total_return": (index[-1] - 1.0) if index else 0.0,
            "spy_return": pct_change(spy_now, inception_spy),
            "week_vs_spy": week_vs_spy,
            "n_positions": len(holdings),
            "invested_pct": (invested / nav) if nav > 0 else None,
            "k_probs": k_probs,
            "regime_features": regime_features,
            "risk": risk,
            "turnover": turnover,
        }
    )
    store.replace_holdings([{**h, "asof": today_str} for h in holdings])
    store.insert_weekly_portfolio(
        asof,
        [
            {"asof_friday": asof, "ticker": t, "target_weight": w, "k_probs": k_probs,
             "company_name": metadata.get(t, {}).get("company_name"),
             "sector": metadata.get(t, {}).get("sector")}
            for t, w in target_weights.items()
        ],
    )
    orders_path = Path(orders_dir) / f"{asof}.json"
    if orders_path.exists():
        exec_rows = compute_execution_quality(_load_json(orders_path))
        store.insert_executions(asof, [{**r, "asof": asof} for r in exec_rows])

    return {"asof": asof, "nav": nav, "n_holdings": len(holdings)}


def is_market_hours(now=None, open_str: str = "09:30", close_str: str = "16:00") -> bool:
    """True if `now` is a weekday within [open, close] America/New_York."""
    if now is None:
        now = pd.Timestamp.now(tz="America/New_York")
    now = pd.Timestamp(now)
    if now.tz is None:
        now = now.tz_localize("America/New_York")
    else:
        now = now.tz_convert("America/New_York")
    if now.dayofweek >= 5:  # Sat/Sun
        return False
    open_t = pd.Timestamp(f"{now.date()} {open_str}", tz="America/New_York")
    close_t = pd.Timestamp(f"{now.date()} {close_str}", tz="America/New_York")
    return open_t <= now <= close_t


def fetch_spy_close() -> float | None:
    """Last SPY close via yfinance (already a dep). Returns None on failure."""
    try:
        import yfinance as yf  # noqa: PLC0415

        hist = yf.Ticker("SPY").history(period="5d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as exc:  # noqa: BLE001
        logger.warning("publish: SPY fetch failed: %s", exc)
        return None


def fetch_spy_last() -> float | None:
    """Latest SPY print via yfinance: the last 1-minute bar (near-live during
    market hours), falling back to the last daily close. Benchmark only — it
    does not need brokerage precision."""
    try:
        import yfinance as yf  # noqa: PLC0415

        intraday = yf.Ticker("SPY").history(period="1d", interval="1m")
        if not intraday.empty:
            return float(intraday["Close"].iloc[-1])
    except Exception as exc:  # noqa: BLE001
        logger.warning("publish: intraday SPY fetch failed (%s) — falling back to close", exc)
    return fetch_spy_close()


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint: `python -m trading.publish [--intraday | --from-audit]`.

    default     EOD broker publish (16:30 timer). NAV/holdings from the live
                Gateway account channel; failures propagate → OnFailure alert.
    --intraday  15-min market-hours tick. Same broker path, but exits 0 quietly
                outside market hours or when the Gateway is unreachable (a
                skipped tick is not an incident; EOD will page if it persists).
    --from-audit  Legacy reconstruction from the order audit + yfinance closes
                (backfill/repair after an outage; rebuilds the whole curve).
    """
    import argparse  # noqa: PLC0415

    import trading.config as config  # noqa: PLC0415
    from trading.data.snapshot import most_recent_friday  # noqa: PLC0415
    from trading.data.sources import fetch_ticker_metadata  # noqa: PLC0415
    from trading.publish.store import SupabaseStore, make_client  # noqa: PLC0415

    parser = argparse.ArgumentParser(prog="python -m trading.publish")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--intraday", action="store_true",
                       help="15-min tick: skip outside market hours / Gateway down")
    group.add_argument("--from-audit", action="store_true",
                       help="rebuild from order audit + yfinance closes (backfill)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    if args.intraday and not is_market_hours():
        logger.info("publish: outside market hours — intraday tick skipped")
        return 0

    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_KEY:
        logger.error("publish: SUPABASE_URL / SUPABASE_SERVICE_KEY not set — aborting")
        return 1

    store = SupabaseStore(make_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY))
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)

    if args.from_audit:
        from trading.data.sources import fetch_close_history  # noqa: PLC0415
        summary = publish_from_audit(
            store,
            weights_dir=config.WEIGHTS_DIR,
            orders_dir=config.ORDERS_DIR,
            asof=most_recent_friday(),
            today=today,
            price_fetch=fetch_close_history,
            fetch_metadata=fetch_ticker_metadata,
            flows_path=config.CAPITAL_FLOWS_PATH,
        )
        logger.info("publish: done (from-audit) — %s", summary)
        return 0

    # Broker path (intraday + EOD). Read-only connection on its own client id so
    # a Monday 15:00 tick can coexist with the rebalance connection.
    from trading.broker.ibkr import IBKRBroker  # noqa: PLC0415
    broker = IBKRBroker(
        host=config.IBKR_HOST,
        port=config.IBKR_PORT,
        client_id=config.IBKR_PUBLISH_CLIENT_ID,
        readonly=True,
    )
    try:
        summary = publish_live(
            broker, store,
            weights_dir=config.WEIGHTS_DIR,
            orders_dir=config.ORDERS_DIR,
            asof=most_recent_friday(),
            today=today,
            spy_last=fetch_spy_last(),
            fetch_metadata=fetch_ticker_metadata,
            flows_path=config.CAPITAL_FLOWS_PATH,
        )
    except (OSError, TimeoutError) as exc:
        if args.intraday:
            logger.warning("publish: Gateway unreachable (%s) — intraday tick skipped", exc)
            return 0
        raise
    logger.info("publish: done — %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
