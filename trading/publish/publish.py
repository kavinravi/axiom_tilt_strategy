"""Orchestrate one publish: broker + audit files -> metrics -> Supabase.

publish_once is fully injectable (broker, store, dirs, dates, spy_close) so tests
run against DryRunBroker + a fake store with no network. main() wires the real
IBKRBroker + SupabaseStore for the systemd timer.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from trading.publish.metrics import (
    compute_day_pnl,
    compute_execution_quality,
    compute_holdings,
    compute_risk,
    compute_turnover,
    pct_change,
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


def publish_once(broker, store, *, weights_dir, orders_dir, asof, today, spy_close,
                 fetch_metadata=None):
    """Compute and write one snapshot. Returns a small summary dict.

    ``fetch_metadata`` is an optional callable ``tickers -> {ticker: {company_name,
    sector}}`` (injected so tests stay network-free; main() passes the real
    Sharadar fetcher). When None, holdings carry None name/sector.
    """
    asof = str(pd.Timestamp(asof).date())
    today = pd.Timestamp(today).normalize()
    if today.tz is not None:  # make the contract explicit: operate on a tz-naive date
        today = today.tz_localize(None)

    # 1. Live account state (connect/disconnect bracket).
    broker.connect()
    try:
        positions = broker.get_positions()
        nav = float(broker.get_nav())
        prices: dict[str, float] = {}
        for ticker in positions:
            try:
                bid, ask = broker.get_quote(ticker)
                prices[ticker] = (bid + ask) / 2.0
            except Exception as exc:  # noqa: BLE001
                logger.warning("publish: no quote for %s: %s", ticker, exc)
    finally:
        broker.disconnect()

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
    curve = store.read_equity_curve()
    prev_nav = _prev_nav(curve, today)
    inception_nav = float(curve[0]["nav"]) if curve else nav
    inception_spy = next(
        (p["spy_close"] for p in curve if p.get("spy_close") is not None), spy_close
    )
    today_str = str(today.date())
    navs = [float(p["nav"]) for p in curve if str(p["date"]) < today_str] + [nav]

    # 4. Metrics.
    holdings = compute_holdings(positions, prices, target_weights, nav, metadata=metadata)
    day_pnl, day_pnl_pct = compute_day_pnl(nav, prev_nav)
    risk = compute_risk(navs)
    invested = sum(h["market_value"] for h in holdings)

    # 5. Writes (equity point first so it is present for the next run).
    store.upsert_equity_point(today_str, nav, spy_close)
    store.upsert_snapshot(
        {
            "asof": today_str,
            "nav": nav,
            "day_pnl": day_pnl,
            "day_pnl_pct": day_pnl_pct,
            "total_return": pct_change(nav, inception_nav),
            "spy_return": pct_change(spy_close, inception_spy),
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


def publish_from_audit(store, *, weights_dir, orders_dir, asof, today, price_fetch,
                       fetch_metadata=None):
    """Compute and write one snapshot from the order audit + injected prices.

    `price_fetch(tickers, start, end) -> DataFrame` returns forward-filled daily
    closes indexed by normalized date, one column per ticker (see
    sources.fetch_close_history). No broker is contacted. Returns a summary dict.
    """
    asof = str(pd.Timestamp(asof).date())
    today = pd.Timestamp(today).normalize()

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

    curve = reconstruct_curve(history, closes, spy_history)
    navs = [p["nav"] for p in curve]
    nav = navs[-1] if navs else 0.0
    prev_nav = navs[-2] if len(navs) >= 2 else None

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
    day_pnl, day_pnl_pct = compute_day_pnl(nav, prev_nav)
    risk = compute_risk(navs)
    invested = sum(h["market_value"] for h in holdings)
    inception_nav = navs[0] if navs else nav
    inception_spy = next((p["spy_close"] for p in curve if p["spy_close"] is not None), None)
    spy_now = curve[-1]["spy_close"] if curve else None
    today_str = str(today.date())

    store.replace_equity_curve(curve)
    store.upsert_snapshot(
        {
            "asof": today_str,
            "nav": nav,
            "day_pnl": day_pnl,
            "day_pnl_pct": day_pnl_pct,
            "total_return": pct_change(nav, inception_nav),
            "spy_return": pct_change(spy_now, inception_spy),
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


def main() -> int:
    """CLI entrypoint for the daily timer: `python -m trading.publish`.

    Broker-free: holdings, NAV, and the equity curve are reconstructed from the
    order audit + yfinance closes. No IBKR connection, no market-hours guard.
    """
    import trading.config as config  # noqa: PLC0415
    from trading.data.snapshot import most_recent_friday  # noqa: PLC0415
    from trading.data.sources import fetch_close_history, fetch_ticker_metadata  # noqa: PLC0415
    from trading.publish.store import SupabaseStore, make_client  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)

    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_KEY:
        logger.error("publish: SUPABASE_URL / SUPABASE_SERVICE_KEY not set — aborting")
        return 1

    store = SupabaseStore(make_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY))
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    summary = publish_from_audit(
        store,
        weights_dir=config.WEIGHTS_DIR,
        orders_dir=config.ORDERS_DIR,
        asof=most_recent_friday(),
        today=today,
        price_fetch=fetch_close_history,
        fetch_metadata=fetch_ticker_metadata,
    )
    logger.info("publish: done — %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
