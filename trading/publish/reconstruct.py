"""Pure reconstruction of holdings, cash, and NAV history from the order audit.

No network, no Supabase, no broker. Daily-close prices are injected (a DataFrame
indexed by normalized date, columns = tickers) so these functions are trivially
unit-testable. The audit files live in trading/audit/orders/<asof>.json and carry
post_positions (exact post-trade holdings) and fills (with avg_price).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_history(orders_dir: Path | str) -> list[dict]:
    """All order-audit records under orders_dir, ascending by 'asof'."""
    records = [json.loads(p.read_text()) for p in Path(orders_dir).glob("*.json")]
    records.sort(key=lambda r: str(r.get("asof", "")))
    return records


def current_holdings(history: list[dict]) -> dict[str, float]:
    """Latest post-trade positions, zero-share names dropped."""
    for rec in reversed(history):
        post = rec.get("post_positions")
        if post is not None:
            return {str(k): float(v) for k, v in post.items() if float(v) != 0.0}
    return {}


def _signed_qty(fill: dict) -> float:
    qty = float(fill.get("quantity", 0.0))
    return qty if fill.get("side", "BUY") == "BUY" else -qty


def _first_build(history: list[dict]) -> dict:
    anchor = next((r for r in history if r.get("first_build")), None)
    if anchor is None:
        raise ValueError("reconstruct: no first_build record in history")
    return anchor


def inception_date(history: list[dict]) -> pd.Timestamp:
    """Normalized asof of the first_build record (when the strategy went live)."""
    return pd.Timestamp(_first_build(history)["asof"]).normalize()


def cash_after(history: list[dict]) -> float:
    """Residual cash = inception NAV - sum(signed fill qty * avg_price).

    BUY spends cash (signed +qty); SELL returns cash (signed -qty, so subtracting
    a negative adds). Inception NAV is the first_build record's all-cash 'nav'.
    """
    cash = float(_first_build(history)["nav"])
    for rec in history:
        for f in rec.get("fills", []):
            cash -= _signed_qty(f) * float(f.get("avg_price") or 0.0)
    return cash


def reconstruct_curve(history: list[dict], close_history: pd.DataFrame, spy_history: pd.Series | None) -> list[dict]:
    """Daily {date, nav, spy_close} from inception through the last price date.

    close_history: DataFrame indexed by normalized date, columns = tickers, values
    = daily close. spy_history: Series indexed by the same dates. For each day d:
    holdings/cash come from the latest rebalance with asof <= d; nav = cash +
    sum(shares * close). A missing/NaN close contributes zero (price comes in via
    the forward-filled frame from fetch_close_history).
    """
    if not history or close_history is None or len(close_history.index) == 0:
        return []
    start = inception_date(history)
    rows: list[dict] = []
    for raw_d in close_history.index:
        d = pd.Timestamp(raw_d).normalize()
        if d < start:
            continue
        applied = [r for r in history if pd.Timestamp(r["asof"]).normalize() <= d]
        if not applied:
            continue
        # holdings/cash are recomputed per date but only change when `applied` grows
        holdings = current_holdings(applied)
        cash = cash_after(applied)
        mv = 0.0
        for ticker, shares in holdings.items():
            if ticker in close_history.columns:
                px = close_history.at[raw_d, ticker]
                if not pd.isna(px):
                    mv += shares * float(px)
        spy = spy_history.get(raw_d) if spy_history is not None else None
        rows.append({
            "date": str(d.date()),
            "nav": cash + mv,
            "spy_close": (float(spy) if spy is not None and not pd.isna(spy) else None),
        })
    return rows
