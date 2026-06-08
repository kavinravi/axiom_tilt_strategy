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


def load_history(orders_dir) -> list[dict]:
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
