"""One-shot backfill: seed weekly_portfolio + executions from existing audit files.

The daily equity_curve cannot be backfilled (no historical NAV) — it builds forward
from go-live. Run once: `python -m trading.publish.backfill`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from trading.publish.metrics import compute_execution_quality

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def backfill(store, *, weights_dir, orders_dir) -> dict:
    """Load every weights + orders audit file into the store. Returns counts."""
    weeks = 0
    for wp in sorted(Path(weights_dir).glob("*.json")):
        asof = wp.stem
        payload = _load_json(wp)
        target_weights = {str(k): float(v) for k, v in (payload.get("weights") or {}).items()}
        k_probs = payload.get("k_probs") or {}
        store.insert_weekly_portfolio(
            asof,
            [
                {"asof_friday": asof, "ticker": t, "target_weight": w, "k_probs": k_probs}
                for t, w in target_weights.items()
            ],
        )
        weeks += 1

    order_files = 0
    for op in sorted(Path(orders_dir).glob("*.json")):
        asof = op.stem
        rows = compute_execution_quality(_load_json(op))
        store.insert_executions(asof, [{**r, "asof": asof} for r in rows])
        order_files += 1

    logger.info("backfill: %d weeks, %d order files", weeks, order_files)
    return {"weeks": weeks, "order_files": order_files}


def main() -> int:
    import trading.config as config  # noqa: PLC0415
    from trading.publish.store import SupabaseStore, make_client  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)
    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_KEY:
        logger.error("backfill: SUPABASE_URL / SUPABASE_SERVICE_KEY not set — aborting")
        return 1
    store = SupabaseStore(make_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY))
    backfill(store, weights_dir=config.WEIGHTS_DIR, orders_dir=config.ORDERS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
