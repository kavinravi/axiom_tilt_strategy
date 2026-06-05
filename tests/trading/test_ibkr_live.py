"""Read-only live smoke test against the IB Gateway (paper account).

Marked ``slow`` — requires a running IB Gateway. Skip automatically when
IBKR_HOST is not set in the environment.

Run manually:
    python -m pytest tests/trading/test_ibkr_live.py -v -m slow

IMPORTANT: This test is strictly READ-ONLY. No orders are placed.
Uses client_id=13 to avoid clashing with other connections.
"""
import pytest

from src.utils.env import get_env


def _has_ibkr_host() -> bool:
    return bool(get_env("IBKR_HOST"))


@pytest.mark.slow
@pytest.mark.skipif(
    not _has_ibkr_host(),
    reason="IBKR_HOST not set — skipping live Gateway test",
)
def test_ibkr_live_readonly():
    """Connect (read-only) to the paper Gateway and validate account data + a quote."""
    from trading.broker.ibkr import IBKRBroker

    host = get_env("IBKR_HOST")
    port = int(get_env("IBKR_PORT", default="4002"))

    broker = IBKRBroker(host=host, port=port, client_id=13, readonly=True)
    broker.connect()

    try:
        # --- NAV ---
        nav = broker.get_nav()
        print(f"\n[LIVE] NAV = ${nav:,.2f}")
        assert nav > 0, f"Expected positive NAV, got {nav}"

        # --- Positions ---
        positions = broker.get_positions()
        print(f"[LIVE] Positions ({len(positions)} holdings): {positions}")
        assert isinstance(positions, dict), "get_positions() must return a dict"
        # On a fresh paper account with no holdings this is an empty dict — that's fine
        for ticker, shares in positions.items():
            assert isinstance(ticker, str) and ticker, "position key must be a non-empty string"
            assert isinstance(shares, (int, float)), "position value must be numeric"

        # --- Quote (AAPL) ---
        # Outside regular trading hours (and without a real-time data subscription)
        # no quote may be available; get_quote correctly RAISES rather than returning
        # nan. Treat that as a skip, not a failure — the quote path is only meaningful
        # when market data is actually flowing.
        try:
            bid, ask = broker.get_quote("AAPL")
        except RuntimeError as exc:
            pytest.skip(f"no market data available (likely outside RTH / no subscription): {exc}")
        print(f"[LIVE] AAPL quote: bid={bid:.4f}  ask={ask:.4f}")
        assert bid > 0, f"Expected positive bid, got {bid}"
        assert ask > 0, f"Expected positive ask, got {ask}"
        assert ask >= bid, f"ask ({ask}) must be >= bid ({bid})"

    finally:
        broker.disconnect()
