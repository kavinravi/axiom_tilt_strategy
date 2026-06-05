"""One-off RTH smoke test: L1 real-time data + a real paper fill.

NOT a unit test — run manually against a LIVE paper IB Gateway during RTH:

    python scripts/paper_smoke.py            # read + quote + tiny fill + flatten
    python scripts/paper_smoke.py --no-trade # read + quote only (no orders)

Places a tiny 1-share MARKET order on a liquid name to prove the paper
trading path fills, then immediately flattens it. Paper money only.
"""
from __future__ import annotations

import argparse
import sys
import time

import math

from ib_async import Stock

from src.utils.env import get_env
from trading.broker.base import Order
from trading.broker.ibkr import IBKRBroker

TEST_TICKER = "AAPL"   # liquid, tight spread, instant market fill during RTH

# IB market-data type codes -> human label
_MDT = {1: "REAL-TIME (live L1)", 2: "frozen (last real-time)",
        3: "DELAYED (15-min)", 4: "delayed-frozen"}
# Error codes that mean "this login is not entitled to the feed" (account-side, not code).
_ENTITLEMENT_ERRS = {10089, 10091, 10168}


def _probe_feed(ib, ticker, mdtype):
    """Request *ticker* at market-data type *mdtype*; return (bid, ask, errors)."""
    errs: list[tuple[int, str]] = []
    handler = lambda reqId, code, msg, c=None: errs.append((code, msg))
    ib.errorEvent += handler
    try:
        c = Stock(ticker, "SMART", "USD")
        [c] = ib.qualifyContracts(c)
        ib.reqMarketDataType(mdtype)
        t = ib.reqMktData(c, "", False, False)
        ib.sleep(4)
        bid, ask = t.bid, t.ask
        ib.cancelMktData(c)
    finally:
        ib.errorEvent -= handler
    # keep only entitlement/data errors, not the benign farm-connection notices
    real = [(code, m) for code, m in errs if code in _ENTITLEMENT_ERRS]
    return bid, ask, real


def report_data_entitlement(broker, ticker) -> bool:
    """Explicitly distinguish REAL-TIME vs DELAYED so the test can't mislabel.

    Returns True iff a real-time (type-1) quote came back with live values.
    """
    ib = broker.ib
    bid, ask, errs = _probe_feed(ib, ticker, 1)   # ask for REAL-TIME
    ok = not (math.isnan(bid) or bid <= 0 or math.isnan(ask) or ask <= 0)
    if ok:
        print(f"[DATA] {ticker} REAL-TIME (type 1): bid={bid:.4f} ask={ask:.4f}  "
              f"✅ real-time L1 ENTITLEMENT ACTIVE")
        return True
    # real-time failed — say WHY, then show what delayed gives
    if any(code in _ENTITLEMENT_ERRS for code, _ in errs):
        print(f"[DATA] {ticker} REAL-TIME (type 1): DENIED — "
              f"Error {errs[0][0]} (login not entitled / share not effective)")
    else:
        print(f"[DATA] {ticker} REAL-TIME (type 1): no live quote (bid={bid} ask={ask}) "
              f"— may just be outside RTH")
    dbid, dask, _ = _probe_feed(ib, ticker, 3)    # delayed fallback
    ib.reqMarketDataType(1)                        # restore
    if not (math.isnan(dbid) or dbid <= 0):
        print(f"[DATA] {ticker} DELAYED  (type 3): bid={dbid:.4f} ask={dask:.4f}  "
              f"⚠️  only delayed available — real-time L1 NOT active")
    else:
        print(f"[DATA] {ticker} DELAYED  (type 3): also NaN — market likely closed; inconclusive")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-trade", action="store_true", help="skip the paper fill test")
    ap.add_argument("--ticker", default=TEST_TICKER)
    ap.add_argument("--client-id", type=int, default=17)
    args = ap.parse_args()

    host = get_env("IBKR_HOST", default="127.0.0.1")
    port = int(get_env("IBKR_PORT", default="4002"))
    print(f"Connecting to IB Gateway {host}:{port} (clientId={args.client_id}) ...")

    # readonly=False so we can place the paper order; only used if --no-trade absent.
    broker = IBKRBroker(host=host, port=port, client_id=args.client_id,
                        readonly=args.no_trade)
    broker.connect()
    print("  connected.\n")

    rc = 0
    try:
        # ---- 1. Read paths -------------------------------------------------
        nav = broker.get_nav()
        print(f"[READ] NAV            = ${nav:,.2f}")
        positions = broker.get_positions()
        print(f"[READ] positions ({len(positions)}) = {positions}\n")

        # ---- 2. Market data: REAL-TIME vs DELAYED (explicit, no mislabeling) --
        print(f"[DATA] checking real-time L1 entitlement for {args.ticker} ...")
        realtime_ok = report_data_entitlement(broker, args.ticker)
        if not realtime_ok:
            rc = 1
        print()

        # ---- 3. Paper fill -------------------------------------------------
        if args.no_trade:
            print("[TRADE] skipped (--no-trade)")
            return rc

        print(f"[TRADE] placing paper MARKET BUY 1 {args.ticker} ...")
        h = broker.submit_market(Order(ticker=args.ticker, side="BUY", quantity=1))
        for _ in range(20):  # up to ~10s
            broker.ib.sleep(0.5)
            f = broker.get_fill(h)
            if f.status == "filled":
                break
        print(f"[TRADE] BUY  fill: status={f.status} qty={f.quantity} "
              f"avg=${f.avg_price:.4f}")
        if f.status != "filled" or f.quantity < 1:
            print("[TRADE] ❌ BUY did not fill")
            broker.cancel(h)
            return 1
        print("[TRADE] ✅ PAPER FILL OK\n")

        # ---- 4. Flatten ----------------------------------------------------
        print(f"[TRADE] flattening: paper MARKET SELL 1 {args.ticker} ...")
        h2 = broker.submit_market(Order(ticker=args.ticker, side="SELL", quantity=1))
        for _ in range(20):
            broker.ib.sleep(0.5)
            f2 = broker.get_fill(h2)
            if f2.status == "filled":
                break
        print(f"[TRADE] SELL fill: status={f2.status} qty={f2.quantity} "
              f"avg=${f2.avg_price:.4f}")
        if f2.status != "filled":
            print("[TRADE] ⚠️  SELL did not confirm filled — CHECK POSITIONS MANUALLY")
            rc = 1
        else:
            print("[TRADE] ✅ flattened\n")

        print(f"[FINAL] positions = {broker.get_positions()}")
    finally:
        broker.disconnect()
        print("disconnected.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
