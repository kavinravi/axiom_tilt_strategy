# Execution + Broker — Implementation Plan (Plan 3 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development / executing-plans. Steps use `- [ ]`.

**Goal:** turn this week's frozen target weights into broker orders: read live positions/NAV, diff to share-level orders, enforce safety rails, and execute via a passive→midpoint→terminal ladder. Built behind a `Broker` interface with a `dryrun` fake (fully testable now) and an `ib_async` implementation (wired now, paper/live-validated once IB Gateway exists).

**Architecture:** `broker/` is the IBKR boundary (interface + dryrun + ib_async). `execution/` is broker-agnostic: `diff` (weights+positions+NAV→orders), `safety` (pre-trade caps + kill switch), `ladder` (staged order placement against any Broker), `rebalance` (orchestrator). `run.py` gains a `rebalance` subcommand gated by `EXECUTION_MODE`.

**Tech Stack:** Python 3.11+, `ib_async` (new dep), pandas/numpy, the Plan-2 `trading/` package. Reuses frozen weights from `trading/audit/weights/<date>.json`.

---

## Rollout gate (from the spec — non-negotiable)
`dryrun` (default; no connection) → `paper` (validate against IB paper) → **supervised manual first live run** (`--confirm`) → enable automated scheduler. The live trigger stays off until a clean paper rebalance + one supervised live run. See `docs/ibkr-account-and-gateway-setup.md`.

## Build phasing
- **Phase A (tonight, no Gateway needed):** `config` additions, `broker/base.py`, `broker/dryrun.py`, `execution/diff.py`, `execution/safety.py` — all unit-tested.
- **Phase B (tonight if capacity; testable against dryrun):** `execution/ladder.py`, `execution/rebalance.py`, `run.py rebalance`.
- **Phase C (needs Gateway — validate with user):** `broker/ibkr.py` (ib_async) live connection, paper rebalance, supervised live run.

---

## config additions (`trading/config.py`)

```python
# IBKR connection (see docs/ibkr-account-and-gateway-setup.md)
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 4002          # 4002 paper / 4001 live (IB Gateway)
IBKR_CLIENT_ID = 11

# Safety rails
KILL_SWITCH_FILE = TRADING_DIR / "KILL_SWITCH"   # if this file exists, abort all order placement
MAX_ORDER_FRAC_NAV = 0.12     # reject if any single order notional > 12% of NAV
MAX_TURNOVER_FRAC = 0.60      # reject the whole rebalance if total traded notional > 60% of NAV

# Execution ladder (Monday)
LADDER_PASSIVE_WAIT_SEC = 180        # stage 1 wait before escalating to MIDPRICE
LADDER_MIDPRICE_WAIT_SEC = 120       # stage 2 wait before terminal cross
LADDER_TERMINAL_CROSS = True         # stage 3: cross the spread near close to guarantee completion
ORDERS_DIR = AUDIT_DIR / "orders"    # per-run order/fill audit logs
```

## Part A1 — `broker/base.py` (interface)

```python
from dataclasses import dataclass

@dataclass
class Order:
    ticker: str
    side: str        # "BUY" | "SELL"
    quantity: float  # shares (fractional allowed)

@dataclass
class Fill:
    ticker: str
    side: str
    quantity: float       # filled qty
    avg_price: float
    status: str           # "filled" | "partial" | "unfilled" | "cancelled"

class Broker(ABC):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_positions(self) -> dict[str, float]: ...    # {ticker: shares}
    def get_nav(self) -> float: ...
    def get_quote(self, ticker: str) -> tuple[float, float]: ...   # (bid, ask)
    def place_limit(self, order: Order, limit_price: float) -> Fill: ...
    def place_midprice(self, order: Order) -> Fill: ...
    def place_market(self, order: Order) -> Fill: ...   # terminal cross
```

**Tasks/tests:** define the dataclasses + ABC; a test that `Broker` can't be instantiated and that a trivial subclass implementing the methods works.

## Part A2 — `broker/dryrun.py` (fake broker)

A `DryRunBroker(Broker)` constructed with synthetic `positions`, `nav`, and a `quotes` dict `{ticker: (bid, ask)}` (defaults: empty positions, a configurable NAV, quotes synthesized as price±half-spread). It does **not** connect; `place_*` log the intended order and return a `Fill` simulating execution at the relevant price (limit→limit price if marketable vs the quote else "unfilled"; midprice→midpoint; market→ask for buys/bid for sells). A `fill_ratio` knob (default 1.0) lets tests simulate partial fills so the ladder's escalation is exercised.

**Tests:** positions/nav/quote accessors; each `place_*` returns the right simulated price + status; `fill_ratio < 1` yields a partial fill.

## Part A3 — `execution/diff.py`

```python
def target_shares(target_weights: dict[str,float], nav: float, prices: dict[str,float]) -> dict[str,float]:
    """shares_i = (w_i * nav) / price_i. Fractional. Skips tickers with no/zero price (logged)."""

def diff_to_orders(target_weights, current_positions, nav, prices,
                   min_order_notional: float = 1.0) -> list[Order]:
    """Compare target shares vs current positions → BUY/SELL Order list.
    - delta_i = target_shares_i - current_shares_i  (current for tickers not in target → target 0 → full sell)
    - skip |delta * price| < min_order_notional (dust)
    - SELL for delta<0, BUY for delta>0."""
```
Whole-share rounding is applied only if `config` says so (default: fractional, per the resolved spec decision). Reuse `MAX_WEIGHT` etc. from config.

**Tests (exact-value):** a 2-name target vs known positions/NAV/prices → expected share deltas + sides; a ticker held but not in target → full liquidation order; dust below `min_order_notional` skipped; a target ticker with no price → skipped + logged.

## Part A4 — `execution/safety.py`

```python
def pre_trade_checks(target_weights, orders, current_positions, nav, prices, *, config) -> list[str]:
    """Return list of problems (empty == safe to trade). Checks:
    - kill switch: config.KILL_SWITCH_FILE exists → 'KILL SWITCH ENGAGED'
    - weights sum ≈ 1 (WEIGHT_SUM_TOL); max target weight ≤ MAX_WEIGHT+1e-9
    - holdings count in [MIN_HOLDINGS, MAX_HOLDINGS]
    - per-order notional ≤ MAX_ORDER_FRAC_NAV * nav
    - total traded notional ≤ MAX_TURNOVER_FRAC * nav
    - nav > 0; every order ticker has a price."""

def assert_safe(...): # raises SafetyError if pre_trade_checks non-empty
```

**Tests:** each violation class fires its message (kill-switch file via tmp; oversized order; excessive turnover; bad weight sum; over-cap weight; zero NAV). Clean inputs → `[]`.

## Part B1 — `execution/ladder.py`

```python
def execute_ladder(broker: Broker, orders: list[Order], *, config, now_fn=None) -> list[Fill]:
    """Per order, stage the execution to capture spread then concede:
    1. PASSIVE: place_limit at the passive side (bid for BUY, ask for SELL) from broker.get_quote.
    2. wait LADDER_PASSIVE_WAIT_SEC; collect fills; compute unfilled remainder.
    3. MIDPRICE: place_midprice for the remainder.
    4. wait LADDER_MIDPRICE_WAIT_SEC; remainder.
    5. TERMINAL (if LADDER_TERMINAL_CROSS): place_market for any remainder near close.
    Returns the aggregated Fill per ticker. Records realized price vs contemporaneous
    bid/midpoint for each stage (audit: did bid-first beat midpoint?)."""
```
Time waits go through an injectable `sleep_fn`/`now_fn` so tests run instantly. Works against any `Broker`; tested with `DryRunBroker(fill_ratio=...)` to drive each stage.

**Tests:** full fill at stage 1 (no escalation); partial at stage 1 → remainder midpriced; still partial → terminal market; the audit record captures per-stage realized vs midpoint.

## Part B2 — `execution/rebalance.py`

```python
def run_rebalance(asof=None, *, mode=None, confirm=False, broker=None) -> dict:
    """Orchestrate the Monday rebalance.
    1. mode = mode or config.EXECUTION_MODE; pick broker: dryrun | IBKRBroker(paper/live).
    2. broker.connect(); reconcile current_positions + nav (RECONCILE-BEFORE-TRADE).
    3. load frozen weights from trading/audit/weights/<asof>.json (fail if missing).
    4. fetch quotes for the union of held + target tickers.
    5. orders = diff_to_orders(...); problems = pre_trade_checks(...); abort if any.
    6. if confirm: print the order table and require explicit 'yes' before placing (first live run).
    7. fills = execute_ladder(broker, orders); write audit to trading/audit/orders/<asof>.json
       (intended orders, fills, realized-vs-mid, pre/post positions, nav).
    8. broker.disconnect(); return a summary dict."""
```

**Tests (against DryRunBroker, no network):** end-to-end with a frozen weights file in a tmp audit dir → produces orders, passes safety, "fills" via ladder, writes the orders audit; a kill-switch file aborts before any place_*; a missing frozen-weights file errors clearly.

## Part C — `broker/ibkr.py` (ib_async; validate with Gateway)

`IBKRBroker(Broker)` using `ib_async`:
- `connect()`: `IB().connect(config.IBKR_HOST, config.IBKR_PORT, clientId=config.IBKR_CLIENT_ID)`.
- `get_positions()`: `ib.positions()` → `{contract.symbol: position}` (US equities).
- `get_nav()`: `accountSummary()` tag `NetLiquidation`.
- `get_quote(ticker)`: qualify a `Stock(ticker, "SMART", "USD")`, `reqMktData`, read bid/ask.
- `place_limit/place_midprice/place_market`: build `LimitOrder` / `Order(orderType="MIDPRICE")` / `MarketOrder`, `ib.placeOrder`, wait for fill/timeout, return `Fill`.
- Contract qualification + multi-class ticker mapping (e.g., `BRK B`); handle `ib.qualifyContracts`.

**Validation (with user + Gateway, not unit tests):** read-only connect on paper (positions/NAV/quotes), then a paper `rebalance`, then supervised live. Mock-based unit tests may cover contract/order construction.

## requirements
Add `ib_async>=1.0` to `requirements.txt`.

## Self-review (planning)
Covers spec Part 5 (execution & broker: MIDPRICE + passive ladder, dryrun, ib_async), Part 7 (safety rails: kill switch, per-order/turnover caps, reconcile-before-trade, audit log). Scheduling (Part 6) is Plan 4. Names consistent: `Order`/`Fill`/`Broker`, `diff_to_orders`/`target_shares`, `pre_trade_checks`/`assert_safe`, `execute_ladder`, `run_rebalance`, `IBKRBroker`/`DryRunBroker`.
```
