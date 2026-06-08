# Audit-Trail Dashboard Publisher (broker-free)

**Date:** 2026-06-08
**Status:** Approved design — ready for implementation plan

## Context

The remote monitoring dashboard reads its holdings, NAV, and equity curve from
Supabase, which is populated by the publisher in `trading/publish/`. The current
publisher (`publish_once`) reads live account state from the IBKR Gateway every
run (`broker.connect()` → `get_positions()` → `get_nav()` → `get_quote()` per
ticker). The systemd timer fires every 20 minutes during market hours, so the
Gateway has to be authenticated and running the whole session for the dashboard
to update.

This couples a *read-only monitoring* concern to a *live brokerage session*. For
a strategy that rebalances **once a week**, that is a heavy, fragile dependency:
the dashboard goes stale whenever the Gateway is down, and IBKR forces a daily
re-auth.

We already write a complete per-rebalance audit (`trading/audit/orders/<asof>.json`)
containing `post_positions` (exact post-trade holdings) and `fills` with
`avg_price`. Combined with yfinance prices (already a dependency, already used for
SPY), this is enough to reconstruct holdings, cash, and a full NAV history
**without ever contacting the broker**.

**Outcome:** the dashboard's data is produced from the audit trail + yfinance.
The Gateway is needed *only* for the Monday rebalance (actual trading), never for
the dashboard to refresh.

## Goal / success criteria

- Dashboard holdings, equity curve, and snapshot populate from audit files +
  yfinance, with **no IBKR connection** in the scheduled publish path.
- Equity curve is the **full daily series from inception**, recomputed each run
  (self-healing, deterministic).
- The scheduled job runs **once daily after US close**, with no market-hours /
  Gateway dependency.
- Reconstruction logic is pure and unit-tested without network access.

## Approach (A — integrated audit publisher)

Add a pure reconstruction module and a new broker-free orchestrator alongside the
existing `publish_once`, reusing the `metrics.py` helpers and the injectable
structure. The scheduled job switches to the new path. The broker-based
`publish_once` stays in place (used by tests and available as an optional
broker "truth-up" later).

Rejected alternatives: an `AuditBroker` faking the `Broker` interface (leaky;
can't drive full-curve reconstruction since the broker path writes one point/day);
a fully standalone script (duplicates snapshot/holdings/metadata orchestration).

## Components

### New: `trading/publish/reconstruct.py` (pure, network-free)

- `load_history(orders_dir) -> list[dict]` — read `*.json` order files, sorted by `asof`.
- `current_holdings(history) -> dict[str, float]` — the latest file's `post_positions`.
- `cash(history) -> float` — `first_build.nav − Σ(signed fills × avg_price)` over all
  files (BUY = +qty, SELL = −qty; cash change = −signed×avg_price). Verified exact
  against current data: `100023.40 − 97346.39 = 2677.01`.
- `reconstruct_curve(history, close_history, spy_history) -> list[{date, nav, spy_close}]`
  — for each trading day `d` from inception → today:
  - effective rebalance = latest file with `asof ≤ d`; holdings/cash from it
    (before the first rebalance: holdings `{}`, cash = inception nav → flat line)
  - `nav(d) = cash(d) + Σ(shares(d)[t] × close(t, d))`
  - `spy_close(d)` from `spy_history`

### Extended: `trading/data/sources.py`

- `fetch_close_history(tickers, start, end) -> pd.DataFrame` — one batched yfinance
  call for daily closes across all tickers ever held (union over history).
- Latest closes derived from the same frame (last row) for holdings pricing.
- Fallbacks: missing close on a day → forward-fill from prior day; ticker with no
  yfinance data at all → fall back to its last known `avg_price` from fills.

### Extended: `trading/publish/store.py`

- New `replace_equity_curve(rows: list[dict]) -> None` — delete-all + insert,
  mirroring the existing `replace_holdings`. Used because we rebuild the whole
  curve each run.

### New orchestrator: `publish_from_audit(...)` in `trading/publish/publish.py`

Signature (injectable for tests, mirrors `publish_once`):
`publish_from_audit(store, *, weights_dir, orders_dir, price_fetch, spy_fetch, today, fetch_metadata=None)`

Flow:
1. `history = load_history(orders_dir)`; `holdings_shares = current_holdings(history)`;
   `c = cash(history)`.
2. Fetch close history for the held/ever-held tickers + SPY via injected fetchers.
3. `curve = reconstruct_curve(...)`; `nav = curve[-1].nav`; `prev_nav = curve[-2].nav`
   (None if the curve has a single point).
4. Holdings rows: **reuse `compute_holdings(holdings_shares, latest_closes, target_weights, nav, metadata)`**,
   where `target_weights` comes from the most recent weights file
   (`weights_dir/<latest asof>.json`), same source as `publish_once`.
5. Snapshot: **reuse** `compute_risk(navs)` (navs from `curve`),
   `compute_day_pnl(nav, prev_nav)`, `compute_turnover(...)` (latest vs prior
   weights file), `pct_change(...)`; `n_positions`/`invested_pct` from holdings;
   `k_probs`/`regime_features` from the latest weights file.
6. Writes: `store.replace_equity_curve(curve)`, `store.replace_holdings(...)`,
   `store.upsert_snapshot(...)`, `store.insert_weekly_portfolio(...)`,
   `store.insert_executions(...)` (these last two unchanged — already broker-free).
7. Metadata (company/sector) via `fetch_ticker_metadata` (Sharadar, not the broker) — unchanged, best-effort.

### Entrypoint & scheduling

- New `main()` path wires `publish_from_audit` with real yfinance fetchers +
  `SupabaseStore`. **Removes the IBKR connection and the `is_market_hours` guard**
  from the scheduled path.
- `deploy/systemd/axiom-publish.*` changes from "every 20 min during market hours"
  to **once daily after US close** (e.g. ~16:30 ET). No Gateway dependency.
- The broker-based `publish_once` and its `main()` wiring remain in the codebase
  for tests and a possible future broker truth-up; they are simply no longer the
  scheduled default.

## Error handling

- yfinance fetch failure for a ticker → forward-fill / last `avg_price` fallback;
  never crash the run. Log a warning (mirrors the existing best-effort metadata pattern).
- No `first_build` file in history → cannot anchor cash; log error and abort the
  run (do not publish a wrong NAV).
- Empty history (no rebalances yet) → holdings empty, flat equity at inception nav
  (acceptable; matches a not-yet-traded account).

## Testing

- `reconstruct.py` functions: unit tests with fixture order-audit JSON + injected
  price frames (no network). Cover: cash exactness, holdings = latest
  `post_positions`, curve flat-before-first-rebalance, multi-rebalance netting,
  forward-fill on missing closes.
- `publish_from_audit`: test against a fake recording store + injected fetchers,
  following the existing `tests/trading/test_publish_orchestrator.py` pattern.
- yfinance helpers: thin, tested with a mocked/injected downloader.

## Known limitations (accepted)

- **Drift from true account:** dividends, interest, fees, corporate actions, and
  any manual/out-of-system trades are not in the audit, so reconstructed
  NAV/holdings gradually diverge from the real IBKR account. Mitigation for later:
  an occasional broker-based truth-up run (the retained `publish_once`).
- **Effective-date approximation:** the audit's only date field is `asof` (the
  Friday); there is no recorded fill/execution timestamp. A rebalance is treated
  as effective from its `asof`, ~1 trading day before the actual Monday execution.
  Negligible for the curve; revisit by adding an execution date to the audit if
  precision is ever needed.

## Out of scope

- Changing how the live rebalance executes or what the audit records (beyond
  possibly an execution-date field, deferred).
- Real-time / intraday NAV (daily granularity only).
- The broker truth-up job (noted as future mitigation, not built here).
