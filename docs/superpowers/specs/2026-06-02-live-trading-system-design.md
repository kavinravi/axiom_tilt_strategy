# Spec: Live IBKR trading system (k-ensemble) — full execution + scheduling

**Date:** 2026-06-02
**Status:** Design pending user review
**Owner:** kavinravi
**Supersedes:** `2026-06-01-trading-codebase-design.md` (that spec scoped only the
skeleton + live weights with a stubbed broker; this one expands scope to the full
live execution + scheduling system the user asked for).

## Goal

Stand up a self-contained `trading/` system that runs the validated **k-ensemble**
strategy end to end against **Interactive Brokers**:

1. **Friday:** fetch a current market snapshot (Friday-close data), run it through a
   **shared strategy core**, and freeze **this week's target weights** `{ticker: weight}`.
2. **Monday, 15:00 America/New_York** (= 12:00 PT; 1 hour before the 16:00 ET close):
   connect to IBKR, reconcile actual positions/NAV, diff against the frozen targets, and
   execute the rebalance with a **passive-first execution ladder** — post a limit at the
   passive side (bid for buys / ask for sells) to try to *capture* the spread, then after
   a short wait re-submit the unfilled remainder at **`MIDPRICE`**, with a terminal cross
   near the close to guarantee completion.

The existing backtest must produce **identical** numbers after the refactor that
extracts the shared core.

## Scope decision

**Full live + scheduling**, built with a **staged-rollout safety gate**. The entire
code path is built now — `ib_async` execution, position/NAV reads, order diffing,
`MIDPRICE` orders, and the scheduler — but execution mode escalates through:

```
dryrun  →  paper  →  live
```

- A single `config.EXECUTION_MODE` controls which broker is used.
- **Default is `dryrun`** (no connection). `paper` and `live` require an explicit,
  deliberate change.
- The **automated** `live` scheduler is **not** enabled until the pipeline is validated
  on paper (one-time mechanical check) and the **first live rebalance has been run
  manually under supervision** with confirm-before-submit (see Rollout plan).

This is the full system — same code path, same scheduler — with the live trigger
gated until paper validates. Nothing is throwaway.

## Current state & prerequisites

- **Chosen strategy:** regime-conditional K-selector **ensemble** (not RL/PPO).
  See `reports/EXECUTIVE_SUMMARY.md` / `WAKE_UP.md`.
- **Strategy math is copy-pasted** between `experiments/regime_k_selector.py` and
  `experiments/regime_k_selector_weights.py`; no single source of truth. The
  water-fill cap lives at `src/utils/rl_env.py:14` (`project_to_simplex`).
- **Model is never persisted** — retrained from scratch each backtest run.
- **Account:** the user has a **funded live IBKR account only** — no paper account
  in active use, and **IB Gateway / TWS is not installed yet**.
  - IBKR auto-provisions a **free paper account** with every live account → activate
    and use it for validation before live.
  - No Gateway means **no live connection is possible yet** → build + validate against
    `dryrun` now; wire the real connection after Gateway is installed and logged in.
- **Dependencies to add:** `ib_async` (not installed), `fredapi` (not installed; FRED
  macro). `nasdaqdatalink` (Sharadar) is already installed. Scheduling uses system
  cron/systemd timers on the VPS (no extra Python dep — see Part 6).

## Backtest-vs-live fidelity notes (important)

- The backtest assumes a **Friday rebalance at Friday close** (decision and fill same
  day). Live decides Friday but **fills Monday ~15:00 ET** — a ~3-day gap with weekend
  + Monday-AM drift the Sharpe-1.28 backtest never modeled. We accept this and
  **measure** the Friday-close → Monday-fill slippage during paper trading.
- We compute weights from the **Friday-close snapshot** (matches the backtest's data
  cadence) and freeze them; Monday only executes toward the frozen targets.
- **Two data planes:** Sharadar EOD drives *what to hold* (weights/market-caps, Friday);
  IBKR live quotes drive *the fill price* (`MIDPRICE`, Monday). Sharadar never sees the
  execution price.
- **Universe is current S&P 500 membership** (live) vs the backtest's PIT membership —
  expected for a live snapshot.

---

## Part 1 — Extract the shared core (`src/strategy/`)

The single source of truth, imported by both backtest and live. Identifier-agnostic:
operates on a generic `id` column (backtest passes `permno`, live passes `ticker`).

- `src/strategy/factors.py`
  - `score_universe(snapshot_df, id_col="id") -> DataFrame[id, score]`
  - `sp = revenue / mcap` (clip lower 0); `fcfa = fcf / assets` (clip [-1, 2], NaN
    where assets ≤ 0); cross-sectional z-scores per date; `score = 0.5·z_sp + 0.5·z_fcfa`.
    NaN z-scores → 0.0.
- `src/strategy/k_selector.py`
  - `build_regime_features(...)` — VIX (`macro_vixcls`), 10Y (`macro_dgs10`), term
    spread (`macro_t10y2y`), trailing SPY 4w/12w returns, 12w/26w annualized vol.
    Shift(1) to avoid look-ahead, exactly as current code.
  - `train_model(regime_X, labels) -> model` (LGBM multiclass; same hyperparameters:
    n_estimators=500, lr=0.03, num_leaves=15, min_data_in_leaf=20, feature_fraction=0.8,
    bagging_fraction=0.8, lambda_l2=2.0, num_class=4).
  - `save_model(model, path)` / `load_model(path)`.
  - `predict_k_probs(model, regime_row) -> {10: p, 20: p, 30: p, 50: p}`.
- `src/strategy/allocate.py`
  - `topk_mcap_weights(scored_df, K, max_weight=0.10) -> {id: weight}` — top-K by score,
    mcap weight via water-fill cap (wraps `project_to_simplex` on `log(mcap)`).
  - `ensemble_weights(scored_df, k_probs, K_candidates=[10,20,30,50]) -> {id: weight}` —
    convex combination `w(i) = Σ_K p(K)·w_K(i)`.
- `src/strategy/__init__.py` — clean public API.

**Refactor the experiments** to import from `src/strategy/` instead of their inline
copies. Forward-return/label/walk-loop code stays in the backtest path; only the pure,
forward-looking-free pieces move to the core.

**Acceptance test (no silent drift):** after the refactor, the backtest's
`weekly_regime_K_ensemble.parquet` and `k_ensemble_weights.parquet` are numerically
identical (within float tolerance) to the pre-refactor outputs.

## Part 2 — Persist a production model

- `src/strategy/train.py` (CLI): train one LGBM on **all history through the latest
  available date**, save to `trading/models/k_selector.txt` + `k_selector.meta.json`
  (train date, feature list, label definition, K candidates, data vintage).
- **Retrain cadence: quarterly, configurable** (a `config.py` toggle allows weekly).
  Rationale: backtest validated annual retraining with a 1–2 year lag, so quarterly
  with current data is strictly fresher; freezing the model between retrains makes each
  week's trades reproducible and auditable.

## Part 3 — `trading/` structure

```
trading/
  __init__.py
  config.py            paths; universe source; data-source toggles; retrain cadence;
                       EXECUTION_MODE (dryrun|paper|live); IBKR conn (host/port/clientId);
                       safety limits; schedule (tz, snapshot day, execute time)
  data/
    __init__.py
    universe.py        current S&P 500 members -> tickers
    sources.py         thin wrappers: Sharadar (NDL), FRED, SPY history
    snapshot.py        assemble one current Friday-close cross-section keyed by ticker
  models/
    k_selector.txt     persisted production model (+ .meta.json)
  weights.py           snapshot -> core weights; freeze this week's targets to disk
  broker/
    __init__.py
    base.py            Broker interface: get_positions(), get_nav(), place_orders(), connect()/disconnect()
    ibkr.py            ib_async implementation — limit + MIDPRICE order primitives
    dryrun.py          fake broker: synthetic/last-known positions+NAV, logs intended orders
  execution/
    __init__.py
    diff.py            frozen target weights + current positions + NAV -> share order list
    ladder.py          staged execution: passive bid/ask -> MIDPRICE -> terminal cross
    safety.py          pre-trade sanity checks + kill switch + caps
    rebalance.py       reconcile -> load frozen weights -> diff -> safety -> ladder -> audit
  schedule/
    __init__.py
    scheduler.py       Friday weights job + Monday 15:00 ET execute job (tz-aware)
  alerts.py            email/push notification on job failure or safety abort
  audit/               per-run logs: weights, intended vs filled orders, NAV (gitignored)
  run.py               CLI: python -m trading.run {weights|rebalance|schedule} --mode dryrun
  README.md
```

## Part 4 — Live data snapshot

`trading/data/snapshot.py` builds one current Friday-close cross-section, reusing
existing keys so `score_universe` runs unchanged (`id`=ticker, `revenue`, `fcf`,
`assets`, `prc`, `shrout`/`marketcap`, plus macro columns).

- **Universe:** current S&P 500 members → tickers. **Preferred source: the Sharadar
  `SP500` constituent table** via the `NASDAQ_DATA_LINK_API_KEY` we already use — same
  vendor as fundamentals/prices, fully programmatic (no HTML scraping), and reliable.
  Verify it's in the current subscription bundle during impl; if not, fall back to
  Wikipedia's S&P 500 table. We only need the *current* set, not history.
- **Fundamentals** (revenue, FCF, total assets): latest Sharadar SF1 ARQ via
  `NASDAQ_DATA_LINK_API_KEY` — same vendor/methodology as the backtest.
- **Prices + shares outstanding** (for mcap): Sharadar SEP (same vendor).
- **Macro** (VIX, DGS10, T10Y2Y): FRED (free, via `fredapi`).
- **SPY history** (regime trailing ret/vol): ~30 weeks of weekly SPY closes.
- **CRSP/WRDS dropped for live** — not needed for a current snapshot; avoids the
  academic-license concern for commercial trading.
- **Timing:** the Friday job runs after Sharadar's Friday EOD is published (Friday
  evening / weekend). Monday execution reuses the frozen weights — no fresh fundamentals.

## Part 5 — Execution & broker (`ib_async`)

- `broker/base.py` defines the interface every broker implements:
  `connect()`, `disconnect()`, `get_positions() -> {ticker: shares}`,
  `get_nav() -> float`, `place_orders(orders) -> fills`.
- `broker/ibkr.py` — `ib_async` implementation exposing two order primitives the ladder
  uses: a **passive limit** (priced at the current bid for buys / ask for sells) and a
  **`MIDPRICE`** order (IBKR's native type that fills at the **NBBO midpoint or better** —
  better = below midpoint for buys, above for sells; built for liquid US stocks). Handles
  connection (host/port/clientId from config), qualifying contracts (S&P 500 US
  equities), and reading fills/cancels.
- `broker/dryrun.py` — no connection; synthetic or last-known positions + NAV; logs the
  orders it *would* place at each ladder stage. Default mode; how we validate before
  Gateway exists.
- `execution/diff.py` — given frozen target weights, current positions, and NAV, compute
  per-ticker **share deltas** to trade (buys + sells). **Fractional shares:** IBKR
  supports them once the account permission is enabled, so we trade fractional where the
  order type allows it; otherwise round to whole shares (dust dropped). Rounding error is
  small for a 10–50 name, 10%-capped book unless the account is tiny.
- `execution/ladder.py` — the **passive-first execution ladder**, the realization of the
  user's "penny-pinch then concede" plan:
  1. **Passive stage:** post a limit at the bid (buys) / ask (sells) to try to capture
     the spread; do *not* chase the quote.
  2. **Wait** a short, configurable window (default ~1–5 min).
  3. **Midpoint stage:** cancel the unfilled remainder and re-submit it as `MIDPRICE`.
  4. **Terminal stage** (near the 16:00 ET close, configurable, e.g. 15:55 ET): cross the
     spread on anything still unfilled (marketable limit) so the book doesn't drift from
     target. Optional toggle to instead carry the remainder to next cycle.
  - **Audit instrumentation:** record the realized fill price vs the contemporaneous
    bid/midpoint per order, so we can *empirically verify* whether the bid-first stage
    actually saves money (vs. just paying midpoint) rather than asserting it. This
    directly answers the dad-wants-midpoint vs. it-loses-money question with live data.

## Part 6 — Scheduling

- `schedule/scheduler.py` runs two tz-aware jobs, anchored to **America/New_York** so
  DST is handled automatically. The user said "12pm PST"; **12:00 PT = 15:00 ET = 1 hour
  before the 16:00 ET close** (PT is 3 hours behind ET), so the robust anchor is
  **15:00 America/New_York**:
  - **Friday job:** build snapshot → compute + freeze weights to `audit/weights/<date>.json`.
  - **Monday 15:00 ET job:** run `execution/rebalance.py` against the configured broker.
- **Deployment: an always-on remote host (small VPS), not the local machine.** The user
  can't guarantee local uptime once school resumes (~Sep 2026) and the strategy runs for
  years, so the system must run unattended. The VPS runs IB Gateway headless via
  **IBC / IBeam** (auto-start, auto-login, and the ~daily Gateway session reset IBKR
  forces) and fires the two jobs via **system `cron` / systemd timers** — robust for a
  twice-weekly job, survives crashes, no babysat long-running process.
- **Failure alerting** (`trading/alerts.py`): if a job fails — Gateway down, login lapsed,
  connection refused, or a safety abort — send an alert (email/push). A missed rebalance
  then degrades to "notify the user to intervene," not silent failure. Tolerable because
  the strategy trades only twice a week.
- **Honest operational risk:** the hard part of years-long unattended IBKR operation is
  the **daily Gateway re-auth / 2FA on a live account**, not the scheduling. IBC/IBeam
  mitigate it, but truly zero-touch 2FA can still require occasional manual re-auth; the
  alerting above is the safety net.
- The **code is deployment-agnostic**: idempotent, re-runnable CLI entrypoints with
  config-driven Gateway host/port. It runs the same on WSL (for the dryrun/paper smoke
  test) and on the VPS (for live).
- The scheduler honors `EXECUTION_MODE`; the **automated Monday live trigger stays
  disabled until the pipeline is validated on paper AND one supervised manual live
  rebalance has run clean** (see Rollout plan).

## Part 7 — Safety rails & rollout gate

- **Execution-mode gate:** `dryrun` (default) → `paper` → `live`, switched in config.
- **Kill switch:** an env var / file flag that hard-aborts any order placement.
- **Pre-trade sanity caps** (`execution/safety.py`), abort the whole rebalance if violated:
  - no single target weight > 10% (IPS cap; should already hold from allocate);
  - no single order > configurable % of NAV;
  - total turnover for the rebalance ≤ configurable %;
  - target ticker count within expected band (e.g. 10–50);
  - weights sum ≈ 1.0.
- **Reconcile-before-trade:** always read actual IBKR positions + NAV first; never trust
  a cached view.
- **Audit log:** every run writes weights, intended orders, fills, and NAV to
  `trading/audit/` (gitignored) for reproducibility and post-trade review.

## Testing

- **Refactor acceptance:** backtest outputs identical pre/post extraction (Part 1).
- **Unit tests** for `src/strategy/`: scoring correctness; `topk_mcap_weights` sums to 1
  and respects the 10% cap; `ensemble_weights` convexity (sums to 1, cap preserved).
- **Snapshot smoke test:** fetch + assemble a current snapshot without error.
- **Execution unit tests:** `diff.py` produces correct share deltas from
  (weights, positions, NAV); `safety.py` blocks each violation class.
- **dryrun end-to-end:** `python -m trading.run rebalance --mode dryrun` runs the full
  pipeline against the fake broker and logs intended orders.

## Rollout plan

The user is going **straight to live for real trading** — paper is used only for a
one-time *mechanical* validation of the pipeline, not an extended paper-trading track
record. The compensating control for skipping a paper track record is a supervised first
live run.

1. **Build now** against `dryrun` (no account/Gateway needed) — validate the full pipeline
   logic end to end (snapshot → weights → diff → safety → intended orders).
2. User **installs IB Gateway**, logs in, **activates the free paper account**.
3. **Paper smoke-test (one-time):** run weights + rebalance against the paper account to
   confirm the *mechanics* work against a real IBKR connection — order placement,
   `MIDPRICE` behavior, fills, NAV reads, position reconciliation, audit logs. No
   multi-week paper track record.
4. **First live run is supervised + manual** with a **confirm-before-submit** prompt:
   `EXECUTION_MODE=live`, run the rebalance by hand, review the intended order list, then
   approve submission. Verify fills/positions afterward.
5. **Enable the automated Monday scheduler on live** only after that one supervised live
   rebalance runs clean.

## Out of scope (for now)

- Multi-strategy / multi-account support.
- Tax-lot optimization, wash-sale handling.
- A UI/dashboard (audit logs are files for now).

## Resolved decisions (2026-06-02)

- **Universe:** Sharadar `SP500` table preferred (verify it's in the bundle), Wikipedia
  fallback. Current set only.
- **Execution:** passive bid/ask → MIDPRICE → terminal cross ladder; instrument realized
  vs midpoint to validate the savings empirically.
- **Fractional shares:** use them (enable the IBKR permission); whole-share rounding as
  fallback where an order type doesn't support fractionals.
- **Deployment:** always-on VPS + IBC/IBeam headless Gateway + cron/systemd timers +
  failure alerting. Code stays deployment-agnostic.

## Open items to confirm during implementation

- Confirm Sharadar `SP500` table is included in the current NDL subscription.
- Confirm IBKR fractional + `MIDPRICE` order-type compatibility (else whole-share for the
  midpoint stage).
- Tune ladder timings (passive wait window, terminal-cross cutoff) during paper testing.
- Choose VPS provider/sizing and IBC vs IBeam for the headless Gateway.
