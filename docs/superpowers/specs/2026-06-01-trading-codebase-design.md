# Spec: Self-contained live trading codebase (k-ensemble → IBKR)

> **SUPERSEDED (2026-06-02)** by `2026-06-02-live-trading-system-design.md`.
> That spec expands scope from "skeleton + live weights, stubbed broker" to the
> full live execution + scheduling system. Parts 1–2 (shared core, model
> persistence) carry over unchanged. Kept for history.

**Date:** 2026-06-01
**Status:** Superseded
**Owner:** kavinravi

## Goal

Stand up a `trading/` folder that, on demand, fetches a current market snapshot,
runs the validated **k-ensemble** strategy through a **shared strategy core**, and
prints **this week's target weights** `{ticker: weight}`. IBKR order execution is
stubbed behind a clean interface for a later milestone (gated on the brokerage
account existing). The existing backtest must produce **identical** numbers after
the refactor.

This is the first build phase of taking the k-ensemble model live via Interactive
Brokers. It deliberately stops short of placing orders.

## Background / current state

- The chosen production strategy is the **regime-conditional K-selector ensemble**
  (not the RL/PPO variant). See `reports/EXECUTIVE_SUMMARY.md`.
- The strategy math is currently **copy-pasted** between
  `experiments/regime_k_selector.py` and `experiments/regime_k_selector_weights.py`
  (`load_data`, factor scoring, regime features, walk-forward training). No single
  source of truth exists today.
- The code is **backtest-shaped**: it needs forward returns (`fwd_ret_5d`), trains
  17 walk-forward models inline, and reads bulk historical parquet panels. Live
  trading needs the opposite: one persisted model + a single current snapshot with
  no forward labels.
- The model is **never persisted** — retrained from scratch each run.
- The backtest used **annual** walk-forward retraining with a ~1-year embargo
  (model for test year N trained on data ≤ N-2). Validated Sharpe ≈ 1.28 was
  achieved with a model routinely 1–2 years stale relative to the traded period.

## Architecture decision

**Shared strategy core** (chosen over self-contained copy or weights-handoff).
A single `src/strategy/` module is the source of truth; both the backtest scripts
and the new `trading/` folder import it. This prevents the live system from
silently diverging from the validated backtest, and removes the existing
copy-paste duplication.

## Scope decision

**Skeleton + live weights.** Extract the shared core, scaffold `trading/`, and
build the live data-snapshot path so `trading/` can produce this week's target
weights. The broker layer is a clean stub/interface. Real IBKR execution,
scheduling, and paper-trading validation are out of scope for this phase.

---

## Part 1 — Extract the shared core (`src/strategy/`)

The single source of truth, imported by both backtest and live. Identifier-agnostic:
operates on a generic `id` column (backtest passes `permno`, live passes `ticker`).

- `src/strategy/factors.py`
  - `score_universe(snapshot_df, id_col="id") -> DataFrame[id, score]`
  - `sp = revenue / mcap` (clip lower 0); `fcfa = fcf / assets`
    (clip [-1, 2], NaN where assets ≤ 0); cross-sectional z-scores per date;
    `score = 0.5 * z_sp + 0.5 * z_fcfa`. NaN z-scores → 0.0.
- `src/strategy/k_selector.py`
  - `build_regime_features(...)` — VIX (`macro_vixcls`), 10Y (`macro_dgs10`),
    term spread (`macro_t10y2y`), trailing SPY 4w/12w returns, 12w/26w annualized
    vol. Shift(1) to avoid look-ahead, exactly as current code.
  - `train_model(regime_X, labels) -> model` (LGBM multiclass, same
    hyperparameters as current: n_estimators=500, lr=0.03, num_leaves=15,
    min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8, lambda_l2=2.0,
    num_class=4).
  - `save_model(model, path)` / `load_model(path)`.
  - `predict_k_probs(model, regime_row) -> {10: p, 20: p, 30: p, 50: p}`.
- `src/strategy/allocate.py`
  - `topk_mcap_weights(scored_df, K, max_weight=0.10) -> {id: weight}` —
    top-K by score, mcap weight via water-fill cap (wraps existing
    `src/utils/rl_env.py:project_to_simplex` on `log(mcap)`).
  - `ensemble_weights(scored_df, k_probs, K_candidates=[10,20,30,50]) -> {id: weight}`
    — convex combination `w(i) = Σ_K p(K) · w_K(i)`.
- `src/strategy/__init__.py` — exposes the clean public API.

**Refactor the experiments:** rewrite `experiments/regime_k_selector.py` and
`experiments/regime_k_selector_weights.py` to import from `src/strategy/` instead
of their inline copies. The forward-return/label/walk-loop code stays in the
backtest path; only the pure, forward-looking-free pieces move to the core.

**Acceptance test (no silent drift):** after the refactor, the backtest's weekly
returns (`weekly_regime_K_ensemble.parquet`) and `k_ensemble_weights.parquet` are
**byte-identical** (or numerically identical within float tolerance) to the
pre-refactor outputs.

## Part 2 — Persist a production model

- `src/strategy/train.py` (CLI): train one LGBM on **all history through the latest
  available date**, save to `trading/models/k_selector.txt` plus
  `k_selector.meta.json` (train date, feature list, label definition, K candidates,
  data vintage).
- **Retrain cadence: quarterly, configurable** (a `config.py` toggle allows weekly).
  Rationale: the backtest validated annual retraining with a 1–2 year lag, so
  quarterly with current data is strictly fresher; retraining is cheap; freezing
  the model between retrains makes each week's trades reproducible and auditable.

## Part 3 — `trading/` structure

```
trading/
  __init__.py
  config.py            paths, universe source, data-source toggles, retrain cadence
  data/
    __init__.py
    universe.py        current S&P 500 members -> tickers
    sources.py         thin wrappers: Sharadar (NDL), FRED, SPY history
    snapshot.py        assemble one current cross-section keyed by ticker
  models/
    k_selector.txt     persisted production model (+ .meta.json)
  broker/
    __init__.py
    base.py            Broker interface: get_positions(), get_nav(), place_orders()
    ibkr.py            ib_async implementation — STUB (NotImplementedError + TODO)
    dryrun.py          prints intended orders, no connection
  rebalance.py         snapshot -> core weights -> diff vs positions -> order list
  run.py               CLI entrypoint: python -m trading.run --dry-run
  README.md
```

## Part 4 — Live data snapshot

`trading/data/snapshot.py` builds one current cross-section reusing existing keys.
Columns must match what the shared core expects (`id`=ticker, `revenue`, `fcf`,
`assets`, `prc`, `shrout`/`marketcap`, plus macro columns), so `score_universe`
runs unchanged.

- **Universe:** current S&P 500 members → tickers. Source: a maintained
  constituents list (e.g., Wikipedia/dataset). [confirm source during impl]
- **Fundamentals** (revenue, FCF, total assets): latest Sharadar SF1 ARQ via
  `NASDAQ_DATA_LINK_API_KEY` — same vendor/methodology as the backtest. *(keep
  paying Sharadar.)*
- **Prices + shares outstanding** (for mcap): Sharadar SEP (same vendor, for
  consistency). Alternatives: yfinance / FMP.
- **Macro** (VIX, DGS10, T10Y2Y): FRED. *(free.)*
- **SPY history** (regime trailing ret/vol): ~30 weeks of weekly SPY closes from
  the same price source.
- **CRSP/WRDS dropped for live** — not needed for a current snapshot, and avoids
  the academic-license concern for commercial trading.

## Part 5 — Dry-run output

`python -m trading.run --dry-run`:
1. Build snapshot.
2. `score_universe` → `predict_k_probs` (loaded model) → `ensemble_weights`.
3. Print this week's target-weight table + sanity checks: weights sum ≈ 1.0,
   max ≤ 10%, holding count.
4. Broker = `dryrun` → prints "would place N orders (broker not yet connected)".

## Data cadence summary (clarification)

| Item | Refresh | Notes |
|------|---------|-------|
| Prices / market caps | Weekly (every rebalance) | Never stale |
| Fundamentals | Fetched weekly; values change ~quarterly | Latest filed; PIT, matches backtest |
| Model retrain | Quarterly (configurable to weekly) | Backtest used annual; quarterly is fresher |

## Out of scope (next milestones, post-account)

- Real `ib_async` connection, position/NAV reads, order diffing & submission.
- Scheduling/automation, IB Gateway / 2FA operations.
- Paper-trading validation runs.

## Testing

- **Refactor acceptance:** backtest outputs identical pre/post extraction.
- **Unit tests** for `src/strategy/`: scoring correctness; `topk_mcap_weights`
  sums to 1 and respects the 10% cap; `ensemble_weights` convexity (sums to 1,
  cap preserved).
- **Snapshot smoke test:** can fetch + assemble a current snapshot without error.

## Resolved defaults

- Folder name: `trading/`.
- Price source: Sharadar (same vendor as fundamentals) for consistency.
- Retrain cadence: quarterly, configurable.
- Live universe: maintained S&P 500 constituents list.
