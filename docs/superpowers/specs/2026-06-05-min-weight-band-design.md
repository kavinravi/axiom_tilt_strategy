# Min-weight band allocator — design

**Date:** 2026-06-05
**Status:** Approved (design); ready for implementation planning
**Author:** brainstorming session (kavinravi)
**Branch:** `experiment/min-weight-band`
**Related:** `src/strategy/allocate.py` (current ensemble blend — what this replaces),
`experiments/regime_k_selector.py` (the walk-forward harness this mirrors),
`trading/weights.py` (live path — ported to only AFTER this validates).

---

## 1. Context & problem

The current system scores the full S&P 500, then a regime LightGBM ("k-selector")
emits a probability distribution over `K ∈ {10,20,30,50}` (how concentrated to be).
Final weights are a **convex blend** `w(i) = Σ_K p(K)·w_K(i)` of the per-K
mcap-weighted (10% cap) portfolios (`allocate.py:ensemble_weights`).

**The blend manufactures a long tail of sub-2% dust.** A name ranked ~#45 only earns
weight in the K=50 branch, so it lands at fractions of a percent. Measured on the two
frozen files:

| Friday | n | max | smallest | **# under 2%** |
|---|---|---|---|---|
| 2026-05-29 | 50 | 10.0% | 0.03% | **35 of 50** |
| 2026-06-05 | 50 | 10.0% | 0.01% | **34 of 50** |

**New rules (from the strategy owner):** every position ≥ **2%**, ≤ **10%**, holding
count `N ∈ [10, 50]`, weights sum to 1. These are self-consistent: `10·10% = 100%`
and `50·2% = 100%`, so the [2%,10%] band *mathematically* pins `N` to [10,50]. The
holding count stops being a model choice and becomes an **output** of a constrained
allocator.

**Key consequence:** a 2% floor + 10% cap squeezes the mcap tilt. At N=10 every name
is forced to exactly 10%; at N=50 every name is forced to exactly 2% (equal weight).
The tilt only breathes at intermediate counts (N=20 → avg 5%, range 2–10%). `N`
becomes the main lever, the tilt a secondary one.

## 2. Goal & non-goals

**Goal:** decide, via a clean walk-forward backtest, whether a min-weight band
allocator that obeys the new rules is a *valid* replacement for the ensemble blend —
i.e. doesn't materially hurt risk-adjusted performance. If it passes and the owner
signs off, port it into prod (`src/strategy` + `trading/`) as a separate step.

**Non-goals:**
- No change to the factor score (`score_universe`), regime features, universe
  selection, data pipeline, or execution/ladder.
- **No edits to `src/strategy/*` or `trading/*` during R&D** (see §3).
- No continuous-N optimizer (the rejected "Option C"); only revisited if this fails.

## 3. Isolation strategy

The live system shares code: `trading/weights.py` imports
`src.strategy.allocate.ensemble_weights`, and the backtest imports `src/strategy/*`.
Editing those files changes prod the moment R&D starts.

- Work on branch `experiment/min-weight-band` (off `trading-codebase`).
- **All new code lives in `experiments/min_weight_band/`.** It imports only the
  *unchanged* shared primitives — `load_data`, `score_universe`,
  `build_regime_features`, `make_k_classifier`, `per_k_weights_and_returns` (or a
  band-aware local copy) — and brings its own band allocator + walk-forward script.
- **Zero edits to `src/strategy/*` or `trading/*`** while validating. The live system
  stays byte-for-byte identical and runnable all weekend (if the owner wants Friday's
  old weights, `python -m trading.run weights` still works on the validated path).
- Revert = `rm -rf experiments/min_weight_band/` or `git checkout trading-codebase`.
- A git *worktree* is **not** used: the 1.2 GB panel (`data/processed/panel/`,
  `training_panel/`) is gitignored and lives only in this checkout, so a fresh
  worktree would have no data.

The 1.2 GB panel never moves; the experiment reads it in place.

## 4. The band allocator (core change)

### 4.1 `band_water_fill(mcaps, floor=0.02, cap=0.10)`

Project the mcap-proportional target onto the feasible set
`{w : floor ≤ w_i ≤ cap, Σ w_i = 1}`. This is the existing
`src/utils/rl_env.py:project_to_simplex` water-fill, generalized to clamp a **floor**
as well as a cap:

1. Base preference `b_i = mcap_i / Σ mcap` (equivalently `softmax(log mcap)`).
2. Iterate (bounded to `K + c` rounds): clamp `w_i = clip(·, floor, cap)`; names at
   floor or cap are *pinned*; the residual `1 − Σ_pinned w_i` is distributed across
   the strictly-interior (free) names proportional to their base preference; repeat
   until no new pin.
3. Final exact renormalization among free names absorbs float drift so pinned weights
   stay exactly on their bound.

Feasibility requires `K·floor ≤ 1 ≤ K·cap` ⇒ `10 ≤ K ≤ 50`, satisfied by the whole
grid (§4.3). Raise on infeasible input (mirrors the current cap-only guard).

**Edge behavior (expected, not a bug):** K=10 → all 10%; K=50 → all 2% (equal
weight); the tilt is active only for K=20/30/40.

### 4.2 Per-K portfolio + select-K inference

- `band_topk(scored_df, K)`: top-K by `score`, then `band_water_fill` on their mcaps
  ⇒ exactly **N=K holdings, each in [2%,10%]**.
- **Inference selects the modal K** — `K* = argmax_K p(K)` from the LightGBM — and
  returns `band_topk(scored, K*)`. **The ensemble blend is removed** (it was the dust
  factory). One K, one clean portfolio, no tail.

### 4.3 K grid

`K ∈ {10, 20, 30, 40, 50}` — adds the **40** that `constants.py` currently lacks,
honoring the owner's stated grid. Five LightGBM classes.

## 5. Retrain target

The per-K weekly returns change (band weighting ≠ cap-only weighting, and K=50 is now
forced equal-weight), so the labels change:

- Recompute per-K returns with `band_topk` weights.
- Label = `argmax-K` of per-K weekly forward return per Friday (same definition as
  today, new inputs).
- **Retrain the LightGBM walk-forward**, identical harness to
  `regime_k_selector.py`: walks 2007→2025, 1y validation / 1y test, early stopping
  (30 rounds), the exact `make_k_classifier` hyperparameters, `num_class=5`.

## 6. Backtest & acceptance

One walk-forward pass (mirroring `regime_k_selector.py`) produces a single OOS
weekly-return series per strategy — Fridays **2009→2025** (walk 1 tests 2009). Returns
are **net of 5 bps × turnover** (turnover = Σ_i |w_t(i) − w_{t-1}(i)|, matching the
existing harness/notebook convention). Per OOS Friday, record the selected-K band
portfolio's weekly return and its weight vector (for turnover).

**Report the head-to-head table over THREE windows** (masks on the same OOS series),
because each tells a different and complementary story:

- **Full 2009–2025** — the most honest record, but includes the GFC-recovery tail,
  which is an anomalous, hard-to-repeat regime.
- **2010–2025** — cuts the GFC off for a cleaner regime read, at the cost of
  completeness.
- **2025 only** — performance in the conditions most like today, but least robust
  (2025 was a strong year, so small sample + favorable tape).

No window is privileged; all three are presented side by side so the owner can weigh
honesty vs. relevance themselves.

**Head-to-head table (per window)** — candidate vs the thing it replaces vs market vs a static ref:

| strategy | ann | vol | **Sharpe** | Sortino | **maxDD** | turnover | avg N | avg min-wt |
|---|---|---|---|---|---|---|---|---|
| new select-K band (candidate) | | | | | | | | |
| old ensemble blend | | | | | | | | |
| SPY | | | | | | — | — | — |
| static K=30 band (ref) | | | | | | | | |

Also report the LightGBM's **K-pick frequency** per walk (sanity: is it actually
varying N, or collapsing to one K?).

**Acceptance:** the owner judges pass/fail; headline metrics are **net Sharpe** and
**max drawdown** vs the old ensemble. No hard threshold encoded. If it fails or shows a
large performance drop, we reconvene on architecture (the rejected Options B/C).

**Outputs** (all under `experiments/min_weight_band/`): the per-strategy weekly-return
parquets, the comparison table (printed + a small `results.md`/`json`), and the
retrained walk models if useful for inspection. Nothing written outside that subdir.

## 7. Porting to prod — SEPARATE step, only if it passes

Documented here so the second step is unambiguous; **not done during R&D**:

- `src/strategy/constants.py`: add `MIN_WEIGHT = 0.02`; `K_CANDIDATES = [10,20,30,40,50]`.
- `src/utils/rl_env.py`: extend `project_to_simplex` with an optional `min_weight`
  (or add `band_project`); keep the cap-only signature working for existing callers.
- `src/strategy/allocate.py`: add `band_topk` + `select_k_weights`; retire / bypass
  `ensemble_weights` on the live path.
- `src/strategy/historical.py`, `train.py`: band-aware per-K returns; retrain and
  overwrite `trading/models/k_selector.txt` (+ `.meta.json`).
- `trading/weights.py`: swap `ensemble_weights` → select-K band; **`validate_weights`
  gains a min-weight check** (every weight ≥ 2% − ε) and tightens n_holdings to [10,50].
- `trading/config.py`: `MIN_HOLDINGS=10`, `MAX_HOLDINGS=50`, add `MIN_WEIGHT`.
- Update tests: `test_allocate`, `test_weights`, `test_k_selector`,
  `test_backtest_acceptance`, `test_train_model`.

## 8. Risks

- **argmax-K turnover.** The modal K can jump week-to-week (e.g. 20→40), and a band
  portfolio resizes hard when N changes — higher turnover than the smoothing blend.
  Mitigation if it bites: select via rounded `E[K] = Σ p(K)·K` instead of argmax
  (smoother, still grid-snapped). Backtest reports turnover so we'll see it.
- **Tilt washout.** The 2% floor flattens the mcap tilt at grid extremes (K=10/50).
  Expected; the point of the new rules.
- **Concentration / downside.** Fewer, larger positions raise idiosyncratic vol; a
  long-only book in a drawdown is exactly the owner's worry. maxDD/Sortino capture it.
- **Noisier labels.** Band weighting compresses per-K return spread (K=50 → equal
  weight), so `argmax-K` labels may be noisier and the LightGBM may collapse to a
  couple of K's. The K-pick-frequency report flags this.
- Look-ahead / survivorship: inherited unchanged from the existing harness.
