# Live Weights Pipeline — Implementation Plan (Plan 2 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Stand up the `trading/` package so `python -m trading.run weights` fetches a current market snapshot, runs it through the validated strategy core, and prints this week's target weights `{ticker: weight}` with sanity checks. No broker/execution yet (Plan 3).

**Architecture:** A thin live data layer (Sharadar SP500/SF1/DAILY + FRED) assembles one current cross-section keyed by ticker with exactly the columns `src/strategy` expects, then reuses the Plan-1 core (`score_universe` → regime features → `predict_k_probs` → `ensemble_weights`) unchanged. The persisted production model (`trading/models/k_selector.txt`) is loaded, never retrained here. Weights are frozen to an audit file.

**Tech Stack:** Python 3.11+, pandas, numpy, `nasdaqdatalink` (Sharadar), `pandas_datareader` (FRED), lightgbm. All already installed.

---

## Data sources (RESOLVED — verified by live API probe 2026-06-02)

| Need | Source | Detail |
|------|--------|--------|
| Universe | `SHARADAR/SP500` | `action=='current'` → 503 tickers (the explicit current-membership marker). |
| Fundamentals: `revenue`, `fcf`, `assets` | `SHARADAR/SF1` dim `ARQ` | latest `datekey` per ticker (PIT). Also exposes `price`, `sharesbas`. |
| `marketcap` (live mcap) | `SHARADAR/DAILY` | latest `date` ≤ asof per ticker. Fresh (through prior trading day). |
| `prc`, `shrout` (fallback cols only) | `SHARADAR/SF1` | `price`, `sharesbas`. Used only if DAILY marketcap missing for a ticker. |
| Macro: `macro_vixcls`, `macro_dgs10`, `macro_t10y2y` | FRED via `pandas_datareader` | series `VIXCLS`, `DGS10`, `T10Y2Y`. |
| SPY regime history | FRED via `pandas_datareader` | series `SP500` (index level), resampled to weekly Fridays. |

**Do NOT use Sharadar SEP/SFP** — subscribed but returns 0 recent rows (stale). `nasdaqdatalink` API key is `NASDAQ_DATA_LINK_API_KEY` (in `.env`); set `ndl.ApiConfig.api_key = get_env("NASDAQ_DATA_LINK_API_KEY", required=True)` (see `src/data/ingest_sharadar.py`). FRED needs no key.

## Notes & gotchas

- **The rebalance date is a Friday.** `compute_target_weights(asof=None)` defaults to the most recent Friday ≤ today (`pd.Timestamp.today().normalize()`; if today isn't Friday, snap back to the previous Friday). Allow an explicit `asof` for testing on historical Fridays.
- **Regime features must match training.** Reuse `src.strategy.build_regime_features` exactly. It needs a weekly Friday DatetimeIndex (~40 weeks ending at the rebalance Friday), a SPY close series sampled at those Fridays, and a macro DataFrame over the same index. Take `.iloc[-1]` as the current regime row. The 4 SPY features are `shift(1)`-lagged inside the function; the 3 macro features are contemporaneous (current Friday's values).
- **Identifier column is `ticker`.** Pass `id_col="ticker"` to `score_universe` / `ensemble_weights`. (The core is identifier-agnostic.)
- **`score_universe` references `df["prc"]` and `df["shrout"]`** even when `marketcap` is present (pandas evaluates the fallback branch), so those columns MUST exist in the snapshot.
- **Network calls are real** (Sharadar + FRED). Add a simple retry (3 attempts, short backoff) around each external call — FRED occasionally times out. Keep `paginate=True` on Sharadar `get_table`.
- **Multi-class tickers** (e.g. `BRK.B`, `GOOG`/`GOOGL`) come straight from the SP500 table and join to SF1/DAILY by the same ticker string (same vendor) — no special handling, but log any universe tickers that fail to get fundamentals or marketcap.
- **Audit dir is gitignored.** Add `trading/audit/` to `.gitignore`. `trading/models/` stays tracked.
- Run from repo root; tests: `python -m pytest tests/trading/<file> -v`.

## File structure

```
trading/
  __init__.py
  config.py            paths, model path, Sharadar table names, FRED series map,
                       REGIME_HISTORY_WEEKS, EXECUTION_MODE scaffold, sanity bounds
  data/
    __init__.py
    universe.py        current S&P 500 tickers from SHARADAR/SP500
    sources.py         Sharadar SF1/DAILY + FRED macro/SPY wrappers (with retry)
    snapshot.py        assemble one current cross-section keyed by ticker
  regime.py            build the current-Friday regime feature row (reuses core)
  weights.py           full pipeline: snapshot -> scores -> regime -> probs -> weights -> freeze
  run.py               CLI: python -m trading.run weights [--asof YYYY-MM-DD]
  README.md
  models/k_selector.txt   (already exists from Plan 1)
  audit/                  (gitignored) frozen weights + run logs
tests/trading/
  test_universe.py
  test_snapshot.py
  test_regime.py
  test_weights.py
  test_live_smoke.py    (marked slow; hits real APIs)
```

---

## Task 1: Scaffold `trading/` package + `config.py`

**Files:** Create `trading/__init__.py`, `trading/config.py`, `trading/data/__init__.py`, `tests/trading/__init__.py`, `tests/trading/test_config.py`. Modify `.gitignore`.

- [ ] **Step 1: Write `trading/config.py`**

```python
"""Configuration for the live trading system (single place for paths + sources)."""
from __future__ import annotations

from src.strategy.constants import K_CANDIDATES, MAX_WEIGHT
from src.utils.io import repo_root

REPO_ROOT = repo_root()
TRADING_DIR = REPO_ROOT / "trading"
MODEL_PATH = TRADING_DIR / "models" / "k_selector.txt"
AUDIT_DIR = TRADING_DIR / "audit"
WEIGHTS_DIR = AUDIT_DIR / "weights"

# Execution mode scaffold (Plan 3+ uses paper/live; Plan 2 only computes weights).
EXECUTION_MODE = "dryrun"  # one of: dryrun | paper | live

# Sharadar tables (Nasdaq Data Link)
SHARADAR_SP500 = "SHARADAR/SP500"
SHARADAR_SF1 = "SHARADAR/SF1"
SHARADAR_DAILY = "SHARADAR/DAILY"
SF1_DIMENSION = "ARQ"  # As-Reported Quarterly, matches the backtest panel

# FRED series -> snapshot column names (via pandas_datareader)
FRED_MACRO_SERIES = {"VIXCLS": "macro_vixcls", "DGS10": "macro_dgs10", "T10Y2Y": "macro_t10y2y"}
FRED_SPY_SERIES = "SP500"  # S&P 500 index level (SPY ETF not freshly available in Sharadar)

# Regime feature window: enough weekly Fridays for the 26w vol + shift(1).
REGIME_HISTORY_WEEKS = 40

# Sanity bounds for the weights output
MIN_HOLDINGS = 10
MAX_HOLDINGS = 503
WEIGHT_SUM_TOL = 1e-6
```

- [ ] **Step 2: Write empty `trading/__init__.py`, `trading/data/__init__.py`, `tests/trading/__init__.py`** (docstring only).

- [ ] **Step 3: Add `trading/audit/` to `.gitignore`**

Append to `.gitignore`:
```
# Live trading run artifacts (regeneratable; may contain account-specific data)
trading/audit/
```

- [ ] **Step 4: Write `tests/trading/test_config.py`**

```python
def test_config_constants():
    from trading import config
    assert config.MODEL_PATH.name == "k_selector.txt"
    assert config.SF1_DIMENSION == "ARQ"
    assert set(config.FRED_MACRO_SERIES.values()) == {"macro_vixcls", "macro_dgs10", "macro_t10y2y"}
    assert config.REGIME_HISTORY_WEEKS >= 30
    assert config.MAX_WEIGHT == 0.10
```

- [ ] **Step 5: Run + commit**

Run: `python -m pytest tests/trading/test_config.py -v` → PASS
```bash
git add trading/__init__.py trading/config.py trading/data/__init__.py tests/trading/ .gitignore
git commit -m "feat(trading): scaffold trading package + config"
```

---

## Task 2: `trading/data/universe.py` — current S&P 500 tickers

**Files:** Create `trading/data/universe.py`, `tests/trading/test_universe.py`.

- [ ] **Step 1: Write the failing test** (pure parsing logic, no network)

```python
import pandas as pd
from trading.data.universe import current_members_from_sp500_table


def test_current_members_from_action_table():
    # 'current' is the explicit membership marker; reconstruction must agree.
    df = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01", "2020-01-01"]),
        "action": ["added", "removed", "current", "current"],
        "ticker": ["AAA", "AAA", "BBB", "CCC"],
    })
    members = current_members_from_sp500_table(df)
    assert members == ["BBB", "CCC"]  # AAA added then removed; BBB/CCC current
```

- [ ] **Step 2: Run → FAIL** (`ModuleNotFoundError`).

- [ ] **Step 3: Write `trading/data/universe.py`**

```python
"""Current S&P 500 membership from the Sharadar SP500 constituents table."""
from __future__ import annotations

import pandas as pd

from trading.config import SHARADAR_SP500


def current_members_from_sp500_table(sp500_df: pd.DataFrame) -> list[str]:
    """Return the sorted current member tickers from a SHARADAR/SP500 action frame.

    The table marks current members explicitly with action=='current'. We use
    that set directly (it is the vendor's current-membership snapshot)."""
    cur = sp500_df.loc[sp500_df["action"] == "current", "ticker"]
    return sorted(cur.dropna().astype(str).unique().tolist())


def get_current_sp500_tickers(ndl=None) -> list[str]:
    """Fetch + parse current S&P 500 tickers from Sharadar. ~503 names."""
    if ndl is None:
        import nasdaqdatalink as ndl  # noqa: PLC0415
        from src.utils.env import get_env
        ndl.ApiConfig.api_key = get_env("NASDAQ_DATA_LINK_API_KEY", required=True)
    df = ndl.get_table(SHARADAR_SP500, paginate=True)
    return current_members_from_sp500_table(df)
```

- [ ] **Step 4: Run → PASS.**

- [ ] **Step 5: Commit** `git commit -m "feat(trading): current S&P 500 universe from Sharadar SP500"`

---

## Task 3: `trading/data/sources.py` — Sharadar + FRED wrappers

**Files:** Create `trading/data/sources.py`, `tests/trading/test_sources.py`.

Each fetch wrapper takes an optional injected client (`ndl`) so tests can pass a fake. Add a small `_retry(fn, attempts=3)` helper (catch Exception, short `time.sleep` backoff, re-raise on final).

Functions (signatures are the contract — later tasks depend on them):
- `latest_fundamentals(tickers, ndl=None) -> pd.DataFrame` — columns `["ticker","datekey","revenue","fcf","assets","price","sharesbas"]`, one row per ticker = the row with the max `datekey` (SF1 ARQ). Query `SHARADAR/SF1` with `ticker=<chunks of 100>`, `dimension="ARQ"`, `paginate=True`; concat; sort by `datekey`; `drop_duplicates("ticker", keep="last")`.
- `latest_marketcap(tickers, asof, ndl=None) -> pd.DataFrame` — columns `["ticker","date","marketcap"]`, one row per ticker = latest `date` ≤ `asof` from `SHARADAR/DAILY` (query with `date={"lte": asof_str}`, chunked tickers, paginate; sort by date; `drop_duplicates("ticker", keep="last")`).
- `fetch_macro_history(index, end) -> pd.DataFrame` — DataFrame indexed by `index` (the weekly Friday DatetimeIndex) with columns `macro_vixcls,macro_dgs10,macro_t10y2y`. Pull FRED `VIXCLS,DGS10,T10Y2Y` from `index.min()-90d` to `end`, forward-fill, reindex to `index`. Rename via `FRED_MACRO_SERIES`.
- `fetch_spy_weekly(index, end) -> pd.Series` — FRED `SP500` from `index.min()-30d` to `end`, ffill, reindex to `index` → close series named "close".

- [ ] **Step 1: Write `tests/trading/test_sources.py`** using a `FakeNDL` whose `get_table` returns canned frames, asserting:
  - `latest_fundamentals` returns exactly one row per ticker and picks the max-`datekey` row.
  - `latest_marketcap` picks the latest `date ≤ asof` per ticker.
  - (Macro/SPY are FRED-backed → cover them in the live smoke test, Task 8, not here.)

```python
import pandas as pd
from trading.data import sources


class FakeNDL:
    def __init__(self, frame): self._frame = frame
    def get_table(self, name, **kw): return self._frame.copy()


def test_latest_fundamentals_picks_max_datekey():
    frame = pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB"],
        "datekey": pd.to_datetime(["2025-05-01", "2026-02-01", "2026-01-15"]),
        "revenue": [10, 20, 5], "fcf": [1, 2, 1], "assets": [100, 110, 50],
        "price": [9, 11, 4], "sharesbas": [1000, 1000, 500],
    })
    out = sources.latest_fundamentals(["AAA", "BBB"], ndl=FakeNDL(frame))
    assert len(out) == 2
    assert out.set_index("ticker").loc["AAA", "revenue"] == 20  # the 2026-02-01 row


def test_latest_marketcap_picks_latest_on_or_before_asof():
    frame = pd.DataFrame({
        "ticker": ["AAA", "AAA", "BBB"],
        "date": pd.to_datetime(["2026-05-29", "2026-06-01", "2026-06-01"]),
        "marketcap": [100, 110, 50],
    })
    out = sources.latest_marketcap(["AAA", "BBB"], asof=pd.Timestamp("2026-06-02"), ndl=FakeNDL(frame))
    assert out.set_index("ticker").loc["AAA", "marketcap"] == 110
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement `trading/data/sources.py`** per the signatures above (chunk tickers by 100 like `src/data/ingest_sharadar.py`; the `_chunk` helper can be copied). Inject `ndl`; when `None`, build it from the env key.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `git commit -m "feat(trading): Sharadar + FRED data source wrappers"`

---

## Task 4: `trading/data/snapshot.py` — assemble the current cross-section

**Files:** Create `trading/data/snapshot.py`, `tests/trading/test_snapshot.py`.

`build_snapshot(asof=None, ndl=None) -> pd.DataFrame` returns one row per current S&P 500 ticker with columns:
`["ticker","date","prc","shrout","marketcap","revenue","fcf","assets"]` where `date` = the rebalance Friday (asof), `prc`=SF1 `price`, `shrout`=SF1 `sharesbas`, `marketcap`=DAILY marketcap (NaN if missing → `score_universe` falls back to prc*shrout). Log any tickers dropped for missing fundamentals.

Also expose `most_recent_friday(today=None) -> pd.Timestamp`.

- [ ] **Step 1: Write `tests/trading/test_snapshot.py`** with an `assemble_snapshot(universe, fundamentals, marketcaps, asof)` PURE helper (no network) that does the merging, and test it on small frames:
  - resulting columns exactly the required set;
  - `marketcap` taken from DAILY; `prc`/`shrout` from SF1;
  - a universe ticker missing fundamentals is dropped (logged);
  - `date` equals `asof` for all rows.
  - Test `most_recent_friday(pd.Timestamp("2026-06-03"))` (a Wednesday) == `2026-05-29`; and on a Friday returns that Friday.

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** `snapshot.py`: `assemble_snapshot` (pure merge) + `build_snapshot` (fetch universe→sources→assemble) + `most_recent_friday`.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `git commit -m "feat(trading): build current cross-section snapshot"`

---

## Task 5: `trading/regime.py` — current-Friday regime feature row

**Files:** Create `trading/regime.py`, `tests/trading/test_regime.py`.

`build_current_regime_row(asof, spy_weekly=None, macro=None) -> np.ndarray` returns the 7 `REGIME_FEATURES` for `asof`:
- weekly Friday index = `pd.date_range(end=asof, periods=REGIME_HISTORY_WEEKS, freq="W-FRI")`
- `spy_at` = `fetch_spy_weekly(index, asof)` (injectable for tests)
- `mbd` = `fetch_macro_history(index, asof)` (injectable for tests)
- `regime_df = build_regime_features(index, spy_at, mbd)` (the Plan-1 core function)
- return `regime_df.iloc[-1].to_numpy(dtype=float)`

- [ ] **Step 1: Write `tests/trading/test_regime.py`** passing synthetic `spy_weekly` (a rising weekly close Series over the index) and `macro` (constant DataFrame), asserting the returned row has length 7, is finite, and column order matches `REGIME_FEATURES`. Verify it equals `build_regime_features(index, spy_weekly, macro).iloc[-1]` (consistency with the core).
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** `regime.py` (inject `spy_weekly`/`macro`; when `None`, call `sources.fetch_spy_weekly`/`fetch_macro_history`).
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `git commit -m "feat(trading): current-Friday regime feature row"`

---

## Task 6: `trading/weights.py` — the full weights pipeline

**Files:** Create `trading/weights.py`, `tests/trading/test_weights.py`.

`compute_target_weights(asof=None, snapshot=None, regime_row=None, model=None) -> dict` (inject snapshot/regime/model for tests):
1. `asof = asof or most_recent_friday()`
2. `snapshot = snapshot if snapshot is not None else build_snapshot(asof)`
3. `scored = score_universe(snapshot, id_col="ticker")`
4. `regime_row = regime_row if regime_row is not None else build_current_regime_row(asof)`
5. `model = model or load_model(MODEL_PATH)`
6. `k_probs = predict_k_probs(model, regime_row)`
7. `weights = ensemble_weights(scored, k_probs, id_col="ticker")`
8. `freeze_weights(weights, k_probs, asof)` → writes `trading/audit/weights/<asof>.json` (`{"asof":..., "k_probs":..., "weights":...}`); make `WEIGHTS_DIR` if needed.
9. return a result dict: `{"asof", "weights", "k_probs", "n_holdings", "weight_sum", "max_weight"}`.

Also `validate_weights(result) -> list[str]` returning a list of human-readable problems (empty == OK): sum within `WEIGHT_SUM_TOL` of 1; max ≤ `MAX_WEIGHT + 1e-9`; `MIN_HOLDINGS ≤ n ≤ MAX_HOLDINGS`.

- [ ] **Step 1: Write `tests/trading/test_weights.py`**:
  - Build a synthetic scored snapshot (≥60 tickers with `ticker,score,mcap`, plus the raw columns so `score_universe` is exercised) and a synthetic `regime_row`; load the REAL model from `MODEL_PATH` (skip if missing).
  - Call `compute_target_weights(asof=<a Friday>, snapshot=<synthetic>, regime_row=<synthetic>)`.
  - Assert: `weight_sum ≈ 1`, `max_weight ≤ 0.10+1e-9`, `MIN_HOLDINGS ≤ n_holdings ≤ MAX_HOLDINGS`, `set(k_probs) == {10,20,30,50}` and probs sum ≈ 1, `validate_weights(result) == []`, and the audit JSON file was written.
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** `weights.py`.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `git commit -m "feat(trading): target-weight pipeline + freeze + validation"`

---

## Task 7: `trading/run.py` — CLI

**Files:** Create `trading/run.py`, `trading/README.md`, `tests/trading/test_run.py`.

`python -m trading.run weights [--asof YYYY-MM-DD]`:
1. `result = compute_target_weights(asof=...)`
2. Print a sorted table (ticker, weight%) descending; print `k_probs`, `asof`, holdings count, weight sum, max weight.
3. Run `validate_weights`; print "✓ sanity checks passed" or the list of problems; exit non-zero if any problem.

- [ ] **Step 1: Write `tests/trading/test_run.py`** that calls the run module's `format_report(result)` PURE helper on a canned result dict and asserts the rendered text contains the holdings count, the weight-sum line, and the top ticker. (Keep network out of this unit test.)
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** `run.py` (argparse subcommand `weights`; `format_report`; `main`) and a short `trading/README.md` documenting `python -m trading.run weights` and the data sources.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `git commit -m "feat(trading): run CLI (weights) + README"`

---

## Task 8: Live end-to-end smoke test

**Files:** Create `tests/trading/test_live_smoke.py`.

- [ ] **Step 1: Write the smoke test** (marked slow; skip if no API key or no model):

```python
import os
import pytest
from trading.config import MODEL_PATH


@pytest.mark.slow
@pytest.mark.skipif(not os.getenv("NASDAQ_DATA_LINK_API_KEY") or not MODEL_PATH.exists(),
                    reason="needs Sharadar key + persisted model")
def test_live_weights_end_to_end():
    from trading.weights import compute_target_weights, validate_weights
    result = compute_target_weights()  # hits Sharadar + FRED for the current Friday
    assert validate_weights(result) == [], result
    assert 10 <= result["n_holdings"] <= 503
    assert abs(result["weight_sum"] - 1.0) < 1e-6
    assert result["max_weight"] <= 0.10 + 1e-9
```

- [ ] **Step 2: Run it for real** from repo root: `python -m pytest tests/trading/test_live_smoke.py -v -m slow` (also run `python -m trading.run weights` and eyeball the table). Confirm it produces a sane current-week portfolio. If a data hiccup occurs (e.g. FRED timeout), the retry should handle it; if a source is genuinely empty, report it.
- [ ] **Step 3: Commit** `git commit -m "test(trading): live end-to-end weights smoke test"`

---

## Final verification
- [ ] `python -m pytest tests/ -v` — Plan-1 tests (22) + Plan-2 unit tests all pass.
- [ ] `python -m trading.run weights` prints this week's target weights with passing sanity checks.

## Self-Review (completed during planning)

**Spec coverage (vs spec Parts 3-5, minus execution):** Part 3 `trading/` structure (config/data/snapshot/run) → Tasks 1-7. Part 4 live snapshot (universe, fundamentals, prices/mcap, macro, SPY history; CRSP dropped) → Tasks 2-5, sources RESOLVED above. Part 5 dry-run output (weights table + sanity checks: sum≈1, max≤10%, holding count) → Tasks 6-7. Broker/execution → deferred to Plan 3 (only the `EXECUTION_MODE` scaffold appears here). ✓

**Placeholder scan:** none — data sources are resolved and verified; every task has concrete signatures, test assertions, and commit messages.

**Type/name consistency:** `current_members_from_sp500_table`/`get_current_sp500_tickers`, `latest_fundamentals`/`latest_marketcap`/`fetch_macro_history`/`fetch_spy_weekly`, `assemble_snapshot`/`build_snapshot`/`most_recent_friday`, `build_current_regime_row`, `compute_target_weights`/`freeze_weights`/`validate_weights`, `format_report` — used consistently across tasks. Snapshot id column is `ticker`; `score_universe`/`ensemble_weights` always called with `id_col="ticker"`.
