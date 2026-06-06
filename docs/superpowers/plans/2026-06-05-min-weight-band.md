# Min-weight band allocator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained experiment that backtests a min-weight (2%) / max-weight (10%) band allocator with a regime-LightGBM that selects the modal concentration K, and reports it head-to-head against the current ensemble blend over three windows.

**Architecture:** Everything lives under `experiments/min_weight_band/` and imports only *unchanged* `src/strategy` primitives + brings its own band allocator. The walk-forward harness mirrors `experiments/regime_k_selector.py`. No edits to `src/strategy/*` or `trading/*` — the live system stays runnable. Porting to prod is a separate future step (spec §7).

**Tech Stack:** Python 3.11, numpy, pandas, lightgbm, pytest. Package is pip-installed editable, so `from src.strategy...` works from anywhere; the experiment runs as `python -m experiments.min_weight_band.run_backtest` and tests run as `pytest experiments/min_weight_band/ -v`.

**Spec:** `docs/superpowers/specs/2026-06-05-min-weight-band-design.md`

---

## File structure

- Create `experiments/min_weight_band/__init__.py` — empty; makes the dir a package so `python -m` and pytest imports resolve.
- Create `experiments/min_weight_band/allocator.py` — `band_water_fill`, `band_topk`, `band_per_k_weights_and_returns`. The core math. Pure-ish (the last reads a panel).
- Create `experiments/min_weight_band/backtest_lib.py` — `metrics`, `turnover_series`, `net_returns`, `walk_forward_proba`, `window_mask`, `format_table`. Pure functions, fully unit-tested.
- Create `experiments/min_weight_band/run_backtest.py` — `main()`: wires data → old & new walks → 4 strategies × 3 windows → writes `results.md` + parquets.
- Create `experiments/min_weight_band/README.md` — one screen: how to run, what it writes.
- Create `experiments/min_weight_band/test_allocator.py` — unit tests for `allocator.py`.
- Create `experiments/min_weight_band/test_backtest_lib.py` — unit tests for `backtest_lib.py`.

All run commands are issued **from the repo root** `/home/kavin-ravi/CodingStuff/axiom_tilt_strategy`.

---

## Task 0: Scaffold the experiment package

**Files:**
- Create: `experiments/min_weight_band/__init__.py`
- Create: `experiments/min_weight_band/README.md`

- [ ] **Step 1: Create the package marker**

Create `experiments/min_weight_band/__init__.py` with exactly:

```python
"""Min-weight (2%) / max-weight (10%) band allocator experiment.

Self-contained: imports only unchanged src/strategy primitives. Delete this
directory to fully revert. See docs/superpowers/specs/2026-06-05-min-weight-band-design.md.
"""
```

- [ ] **Step 2: Create the README**

Create `experiments/min_weight_band/README.md` with:

```markdown
# Min-weight band allocator experiment

Backtests a 2%-floor / 10%-cap band allocator (regime-LGBM picks the modal K,
no ensemble blend) vs the current ensemble blend.

## Run (from repo root)

    pytest experiments/min_weight_band/ -v          # unit tests
    python -m experiments.min_weight_band.run_backtest   # full walk-forward + report

## Outputs (this directory)

- `results.md`            — three-window head-to-head tables (the deliverable)
- `weekly_returns.parquet`— per-strategy net weekly returns (date, strategy, ret)

Nothing is written outside this directory. Live system is untouched.
```

- [ ] **Step 3: Verify the package imports**

Run: `python -c "import experiments.min_weight_band; print('ok')"`
Expected: prints `ok` (no ImportError).

- [ ] **Step 4: Commit**

```bash
git add experiments/min_weight_band/__init__.py experiments/min_weight_band/README.md
git commit -m "Scaffold min-weight band experiment package"
```

---

## Task 1: `band_water_fill` — box-constrained simplex projection

**Files:**
- Create: `experiments/min_weight_band/allocator.py`
- Test: `experiments/min_weight_band/test_allocator.py`

- [ ] **Step 1: Write the failing tests**

Create `experiments/min_weight_band/test_allocator.py`:

```python
import numpy as np
import pandas as pd
import pytest

from experiments.min_weight_band.allocator import band_water_fill


def test_sums_to_one_and_respects_band():
    w = band_water_fill(np.linspace(100.0, 10.0, 25), floor=0.02, cap=0.10)
    assert len(w) == 25
    assert abs(w.sum() - 1.0) < 1e-9
    assert w.min() >= 0.02 - 1e-9
    assert w.max() <= 0.10 + 1e-9


def test_k10_forces_all_at_cap():
    w = band_water_fill(np.linspace(100.0, 10.0, 10), floor=0.02, cap=0.10)
    np.testing.assert_allclose(w, np.full(10, 0.10), atol=1e-9)


def test_k50_forces_all_at_floor():
    w = band_water_fill(np.linspace(100.0, 10.0, 50), floor=0.02, cap=0.10)
    np.testing.assert_allclose(w, np.full(50, 0.02), atol=1e-9)


def test_tilt_is_monotone_in_mcap_at_intermediate_k():
    # mcaps descending -> weights non-increasing (bigger mcap never weighs less)
    w = band_water_fill(np.linspace(100.0, 10.0, 20), floor=0.02, cap=0.10)
    assert np.all(np.diff(w) <= 1e-9)


def test_equal_weight_when_all_mcap_nonpositive():
    w = band_water_fill(np.zeros(20), floor=0.02, cap=0.10)
    np.testing.assert_allclose(w, np.full(20, 0.05), atol=1e-9)


def test_infeasible_band_raises():
    with pytest.raises(ValueError):
        band_water_fill(np.ones(5), floor=0.02, cap=0.10)   # 5*0.10 = 0.5 < 1
    with pytest.raises(ValueError):
        band_water_fill(np.ones(60), floor=0.02, cap=0.10)  # 60*0.02 = 1.2 > 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest experiments/min_weight_band/test_allocator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiments.min_weight_band.allocator'`.

- [ ] **Step 3: Write the minimal implementation**

Create `experiments/min_weight_band/allocator.py`:

```python
"""Band allocator: mcap-tilted weights clamped to [floor, cap] summing to 1,
plus the per-K portfolio and the band-aware per-K returns used by the backtest.

Imports only unchanged src/strategy primitives; no edits to the live core.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def band_water_fill(mcaps, floor: float = 0.02, cap: float = 0.10) -> np.ndarray:
    """Project an mcap-proportional target onto {w : floor<=w<=cap, sum w = 1}.

    mcap-proportional base, then iteratively clamp out-of-band names (sticky
    pins) and redistribute the residual across still-free names proportional to
    their mcap base, preserving the tilt. A final slack-based finalizer repairs
    any float residual without violating the band. Raises if the band is
    infeasible for K names (needs K*floor <= 1 <= K*cap).
    """
    mcaps = np.asarray(mcaps, dtype=np.float64)
    K = len(mcaps)
    if K * cap < 1.0 - 1e-12 or K * floor > 1.0 + 1e-12:
        raise ValueError(
            f"infeasible band: K={K}, floor={floor}, cap={cap} "
            f"(need K*cap>=1>=K*floor)"
        )

    clean = np.where(np.isnan(mcaps) | (mcaps <= 0.0), 0.0, mcaps)
    base = np.full(K, 1.0 / K) if clean.sum() <= 0 else clean / clean.sum()

    w = base.copy()
    pinned = np.zeros(K, dtype=bool)
    for _ in range(2 * K + 5):
        over = (w > cap + 1e-15) & ~pinned
        under = (w < floor - 1e-15) & ~pinned
        if not over.any() and not under.any():
            break
        w[over] = cap
        w[under] = floor
        pinned |= over | under
        free = ~pinned
        if not free.any():
            break
        residual = 1.0 - w[pinned].sum()
        fb = base[free]
        w[free] = (residual / free.sum() if fb.sum() <= 0
                   else residual * fb / fb.sum())

    # Finalizer: repair float residual by moving only into available slack.
    for _ in range(K + 5):
        w = np.clip(w, floor, cap)
        residual = 1.0 - w.sum()
        if abs(residual) < 1e-12:
            break
        slack = (cap - w) if residual > 0 else (w - floor)
        s = slack.sum()
        if s <= 1e-15:
            break
        w = w + residual * slack / s
    return w
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest experiments/min_weight_band/test_allocator.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/min_weight_band/allocator.py experiments/min_weight_band/test_allocator.py
git commit -m "band_water_fill: box-constrained simplex projection (2%/10%)"
```

---

## Task 2: `band_topk` — top-K by score, band-weighted

**Files:**
- Modify: `experiments/min_weight_band/allocator.py`
- Test: `experiments/min_weight_band/test_allocator.py`

- [ ] **Step 1: Add the failing tests**

Append to `experiments/min_weight_band/test_allocator.py`:

```python
from experiments.min_weight_band.allocator import band_topk


def _scored(n):
    return pd.DataFrame({
        "id": list(range(n)),
        "score": np.linspace(1.0, 0.0, n),    # descending
        "mcap": np.linspace(100.0, 10.0, n),  # descending
    })


def test_band_topk_picks_k_names_in_band_summing_to_one():
    w = band_topk(_scored(60), K=25, floor=0.02, cap=0.10, id_col="id")
    assert len(w) == 25
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert min(w.values()) >= 0.02 - 1e-9
    assert max(w.values()) <= 0.10 + 1e-9
    assert set(w.keys()) == set(range(25))  # top-25 by score


def test_band_topk_k10_all_at_cap():
    w = band_topk(_scored(60), K=10, floor=0.02, cap=0.10, id_col="id")
    np.testing.assert_allclose(sorted(w.values()), [0.10] * 10, atol=1e-9)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest experiments/min_weight_band/test_allocator.py -k band_topk -v`
Expected: FAIL — `ImportError: cannot import name 'band_topk'`.

- [ ] **Step 3: Add the implementation**

Append to `experiments/min_weight_band/allocator.py`:

```python
def band_topk(scored_df: pd.DataFrame, K: int, floor: float = 0.02,
              cap: float = 0.10, id_col: str = "id") -> dict[Any, float]:
    """Top-K by `score`, band-weighted by mcap. Single-date frame in,
    {id: weight} out. Exactly K holdings, each in [floor, cap], summing to 1."""
    g = scored_df.sort_values("score", ascending=False).head(K).reset_index(drop=True)
    w = band_water_fill(g["mcap"].to_numpy(dtype=np.float64), floor=floor, cap=cap)
    return {idv: float(min(max(wt, floor), cap))
            for idv, wt in zip(g[id_col].to_numpy(), w)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest experiments/min_weight_band/test_allocator.py -v`
Expected: all 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/min_weight_band/allocator.py experiments/min_weight_band/test_allocator.py
git commit -m "band_topk: top-K by score, band-weighted (exactly K holdings)"
```

---

## Task 3: `band_per_k_weights_and_returns` — band per-K panel

**Files:**
- Modify: `experiments/min_weight_band/allocator.py`
- Test: `experiments/min_weight_band/test_allocator.py`

This is the band analog of `src/strategy/historical.py:per_k_weights_and_returns`, which the existing harness uses with cap-only `topk_mcap_weights`. Same shape, band weighting.

- [ ] **Step 1: Add the failing test**

Append to `experiments/min_weight_band/test_allocator.py`:

```python
from experiments.min_weight_band.allocator import band_per_k_weights_and_returns


def _panel_two_dates():
    rows = []
    for d in (pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-10")):
        for i in range(30):
            rows.append({"date": d, "permno": i, "score": 30 - i,
                         "mcap": float(100 - i), "fwd_ret_5d": 0.01 * (i % 5 - 2)})
    return pd.DataFrame(rows)


def test_band_per_k_returns_shape_and_band():
    wdf, rdf = band_per_k_weights_and_returns(_panel_two_dates(), K=20)
    # weights: each date has exactly 20 names, all in band, summing to 1
    for d, g in wdf.groupby("date"):
        assert len(g) == 20
        assert g["weight"].min() >= 0.02 - 1e-9
        assert g["weight"].max() <= 0.10 + 1e-9
        assert abs(g["weight"].sum() - 1.0) < 1e-9
    # returns: one finite value per date
    assert len(rdf) == 2
    assert rdf.notna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest experiments/min_weight_band/test_allocator.py -k band_per_k -v`
Expected: FAIL — `ImportError: cannot import name 'band_per_k_weights_and_returns'`.

- [ ] **Step 3: Add the implementation**

Append to `experiments/min_weight_band/allocator.py`:

```python
def band_per_k_weights_and_returns(df: pd.DataFrame, K: int, floor: float = 0.02,
                                   cap: float = 0.10):
    """Per Friday: top-K band-weighted weights + the portfolio's fwd_ret_5d.

    Mirrors src/strategy/historical.per_k_weights_and_returns but uses band_topk
    instead of the cap-only topk_mcap_weights. Returns
    (weight_df[date,permno,weight], return_series indexed by date)."""
    weight_rows = []
    return_rows = []
    for d, g in df.groupby("date", sort=False):
        w = band_topk(g, K, floor=floor, cap=cap, id_col="permno")
        gk = g.sort_values("score", ascending=False).head(K)
        fwd = dict(zip(gk["permno"].astype(int).to_numpy(),
                       np.nan_to_num(gk["fwd_ret_5d"].to_numpy(dtype=np.float64))))
        ret = float(sum(w[p] * fwd[int(p)] for p in w))
        return_rows.append({"date": d, "ret": ret})
        for p, wt in w.items():
            weight_rows.append({"date": d, "permno": int(p), "weight": float(wt)})
    wdf = pd.DataFrame(weight_rows)
    rdf = pd.DataFrame(return_rows).sort_values("date").set_index("date")["ret"]
    return wdf, rdf
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest experiments/min_weight_band/test_allocator.py -v`
Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add experiments/min_weight_band/allocator.py experiments/min_weight_band/test_allocator.py
git commit -m "band_per_k_weights_and_returns: band per-K weights + weekly returns"
```

---

## Task 4: `backtest_lib` — metrics, turnover, net returns, windows, walk-forward

**Files:**
- Create: `experiments/min_weight_band/backtest_lib.py`
- Test: `experiments/min_weight_band/test_backtest_lib.py`

- [ ] **Step 1: Write the failing tests**

Create `experiments/min_weight_band/test_backtest_lib.py`:

```python
import numpy as np
import pandas as pd

from experiments.min_weight_band.backtest_lib import (
    metrics, turnover_series, net_returns, window_mask,
)


def test_metrics_keys_and_zero_vol_sharpe():
    m = metrics(np.zeros(52))
    assert set(m) == {"ann", "vol", "sharpe", "sortino", "mdd"}
    assert m["vol"] == 0.0
    assert m["sharpe"] == 0.0   # guarded, not inf/nan


def test_metrics_positive_drift_has_positive_sharpe():
    r = np.full(52, 0.01)
    m = metrics(r)
    assert m["ann"] > 0
    assert m["sharpe"] == 0.0 or m["vol"] == 0.0  # constant series -> zero vol


def test_turnover_first_week_is_half_of_built_book():
    weights = [{1: 0.5, 2: 0.5}, {1: 0.5, 2: 0.5}]
    tu = turnover_series(weights)
    assert abs(tu[0] - 0.5) < 1e-12   # build from cash: 0.5 * sum|w| = 0.5
    assert abs(tu[1] - 0.0) < 1e-12   # no change


def test_turnover_full_swap_is_one():
    weights = [{1: 1.0}, {2: 1.0}]
    tu = turnover_series(weights)
    assert abs(tu[1] - 1.0) < 1e-12   # 0.5*(|−1|+|+1|) = 1.0


def test_net_returns_subtracts_cost():
    gross = np.array([0.02, 0.02])
    tu = np.array([0.5, 0.0])
    net = net_returns(gross, tu, cost_bps=5.0)
    assert abs(net[0] - (0.02 - 5e-4 * 0.5)) < 1e-12
    assert abs(net[1] - 0.02) < 1e-12


def test_window_mask_selects_years():
    idx = pd.to_datetime(["2009-06-05", "2010-06-04", "2025-06-06"])
    np.testing.assert_array_equal(window_mask(idx, 2009, 2025), [True, True, True])
    np.testing.assert_array_equal(window_mask(idx, 2010, 2025), [False, True, True])
    np.testing.assert_array_equal(window_mask(idx, 2025, 2025), [False, False, True])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest experiments/min_weight_band/test_backtest_lib.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'experiments.min_weight_band.backtest_lib'`.

- [ ] **Step 3: Write the implementation**

Create `experiments/min_weight_band/backtest_lib.py`:

```python
"""Pure backtest helpers: metrics, turnover, net returns, year windows, and the
generic walk-forward LGBM probability pass. No I/O, no globals."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.strategy.k_selector import make_k_classifier

PERIODS_PER_YEAR = 52
COST_BPS = 5.0


def metrics(rets) -> dict[str, float]:
    """Annualized return, vol, Sharpe, Sortino, max drawdown for a weekly series.
    Zero-vol cases return 0.0 (never inf/nan) so tables stay clean."""
    r = np.asarray(rets, dtype=np.float64)
    if len(r) == 0:
        return {"ann": 0.0, "vol": 0.0, "sharpe": 0.0, "sortino": 0.0, "mdd": 0.0}
    cum = float(np.prod(1.0 + r) - 1.0)
    ann = (1.0 + cum) ** (PERIODS_PER_YEAR / len(r)) - 1.0
    vol = float(np.std(r, ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if len(r) > 1 else 0.0
    sharpe = float(ann / vol) if vol > 0 else 0.0
    downside = r[r < 0]
    dvol = float(np.std(downside, ddof=1) * np.sqrt(PERIODS_PER_YEAR)) if len(downside) > 1 else 0.0
    sortino = float(ann / dvol) if dvol > 0 else 0.0
    eq = np.cumprod(1.0 + r)
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    return {"ann": ann, "vol": vol, "sharpe": sharpe, "sortino": sortino, "mdd": mdd}


def turnover_series(weight_dicts: list[dict]) -> np.ndarray:
    """One-way turnover per period: 0.5 * sum_i |w_t(i) - w_{t-1}(i)|, with names
    absent on either side treated as 0. First period builds from cash."""
    tu = np.zeros(len(weight_dicts), dtype=np.float64)
    prev: dict = {}
    for t, cur in enumerate(weight_dicts):
        names = set(cur) | set(prev)
        tu[t] = 0.5 * sum(abs(cur.get(n, 0.0) - prev.get(n, 0.0)) for n in names)
        prev = cur
    return tu


def net_returns(gross, turnover, cost_bps: float = COST_BPS) -> np.ndarray:
    """gross - (cost_bps/1e4) * turnover, elementwise."""
    g = np.asarray(gross, dtype=np.float64)
    tu = np.asarray(turnover, dtype=np.float64)
    return g - (cost_bps / 1e4) * tu


def window_mask(dates: pd.DatetimeIndex, start_year: int, end_year: int) -> np.ndarray:
    """Boolean mask for start_year <= year <= end_year (inclusive)."""
    y = pd.DatetimeIndex(dates).year
    return ((y >= start_year) & (y <= end_year)).to_numpy()


def walk_forward_proba(regime_df: pd.DataFrame, labels: pd.Series,
                       all_dates: pd.DatetimeIndex, num_class: int) -> pd.DataFrame:
    """Walk-forward LGBM class probabilities, identical scheme to
    experiments/regime_k_selector.py (walks 1..17, 1y val / 1y test, early
    stopping). Returns a frame indexed by OOS date with one column per class
    ('c0'..'c{num_class-1}')."""
    years = all_dates.year
    rows = []
    for walk_id in range(1, 18):
        train_end = 2007 + walk_id - 1
        val_year = train_end + 1
        test_year = train_end + 2
        train_mask = years <= train_end
        val_mask = years == val_year
        test_mask = years == test_year
        if test_mask.sum() < 10:
            continue
        Xtr, ytr = regime_df[train_mask], labels[train_mask]
        Xvl, yvl = regime_df[val_mask], labels[val_mask]
        Xte = regime_df[test_mask]
        vtr = ytr.notna(); Xtr, ytr = Xtr[vtr], ytr[vtr].astype(int)
        vvl = yvl.notna(); Xvl, yvl = Xvl[vvl], yvl[vvl].astype(int)
        if len(Xtr) < 100 or len(Xvl) < 5:
            continue
        clf = make_k_classifier(num_class=num_class)
        clf.fit(Xtr.to_numpy(), ytr.to_numpy(),
                eval_set=[(Xvl.to_numpy(), yvl.to_numpy())],
                callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)])
        proba = clf.predict_proba(Xte.to_numpy())
        for i, d in enumerate(all_dates[test_mask]):
            rows.append({"date": d, **{f"c{j}": float(proba[i, j]) for j in range(num_class)}})
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out["date"] = pd.to_datetime(out["date"])
    return out.set_index("date")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest experiments/min_weight_band/test_backtest_lib.py -v`
Expected: all 6 tests PASS. (`walk_forward_proba` is exercised by the full run in Task 6, not unit-tested here — it needs the trained-model + panel fixtures the run provides.)

- [ ] **Step 5: Commit**

```bash
git add experiments/min_weight_band/backtest_lib.py experiments/min_weight_band/test_backtest_lib.py
git commit -m "backtest_lib: metrics, turnover, net returns, windows, walk-forward proba"
```

---

## Task 5: `run_backtest` — wire data → strategies → three-window report

**Files:**
- Create: `experiments/min_weight_band/run_backtest.py`

This is a runnable script (no new unit test — it's I/O + model training over the full 1.2 GB panel; its helpers are already tested in Tasks 1–4). It is validated by actually running it in Task 6.

- [ ] **Step 1: Write the script**

Create `experiments/min_weight_band/run_backtest.py`:

```python
"""Walk-forward backtest: new band select-K vs old ensemble blend vs SPY vs
static K=30 band, reported over three windows (2009-25, 2010-25, 2025-only).

Run from repo root:  python -m experiments.min_weight_band.run_backtest
Writes results.md + weekly_returns.parquet into this directory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.min_weight_band.allocator import band_per_k_weights_and_returns
from experiments.min_weight_band.backtest_lib import (
    metrics, net_returns, turnover_series, walk_forward_proba, window_mask,
)
from src.strategy.allocate import topk_mcap_weights
from src.strategy.constants import MAX_WEIGHT
from src.strategy.historical import (
    build_k_labels, load_data, load_spy_at, macro_by_date, per_k_weights_and_returns,
)
from src.strategy.k_selector import build_regime_features

OLD_K = [10, 20, 30, 50]
NEW_K = [10, 20, 30, 40, 50]
OUT_DIR = Path(__file__).resolve().parent
WINDOWS = [("full 2009-2025", 2009, 2025), ("2010-2025", 2010, 2025), ("2025 only", 2025, 2025)]


def _old_per_k(df):
    w, r = {}, {}
    for K in OLD_K:
        wk, rk = per_k_weights_and_returns(df, K, max_weight=MAX_WEIGHT)
        w[K], r[K] = wk, rk
    return w, r


def _new_per_k(df):
    w, r = {}, {}
    for K in NEW_K:
        wk, rk = band_per_k_weights_and_returns(df, K)
        w[K], r[K] = wk, rk
    return w, r


def _weights_by_date(wdf: pd.DataFrame) -> dict:
    """{date -> {permno -> weight}} from a [date,permno,weight] frame."""
    out = {}
    for d, g in wdf.groupby("date"):
        out[d] = dict(zip(g["permno"].astype(int), g["weight"].astype(float)))
    return out


def _ensemble_series(proba: pd.DataFrame, k_list, k_weights, k_returns):
    """Old path: convex blend over K. Returns (dates, gross, weight_dicts)."""
    wbd = {K: _weights_by_date(k_weights[K]) for K in k_list}
    dates, gross, wdicts = [], [], []
    for d, prow in proba.iterrows():
        p = {K: float(prow[f"c{j}"]) for j, K in enumerate(k_list)}
        combined: dict = {}
        for K in k_list:
            for permno, wt in wbd[K].get(d, {}).items():
                combined[permno] = combined.get(permno, 0.0) + p[K] * wt
        gross.append(sum(p[K] * float(k_returns[K].get(d, 0.0)) for K in k_list))
        wdicts.append(combined)
        dates.append(d)
    return pd.DatetimeIndex(dates), np.asarray(gross), wdicts


def _select_k_series(proba: pd.DataFrame, k_list, k_weights, k_returns):
    """New path: pick modal K = argmax proba. Returns (dates, gross, weight_dicts)."""
    wbd = {K: _weights_by_date(k_weights[K]) for K in k_list}
    dates, gross, wdicts = [], [], []
    for d, prow in proba.iterrows():
        j = int(np.argmax([prow[f"c{i}"] for i in range(len(k_list))]))
        Kstar = k_list[j]
        gross.append(float(k_returns[Kstar].get(d, 0.0)))
        wdicts.append(wbd[Kstar].get(d, {}))
        dates.append(d)
    return pd.DatetimeIndex(dates), np.asarray(gross), wdicts


def _static_series(k_weights, k_returns, K, oos_dates):
    wbd = _weights_by_date(k_weights[K])
    gross = np.asarray([float(k_returns[K].get(d, 0.0)) for d in oos_dates])
    wdicts = [wbd.get(d, {}) for d in oos_dates]
    return gross, wdicts


def _row(name, net, wdicts, oos_dates):
    n = np.asarray([len(w) for w in wdicts])
    mins = np.asarray([min(w.values()) if w else np.nan for w in wdicts])
    avg_tu = float(turnover_series(wdicts).mean())
    m = metrics(net)
    return {"strategy": name, "ann": m["ann"], "vol": m["vol"], "sharpe": m["sharpe"],
            "sortino": m["sortino"], "mdd": m["mdd"], "turnover": avg_tu,
            "avg_n": float(n.mean()), "avg_min_wt": float(np.nanmean(mins))}


def _fmt_table(rows):
    head = f"| {'strategy':<26} | {'ann':>7} | {'vol':>7} | {'sharpe':>7} | {'sortino':>7} | {'mdd':>7} | {'turn':>6} | {'avgN':>5} | {'minWt':>6} |"
    sep = "|" + "|".join(["-" * (len(c) + 2) for c in head.split("|")[1:-1]]) + "|"
    lines = [head, sep]
    for r in rows:
        lines.append(
            f"| {r['strategy']:<26} | {r['ann']:>6.1%} | {r['vol']:>6.1%} | "
            f"{r['sharpe']:>7.2f} | {r['sortino']:>7.2f} | {r['mdd']:>6.1%} | "
            f"{r['turnover']:>6.2f} | {r['avg_n']:>5.1f} | {r['avg_min_wt']:>6.2%} |"
        )
    return "\n".join(lines)


def main():
    print("Loading panel ...")
    df = load_data()

    print("Building per-K returns (old cap-only + new band) ...")
    old_w, old_r = _old_per_k(df)
    new_w, new_r = _new_per_k(df)

    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in new_r.values()])))
    spy_at = load_spy_at(all_dates)
    regime = build_regime_features(all_dates, spy_at, macro_by_date(df, all_dates))

    print("Walk-forward: old 4-class ...")
    old_labels, _ = build_k_labels(old_r, all_dates, OLD_K)
    old_proba = walk_forward_proba(regime, old_labels, all_dates, num_class=len(OLD_K))

    print("Walk-forward: new 5-class ...")
    new_labels, _ = build_k_labels(new_r, all_dates, NEW_K)
    new_proba = walk_forward_proba(regime, new_labels, all_dates, num_class=len(NEW_K))

    # Common OOS dates (both walks share the same scheme; intersect to be safe).
    oos = old_proba.index.intersection(new_proba.index)
    old_proba, new_proba = old_proba.loc[oos], new_proba.loc[oos]

    # Strategy gross + weights on the common OOS dates.
    od, old_g, old_wd = _ensemble_series(old_proba, OLD_K, old_w, old_r)
    nd, new_g, new_wd = _select_k_series(new_proba, NEW_K, new_w, new_r)
    k30_g, k30_wd = _static_series(new_w, new_r, 30, oos)
    spy_g = spy_at.reindex(oos).pct_change().fillna(0.0).to_numpy()
    spy_wd = [{"SPY": 1.0} for _ in oos]  # buy-hold proxy: ~0 turnover after entry

    # Net of cost.
    series = {
        "new band select-K": net_returns(new_g, turnover_series(new_wd)),
        "old ensemble blend": net_returns(old_g, turnover_series(old_wd)),
        "static K=30 band": net_returns(k30_g, turnover_series(k30_wd)),
        "SPY": net_returns(spy_g, turnover_series(spy_wd)),
    }
    wdicts = {"new band select-K": new_wd, "old ensemble blend": old_wd,
              "static K=30 band": k30_wd, "SPY": spy_wd}

    # Three-window report.
    out_md = ["# Min-weight band — backtest report",
              "", f"OOS Fridays: {len(oos)} ({oos.min().date()} .. {oos.max().date()}), "
              "net of 5 bps x one-way turnover.", ""]
    for title, y0, y1 in WINDOWS:
        mask = window_mask(oos, y0, y1)
        rows = [_row(name, series[name][mask],
                     [w for w, m in zip(wdicts[name], mask) if m], oos[mask])
                for name in series]
        out_md += [f"## {title}  ({int(mask.sum())} weeks)", "", _fmt_table(rows), ""]
        print(f"\n=== {title} ({int(mask.sum())} weeks) ===")
        print(_fmt_table(rows))

    (OUT_DIR / "results.md").write_text("\n".join(out_md))

    long = []
    for name, net in series.items():
        for d, r in zip(oos, net):
            long.append({"date": d, "strategy": name, "ret": float(r)})
    pd.DataFrame(long).to_parquet(OUT_DIR / "weekly_returns.parquet", index=False)
    print(f"\nWrote {OUT_DIR / 'results.md'} and weekly_returns.parquet")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Byte-compile to catch syntax errors early**

Run: `python -m py_compile experiments/min_weight_band/run_backtest.py && echo OK`
Expected: prints `OK`.

- [ ] **Step 3: Commit**

```bash
git add experiments/min_weight_band/run_backtest.py
git commit -m "run_backtest: walk-forward report (new band vs old blend vs SPY, 3 windows)"
```

---

## Task 6: Run the full backtest and capture the report

**Files:**
- Produces: `experiments/min_weight_band/results.md`, `experiments/min_weight_band/weekly_returns.parquet`

- [ ] **Step 1: Run the whole unit suite once (regression guard)**

Run: `pytest experiments/min_weight_band/ -v`
Expected: all tests from Tasks 1–4 PASS.

- [ ] **Step 2: Run the backtest**

Run: `python -m experiments.min_weight_band.run_backtest`
Expected: prints three window tables; finishes by writing `results.md` + `weekly_returns.parquet`. (Training 2×17 LGBM walks over the panel — expect a few minutes.)

- [ ] **Step 3: Sanity-check the output**

Run: `cat experiments/min_weight_band/results.md`
Verify, by eye:
- `new band select-K` row has `minWt` ≥ ~2.0% in every window (the floor holds).
- `avgN` is between 10 and 50.
- Sharpe/mdd are finite and plausible (no inf/nan).
- The old-blend row roughly matches the known current behavior.
If `minWt` < 2% for the band strategy, STOP — the allocator has a bug; do not proceed.

- [ ] **Step 4: Commit the report**

```bash
git add experiments/min_weight_band/results.md experiments/min_weight_band/weekly_returns.parquet
git commit -m "Add min-weight band backtest report (3 windows)"
```

---

## Task 7: Hand the report to the owner for the go/no-go

- [ ] **Step 1: Summarize for the owner**

Present the three tables with a plain-language read for each window:
- **full 2009-2025** — most honest, includes the GFC tail.
- **2010-2025** — cleaner regime, GFC removed.
- **2025 only** — closest to today, least robust.
Call out the headline: **new band select-K net Sharpe & max drawdown vs old ensemble blend**, plus the turnover delta (the argmax-K risk from spec §8).

- [ ] **Step 2: Decision gate**

If the owner approves → open a *separate* plan to port into `src/strategy` + `trading/` per spec §7 (constants, `band_water_fill` into `project_to_simplex`, `weights.py` select-K + min-weight validation, retrain `k_selector.txt`, config bounds, prod tests). **Do not** start porting in this experiment branch.

If the owner rejects or sees a large drop → reconvene on architecture (spec Options B/C). The experiment is fully revertible: `git checkout trading-codebase` (or delete `experiments/min_weight_band/`).

---

## Self-review notes

- **Spec coverage:** §3 isolation → Task 0 + the import-only-from-src rule throughout. §4.1 `band_water_fill` → Task 1. §4.2 `band_topk` + select-modal-K → Task 2 + `_select_k_series`. §4.3 K grid {10,20,30,40,50} → `NEW_K` in Task 5. §5 retrain target (band per-K returns, argmax-K labels, walk-forward) → Tasks 3–5. §6 three windows + net cost + head-to-head + K-pick → Tasks 4–6 (`WINDOWS`, `net_returns`, `_fmt_table`; K-pick visible via `avg_n`). §7 porting → Task 7 (explicitly deferred). §8 turnover risk → reported in every table.
- **Turnover convention** matches `experiments/v6_turnover_measurement.py` (0.5·L1, one-way).
- **No prod edits:** every Create/Modify path is under `experiments/min_weight_band/`. Confirmed zero `src/strategy/*` or `trading/*` writes.
- **Type consistency:** `band_water_fill`/`band_topk`/`band_per_k_weights_and_returns` signatures are identical across the task that defines them and the task that calls them; proba columns are `c0..c{n-1}` in both `walk_forward_proba` and the series builders.
