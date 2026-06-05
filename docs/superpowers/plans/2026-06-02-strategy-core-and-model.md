# Strategy Core + Persisted Model — Implementation Plan (Plan 1 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the k-ensemble strategy math into a single source of truth (`src/strategy/`) imported by both the backtest and (later) live trading, prove the refactor causes zero drift in backtest outputs, and persist one production model.

**Architecture:** Pull the pure, forward-looking-free pieces (factor scoring, allocation/water-fill, regime features, the LGBM classifier factory) into focused modules under `src/strategy/`. A separate `historical.py` holds the backtest-only bulk-panel loader + label builders (shared by the two experiment scripts and the production trainer, killing the current copy-paste). Refactor the two `experiments/regime_k_selector*.py` scripts to import the core; gate on their outputs matching a freshly-captured pre-refactor baseline. Add `train.py` to persist one production model.

**Tech Stack:** Python 3.11+, pandas, numpy, lightgbm 4.6, pytest 9. Reuses `src/utils/rl_env.py:project_to_simplex`, `src/utils/io.py`, `src/utils/ranker.py:friday_only`.

---

## Prerequisites (verified 2026-06-02)

- Data present: `data/processed/panel/year=*/`, `data/processed/training_panel/year=*/`, `artifacts/benchmarks/spy_daily.parquet`.
- Baseline artifacts present: `artifacts/backtest_factor_v1/{weekly_regime_K_ensemble,weekly_regime_K_argmax,k_ensemble_weights,k_ensemble_probas}.parquet`.
- `lightgbm==4.6.0`, `pytest==9.0.3` installed; package installed editable (`import src...` works from anywhere).

## Notes & gotchas (read before starting)

- **LightGBM determinism:** LGBM 4.6 with default multi-threading is *usually* but not *guaranteed* bitwise-reproducible. The acceptance test therefore (a) captures a baseline by re-running the **unmodified** scripts in this same environment, then (b) compares the post-refactor run to that baseline with `np.allclose(rtol=1e-5, atol=1e-8)`. The deterministic core (scores, weights, regime features, labels) is covered exactly by unit tests. If `allclose` fails after a *pure* refactor, that is real drift — debug it with superpowers:systematic-debugging, do not loosen the tolerance.
- **Row-count drift in weights/probas:** `k_ensemble_weights.parquet` keeps only weights `> 1e-8`. If LGBM nondeterminism nudges a borderline weight across that threshold, row counts can differ by a handful even with identical code. If that happens (and only then), switch the acceptance comparison for that file to an outer-merge on `(date, permno)` and assert allclose on the overlap + that non-overlap weights are all `< 1e-6`.
- **Run scripts from the repo root** so the editable `src` package imports cleanly: `python experiments/regime_k_selector.py`.
- `artifacts/` is gitignored — the baseline copy and regenerated outputs are never committed; only code + tests are.
- **No `random_state` is set** in the current LGBM call; do **not** add one (it would change outputs vs. the validated backtest). Keep the hyperparameters byte-for-byte as they are.

## File structure

```
src/strategy/
  __init__.py        public LIVE-core API: score_universe, topk_mcap_weights,
                     ensemble_weights, build_regime_features, make_k_classifier,
                     train_model, save_model, load_model, predict_k_probs, constants
  constants.py       K_CANDIDATES, MAX_WEIGHT, EPS, REGIME_FEATURES
  factors.py         score_universe (id-agnostic factor scoring)
  allocate.py        topk_mcap_weights, ensemble_weights (wrap project_to_simplex)
  k_selector.py      build_regime_features, make_k_classifier, train_model,
                     save_model, load_model, predict_k_probs
  historical.py      BACKTEST-ONLY: load_data, per_k_weights_and_returns,
                     build_k_labels, load_spy_at, macro_by_date  (not imported by __init__)
  train.py           production trainer CLI -> trading/models/k_selector.txt + .meta.json
tests/
  test_factors.py
  test_allocate.py
  test_k_selector.py
  test_strategy_api.py
  test_historical_smoke.py
  test_backtest_acceptance.py
  test_train_model.py
experiments/
  regime_k_selector.py          (modified: import core, drop inline copies)
  regime_k_selector_weights.py  (modified: import core, drop inline copies)
trading/models/                 (created by train.py)
```

To avoid import cycles: every module imports siblings **directly** (`from src.strategy.factors import score_universe`), never from the package root. `__init__.py` exposes the live core only; `historical.py` and `train.py` are imported explicitly by their consumers.

---

## Task 1: Scaffold the `src/strategy/` package, constants, and tests dir

**Files:**
- Create: `src/strategy/__init__.py` (temporary minimal), `src/strategy/constants.py`
- Create: `tests/test_strategy_api.py` (temporary smoke)

- [ ] **Step 1: Write `src/strategy/constants.py`**

```python
"""Shared constants for the strategy core (single source of truth)."""
from __future__ import annotations

K_CANDIDATES = [10, 20, 30, 50]
MAX_WEIGHT = 0.10
EPS = 1e-8
REGIME_FEATURES = [
    "macro_vixcls", "macro_dgs10", "macro_t10y2y",
    "spy_ret_4w", "spy_ret_12w", "spy_vol_12w", "spy_vol_26w",
]
```

- [ ] **Step 2: Write a minimal `src/strategy/__init__.py`**

```python
"""Strategy core: single source of truth for the k-ensemble math.

Imported by both the backtest (experiments/) and the live trading system.
"""
from __future__ import annotations

from src.strategy.constants import EPS, K_CANDIDATES, MAX_WEIGHT, REGIME_FEATURES

__all__ = ["K_CANDIDATES", "MAX_WEIGHT", "EPS", "REGIME_FEATURES"]
```

- [ ] **Step 3: Write a temporary smoke test `tests/test_strategy_api.py`**

```python
def test_constants_importable():
    from src.strategy import K_CANDIDATES, MAX_WEIGHT, REGIME_FEATURES
    assert K_CANDIDATES == [10, 20, 30, 50]
    assert MAX_WEIGHT == 0.10
    assert len(REGIME_FEATURES) == 7
```

- [ ] **Step 4: Run the test**

Run: `cd /home/kavin-ravi/CodingStuff/axiom_tilt_strategy && python -m pytest tests/test_strategy_api.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add src/strategy/__init__.py src/strategy/constants.py tests/test_strategy_api.py
git commit -m "feat(strategy): scaffold strategy core package + constants"
```

---

## Task 2: `factors.py` — `score_universe`

Extracts the factor-scoring block from `experiments/regime_k_selector.py:58-67`, made identifier-agnostic (the id column is carried, never used in the math).

**Files:**
- Create: `src/strategy/factors.py`
- Test: `tests/test_factors.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd

from src.strategy.factors import score_universe


def _snapshot():
    return pd.DataFrame({
        "id": [1, 2, 3],
        "date": pd.to_datetime(["2020-01-03"] * 3),
        "prc": [10.0, 20.0, 5.0],
        "shrout": [100.0, 50.0, 200.0],
        "marketcap": [1000.0, np.nan, 900.0],   # id=2 falls back to |prc|*shrout = 1000
        "revenue": [500.0, 200.0, 450.0],
        "fcf": [50.0, 20.0, -10.0],
        "assets": [1000.0, 400.0, 0.0],          # id=3 assets<=0 -> fcfa NaN -> z 0
    })


def test_mcap_fallback_when_marketcap_missing():
    out = score_universe(_snapshot(), id_col="id")
    assert out.loc[out["id"] == 2, "mcap"].iloc[0] == 1000.0


def test_sp_and_fcfa_computed_and_clipped():
    out = score_universe(_snapshot(), id_col="id")
    np.testing.assert_allclose(out["sp"].to_numpy(), [0.5, 0.2, 0.5])
    # id=3 has assets<=0 -> fcfa is NaN
    assert np.isnan(out.loc[out["id"] == 3, "fcfa"].iloc[0])


def test_score_is_finite_and_orders_by_value_quality():
    out = score_universe(_snapshot(), id_col="id").set_index("id")
    assert out["score"].notna().all()
    # id=2 has the lowest sp and no quality edge -> lowest score
    assert out.loc[2, "score"] == out["score"].min()


def test_is_identifier_agnostic():
    df = _snapshot().rename(columns={"id": "ticker"})
    out = score_universe(df, id_col="ticker")
    assert "score" in out.columns and len(out) == 3
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_factors.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.strategy.factors'`

- [ ] **Step 3: Write `src/strategy/factors.py`**

```python
"""Deterministic factor scoring (value + quality composite).

Identifier-agnostic: operates on a generic id column (backtest passes `permno`,
live passes `ticker`). The id is carried through, never used in the math.
Mirrors experiments/regime_k_selector.py:58-67 exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def score_universe(snapshot_df: pd.DataFrame, id_col: str = "id",
                   date_col: str = "date") -> pd.DataFrame:
    """Add mcap, sp, fcfa, z_sp, z_fcfa, score columns to a snapshot.

    Requires columns: prc, shrout, marketcap, revenue, fcf, assets, plus
    `id_col` and `date_col`. Z-scores are cross-sectional per date.
    """
    df = snapshot_df.copy()
    df["mcap"] = df["marketcap"].where(df["marketcap"].notna(),
                                       np.abs(df["prc"]) * df["shrout"])
    df["sp"] = (df["revenue"] / df["mcap"]).clip(lower=0)
    df["fcfa"] = (df["fcf"] / df["assets"]).clip(lower=-1.0, upper=2.0)
    df.loc[df["assets"] <= 0, "fcfa"] = np.nan
    for c in ["sp", "fcfa"]:
        g = df.groupby(date_col, sort=False)[c]
        df[f"z_{c}"] = (df[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        df[f"z_{c}"] = df[f"z_{c}"].fillna(0.0)
    df["score"] = 0.5 * df["z_sp"] + 0.5 * df["z_fcfa"]
    return df
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_factors.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/strategy/factors.py tests/test_factors.py
git commit -m "feat(strategy): add score_universe factor scoring"
```

---

## Task 3: `allocate.py` — `topk_mcap_weights` + `ensemble_weights`

Wraps `src/utils/rl_env.py:project_to_simplex` (the water-fill cap). `topk_mcap_weights` mirrors the per-date weight logic in `experiments/regime_k_selector.py:82-92`; `ensemble_weights` mirrors the convex blend in `experiments/regime_k_selector_weights.py:195-204`.

**Files:**
- Create: `src/strategy/allocate.py`
- Test: `tests/test_allocate.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd

from src.strategy.allocate import ensemble_weights, topk_mcap_weights


def _scored(n):
    return pd.DataFrame({
        "id": list(range(n)),
        "score": np.linspace(1.0, 0.0, n),   # descending
        "mcap": np.linspace(100.0, 10.0, n),  # descending
    })


def test_topk_picks_k_names_sums_to_one_respects_cap():
    w = topk_mcap_weights(_scored(20), K=10, max_weight=0.10, id_col="id")
    assert len(w) == 10
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert max(w.values()) <= 0.10 + 1e-9
    # picks the top-10 by score (ids 0..9)
    assert set(w.keys()) == set(range(10))


def test_topk_equal_weight_fallback_when_all_mcap_zero():
    df = _scored(10)
    df["mcap"] = 0.0
    w = topk_mcap_weights(df, K=10, max_weight=0.10, id_col="id")
    np.testing.assert_allclose(sorted(w.values()), [0.1] * 10)


def test_ensemble_is_convex_sums_to_one_and_preserves_cap():
    df = _scored(60)
    k_probs = {10: 0.25, 20: 0.25, 30: 0.25, 50: 0.25}
    w = ensemble_weights(df, k_probs, id_col="id")
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert max(w.values()) <= 0.10 + 1e-9


def test_ensemble_concentrated_prob_matches_single_k():
    df = _scored(60)
    only10 = ensemble_weights(df, {10: 1.0, 20: 0.0, 30: 0.0, 50: 0.0}, id_col="id")
    direct10 = topk_mcap_weights(df, K=10, max_weight=0.10, id_col="id")
    assert set(only10.keys()) == set(direct10.keys())
    for k in direct10:
        assert abs(only10[k] - direct10[k]) < 1e-9
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_allocate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.strategy.allocate'`

- [ ] **Step 3: Write `src/strategy/allocate.py`**

```python
"""Allocation: top-K mcap weighting with 10% water-fill cap, and the
probability-weighted ensemble blend.

Wraps src/utils/rl_env.py:project_to_simplex. Because each per-K portfolio
already satisfies w_K(i) <= max_weight and the probabilities sum to 1, the
convex blend sum_K p_K * w_K(i) also satisfies the cap and sums to 1 — no
re-cap needed (matches the backtest).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.constants import EPS, K_CANDIDATES, MAX_WEIGHT
from src.utils.rl_env import project_to_simplex


def topk_mcap_weights(scored_df: pd.DataFrame, K: int,
                      max_weight: float = MAX_WEIGHT, id_col: str = "id") -> dict:
    """Top-K by score, mcap-weighted via water-fill cap. Single-date frame in,
    {id: weight} out. Requires columns: `id_col`, score, mcap."""
    g = scored_df.sort_values("score", ascending=False).head(K).reset_index(drop=True)
    mcaps = g["mcap"].to_numpy(dtype=np.float64)
    mcaps = np.where(np.isnan(mcaps), 0.0, mcaps)
    if mcaps.sum() <= 0:
        n = len(g)
        w = np.full(n, 1.0 / n)
    else:
        w = project_to_simplex(np.log(np.maximum(mcaps, EPS)), max_weight=max_weight)
    return {idv: float(wt) for idv, wt in zip(g[id_col].to_numpy(), w)}


def ensemble_weights(scored_df: pd.DataFrame, k_probs: dict,
                     K_candidates: list | None = None,
                     max_weight: float = MAX_WEIGHT, id_col: str = "id") -> dict:
    """Convex combination w(i) = sum_K p(K) * w_K(i). Single-date frame in,
    {id: weight} out. `k_probs` maps K -> probability."""
    if K_candidates is None:
        K_candidates = K_CANDIDATES
    combined: dict = {}
    for K in K_candidates:
        p = float(k_probs[K])
        wK = topk_mcap_weights(scored_df, K, max_weight=max_weight, id_col=id_col)
        for idv, wt in wK.items():
            combined[idv] = combined.get(idv, 0.0) + p * wt
    return {idv: wt for idv, wt in combined.items() if wt > EPS}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_allocate.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/strategy/allocate.py tests/test_allocate.py
git commit -m "feat(strategy): add topk_mcap_weights + ensemble_weights"
```

---

## Task 4: `k_selector.py` — regime features, classifier, train/save/load/predict

`build_regime_features` mirrors `experiments/regime_k_selector.py:105-115`. `make_k_classifier` reproduces the exact LGBM hyperparameters from lines 152-156. `train_model`/`save_model`/`load_model`/`predict_k_probs` are the production helpers.

**Files:**
- Create: `src/strategy/k_selector.py`
- Test: `tests/test_k_selector.py`

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd

from src.strategy.constants import K_CANDIDATES, REGIME_FEATURES
from src.strategy.k_selector import (
    build_regime_features, load_model, make_k_classifier,
    predict_k_probs, save_model, train_model,
)


def _spy_macro(n=40):
    dates = pd.date_range("2020-01-03", periods=n, freq="7D")
    spy_at = pd.Series(np.linspace(100.0, 150.0, n), index=dates)
    macro = pd.DataFrame(
        {"macro_vixcls": 15.0, "macro_dgs10": 2.0, "macro_t10y2y": 0.5}, index=dates
    )
    return dates, spy_at, macro


def test_regime_features_columns_and_no_nans():
    dates, spy_at, macro = _spy_macro()
    rf = build_regime_features(dates, spy_at, macro)
    assert list(rf.columns) == REGIME_FEATURES
    assert len(rf) == len(dates)
    assert rf.notna().all().all()


def test_regime_features_are_lagged_one_period():
    # spy_ret_4w at row i must use returns through row i-1 (shift(1) => no look-ahead).
    dates, spy_at, macro = _spy_macro()
    rf = build_regime_features(dates, spy_at, macro)
    spy_w = spy_at.pct_change().fillna(0.0)
    unshifted = (1 + spy_w).rolling(4).apply(lambda x: x.prod() - 1, raw=False)
    # row 10 of the feature equals the UNSHIFTED value at row 9
    np.testing.assert_allclose(rf["spy_ret_4w"].iloc[10], unshifted.iloc[9])


def test_make_classifier_has_exact_hyperparameters():
    p = make_k_classifier(num_class=4).get_params()
    assert p["n_estimators"] == 500
    assert p["learning_rate"] == 0.03
    assert p["num_leaves"] == 15
    assert p["min_data_in_leaf"] == 20
    assert p["feature_fraction"] == 0.8
    assert p["bagging_fraction"] == 0.8
    assert p["lambda_l2"] == 2.0
    assert p["objective"] == "multiclass"
    assert p["num_class"] == 4


def test_train_save_load_predict_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, len(REGIME_FEATURES)))
    y = rng.integers(0, 4, size=300)
    model = train_model(X, y, num_class=4)
    path = tmp_path / "k_selector.txt"
    save_model(model, path)
    assert path.exists()
    loaded = load_model(path)
    probs = predict_k_probs(loaded, X[0])
    assert set(probs.keys()) == set(K_CANDIDATES)
    assert abs(sum(probs.values()) - 1.0) < 1e-6
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_k_selector.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.strategy.k_selector'`

- [ ] **Step 3: Write `src/strategy/k_selector.py`**

```python
"""Regime feature construction + the LGBM K-selector model lifecycle.

build_regime_features mirrors experiments/regime_k_selector.py:105-115.
make_k_classifier reproduces the exact hyperparameters from the backtest;
DO NOT add random_state (it would change validated outputs).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.strategy.constants import K_CANDIDATES, REGIME_FEATURES


def build_regime_features(all_dates: pd.DatetimeIndex, spy_at: pd.Series,
                          macro_by_date: pd.DataFrame) -> pd.DataFrame:
    """Build the 7 regime features indexed by `all_dates`.

    `spy_at`: SPY close sampled at all_dates (Friday frequency).
    `macro_by_date`: DataFrame indexed by all_dates with the 3 macro columns.
    All trailing features are shift(1)-lagged to avoid look-ahead.
    """
    spy_w_ret = spy_at.pct_change().fillna(0.0)
    regime_df = pd.DataFrame({
        "macro_vixcls": macro_by_date["macro_vixcls"].values,
        "macro_dgs10":  macro_by_date["macro_dgs10"].values,
        "macro_t10y2y": macro_by_date["macro_t10y2y"].values,
        "spy_ret_4w":   (1 + spy_w_ret).rolling(4).apply(lambda x: x.prod() - 1, raw=False).shift(1).values,
        "spy_ret_12w":  (1 + spy_w_ret).rolling(12).apply(lambda x: x.prod() - 1, raw=False).shift(1).values,
        "spy_vol_12w":  (spy_w_ret.rolling(12).std() * np.sqrt(52)).shift(1).values,
        "spy_vol_26w":  (spy_w_ret.rolling(26).std() * np.sqrt(52)).shift(1).values,
    }, index=all_dates).ffill().bfill().fillna(0.0)
    return regime_df[REGIME_FEATURES]


def make_k_classifier(num_class: int = 4) -> lgb.LGBMClassifier:
    """Unfitted LGBM multiclass classifier with the exact validated hyperparameters."""
    return lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.03, num_leaves=15,
        min_data_in_leaf=20, feature_fraction=0.8, bagging_fraction=0.8,
        lambda_l2=2.0, verbose=-1, objective="multiclass", num_class=num_class,
    )


def train_model(regime_X, labels, num_class: int = 4) -> lgb.LGBMClassifier:
    """Fit one production classifier on all data (no early stopping / holdout)."""
    clf = make_k_classifier(num_class=num_class)
    clf.fit(np.asarray(regime_X), np.asarray(labels))
    return clf


def save_model(model, path) -> None:
    """Persist the underlying booster as an LGBM text model."""
    booster = model.booster_ if hasattr(model, "booster_") else model
    booster.save_model(str(path))


def load_model(path) -> lgb.Booster:
    """Load a persisted LGBM text model as a Booster."""
    return lgb.Booster(model_file=str(path))


def predict_k_probs(model, regime_row, K_candidates: list | None = None) -> dict:
    """Predict K-probabilities for one regime row -> {K: prob}.

    Works for both an LGBMClassifier (predict_proba) and a raw Booster
    (predict returns class probabilities for a multiclass model)."""
    if K_candidates is None:
        K_candidates = K_CANDIDATES
    x = np.asarray(regime_row, dtype=float).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x))[0]
    else:
        proba = np.asarray(model.predict(x))[0]
    return {K: float(p) for K, p in zip(K_candidates, proba)}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_k_selector.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/strategy/k_selector.py tests/test_k_selector.py
git commit -m "feat(strategy): add regime features + LGBM K-selector lifecycle"
```

---

## Task 5: Expose the full public API in `__init__.py`

**Files:**
- Modify: `src/strategy/__init__.py`
- Modify: `tests/test_strategy_api.py`

- [ ] **Step 1: Update the test to assert the full API**

```python
def test_constants_importable():
    from src.strategy import K_CANDIDATES, MAX_WEIGHT, REGIME_FEATURES
    assert K_CANDIDATES == [10, 20, 30, 50]
    assert MAX_WEIGHT == 0.10
    assert len(REGIME_FEATURES) == 7


def test_full_public_api_importable():
    from src.strategy import (
        score_universe, topk_mcap_weights, ensemble_weights,
        build_regime_features, make_k_classifier, train_model,
        save_model, load_model, predict_k_probs,
    )
    for fn in (score_universe, topk_mcap_weights, ensemble_weights,
               build_regime_features, make_k_classifier, train_model,
               save_model, load_model, predict_k_probs):
        assert callable(fn)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_strategy_api.py -v`
Expected: FAIL with `ImportError: cannot import name 'score_universe'`

- [ ] **Step 3: Update `src/strategy/__init__.py`**

```python
"""Strategy core: single source of truth for the k-ensemble math.

Imported by both the backtest (experiments/) and the live trading system.
This package exposes the LIVE core; the backtest-only bulk-panel loader lives
in src/strategy/historical.py and is imported explicitly by its consumers.
"""
from __future__ import annotations

from src.strategy.allocate import ensemble_weights, topk_mcap_weights
from src.strategy.constants import EPS, K_CANDIDATES, MAX_WEIGHT, REGIME_FEATURES
from src.strategy.factors import score_universe
from src.strategy.k_selector import (
    build_regime_features, load_model, make_k_classifier,
    predict_k_probs, save_model, train_model,
)

__all__ = [
    "K_CANDIDATES", "MAX_WEIGHT", "EPS", "REGIME_FEATURES",
    "score_universe", "topk_mcap_weights", "ensemble_weights",
    "build_regime_features", "make_k_classifier", "train_model",
    "save_model", "load_model", "predict_k_probs",
]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/ -v`
Expected: PASS (all tests so far)

- [ ] **Step 5: Commit**

```bash
git add src/strategy/__init__.py tests/test_strategy_api.py
git commit -m "feat(strategy): expose full public core API"
```

---

## Task 6: `historical.py` — shared backtest loader + label builders

Extracts `load_data` (currently duplicated in both experiments) and the label/return helpers, so the two experiments AND `train.py` share one implementation. `load_data` now calls `score_universe` instead of inlining the score math.

**Files:**
- Create: `src/strategy/historical.py`
- Test: `tests/test_historical_smoke.py`

- [ ] **Step 1: Write the failing smoke test (runs against real data)**

```python
import pandas as pd
import pytest

from src.strategy.constants import K_CANDIDATES
from src.utils.io import processed_dir

pytestmark = pytest.mark.skipif(
    not (processed_dir() / "panel").exists(),
    reason="processed panel data not present",
)


def test_load_data_has_scores_and_friday_rows():
    from src.strategy.historical import load_data
    df = load_data()
    assert {"permno", "date", "score", "mcap"}.issubset(df.columns)
    assert len(df) > 1000
    assert df["score"].notna().all()


def test_per_k_returns_and_labels_align():
    from src.strategy.historical import build_k_labels, load_data, per_k_weights_and_returns
    df = load_data()
    k_returns = {K: per_k_weights_and_returns(df, K)[1] for K in K_CANDIDATES}
    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in k_returns.values()])))
    labels, k_mat = build_k_labels(k_returns, all_dates)
    assert len(labels) == len(all_dates)
    assert set(labels.dropna().unique()).issubset({0, 1, 2, 3})
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_historical_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.strategy.historical'`

- [ ] **Step 3: Write `src/strategy/historical.py`**

```python
"""BACKTEST-ONLY data path: bulk historical panel loader + label builders.

Shared by experiments/regime_k_selector*.py and src/strategy/train.py to remove
the prior copy-paste. NOT for live trading — the live snapshot is built
separately (trading/data/snapshot.py, Plan 2). Not imported by the package
__init__ to keep the live core import light.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.allocate import topk_mcap_weights
from src.strategy.constants import K_CANDIDATES, MAX_WEIGHT
from src.strategy.factors import score_universe
from src.utils.io import processed_dir, repo_root
from src.utils.ranker import friday_only

REPO_ROOT = repo_root()
PANEL_DIR = processed_dir() / "panel"
TRAIN_PANEL_DIR = processed_dir() / "training_panel"
SPY_PATH = REPO_ROOT / "artifacts" / "benchmarks" / "spy_daily.parquet"


def load_data() -> pd.DataFrame:
    """Load + merge the historical panel, keep Friday in-universe rows, and
    attach factor scores. Mirrors the prior inline load_data, but scoring is
    delegated to score_universe (single source of truth)."""
    cols = ["permno", "date", "prc", "shrout", "marketcap", "in_universe",
            "revenue", "fcf", "assets"]
    frames = []
    for y in range(2001, 2026):
        for p in sorted((PANEL_DIR / f"year={y}").glob("*.parquet")):
            d = pd.read_parquet(p, columns=cols)
            d["date"] = pd.to_datetime(d["date"])
            d["permno"] = d["permno"].astype("int64")
            frames.append(d)
    daily = pd.concat(frames, ignore_index=True)
    tframes = []
    for y in range(2002, 2026):
        for p in sorted((TRAIN_PANEL_DIR / f"year={y}").glob("*.parquet")):
            d = pd.read_parquet(p, columns=["permno", "date", "fwd_ret_5d",
                                            "macro_vixcls", "macro_dgs10", "macro_t10y2y"])
            d["date"] = pd.to_datetime(d["date"])
            d["permno"] = d["permno"].astype("int64")
            tframes.append(d)
    fri = pd.concat(tframes, ignore_index=True)
    df = daily.merge(fri, on=["permno", "date"], how="inner")
    df = df.dropna(subset=["fwd_ret_5d"]).copy()
    df = friday_only(df).reset_index(drop=True)
    df = df[df["in_universe"]].copy()
    df = score_universe(df, id_col="permno")
    return df


def per_k_weights_and_returns(df: pd.DataFrame, K: int,
                              max_weight: float = MAX_WEIGHT):
    """Per Friday: top-K mcap-weighted (cap10) weights + the portfolio's
    fwd_ret_5d. Returns (weight_df[date,permno,weight], return_series[date])."""
    weight_rows = []
    return_rows = []
    for d, g in df.groupby("date", sort=False):
        w = topk_mcap_weights(g, K, max_weight=max_weight, id_col="permno")
        gk = g.sort_values("score", ascending=False).head(K)
        fwd = dict(zip(gk["permno"].astype(int).to_numpy(),
                       np.nan_to_num(gk["fwd_ret_5d"].to_numpy(dtype=np.float64))))
        ret = float(sum(w[p] * fwd[int(p)] for p in w))
        return_rows.append({"date": d, "ret": ret})
        for p, wt in w.items():
            if wt > 0:
                weight_rows.append({"date": d, "permno": int(p), "weight": float(wt)})
    wdf = pd.DataFrame(weight_rows)
    rdf = pd.DataFrame(return_rows).sort_values("date").set_index("date")["ret"]
    return wdf, rdf


def build_k_labels(k_returns: dict, all_dates: pd.DatetimeIndex,
                   K_candidates: list | None = None):
    """Label = argmax-K of per-K weekly returns per Friday. Returns (labels, k_mat).
    labels are int class indices (0..len(K)-1), NaN where all K returns are NaN."""
    if K_candidates is None:
        K_candidates = K_CANDIDATES
    k_mat = pd.DataFrame(
        {f"K{K}": k_returns[K].reindex(all_dates).values for K in K_candidates},
        index=all_dates,
    )
    k_to_idx = {K: i for i, K in enumerate(K_candidates)}
    labels = k_mat.idxmax(axis=1).str[1:].astype("Int64").map(k_to_idx)
    return labels, k_mat


def load_spy_at(all_dates: pd.DatetimeIndex) -> pd.Series:
    """SPY close sampled at all_dates (ffilled)."""
    spy = pd.read_parquet(SPY_PATH).reset_index()[["Date", "close"]].rename(columns={"Date": "date"})
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").set_index("date")["close"]
    return spy.reindex(spy.index.union(all_dates)).sort_index().ffill().reindex(all_dates)


def macro_by_date(df: pd.DataFrame, all_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """First macro reading per date, reindexed to all_dates."""
    return (df.groupby("date", sort=False)[["macro_vixcls", "macro_dgs10", "macro_t10y2y"]]
              .first().reindex(all_dates))
```

> Note on `build_k_labels`: the original used `.astype(int).map(...)`. Using `astype("Int64")` (nullable) preserves NaN labels safely; the downstream `.dropna()` + `.astype(int)` before training is unchanged in behavior. Verify in Task 10 that labels/outputs still match the baseline.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_historical_smoke.py -v`
Expected: PASS (2 passed). (Takes ~10-30s — it reads the full panel.)

- [ ] **Step 5: Commit**

```bash
git add src/strategy/historical.py tests/test_historical_smoke.py
git commit -m "feat(strategy): add shared historical loader + label builders"
```

---

## Task 7: Capture the pre-refactor acceptance baseline

The experiments are still **unmodified** at this point (Tasks 1-6 only added new files). Re-run them in this environment and snapshot the outputs — this is the reference the refactor must reproduce.

**Files:** none (produces local artifacts under `artifacts/backtest_factor_v1/baseline_prerefactor/`, gitignored).

- [ ] **Step 1: Run the two unmodified experiments**

Run:
```bash
cd /home/kavin-ravi/CodingStuff/axiom_tilt_strategy
python experiments/regime_k_selector.py
python experiments/regime_k_selector_weights.py
```
Expected: both complete; console prints the metrics table / weight-sum logs. (Each trains 17 LGBM walks — expect ~1-4 min each.)

- [ ] **Step 2: Snapshot the four outputs as the baseline**

Run:
```bash
cd /home/kavin-ravi/CodingStuff/axiom_tilt_strategy/artifacts/backtest_factor_v1
mkdir -p baseline_prerefactor
cp weekly_regime_K_ensemble.parquet weekly_regime_K_argmax.parquet \
   k_ensemble_weights.parquet k_ensemble_probas.parquet baseline_prerefactor/
ls -la baseline_prerefactor/
```
Expected: four `.parquet` files listed in `baseline_prerefactor/`.

- [ ] **Step 3: No commit** (artifacts are gitignored). Confirm:

Run: `git status --short artifacts/` → Expected: no output (artifacts ignored).

---

## Task 8: Refactor `experiments/regime_k_selector.py` to import the core

Replace the inline copies with imports. Keep the walk-loop, labels, metrics, and output-writing exactly as they are.

**Files:**
- Modify: `experiments/regime_k_selector.py`

- [ ] **Step 1: Replace the imports + delete the inline `load_data`**

Replace lines 19-68 (the `src.utils...` imports, constants, and the whole `def load_data():` block) with:

```python
from src.strategy import build_regime_features, make_k_classifier
from src.strategy.constants import K_CANDIDATES
from src.strategy.historical import load_data, load_spy_at, macro_by_date, per_k_weights_and_returns
from src.utils.io import repo_root
from src.utils.logging_utils import get_logger

log = get_logger(__name__)
REPO_ROOT = repo_root()
```

(The module no longer needs `time`, `lightgbm as lgb`, `processed_dir`, `friday_only`, `project_to_simplex`, `PANEL_DIR`, `TRAIN_PANEL_DIR`, `SPY_PATH`, `MAX_WEIGHT`, `EPS` at module scope — `lgb` is still used by the early-stopping fit below, so keep `import lightgbm as lgb`. Keep `import numpy as np` and `import pandas as pd`.)

- [ ] **Step 2: Replace `k_weekly_returns` usage with the shared helper**

Delete the local `def k_weekly_returns(K_val):` block (lines 75-92). Replace the build line (line 96):

```python
print("Building per-K weekly returns ...")
k_returns = {K: per_k_weights_and_returns(df, K)[1] for K in K_CANDIDATES}
```

- [ ] **Step 3: Replace the inline regime-feature block**

Replace lines 100-115 (the SPY load + `regime_df = pd.DataFrame({...})` block) with:

```python
# Build regime features (shared core)
spy_at = load_spy_at(all_dates)
regime_df = build_regime_features(all_dates, macro_by_date(df, all_dates).pipe(lambda m: m), all_dates_spy := spy_at)
```

Wait — keep it simple and explicit instead:

```python
# Build regime features (shared core)
spy_at = load_spy_at(all_dates)
mbd = macro_by_date(df, all_dates)
regime_df = build_regime_features(all_dates, spy_at, mbd)
```

Also keep a `spy_at` reference for line ~180 (`spy_aligned = spy_at.reindex(...).pct_change()...`) — it already uses `spy_at`, which now comes from `load_spy_at`. Leave that line unchanged.

- [ ] **Step 4: Replace the inline classifier construction**

Replace the `clf = lgb.LGBMClassifier(...)` block (lines 152-156) with:

```python
        clf = make_k_classifier(num_class=len(K_CANDIDATES))
```

Leave the `clf.fit(... early_stopping ...)` call and everything after unchanged.

- [ ] **Step 5: Run the refactored script**

Run: `python experiments/regime_k_selector.py`
Expected: completes, prints the metrics table. Sanity-check the printed Sharpe for "regime-LGBM K ensemble" is ~1.28.

- [ ] **Step 6: Commit**

```bash
git add experiments/regime_k_selector.py
git commit -m "refactor(backtest): regime_k_selector imports strategy core"
```

---

## Task 9: Refactor `experiments/regime_k_selector_weights.py` to import the core

**Files:**
- Modify: `experiments/regime_k_selector_weights.py`

- [ ] **Step 1: Replace imports + delete inline `load_data` and `k_weights_and_returns`**

Replace lines 15-104 (imports, constants, `def load_data()`, `def k_weights_and_returns()`) with:

```python
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.strategy import build_regime_features, make_k_classifier
from src.strategy.constants import K_CANDIDATES
from src.strategy.historical import (
    build_k_labels, load_data, load_spy_at, macro_by_date, per_k_weights_and_returns,
)
from src.utils.io import repo_root
from src.utils.logging_utils import configure_logging, get_logger

log = get_logger(__name__)
REPO_ROOT = repo_root()
OUT_DIR = REPO_ROOT / "artifacts" / "backtest_factor_v1"
```

- [ ] **Step 2: Use shared per-K weights/returns in `main()`**

Replace the loop that builds `k_weights`/`k_returns` (lines 114-120) with:

```python
    k_weights = {}
    k_returns = {}
    for K in K_CANDIDATES:
        log.info("  K=%d", K)
        kw, kr = per_k_weights_and_returns(df, K)
        k_weights[K] = kw
        k_returns[K] = kr
```

- [ ] **Step 3: Replace the inline regime-feature block + labels**

Replace lines 125-143 (the SPY load, `regime_df = pd.DataFrame({...})`, and the `labels = ...` block) with:

```python
    # Regime features + labels (shared core)
    spy_at = load_spy_at(all_dates)
    mbd = macro_by_date(df, all_dates)
    regime_df = build_regime_features(all_dates, spy_at, mbd)
    labels, k_mat = build_k_labels(k_returns, all_dates)
```

- [ ] **Step 4: Replace the inline classifier construction**

Replace the `clf = lgb.LGBMClassifier(...)` block (lines 163-167) with:

```python
        clf = make_k_classifier(num_class=len(K_CANDIDATES))
```

(`lgb` is otherwise unused now; remove `import lightgbm as lgb` if present and keep `lgb.early_stopping` — wait: the fit still calls `lgb.early_stopping(...)`. Keep `import lightgbm as lgb`.)

- [ ] **Step 5: Run the refactored script**

Run: `python experiments/regime_k_selector_weights.py`
Expected: completes; logs "Weight sums per Friday: min=… max=… mean=…" all ≈ 1.0; writes `k_ensemble_weights.parquet` + `k_ensemble_probas.parquet`.

- [ ] **Step 6: Commit**

```bash
git add experiments/regime_k_selector_weights.py
git commit -m "refactor(backtest): regime_k_selector_weights imports strategy core"
```

---

## Task 10: Acceptance test — outputs match the pre-refactor baseline

**Files:**
- Create: `tests/test_backtest_acceptance.py`

- [ ] **Step 1: Write the acceptance test**

```python
import numpy as np
import pandas as pd
import pytest

from src.utils.io import repo_root

ART = repo_root() / "artifacts" / "backtest_factor_v1"
BASE = ART / "baseline_prerefactor"
RTOL, ATOL = 1e-5, 1e-8

CASES = [
    ("weekly_regime_K_ensemble.parquet", ["weekly_ret"]),
    ("weekly_regime_K_argmax.parquet", ["weekly_ret"]),
    ("k_ensemble_weights.parquet", ["weight"]),
    ("k_ensemble_probas.parquet", ["K10_prob", "K20_prob", "K30_prob", "K50_prob"]),
]


def _load_sorted(p):
    df = pd.read_parquet(p)
    keys = [c for c in ["date", "permno"] if c in df.columns]
    return df.sort_values(keys).reset_index(drop=True) if keys else df


@pytest.mark.parametrize("fname,valcols", CASES)
def test_refactor_matches_baseline(fname, valcols):
    base_p, new_p = BASE / fname, ART / fname
    if not base_p.exists() or not new_p.exists():
        pytest.skip(f"baseline or current output missing for {fname}")
    b, n = _load_sorted(base_p), _load_sorted(new_p)
    assert len(b) == len(n), f"{fname}: row count {len(b)} != {len(n)} (see Notes on row-count drift)"
    for c in valcols:
        np.testing.assert_allclose(
            n[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float),
            rtol=RTOL, atol=ATOL, err_msg=f"{fname}:{c} drifted from baseline",
        )
```

- [ ] **Step 2: Run the acceptance test**

Run: `python -m pytest tests/test_backtest_acceptance.py -v`
Expected: PASS (4 passed). The refactored experiments (Tasks 8-9) already regenerated the four current artifacts; this compares them to the Task-7 baseline.

- [ ] **Step 3: If it FAILS**

Do **not** loosen the tolerance. Use superpowers:systematic-debugging. Most likely causes, in order: (a) a regime-feature column mis-ordered (compare `regime_df.columns`), (b) `build_k_labels` Int64 change altered a tie-break (compare `labels`), (c) genuine LGBM nondeterminism on a borderline weight (apply the merge-on-(date,permno) comparison from Notes — only for the weights file).

- [ ] **Step 4: Commit**

```bash
git add tests/test_backtest_acceptance.py
git commit -m "test(backtest): acceptance gate — refactor matches baseline"
```

---

## Task 11: `train.py` — persist the production model

**Files:**
- Create: `src/strategy/train.py`
- Test: `tests/test_train_model.py`

- [ ] **Step 1: Write the failing test**

```python
import json

import pandas as pd
import pytest

from src.utils.io import processed_dir

pytestmark = pytest.mark.skipif(
    not (processed_dir() / "panel").exists(),
    reason="processed panel data not present",
)


def test_train_writes_model_and_meta(tmp_path):
    from src.strategy.k_selector import load_model, predict_k_probs
    from src.strategy.train import train_production_model

    out = tmp_path / "k_selector.txt"
    meta = train_production_model(out_path=out)
    assert out.exists()
    meta_path = out.with_suffix(".meta.json")
    assert meta_path.exists()

    loaded = json.loads(meta_path.read_text())
    assert loaded["features"] == meta["features"]
    assert loaded["K_candidates"] == [10, 20, 30, 50]
    assert loaded["n_train_fridays"] > 100

    model = load_model(out)
    # 7 regime features -> a valid probability dict over the 4 K classes
    probs = predict_k_probs(model, [20.0, 2.5, 0.4, 0.01, 0.03, 0.15, 0.18])
    assert abs(sum(probs.values()) - 1.0) < 1e-6
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_train_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.strategy.train'`

- [ ] **Step 3: Write `src/strategy/train.py`**

```python
"""Train + persist ONE production K-selector model on all history.

CLI:  python -m src.strategy.train [--out trading/models/k_selector.txt]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import lightgbm
import pandas as pd

from src.strategy.constants import K_CANDIDATES, REGIME_FEATURES
from src.strategy.historical import (
    build_k_labels, load_data, load_spy_at, macro_by_date, per_k_weights_and_returns,
)
from src.strategy.k_selector import build_regime_features, save_model, train_model
from src.utils.io import repo_root

DEFAULT_OUT = repo_root() / "trading" / "models" / "k_selector.txt"


def train_production_model(out_path: Path | str = DEFAULT_OUT) -> dict:
    """Train one LGBM on all history through the latest date; persist model + meta.
    Returns the meta dict."""
    out_path = Path(out_path)
    df = load_data()
    k_returns = {K: per_k_weights_and_returns(df, K)[1] for K in K_CANDIDATES}
    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in k_returns.values()])))
    labels, _ = build_k_labels(k_returns, all_dates)
    spy_at = load_spy_at(all_dates)
    regime = build_regime_features(all_dates, spy_at, macro_by_date(df, all_dates))

    valid = labels.notna()
    model = train_model(
        regime[valid.to_numpy()].to_numpy(),
        labels[valid].astype(int).to_numpy(),
        num_class=len(K_CANDIDATES),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, out_path)
    meta = {
        "train_date_utc": datetime.now(timezone.utc).isoformat(),
        "n_train_fridays": int(valid.sum()),
        "first_date": str(all_dates.min().date()),
        "last_date": str(all_dates.max().date()),
        "features": REGIME_FEATURES,
        "K_candidates": K_CANDIDATES,
        "label": "argmax K of per-K weekly fwd_ret_5d",
        "lightgbm_version": lightgbm.__version__,
    }
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    meta = train_production_model(out_path=args.out)
    print(f"Saved model -> {args.out}")
    print(f"Saved meta  -> {Path(args.out).with_suffix('.meta.json')}")
    print(f"Trained on {meta['n_train_fridays']} Fridays "
          f"{meta['first_date']}..{meta['last_date']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_train_model.py -v`
Expected: PASS (1 passed). (Reads the full panel + trains one model — ~30-60s.)

- [ ] **Step 5: Generate the real production model**

Run: `python -m src.strategy.train`
Expected: prints "Saved model -> .../trading/models/k_selector.txt" and "Trained on N Fridays …".

- [ ] **Step 6: Decide model tracking + commit**

Run: `ls -la trading/models/` and check the `.txt` size.
- If `k_selector.txt` is < ~5 MB: commit both files (the deployed model should be versioned/auditable).
- If larger: add `trading/models/*.txt` to `.gitignore`, commit only `k_selector.meta.json`.

```bash
git add src/strategy/train.py tests/test_train_model.py
git add trading/models/k_selector.meta.json   # + k_selector.txt if small enough
git commit -m "feat(strategy): production model trainer + persisted k_selector"
```

---

## Final verification

- [ ] **Run the full suite**

Run: `python -m pytest tests/ -v`
Expected: all tests pass (unit + smoke + acceptance + train).

- [ ] **Confirm the experiments still run end-to-end** (already verified in Tasks 8-9).

---

## Self-Review (completed during planning)

**Spec coverage (vs `2026-06-02-live-trading-system-design.md` Parts 1-2):**
- Part 1 `factors.py / k_selector.py / allocate.py / __init__.py` → Tasks 2,3,4,5. ✓
- Part 1 "refactor the experiments to import the core" → Tasks 8,9. ✓
- Part 1 "remove copy-paste duplication" → `historical.py` (Task 6) unifies `load_data`. ✓
- Part 1 acceptance "outputs numerically identical" → Tasks 7,10. ✓
- Part 1 unit tests (scoring; topk sums-to-1 + cap; ensemble convexity) → Tasks 2,3. ✓
- Part 2 `train.py`, save model + `meta.json`, train on all history → Task 11. ✓
- Part 2 retrain cadence (quarterly, configurable) → **deferred to Plan 2** (lives in `trading/config.py`; train.py itself is cadence-agnostic). Noted, not a gap.

**Placeholder scan:** no TBD/TODO; every code step has complete code; every command has expected output. ✓

**Type/name consistency:** `score_universe`, `topk_mcap_weights`, `ensemble_weights`, `build_regime_features`, `make_k_classifier`, `train_model`, `save_model`, `load_model`, `predict_k_probs`, `load_data`, `per_k_weights_and_returns`, `build_k_labels`, `load_spy_at`, `macro_by_date`, `train_production_model` are used identically across tasks. `id_col` default `"id"` in the core; backtest callers pass `"permno"`. ✓
