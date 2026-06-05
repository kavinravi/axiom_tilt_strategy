"""Tests for trading/weights.py — Task 6.

All network calls are injected (snapshot, regime_row).  The REAL model is
loaded from MODEL_PATH; the test is skipped if the file is absent.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from trading.config import MAX_WEIGHT, MIN_HOLDINGS, MAX_HOLDINGS, MODEL_PATH, WEIGHTS_DIR


# ---------------------------------------------------------------------------
# Helpers: build synthetic inputs that satisfy score_universe's column contract
# ---------------------------------------------------------------------------

def _make_synthetic_snapshot(n: int = 65, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """Return a snapshot DataFrame with >= 60 tickers and all required columns."""
    if asof is None:
        asof = pd.Timestamp("2026-05-29")  # a known Friday
    rng = np.random.default_rng(seed=42)
    tickers = [f"T{i:04d}" for i in range(n)]
    return pd.DataFrame({
        "ticker": tickers,
        "date": asof,
        "prc": rng.uniform(10, 500, n),
        "shrout": rng.uniform(100, 10_000, n),
        "marketcap": rng.uniform(1e9, 5e11, n),
        "revenue": rng.uniform(1e8, 1e11, n),
        "fcf": rng.uniform(-1e9, 1e10, n),
        "assets": rng.uniform(1e9, 5e11, n),
    })


def _make_synthetic_regime_row() -> np.ndarray:
    """Return a plausible 7-element regime feature vector."""
    return np.array([20.0, 4.5, 0.5, 0.03, 0.12, 0.12, 0.15])


# ---------------------------------------------------------------------------
# Task 6 tests
# ---------------------------------------------------------------------------

@pytest.fixture
def asof():
    return pd.Timestamp("2026-05-29")


@pytest.fixture
def snapshot(asof):
    return _make_synthetic_snapshot(asof=asof)


@pytest.fixture
def regime_row():
    return _make_synthetic_regime_row()


@pytest.fixture
def real_model():
    """Load the persisted k_selector model; skip if not present."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}; skipping.")
    from src.strategy.k_selector import load_model
    return load_model(MODEL_PATH)


@pytest.fixture(autouse=True)
def _isolate_audit_dir(tmp_path, monkeypatch):
    """Redirect frozen-weights output to a tmp dir so tests never touch the real audit/."""
    monkeypatch.setattr("trading.weights.WEIGHTS_DIR", tmp_path / "weights")


# --- compute_target_weights ---

def test_weight_sum_close_to_one(asof, snapshot, regime_row, real_model):
    from trading.weights import compute_target_weights
    result = compute_target_weights(asof=asof, snapshot=snapshot,
                                    regime_row=regime_row, model=real_model)
    assert abs(result["weight_sum"] - 1.0) < 1e-6, (
        f"weight_sum = {result['weight_sum']}"
    )


def test_max_weight_at_most_cap(asof, snapshot, regime_row, real_model):
    from trading.weights import compute_target_weights
    result = compute_target_weights(asof=asof, snapshot=snapshot,
                                    regime_row=regime_row, model=real_model)
    assert result["max_weight"] <= MAX_WEIGHT + 1e-9, (
        f"max_weight = {result['max_weight']}"
    )


def test_holdings_in_bounds(asof, snapshot, regime_row, real_model):
    from trading.weights import compute_target_weights
    result = compute_target_weights(asof=asof, snapshot=snapshot,
                                    regime_row=regime_row, model=real_model)
    assert MIN_HOLDINGS <= result["n_holdings"] <= MAX_HOLDINGS, (
        f"n_holdings = {result['n_holdings']}"
    )


def test_k_probs_keys_and_sum(asof, snapshot, regime_row, real_model):
    from trading.weights import compute_target_weights
    result = compute_target_weights(asof=asof, snapshot=snapshot,
                                    regime_row=regime_row, model=real_model)
    k_probs = result["k_probs"]
    assert set(k_probs.keys()) == {10, 20, 30, 50}, f"k_probs keys: {set(k_probs.keys())}"
    assert abs(sum(k_probs.values()) - 1.0) < 1e-6, (
        f"k_probs sum = {sum(k_probs.values())}"
    )


def test_validate_weights_returns_empty_list(asof, snapshot, regime_row, real_model):
    from trading.weights import compute_target_weights, validate_weights
    result = compute_target_weights(asof=asof, snapshot=snapshot,
                                    regime_row=regime_row, model=real_model)
    problems = validate_weights(result)
    assert problems == [], f"validate_weights returned: {problems}"


def test_audit_json_written(asof, snapshot, regime_row, real_model):
    import trading.weights as tw
    from trading.weights import compute_target_weights
    compute_target_weights(asof=asof, snapshot=snapshot,
                           regime_row=regime_row, model=real_model)
    audit_file = tw.WEIGHTS_DIR / f"{asof.date()}.json"
    assert audit_file.exists(), f"Audit file not found: {audit_file}"
    with audit_file.open() as f:
        data = json.load(f)
    assert "asof" in data
    assert "k_probs" in data
    assert "weights" in data


def test_result_dict_has_required_keys(asof, snapshot, regime_row, real_model):
    from trading.weights import compute_target_weights
    result = compute_target_weights(asof=asof, snapshot=snapshot,
                                    regime_row=regime_row, model=real_model)
    for key in ("asof", "weights", "k_probs", "n_holdings", "weight_sum", "max_weight"):
        assert key in result, f"Missing key: {key}"


# --- validate_weights ---

def test_validate_weights_sum_too_far():
    from trading.weights import validate_weights
    result = {
        "asof": pd.Timestamp("2026-05-29"),
        "weights": {"A": 0.6, "B": 0.3},   # sum=0.9, not ≈1
        "k_probs": {10: 0.25, 20: 0.25, 30: 0.25, 50: 0.25},
        "n_holdings": 2,
        "weight_sum": 0.9,
        "max_weight": 0.6,
    }
    problems = validate_weights(result)
    assert any("sum" in p.lower() for p in problems), problems


def test_validate_weights_max_exceeds_cap():
    from trading.weights import validate_weights
    result = {
        "asof": pd.Timestamp("2026-05-29"),
        "weights": {"A": 0.5, "B": 0.5},   # max=0.5 > 0.10
        "k_probs": {10: 0.25, 20: 0.25, 30: 0.25, 50: 0.25},
        "n_holdings": 2,
        "weight_sum": 1.0,
        "max_weight": 0.5,
    }
    problems = validate_weights(result)
    assert any("max" in p.lower() or "weight" in p.lower() for p in problems), problems


def test_validate_weights_too_few_holdings():
    from trading.weights import validate_weights
    # Build tiny weights that sum to 1 but have fewer than MIN_HOLDINGS
    result = {
        "asof": pd.Timestamp("2026-05-29"),
        "weights": {"A": 0.5, "B": 0.5},
        "k_probs": {10: 0.25, 20: 0.25, 30: 0.25, 50: 0.25},
        "n_holdings": 2,        # below MIN_HOLDINGS (10)
        "weight_sum": 1.0,
        "max_weight": 0.5,
    }
    problems = validate_weights(result)
    assert any("holding" in p.lower() for p in problems), problems


def test_validate_weights_ok():
    from trading.weights import validate_weights
    weights = {f"T{i}": 1 / 20 for i in range(20)}
    result = {
        "asof": pd.Timestamp("2026-05-29"),
        "weights": weights,
        "k_probs": {10: 0.25, 20: 0.25, 30: 0.25, 50: 0.25},
        "n_holdings": 20,
        "weight_sum": sum(weights.values()),
        "max_weight": max(weights.values()),
    }
    assert validate_weights(result) == []
