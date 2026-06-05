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
    The four SPY trailing features (spy_ret_*, spy_vol_*) are shift(1)-lagged to avoid look-ahead.
    The three macro level features are NOT shifted — they are contemporaneous Friday readings,
    consistent with the original experiment.
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
    """Fit one production classifier on all data (no early stopping / holdout).

    Deliberately trains all n_estimators trees — unlike the walk-forward backtest
    which uses early stopping on a held-out validation window. Same hyperparameters,
    so model capacity is equivalent.
    """
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
