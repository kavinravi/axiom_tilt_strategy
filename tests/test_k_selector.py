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
