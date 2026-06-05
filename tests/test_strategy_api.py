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
