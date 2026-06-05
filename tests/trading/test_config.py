def test_config_constants():
    from trading import config
    assert config.MODEL_PATH.name == "k_selector.txt"
    assert config.SF1_DIMENSION == "ARQ"
    assert set(config.FRED_MACRO_SERIES.values()) == {"macro_vixcls", "macro_dgs10", "macro_t10y2y"}
    assert config.REGIME_HISTORY_WEEKS >= 30
    assert config.MAX_WEIGHT == 0.10
