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
