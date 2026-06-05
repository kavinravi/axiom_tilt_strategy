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
