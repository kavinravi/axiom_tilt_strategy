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
