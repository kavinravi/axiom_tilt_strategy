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
