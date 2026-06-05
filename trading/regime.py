"""Build the current-Friday regime feature row for the K-selector model.

Reuses ``src.strategy.build_regime_features`` exactly — no regime math here.
The weekly Friday index spans REGIME_HISTORY_WEEKS periods ending at asof so
that the rolling window functions inside the core have enough history.

SPY and macro series are injectable for testing (no network in unit tests).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.strategy.k_selector import build_regime_features
from trading.config import REGIME_HISTORY_WEEKS


def build_current_regime_row(
    asof: pd.Timestamp,
    spy_weekly: pd.Series | None = None,
    macro: pd.DataFrame | None = None,
) -> np.ndarray:
    """Return the 7 REGIME_FEATURES for the rebalance Friday ``asof``.

    Args:
        asof:       The rebalance Friday (pd.Timestamp, time-normalized).
        spy_weekly: Optional pre-fetched weekly SPY close series indexed by the
                    same Friday DatetimeIndex used internally. When None, fetches
                    from FRED via ``trading.data.sources.fetch_spy_weekly``.
        macro:      Optional pre-fetched macro DataFrame indexed by the same
                    Friday DatetimeIndex. When None, fetches from FRED via
                    ``trading.data.sources.fetch_macro_history``.

    Returns:
        1-D numpy array of length 7, in the order defined by REGIME_FEATURES.
    """
    # Build the weekly Friday index: REGIME_HISTORY_WEEKS periods ending at asof
    index: pd.DatetimeIndex = pd.date_range(
        end=asof, periods=REGIME_HISTORY_WEEKS, freq="W-FRI"
    )

    # Fetch SPY and macro if not injected
    if spy_weekly is None:
        from trading.data.sources import fetch_spy_weekly  # noqa: PLC0415
        spy_weekly = fetch_spy_weekly(index, end=asof)

    if macro is None:
        from trading.data.sources import fetch_macro_history  # noqa: PLC0415
        macro = fetch_macro_history(index, end=asof)

    # Delegate all regime math to the validated core function
    regime_df = build_regime_features(index, spy_weekly, macro)

    # Return the last row (= the asof Friday) as a float numpy array
    return regime_df.iloc[-1].to_numpy(dtype=float)
