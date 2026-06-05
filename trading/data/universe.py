"""Current S&P 500 membership from the Sharadar SP500 constituents table."""
from __future__ import annotations

import pandas as pd

from trading.config import SHARADAR_SP500


def current_members_from_sp500_table(sp500_df: pd.DataFrame) -> list[str]:
    """Return the sorted current member tickers from a SHARADAR/SP500 action frame.

    The table marks current members explicitly with action=='current'. We use
    that set directly (it is the vendor's current-membership snapshot)."""
    cur = sp500_df.loc[sp500_df["action"] == "current", "ticker"]
    return sorted(cur.dropna().astype(str).unique().tolist())


def get_current_sp500_tickers(ndl=None) -> list[str]:
    """Fetch + parse current S&P 500 tickers from Sharadar. ~503 names."""
    if ndl is None:
        import nasdaqdatalink as ndl  # noqa: PLC0415
        from src.utils.env import get_env
        ndl.ApiConfig.api_key = get_env("NASDAQ_DATA_LINK_API_KEY", required=True)
    df = ndl.get_table(SHARADAR_SP500, paginate=True)
    return current_members_from_sp500_table(df)
