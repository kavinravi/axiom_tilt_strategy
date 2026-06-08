"""Tests for fetch_ticker_metadata (SHARADAR/TICKERS -> {ticker: name, sector}).

Uses a fake NDL client so no network is touched.
"""
from __future__ import annotations

import pandas as pd

from trading.data.sources import fetch_ticker_metadata


class _FakeNDL:
    """Minimal stand-in: get_table filters the canned frame by the ticker kwarg."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def get_table(self, datatable: str, **kw):
        # mirror the real ndl.get_table(code, table="SF1", ticker=[...], paginate=True)
        tickers = kw.get("ticker")
        if tickers is None:
            return self._df.copy()
        return self._df[self._df["ticker"].isin(tickers)].copy()


def _frame():
    return pd.DataFrame({
        "ticker": ["NVDA", "CVS"],
        "name": ["NVIDIA CORP", "CVS HEALTH CORP"],
        "sector": ["Technology", "Healthcare"],
    })


def test_fetch_ticker_metadata_maps_name_and_sector():
    out = fetch_ticker_metadata(["NVDA", "CVS"], ndl=_FakeNDL(_frame()))
    assert out["NVDA"] == {"company_name": "NVIDIA CORP", "sector": "Technology"}
    assert out["CVS"]["sector"] == "Healthcare"


def test_fetch_ticker_metadata_empty_input_returns_empty():
    assert fetch_ticker_metadata([], ndl=_FakeNDL(_frame())) == {}


def test_fetch_ticker_metadata_unknown_ticker_omitted():
    out = fetch_ticker_metadata(["NVDA", "ZZZZ"], ndl=_FakeNDL(_frame()))
    assert "NVDA" in out and "ZZZZ" not in out
