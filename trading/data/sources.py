"""Sharadar (fundamentals + marketcap) and FRED (macro + SPY) data-source wrappers.

Each function takes an optional injected NDL client (`ndl=None`) so that tests
can pass a fake without hitting real network.  When `ndl` is None the real
`nasdaqdatalink` client is built from the NASDAQ_DATA_LINK_API_KEY env var.

FRED calls use pandas_datareader and need no API key.  A `_retry` wrapper
handles the occasional FRED timeout.
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Iterator, TypeVar

import pandas as pd

from trading.config import (
    FRED_MACRO_SERIES,
    FRED_SPY_SERIES,
    SF1_DIMENSION,
    SHARADAR_DAILY,
    SHARADAR_SF1,
)

logger = logging.getLogger(__name__)
T = TypeVar("T")

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _chunk(seq: list, n: int) -> Iterator[list]:
    """Yield successive n-sized chunks from a list."""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _retry(fn: Callable[[], T], attempts: int = 3, delay: float = 2.0) -> T:
    """Call `fn()`, retrying up to `attempts` times on any Exception."""
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(delay * (attempt + 1))
    raise last_exc  # type: ignore[misc]


def _build_ndl():
    """Build and configure the real nasdaqdatalink client from the env key."""
    import nasdaqdatalink as ndl  # noqa: PLC0415
    from src.utils.env import get_env  # noqa: PLC0415
    ndl.ApiConfig.api_key = get_env("NASDAQ_DATA_LINK_API_KEY", required=True)
    return ndl


# ------------------------------------------------------------------
# Sharadar wrappers
# ------------------------------------------------------------------

def latest_fundamentals(tickers: list[str], asof: pd.Timestamp | None = None, ndl=None) -> pd.DataFrame:
    """Return one row per ticker = the most-recent SF1 ARQ filing.

    Columns: ticker, datekey, revenue, fcf, assets, price, sharesbas.
    Tickers are chunked by 100 to stay within the Sharadar filter-size cap.

    When ``asof`` is given, the query is date-bounded to ``datekey`` in
    [asof-450d, asof] so SF1 returns only the few recent filings rather than each
    ticker's full multi-decade ARQ history (a major speedup for the live run) and
    stays point-in-time for a historical ``asof``.
    """
    if ndl is None:
        ndl = _build_ndl()

    _COLS = ["ticker", "datekey", "revenue", "fcf", "assets", "price", "sharesbas"]

    datekey_filter = None
    if asof is not None:
        asof = pd.Timestamp(asof)
        datekey_filter = {
            "gte": (asof - pd.Timedelta(days=450)).strftime("%Y-%m-%d"),
            "lte": asof.strftime("%Y-%m-%d"),
        }

    dfs: list[pd.DataFrame] = []
    for chunk in _chunk(tickers, 100):
        kwargs = {"ticker": chunk, "dimension": SF1_DIMENSION, "paginate": True}
        if datekey_filter is not None:
            kwargs["datekey"] = datekey_filter
        df = _retry(lambda kw=kwargs: ndl.get_table(SHARADAR_SF1, **kw))  # type: ignore[return-value]
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=_COLS)

    combined = pd.concat(dfs, ignore_index=True)
    # Keep only the columns we need (the table has many more)
    available = [c for c in _COLS if c in combined.columns]
    combined = combined[available].copy()

    # Ensure datekey is datetime for sorting
    combined["datekey"] = pd.to_datetime(combined["datekey"], errors="coerce")

    # Latest filing per ticker = max datekey
    combined = combined.sort_values("datekey")
    combined = combined.drop_duplicates("ticker", keep="last")

    return combined.reset_index(drop=True)


def fetch_ticker_metadata(tickers, ndl=None) -> dict[str, dict]:
    """Map ``ticker -> {"company_name", "sector"}`` from SHARADAR/TICKERS.

    Best-effort: returns {} (or a partial map) on any failure so the publisher
    never breaks just because metadata is unavailable. Unknown tickers are simply
    omitted. ``ndl`` is injectable for tests."""
    tickers = list(tickers)
    if not tickers:
        return {}
    if ndl is None:
        try:
            ndl = _build_ndl()
        except Exception as exc:  # noqa: BLE001
            logger.warning("ticker metadata: NDL client unavailable (%s)", exc)
            return {}

    out: dict[str, dict] = {}
    for chunk in _chunk(tickers, 100):
        try:
            df = _retry(lambda c=chunk: ndl.get_table(
                "SHARADAR/TICKERS", table="SF1", ticker=c, paginate=True))
        except Exception as exc:  # noqa: BLE001
            logger.warning("ticker metadata fetch failed for %d tickers (%s)", len(chunk), exc)
            continue
        if df is None or len(df) == 0:
            continue
        df = df.drop_duplicates("ticker")
        for _, row in df.iterrows():
            name = row["name"] if "name" in df.columns else None
            sector = row["sector"] if "sector" in df.columns else None
            out[str(row["ticker"])] = {
                "company_name": str(name) if pd.notna(name) else None,
                "sector": str(sector) if pd.notna(sector) else None,
            }
    return out


def latest_marketcap(
    tickers: list[str],
    asof: pd.Timestamp,
    ndl=None,
) -> pd.DataFrame:
    """Return one row per ticker = the most recent DAILY marketcap on or before `asof`.

    Columns: ticker, date, marketcap.
    """
    if ndl is None:
        ndl = _build_ndl()

    asof_str = asof.strftime("%Y-%m-%d")
    # Lower-bound the query (~10 calendar days covers weekends/holidays) so DAILY
    # returns only the last few rows per ticker, not years of history.
    start_str = (asof - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    _COLS = ["ticker", "date", "marketcap"]

    dfs: list[pd.DataFrame] = []
    for chunk in _chunk(tickers, 100):
        df = _retry(lambda c=chunk: ndl.get_table(  # type: ignore[return-value]
            SHARADAR_DAILY,
            ticker=c,
            date={"gte": start_str, "lte": asof_str},
            paginate=True,
        ))
        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=_COLS)

    combined = pd.concat(dfs, ignore_index=True)
    available = [c for c in _COLS if c in combined.columns]
    combined = combined[available].copy()

    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")

    # Filter to dates on or before asof (in case the API doesn't enforce it perfectly)
    combined = combined[combined["date"] <= asof]

    # Latest date per ticker
    combined = combined.sort_values("date")
    combined = combined.drop_duplicates("ticker", keep="last")

    return combined.reset_index(drop=True)


# ------------------------------------------------------------------
# Macro + SPY  (FRED primary; yfinance + AlphaVantage fallback)
# ------------------------------------------------------------------
# FRED's pandas_datareader CSV endpoint is occasionally unreachable; for a
# years-long autonomous system we fall back to yfinance (SPY ETF, ^VIX, ^TNX)
# plus AlphaVantage (2y treasury, for the 10y-2y spread). The fallback series are
# faithful proxies of the FRED series the model trained on: VIXCLS~^VIX,
# DGS10~^TNX, T10Y2Y~(^TNX - AV 2y), SP500~SPY ETF (the training SPY).

def _reindex_weekly(obj, index):
    """Forward-fill a daily-indexed Series/DataFrame onto the weekly-Friday index."""
    combined = obj.index.union(index).sort_values()
    return obj.reindex(combined).ffill().reindex(index)


# --- FRED REST API (api.stlouisfed.org) -----------------------------------
# Used when FRED_API_KEY is set. This host is reachable even where the
# pandas_datareader CSV host (fred.stlouisfed.org) is firewalled, and it returns
# the exact training series (VIXCLS, DGS10, T10Y2Y, SP500) with stable official
# closes — no yfinance/AlphaVantage drift.

def _parse_fred_observations(payload: dict) -> pd.Series:
    """Turn a FRED /series/observations JSON payload into a sorted float Series.

    Skips FRED's "." missing-value placeholder. Raises (fail-loud) when there are
    no usable observations, so the caller falls back instead of silently zeroing
    a regime feature."""
    obs = payload.get("observations") if isinstance(payload, dict) else None
    if not obs:
        raise RuntimeError(f"FRED API: no observations in response: {str(payload)[:160]}")
    s = pd.Series({pd.Timestamp(o["date"]): float(o["value"])
                   for o in obs if o.get("value") not in (".", None, "")})
    if s.empty:
        raise RuntimeError("FRED API: observations parsed to an empty series (all placeholders?)")
    return s.sort_index()


def _fred_api_series(series_id: str, start, end, key: str) -> pd.Series:
    """Fetch one FRED series from the REST API host as a daily float Series."""
    import json  # noqa: PLC0415
    import urllib.parse  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415
    params = urllib.parse.urlencode({
        "series_id": series_id, "api_key": key, "file_type": "json",
        "observation_start": pd.Timestamp(start).strftime("%Y-%m-%d"),
        "observation_end": pd.Timestamp(end).strftime("%Y-%m-%d"),
    })
    url = f"https://api.stlouisfed.org/fred/series/observations?{params}"

    def _pull():
        with urllib.request.urlopen(url, timeout=25) as resp:
            return json.load(resp)

    return _parse_fred_observations(_retry(_pull, attempts=3, delay=2.0))


def _fred_api_macro(index: pd.DatetimeIndex, end: pd.Timestamp, key: str) -> pd.DataFrame:
    start = index.min() - pd.Timedelta(days=90)
    cols = {col: _fred_api_series(fred_id, start, end, key)
            for fred_id, col in FRED_MACRO_SERIES.items()}
    df = pd.DataFrame(cols)
    return _reindex_weekly(df.ffill(), index)[list(FRED_MACRO_SERIES.values())]


def _fred_api_spy(index: pd.DatetimeIndex, end: pd.Timestamp, key: str) -> pd.Series:
    start = index.min() - pd.Timedelta(days=30)
    s = _reindex_weekly(_fred_api_series(FRED_SPY_SERIES, start, end, key).ffill(), index)
    s.name = "close"
    return s


def _fred_macro(index: pd.DatetimeIndex, end: pd.Timestamp) -> pd.DataFrame:
    from pandas_datareader import data as pdr  # noqa: PLC0415
    start = index.min() - pd.Timedelta(days=90)
    names = list(FRED_MACRO_SERIES.keys())  # ["VIXCLS", "DGS10", "T10Y2Y"]
    raw = _retry(lambda: pdr.DataReader(names, "fred", start, end), attempts=1)
    raw = _reindex_weekly(raw.ffill(), index).rename(columns=FRED_MACRO_SERIES)
    return raw[list(FRED_MACRO_SERIES.values())]


def _av_treasury_2year() -> pd.Series:
    """Daily 2-year constant-maturity treasury yield (percent) from AlphaVantage."""
    import json  # noqa: PLC0415
    import urllib.request  # noqa: PLC0415
    from src.utils.env import get_env  # noqa: PLC0415
    key = get_env("ALPHAVANTAGE_API_KEY", required=True)
    url = ("https://www.alphavantage.co/query?function=TREASURY_YIELD"
           f"&interval=daily&maturity=2year&apikey={key}")

    def _pull():
        with urllib.request.urlopen(url, timeout=25) as resp:
            payload = json.load(resp)
        data = payload.get("data") if isinstance(payload, dict) else None
        if not data:  # rate-limit / error responses lack "data" — raise so _retry waits
            raise RuntimeError(f"AlphaVantage 2y treasury: {str(payload)[:160]}")
        return data

    data = _retry(_pull, attempts=4, delay=15.0)  # free tier rate-limits; space out
    s = pd.Series({pd.Timestamp(x["date"]): float(x["value"])
                   for x in data if x.get("value") not in (".", None, "")})
    if s.empty:  # fail loud — never let an empty 2y series silently zero the t10y2y feature
        raise RuntimeError("AlphaVantage 2y treasury parsed to an empty series (all placeholders?)")
    return s.sort_index()


def _yf_macro(index: pd.DatetimeIndex, end: pd.Timestamp) -> pd.DataFrame:
    import yfinance as yf  # noqa: PLC0415
    start = (index.min() - pd.Timedelta(days=120)).strftime("%Y-%m-%d")
    end_str = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = _retry(lambda: yf.download(["^VIX", "^TNX"], start=start, end=end_str,
                                     interval="1d", progress=False, threads=False,
                                     auto_adjust=False))
    close = raw["Close"]
    df = pd.DataFrame({"macro_vixcls": close["^VIX"], "macro_dgs10": close["^TNX"]})
    _idx = pd.to_datetime(df.index)
    df.index = _idx.tz_localize(None) if _idx.tz is None else _idx.tz_convert(None)
    two_y = _av_treasury_2year()
    all_idx = df.index.union(two_y.index).sort_values()
    df = df.reindex(all_idx).ffill()
    df["macro_t10y2y"] = df["macro_dgs10"] - two_y.reindex(all_idx).ffill()
    df = df[["macro_vixcls", "macro_dgs10", "macro_t10y2y"]]
    return _reindex_weekly(df, index)


def fetch_macro_history(index: pd.DatetimeIndex, end: pd.Timestamp) -> pd.DataFrame:
    """Macro features (macro_vixcls, macro_dgs10, macro_t10y2y) on the weekly index.

    Order: FRED REST API (api.stlouisfed.org, needs FRED_API_KEY) → FRED CSV via
    pandas_datareader → yfinance (^VIX, ^TNX) + AlphaVantage (2y) fallback. The
    REST API is preferred because it is reachable where the CSV host is firewalled
    and returns the exact, stable training series.
    """
    from src.utils.env import get_env  # noqa: PLC0415
    key = get_env("FRED_API_KEY", default="")
    if key:
        try:
            return _fred_api_macro(index, end, key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FRED API macro unavailable (%s); trying CSV then yfinance", exc)
    try:
        return _fred_macro(index, end)
    except Exception as exc:  # noqa: BLE001
        logger.warning("FRED macro unavailable (%s); falling back to yfinance+AlphaVantage", exc)
        return _yf_macro(index, end)


def _fred_spy(index: pd.DatetimeIndex, end: pd.Timestamp) -> pd.Series:
    from pandas_datareader import data as pdr  # noqa: PLC0415
    start = index.min() - pd.Timedelta(days=30)
    raw = _retry(lambda: pdr.DataReader(FRED_SPY_SERIES, "fred", start, end), attempts=1)
    s = _reindex_weekly(raw.squeeze().ffill(), index)
    s.name = "close"
    return s


def _yf_spy(index: pd.DatetimeIndex, end: pd.Timestamp) -> pd.Series:
    import yfinance as yf  # noqa: PLC0415
    start = (index.min() - pd.Timedelta(days=45)).strftime("%Y-%m-%d")
    end_str = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = _retry(lambda: yf.download("SPY", start=start, end=end_str, interval="1d",
                                     progress=False, threads=False, auto_adjust=False))
    s = raw["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    _idx = pd.to_datetime(s.index)
    s.index = _idx.tz_localize(None) if _idx.tz is None else _idx.tz_convert(None)
    s = _reindex_weekly(s.ffill(), index)
    s.name = "close"
    return s


def fetch_spy_weekly(index: pd.DatetimeIndex, end: pd.Timestamp) -> pd.Series:
    """Weekly Friday SPY close series.

    Order: FRED REST API (SP500, needs FRED_API_KEY) → FRED CSV via
    pandas_datareader → yfinance SPY fallback.
    """
    from src.utils.env import get_env  # noqa: PLC0415
    key = get_env("FRED_API_KEY", default="")
    if key:
        try:
            return _fred_api_spy(index, end, key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FRED API SPY unavailable (%s); trying CSV then yfinance", exc)
    try:
        return _fred_spy(index, end)
    except Exception as exc:  # noqa: BLE001
        logger.warning("FRED SPY unavailable (%s); falling back to yfinance SPY", exc)
        return _yf_spy(index, end)
