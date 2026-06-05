"""Assemble the current cross-section snapshot keyed by ticker.

The snapshot has exactly the columns expected by src.strategy.score_universe:
    ticker, date, prc, shrout, marketcap, revenue, fcf, assets

- prc / shrout come from SF1 (price / sharesbas).
- marketcap comes from SHARADAR/DAILY (live market cap); NaN when missing
  → score_universe falls back to prc * shrout (SF1 price * sharesbas, which equals
  market cap; sharesbas is an actual share count, not thousands).
- date is the rebalance Friday (asof) for every row.

Universe tickers with no fundamentals row are dropped with a logged warning.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Ordered column contract — consumed by score_universe
_SNAPSHOT_COLS = ["ticker", "date", "prc", "shrout", "marketcap", "revenue", "fcf", "assets"]


# ---------------------------------------------------------------------------
# Date utility
# ---------------------------------------------------------------------------

def most_recent_friday(today: pd.Timestamp | None = None) -> pd.Timestamp:
    """Return the most recent Friday ≤ today (inclusive if today is Friday).

    Args:
        today: Reference date. Defaults to the current wall-clock date (normalized).

    Returns:
        A normalized pd.Timestamp that falls on a Friday.
    """
    if today is None:
        today = pd.Timestamp.today().normalize()
    today = pd.Timestamp(today).normalize()
    # dayofweek: Monday=0 … Friday=4 … Sunday=6
    days_since_friday = (today.dayofweek - 4) % 7
    return today - pd.Timedelta(days=days_since_friday)


# ---------------------------------------------------------------------------
# Pure merge helper (no network; fully testable with small frames)
# ---------------------------------------------------------------------------

def assemble_snapshot(
    universe: list[str],
    fundamentals: pd.DataFrame,
    marketcaps: pd.DataFrame,
    asof: pd.Timestamp,
) -> pd.DataFrame:
    """Merge universe, fundamentals, and marketcaps into one snapshot row per ticker.

    Args:
        universe:     List of ticker strings (the current S&P 500 members).
        fundamentals: DataFrame with columns ticker, revenue, fcf, assets, price, sharesbas
                      (one row per ticker — the most-recent SF1 ARQ filing).
        marketcaps:   DataFrame with columns ticker, marketcap
                      (one row per ticker — the most-recent DAILY row ≤ asof).
        asof:         The rebalance Friday; becomes the ``date`` column for every row.

    Returns:
        DataFrame with exactly ``_SNAPSHOT_COLS`` columns, one row per universe ticker
        that has a fundamentals entry.  Tickers with no fundamentals are dropped and
        logged.  Tickers with no marketcap row get NaN in the marketcap column.
    """
    uni_df = pd.DataFrame({"ticker": universe})

    # --- Join fundamentals (inner — drops universe tickers with no filing) ---
    fund_cols = ["ticker", "revenue", "fcf", "assets", "price", "sharesbas"]
    fund = fundamentals[
        [c for c in fund_cols if c in fundamentals.columns]
    ].copy()

    merged = uni_df.merge(fund, on="ticker", how="left")

    # Detect and log/drop missing fundamentals
    missing_mask = merged["revenue"].isna() | merged["assets"].isna()
    if missing_mask.any():
        dropped = merged.loc[missing_mask, "ticker"].tolist()
        logger.warning(
            "Dropped %d universe ticker(s) with no fundamentals: %s",
            len(dropped),
            dropped,
        )
        merged = merged[~missing_mask].copy()

    # --- Left-join marketcaps (NaN where missing — score_universe handles fallback) ---
    mktcap_cols = ["ticker", "marketcap"]
    mktcap = marketcaps[
        [c for c in mktcap_cols if c in marketcaps.columns]
    ].copy()

    merged = merged.merge(mktcap, on="ticker", how="left")

    # --- Rename SF1 price/sharesbas to prc/shrout ---
    merged = merged.rename(columns={"price": "prc", "sharesbas": "shrout"})

    # --- Set date = asof for every row ---
    merged["date"] = asof

    # --- Return with exact column order ---
    return merged[_SNAPSHOT_COLS].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Wired builder (calls universe + sources + assemble)
# ---------------------------------------------------------------------------

def build_snapshot(asof: pd.Timestamp | None = None, ndl=None) -> pd.DataFrame:
    """Fetch current data from Sharadar and assemble the snapshot.

    Args:
        asof: Rebalance Friday. Defaults to the most recent Friday ≤ today.
        ndl:  Optional injected nasdaqdatalink client (for testing).

    Returns:
        DataFrame with columns ``_SNAPSHOT_COLS``, one row per current S&P 500
        ticker that has SF1 fundamentals available.
    """
    from trading.data.sources import latest_fundamentals, latest_marketcap  # noqa: PLC0415
    from trading.data.universe import get_current_sp500_tickers  # noqa: PLC0415

    if asof is None:
        asof = most_recent_friday()

    universe = get_current_sp500_tickers(ndl=ndl)
    fundamentals = latest_fundamentals(universe, asof=asof, ndl=ndl)
    marketcaps = latest_marketcap(universe, asof=asof, ndl=ndl)

    return assemble_snapshot(universe, fundamentals, marketcaps, asof)
