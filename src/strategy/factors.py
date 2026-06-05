"""Deterministic factor scoring (value + quality composite).

Identifier-agnostic: operates on a generic id column (backtest passes `permno`,
live passes `ticker`). The id is carried through, never used in the math.
Mirrors experiments/regime_k_selector.py:58-67 exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def score_universe(snapshot_df: pd.DataFrame, id_col: str = "id",
                   date_col: str = "date") -> pd.DataFrame:
    """Add mcap, sp, fcfa, z_sp, z_fcfa, score columns to a snapshot.

    Requires columns: prc, shrout, marketcap, revenue, fcf, assets, plus
    `id_col` and `date_col`. Z-scores are cross-sectional per date.

    Assumes `marketcap` is either NaN (uses the |prc|*shrout fallback) or positive.
    A zero/negative marketcap yields sp=inf and zeroes z_sp for the entire snapshot
    date; the historical backtest is protected by upstream in_universe filtering, so
    live callers must ensure a positive market cap.
    """
    df = snapshot_df.copy()
    df["mcap"] = df["marketcap"].where(df["marketcap"].notna(),
                                       np.abs(df["prc"]) * df["shrout"])
    df["sp"] = (df["revenue"] / df["mcap"]).clip(lower=0)
    df["fcfa"] = (df["fcf"] / df["assets"]).clip(lower=-1.0, upper=2.0)
    df.loc[df["assets"] <= 0, "fcfa"] = np.nan
    for c in ["sp", "fcfa"]:
        g = df.groupby(date_col, sort=False)[c]
        df[f"z_{c}"] = (df[c] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        df[f"z_{c}"] = df[f"z_{c}"].fillna(0.0)
    df["score"] = 0.5 * df["z_sp"] + 0.5 * df["z_fcfa"]
    return df
