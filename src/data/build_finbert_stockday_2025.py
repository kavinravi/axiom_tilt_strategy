"""Build finbert_stockday_embed shards for year=2025 only.

Mirrors notebook 03's aggregation (30-day window, 14-day halflife exponential
decay weighting of doc embeds onto trading days), but restricts to 2025 panel
dates + 2024-Q4 lookback for the window. Writes per-permno per-year parquets
to data/processed/finbert_stockday_embed/year=2025/.

usage: python -m src.data.build_finbert_stockday_2025
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.io import processed_dir
from src.utils.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

WINDOW_DAYS = 30
HALFLIFE_DAYS = 14.0
TARGET_YEAR = 2025
LOOKBACK_START = pd.Timestamp(f"{TARGET_YEAR - 1}-12-01")  # need ~30d back into prior year

DOC_EMBED_DIR = processed_dir() / "finbert_doc_embed"
PANEL_DIR = processed_dir() / "panel"
OUT_DIR = processed_dir() / "finbert_stockday_embed"


def aggregate_permno(panel_g: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    """30d EWMA of doc embeds onto trading days. Forward-fill last good vec
    when current day has no in-window filing (matches notebook 03 logic)."""
    ciks = panel_g["cik"].unique()
    f = docs[docs["cik"].isin(ciks)].sort_values("filing_date")
    if f.empty:
        return pd.DataFrame(columns=["permno", "date", "vec"])
    fvecs = np.stack(f["vec"].values).astype(np.float32)
    fdates = f["filing_date"].values.astype("datetime64[D]")
    pdates = panel_g["date"].values.astype("datetime64[D]")
    days_lag = (pdates[:, None] - fdates[None, :]).astype("int64")
    in_window = (days_lag >= 0) & (days_lag <= WINDOW_DAYS)
    weights = np.where(in_window, 0.5 ** (days_lag / HALFLIFE_DAYS), 0.0).astype(np.float32)
    w_sum = weights.sum(axis=1)
    has_filing = w_sum > 0
    agg = (weights @ fvecs) / np.maximum(w_sum[:, None], 1e-12)
    agg = np.where(has_filing[:, None], agg, np.nan).astype(np.float32)
    last = None
    for i in range(agg.shape[0]):
        if has_filing[i]:
            last = agg[i]
        elif last is not None:
            agg[i] = last
    valid = ~np.isnan(agg[:, 0])
    if not valid.any():
        return pd.DataFrame(columns=["permno", "date", "vec"])
    return pd.DataFrame({
        "permno": panel_g["permno"].values[valid],
        "date": pdates[valid],
        "vec": [agg[i] for i in np.where(valid)[0]],
    })


SCHEMA = pa.schema([
    ("permno", pa.int64()),
    ("date", pa.timestamp("ns")),
    ("vec", pa.list_(pa.float32())),
])


def write_permno_year_2025(out_df: pd.DataFrame, permno: int) -> int:
    """Write only 2025 rows (post-lookback) for this permno."""
    if out_df.empty:
        return 0
    out_df = out_df.copy()
    out_df["date"] = pd.to_datetime(out_df["date"])
    out_df = out_df[out_df["date"].dt.year == TARGET_YEAR]
    if out_df.empty:
        return 0
    year_dir = OUT_DIR / f"year={TARGET_YEAR}"
    year_dir.mkdir(parents=True, exist_ok=True)
    out_path = year_dir / f"part-permno-{permno:08d}.parquet"
    records = [
        {"permno": int(r.permno), "date": pd.Timestamp(r.date), "vec": r.vec.tolist()}
        for r in out_df.itertuples(index=False)
    ]
    table = pa.Table.from_pylist(records, schema=SCHEMA)
    pq.write_table(table, out_path, compression="zstd")
    return len(records)


def main():
    configure_logging()

    log.info("Loading doc embeds (need 2024-Q4 + 2025)...")
    docs = duckdb.sql(
        f"SELECT cik, filing_date, vec FROM '{DOC_EMBED_DIR}/year=*/*.parquet' "
        f"WHERE filing_date >= DATE '{LOOKBACK_START.date()}'"
    ).df()
    docs["filing_date"] = pd.to_datetime(docs["filing_date"]).dt.normalize()
    docs["vec"] = [np.asarray(v, dtype=np.float32) for v in docs["vec"].values]
    docs["cik"] = docs["cik"].astype("int64").map(lambda x: f"{x:010d}")
    log.info("  %d doc embed rows | %d unique ciks", len(docs), docs["cik"].nunique())

    log.info("Loading panel rows for %s (with lookback)...", TARGET_YEAR)
    panel = duckdb.sql(
        f"SELECT permno, date, cik FROM '{PANEL_DIR}/year=*/*.parquet' "
        f"WHERE cik IS NOT NULL AND date >= DATE '{LOOKBACK_START.date()}'"
    ).df()
    panel["cik"] = panel["cik"].astype("int64").map(lambda x: f"{x:010d}")
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)
    log.info("  %d panel rows | %d permnos | %s -> %s",
             len(panel), panel["permno"].nunique(),
             panel["date"].min().date(), panel["date"].max().date())

    permnos = sorted(panel["permno"].unique())
    log.info("Processing %d permnos for year=%d shards...", len(permnos), TARGET_YEAR)

    n_perms_with_out, n_rows_out = 0, 0
    for i, pn in enumerate(permnos, 1):
        pg = panel[panel["permno"] == pn]
        out = aggregate_permno(pg, docs)
        n_written = write_permno_year_2025(out, pn)
        if n_written > 0:
            n_perms_with_out += 1
            n_rows_out += n_written
        if i % 100 == 0:
            log.info("  ...%d/%d permnos processed | %d wrote 2025 rows",
                     i, len(permnos), n_perms_with_out)

    log.info("Done. Wrote %d rows across %d permnos to %s/year=%d/",
             n_rows_out, n_perms_with_out, OUT_DIR.relative_to(processed_dir()), TARGET_YEAR)


if __name__ == "__main__":
    main()
