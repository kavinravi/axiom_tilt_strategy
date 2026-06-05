"""Ingest 2025 S&P 500 universe daily prices from WRDS crsp.dsf_v2 (CIZ schema).

The parent axiom_tilt project's existing `ingest_wrds.py` uses the `wrds` Python
package which falls through to an interactive prompt in non-interactive runs.
That's fine for legacy 2024-and-prior pulls, but the 2025 data lives in
`crsp.dsf_v2` (CIZ schema) under the current subscription rather than the
legacy `crsp.dsf`. This module mirrors the Dow project's `ingest_dow_crsp_2025`
approach (raw psycopg2 + DictCursor) but pulls the S&P 500 universe via
`universe_ids.parquet` instead of the Dow constituents.

Outputs to data/processed/crsp_daily/year=2025/part-0.parquet using the same
legacy column schema as crsp.dsf so all downstream code stays unchanged.

Usage:
    python -m src.data.ingest_crsp_2025
    python -m src.data.ingest_crsp_2025 --start 2025-01-01 --end 2025-12-31
"""
from __future__ import annotations

import argparse
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg2
import psycopg2.extras

from src.utils.env import get_env
from src.utils.io import processed_dir
from src.utils.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

_LEGACY_COLS = [
    "permno", "date", "prc", "ret", "vol", "shrout",
    "openprc", "askhi", "bidlo", "cfacpr", "cfacshr",
    "dlret", "dlstcd",
]

_CIZ_TO_LEGACY = {
    "permno": "permno",
    "dlycaldt": "date",
    "dlyprc": "prc",
    "dlyret": "ret",
    "dlyvol": "vol",
    "shrout": "shrout",
    "dlyopen": "openprc",
    "dlyhigh": "askhi",
    "dlylow": "bidlo",
    "dlycumfacpr": "cfacpr",
    "dlycumfacshr": "cfacshr",
}

_DECIMAL_FLOAT_COLS = {"prc", "ret", "vol", "cfacpr", "cfacshr", "openprc", "askhi", "bidlo"}


def map_dsf_v2_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=_LEGACY_COLS)
    df = pd.DataFrame(rows)
    rename = {k: v for k, v in _CIZ_TO_LEGACY.items() if k in df.columns}
    df = df.rename(columns=rename)
    for c in _DECIMAL_FLOAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype(float)
    if "permno" in df.columns:
        df["permno"] = df["permno"].astype("int64")
    if "shrout" in df.columns:
        df["shrout"] = pd.to_numeric(df["shrout"], errors="coerce").astype("Int64")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    # Delisting fields are NA for normal 2025 trading days; legacy schema kept for compat.
    for c in ("dlret", "dlstcd"):
        if c not in df.columns:
            df[c] = pd.NA
    df = df[_LEGACY_COLS]
    return df


def _connect():
    username = get_env("WRDS_USERNAME", required=True)
    return psycopg2.connect(
        host="wrds-pgdata.wharton.upenn.edu",
        port=9737,
        dbname="wrds",
        user=username,
        sslmode="require",
    )


def fetch_dsf_v2(
    conn,
    permnos: list[int],
    start: str,
    end: str,
    chunk_size: int = 100,
) -> pd.DataFrame:
    sql_cols = (
        "permno, dlycaldt, dlyprc, dlyret, dlyvol, shrout, "
        "dlyopen, dlyhigh, dlylow, dlycumfacpr, dlycumfacshr"
    )
    chunks: list[pd.DataFrame] = []
    for i in range(0, len(permnos), chunk_size):
        batch = permnos[i : i + chunk_size]
        placeholders = ", ".join(["%s"] * len(batch))
        query = (
            f"SELECT {sql_cols} "
            f"FROM crsp.dsf_v2 "
            f"WHERE permno IN ({placeholders}) "
            f"  AND dlycaldt BETWEEN %s AND %s "
            f"ORDER BY permno, dlycaldt"
        )
        params = list(batch) + [start, end]
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        log.info("batch %d/%d: %d permnos -> %d rows",
                 i // chunk_size + 1, (len(permnos) + chunk_size - 1) // chunk_size,
                 len(batch), len(rows))
        if rows:
            chunks.append(map_dsf_v2_rows([dict(r) for r in rows]))
    if not chunks:
        return pd.DataFrame(columns=_LEGACY_COLS)
    return pd.concat(chunks, ignore_index=True).sort_values(["permno", "date"]).reset_index(drop=True)


def main(start: str = "2025-01-01", end: str = "2025-12-31") -> None:
    configure_logging()
    universe_path = processed_dir() / "universe_ids.parquet"
    universe = pd.read_parquet(universe_path)
    permnos = sorted(int(p) for p in universe["permno"].dropna().unique())
    log.info("S&P 500 universe: %d unique non-NA permnos", len(permnos))

    log.info("Connecting to WRDS crsp.dsf_v2 ...")
    conn = _connect()
    try:
        df = fetch_dsf_v2(conn, permnos, start=start, end=end)
    finally:
        conn.close()

    if len(df) == 0:
        raise RuntimeError(f"fetch_dsf_v2 returned 0 rows for {len(permnos)} permnos {start}-{end}")

    log.info("Fetched %d rows | %d permnos | %s -> %s",
             len(df), df["permno"].nunique(), df["date"].min().date(), df["date"].max().date())

    out_dir = processed_dir() / "crsp_daily" / "year=2025"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "part-0.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Wrote %d rows -> %s", len(df), out_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2025-12-31")
    args = p.parse_args()
    main(start=args.start, end=args.end)
