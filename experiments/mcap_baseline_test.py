"""Compute weekly returns of the 'mcap-weighted top-30 with 10% cap' strategy.

For each walk N (1..17):
  - load walk-N scoreboard (already top-30 per Friday, has permno + date + fwd_ret_5d)
  - filter to test year = 2008 + N
  - join with panel to get mcap = prc * shrout for each (permno, date)
  - mcap-weights via project_to_simplex(log(mcap), max_weight=0.10) — softmax of
    log-mcap gives mcap-proportional, water-fill clips at 10%
  - portfolio_return = weights . fwd_ret_5d
Concatenate across walks → weekly file. Run cap10_vs_spy comparison.

usage: python -m experiments.mcap_baseline_test
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import processed_dir, repo_root
from src.utils.rl_env import project_to_simplex

REPO_ROOT = repo_root()
SCOREBOARD_ROOT = REPO_ROOT / "artifacts" / "rl"
PANEL_DIR = processed_dir() / "panel"
OUT_PATH = REPO_ROOT / "artifacts" / "backtest_046_cap10" / "weekly_mcap_baseline_cap10.parquet"

TOP_K = 30
MAX_WEIGHT = 0.10


def load_panel_year_mcap(year: int) -> pd.DataFrame:
    """Return (permno, date, mcap) for a given year. mcap = prc * shrout."""
    frames = []
    for p in sorted((PANEL_DIR / f"year={year}").glob("*.parquet")):
        df = pd.read_parquet(p, columns=["permno", "date", "prc", "shrout"])
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["permno", "date", "mcap"])
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["permno"] = panel["permno"].astype("int64")
    # mcap = prc * shrout (shrout in thousands per CRSP convention; abs for sign-flipped quotes)
    panel["mcap"] = np.abs(panel["prc"].astype(float)) * panel["shrout"].astype(float)
    return panel[["permno", "date", "mcap"]]


def mcap_weighted_top30(mcaps: np.ndarray) -> np.ndarray:
    """Softmax(log(mcap)) -> water-fill at 10%. Equivalent to mcap-proportional + cap."""
    safe = np.maximum(mcaps, 1e-8)
    return project_to_simplex(np.log(safe), max_weight=MAX_WEIGHT)


def main():
    all_rows = []
    for walk_id in range(1, 18):
        test_year = 2008 + walk_id
        sb_path = SCOREBOARD_ROOT / f"walk-{walk_id:03d}" / "scoreboard.parquet"
        if not sb_path.exists():
            print(f"walk {walk_id}: no scoreboard, skipping")
            continue
        sb = pd.read_parquet(sb_path)
        sb["date"] = pd.to_datetime(sb["date"])
        sb["permno"] = sb["permno"].astype("int64")
        sb_test = sb[(sb["date"] >= f"{test_year}-01-01") &
                     (sb["date"] <= f"{test_year}-12-31")].copy().reset_index(drop=True)
        if sb_test.empty:
            print(f"walk {walk_id}: empty test year {test_year}")
            continue

        # Look up mcap from panel for the test year (+ a small buffer in case
        # of cross-year dates after rolling joins)
        mcap_df = load_panel_year_mcap(test_year)
        sb_test = sb_test.merge(mcap_df, on=["permno", "date"], how="left")

        miss_rate = sb_test["mcap"].isna().mean()
        if miss_rate > 0.05:
            print(f"walk {walk_id} ({test_year}): {miss_rate:.1%} of rows missing mcap "
                  "(panel lookup failed). May need year-1 or year+1 panel partition.")

        by_date = {d: g.reset_index(drop=True) for d, g in sb_test.groupby("date")}
        dates = sorted(by_date.keys())
        rets = []
        for d in dates:
            cur = by_date[d]
            mcaps = cur["mcap"].to_numpy(dtype=np.float64)[:TOP_K]
            mcaps = np.where(np.isnan(mcaps), 0.0, mcaps)
            if mcaps.sum() <= 0:
                # All missing — fall back to equal-weight, log warning
                w = np.full(TOP_K, 1.0 / TOP_K)
            else:
                w = mcap_weighted_top30(mcaps)
            fwd = cur["fwd_ret_5d"].to_numpy(dtype=np.float64)[:TOP_K]
            fwd = np.where(np.isnan(fwd), 0.0, fwd)
            r = float(np.dot(w, fwd))
            rets.append({"date": d, "mcap_return_gross": r,
                         "max_weight_hit": float((w >= MAX_WEIGHT - 1e-6).sum())})

        ret_df = pd.DataFrame(rets)
        all_rows.append(ret_df)
        n_fri = len(dates)
        ann_ret = (1.0 + ret_df["mcap_return_gross"].mean()) ** 52 - 1.0
        print(f"walk {walk_id:>2} ({test_year}): {n_fri} Fri, mcap ann={ann_ret:.3f}")

    out = pd.concat(all_rows, ignore_index=True).sort_values("date").reset_index(drop=True)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, compression="zstd", index=False)
    print(f"\nwrote -> {OUT_PATH.relative_to(REPO_ROOT)} ({len(out)} weeks)")

    # Quick head-to-head print
    out["year"] = out["date"].dt.year
    def _m(rets):
        rets = np.asarray(rets, dtype=float)
        cum = float(np.prod(1.0 + rets) - 1.0)
        ann = (1.0 + cum) ** (52.0 / len(rets)) - 1.0
        vol = float(np.std(rets, ddof=1) * np.sqrt(52.0))
        sharpe = ann / vol if vol > 0 else 0.0
        eq = np.cumprod(1.0 + rets); peak = np.maximum.accumulate(eq)
        mdd = float((eq / peak - 1.0).min())
        return ann, vol, sharpe, mdd

    print(f"\n{'window':<14} {'weeks':>6} {'annret':>8} {'vol':>8} {'sharpe':>8} {'mdd':>8}")
    for label, sub in [
        ("2009-2025", out),
        ("2010-2024", out[(out["year"] >= 2010) & (out["year"] <= 2024)]),
        ("2010-2025", out[out["year"] >= 2010]),
    ]:
        ann, vol, sh, mdd = _m(sub["mcap_return_gross"])
        print(f"{label:<14} {len(sub):>6} {ann:>8.2%} {vol:>8.2%} {sh:>8.3f} {mdd:>8.2%}")


if __name__ == "__main__":
    main()
