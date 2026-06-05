"""Rebuild scoreboards for all 17 walks using the pruned rankers.

For each walk:
  - Load walk-NNN_pruned/model.joblib (just retrained, 154 features)
  - Load walk PCA + panel + embeds spanning 2002 → walk's test end year
  - Project embeds to PCA, Friday-filter, predict scores
  - Build top-30 scoreboard, save to artifacts/rl_pruned/walk-NNN/scoreboard.parquet
    (separate from existing artifacts/rl/walk-NNN/ to avoid clobbering)

Then runs deterministic mcap-top30 backtest on the new scoreboards across all walks
and compares to the original (non-pruned) mcap-top30 numbers.

usage: python experiments/rebuild_scoreboards_pruned.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.io import processed_dir, repo_root
from src.utils.ranker import friday_only, load_walk_pca, project_text_to_pca
from src.utils.rl_env import build_scoreboard_from_scored_panel, project_to_simplex

REPO_ROOT = repo_root()
PANEL_DIR = processed_dir() / "training_panel"
EMBED_DIR = processed_dir() / "finbert_stockday_embed"
RANKER_ROOT = REPO_ROOT / "artifacts" / "ranker"
OUT_ROOT = REPO_ROOT / "artifacts" / "rl_pruned"
TOP_K = 30
MAX_WEIGHT = 0.10


def load_years(dir_: Path, start: str, end: str, cols=None) -> pd.DataFrame:
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    frames = []
    for y in range(s.year, e.year + 1):
        for p in sorted((dir_ / f"year={y}").glob("*.parquet")):
            df = pd.read_parquet(p, columns=cols)
            df["date"] = pd.to_datetime(df["date"])
            df = df[(df["date"] >= s) & (df["date"] <= e)]
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_year_mcap(year: int) -> pd.DataFrame:
    frames = []
    panel_dir = processed_dir() / "panel"
    for p in sorted((panel_dir / f"year={year}").glob("*.parquet")):
        df = pd.read_parquet(p, columns=["permno", "date", "prc", "shrout"])
        df["date"] = pd.to_datetime(df["date"])
        df["permno"] = df["permno"].astype("int64")
        df["mcap"] = np.abs(df["prc"].astype(float)) * df["shrout"].astype(float)
        frames.append(df[["permno", "date", "mcap"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_scoreboard(walk_id: int) -> pd.DataFrame:
    """Build full (train+val+test) scoreboard for walk_id using pruned ranker."""
    pruned_dir = RANKER_ROOT / f"walk-{walk_id:03d}_pruned"
    bundle = joblib.load(pruned_dir / "model.joblib")
    model = bundle["model"]
    features = bundle["feature_names"]

    pca, n_pca = load_walk_pca(walk_id)
    test_year = 2008 + walk_id  # walk-1 → 2009 ... walk-17 → 2025

    panel = load_years(PANEL_DIR, "2002-01-01", f"{test_year}-12-31")
    embed = load_years(EMBED_DIR, "2002-01-01", f"{test_year}-12-31",
                       cols=["permno", "date", "vec"])
    embed_pca = project_text_to_pca(embed, pca)

    fri = friday_only(panel).merge(embed_pca, on=["permno", "date"], how="inner")
    fri = fri.dropna(subset=["fwd_ret_5d"]).copy()

    X = pd.DataFrame({c: fri[c] if c in fri.columns else np.nan for c in features})
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    fri["score"] = model.predict(X.to_numpy())

    sb = build_scoreboard_from_scored_panel(fri, top_k=TOP_K)
    # Join mcap (= |prc| * shrout from panel) onto the top-30 set
    sb["date"] = pd.to_datetime(sb["date"])
    sb["permno"] = sb["permno"].astype("int64")
    years = sorted({int(y) for y in sb["date"].dt.year.unique()})
    mcap = pd.concat([load_year_mcap(y) for y in years], ignore_index=True)
    sb = sb.merge(mcap, on=["permno", "date"], how="left")
    sb["mcap"] = sb.sort_values(["permno", "date"]).groupby("permno")["mcap"].ffill()
    sb["mcap"] = sb["mcap"].fillna(1.0)
    return sb


def mcap_weighted_top30(mcaps: np.ndarray) -> np.ndarray:
    safe = np.maximum(mcaps, 1e-8)
    return project_to_simplex(np.log(safe), max_weight=MAX_WEIGHT)


def deterministic_backtest(sb_test: pd.DataFrame, test_year: int) -> tuple[list, list]:
    """Returns (dates, weekly_returns) for mcap-weighted top-30 on test_year."""
    df = sb_test[(sb_test["date"] >= f"{test_year}-01-01") &
                 (sb_test["date"] <= f"{test_year}-12-31")].copy().reset_index(drop=True)
    by_date = {d: g.reset_index(drop=True) for d, g in df.groupby("date")}
    dates = sorted(by_date.keys())
    rets = []
    for d in dates:
        cur = by_date[d]
        mcaps = cur["mcap"].to_numpy(dtype=np.float64)[:TOP_K]
        mcaps = np.where(np.isnan(mcaps), 0.0, mcaps)
        if mcaps.sum() <= 0:
            w = np.full(TOP_K, 1.0 / TOP_K)
        else:
            w = mcap_weighted_top30(mcaps)
        fwd = cur["fwd_ret_5d"].to_numpy(dtype=np.float64)[:TOP_K]
        fwd = np.where(np.isnan(fwd), 0.0, fwd)
        rets.append(float(np.dot(w, fwd)))
    return dates, rets


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_dates, all_rets = [], []

    for walk_id in range(1, 18):
        t0 = time.time()
        out_path = OUT_ROOT / f"walk-{walk_id:03d}" / "scoreboard.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        sb = build_scoreboard(walk_id)
        sb.to_parquet(out_path, compression="zstd", index=False)
        test_year = 2008 + walk_id
        dates, rets = deterministic_backtest(sb, test_year)
        all_dates.extend(dates)
        all_rets.extend(rets)
        elapsed = time.time() - t0
        ann = (1.0 + np.mean(rets)) ** 52 - 1.0
        print(f"walk {walk_id:2d} ({test_year}): {len(dates)} Fri, "
              f"mcap-top30 ann={ann:+.3f}, sb_rows={len(sb)} [{elapsed:.0f}s]")

    out = pd.DataFrame({"date": all_dates, "mcap_return_pruned": all_rets})
    out.to_parquet(REPO_ROOT / "artifacts" / "backtest_046_cap10" /
                   "weekly_mcap_baseline_cap10_pruned.parquet",
                   compression="zstd", index=False)

    # Compare to original (non-pruned) mcap-top30 backtest
    orig_path = REPO_ROOT / "artifacts" / "backtest_046_cap10" / "weekly_mcap_baseline_cap10.parquet"
    orig = pd.read_parquet(orig_path)
    orig["date"] = pd.to_datetime(orig["date"])
    out["date"] = pd.to_datetime(out["date"])
    m = orig.merge(out, on="date", how="inner")
    m["year"] = m["date"].dt.year

    def metrics(rets):
        rets = np.asarray(rets, dtype=float)
        cum = float(np.prod(1.0 + rets) - 1.0)
        ann = (1.0 + cum) ** (52.0 / len(rets)) - 1.0
        vol = float(np.std(rets, ddof=1) * np.sqrt(52.0))
        sh = ann / vol if vol > 0 else 0.0
        eq = np.cumprod(1.0 + rets); peak = np.maximum.accumulate(eq)
        mdd = float((eq / peak - 1.0).min())
        return ann, vol, sh, mdd

    print("\n=== DETERMINISTIC MCAP-TOP30 vs SPY (pruned ranker) ===")
    print(f"{'window':<12} {'strat':<14} {'ann':>8} {'vol':>8} {'sharpe':>8} {'mdd':>8}")
    for label, sub in [
        ("2009-2025", m),
        ("2010-2024", m[(m["year"] >= 2010) & (m["year"] <= 2024)]),
        ("2010-2025", m[m["year"] >= 2010]),
    ]:
        for name, col in [("original (190)", "mcap_return_gross"),
                          ("pruned (154) ", "mcap_return_pruned")]:
            ann, vol, sh, mdd = metrics(sub[col])
            print(f"{label:<12} {name:<14} {ann:>8.2%} {vol:>8.2%} {sh:>8.3f} {mdd:>8.2%}")
        print()


if __name__ == "__main__":
    main()
