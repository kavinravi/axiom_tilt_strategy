"""Multi-strategy deterministic backtest on factor-screen-v1 picks.

For each walk's test year, takes the top-30 picks from artifacts/rl_factor_v1/
and applies several weighting schemes:

  - mcap        : mcap-weighted with 10% cap (drop-in vs current best)
  - equal       : equal-weight (1/30, never hits cap)
  - invvol      : inverse-trailing-vol weights with 10% cap
  - minvar      : portfolio min-variance MVO with PIT shrinkage covariance + 10% cap
  - maxsharpe   : portfolio max-Sharpe MVO with PIT mu/cov + 10% cap
  - voltarget   : maxsharpe + target_vol=18% (cash buffer if needed)

Compares each to SPY across 2009-2025 / 2010-2024 / 2010-2025 windows.
Writes weekly returns per-strategy to artifacts/backtest_factor_v1/.

Includes a side-by-side with current best (mcap-top30 on LightGBM ranker picks).

usage: python experiments/factor_screen_backtests.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import processed_dir, repo_root
from src.utils.rl_env import project_to_simplex
from src.utils.logging_utils import configure_logging, get_logger

log = get_logger(__name__)

REPO_ROOT = repo_root()
PANEL_DIR = processed_dir() / "panel"
FACTOR_SB = REPO_ROOT / "artifacts" / "rl_factor_v1"
RANKER_SB = REPO_ROOT / "artifacts" / "rl"  # original LightGBM ranker scoreboards
SPY_PATH = REPO_ROOT / "artifacts" / "benchmarks" / "spy_daily.parquet"
OUT_DIR = REPO_ROOT / "artifacts" / "backtest_factor_v1"
TOP_K = 30
MAX_WEIGHT = 0.10
VOL_LOOKBACK = 252
EPS = 1e-8


# ----------- Weighting schemes -----------

def w_equal(_mcaps, _cov, _mu):
    return np.full(TOP_K, 1.0 / TOP_K)


def w_mcap(mcaps, _cov, _mu):
    safe = np.maximum(mcaps, EPS)
    return project_to_simplex(np.log(safe), max_weight=MAX_WEIGHT)


def w_invvol(_mcaps, cov, _mu):
    vols = np.sqrt(np.diag(cov))
    safe = np.maximum(vols, EPS)
    raw = 1.0 / safe
    return project_to_simplex(np.log(raw), max_weight=MAX_WEIGHT)


def _solve_qp_box(cov, mu_target_max, max_w=MAX_WEIGHT, n_iter=200, lr=0.01,
                  obj="minvar"):
    """Simple projected-gradient solver for the long-only, capped MVO.

    objectives:
      minvar    : min w' cov w
      maxsharpe : max mu' w / sqrt(w' cov w) — projected gradient ascent on mu/vol

    Box constraints: 0 <= w_i <= max_w
    Simplex constraint: sum(w) = 1, enforced via project_to_simplex with cap.
    """
    K = cov.shape[0]
    w = np.full(K, 1.0 / K, dtype=np.float64)

    for _ in range(n_iter):
        if obj == "minvar":
            grad = 2.0 * cov @ w  # gradient of w' cov w
            w_new = w - lr * grad
        elif obj == "maxsharpe":
            # gradient of mu' w / sqrt(w' cov w)
            port_var = float(w @ cov @ w) + EPS
            port_vol = np.sqrt(port_var)
            grad = mu_target_max / port_vol - (float(mu_target_max @ w) / (port_var * port_vol)) * (cov @ w)
            w_new = w + lr * grad
        else:
            raise ValueError(obj)

        # Project to simplex with cap (uses the same softmax+water-fill as the rest of the code)
        # but we want a more direct simplex projection here since w_new is already roughly
        # on the simplex. Use box-and-rescale projection.
        w_new = np.clip(w_new, 0.0, max_w)
        s = w_new.sum()
        if s == 0:
            w_new = np.full(K, 1.0 / K)
        else:
            # Rescale to sum=1, then re-clip if any went over cap, iterate water-fill
            w_new = w_new / s
            for _it in range(50):
                over = w_new > max_w + EPS
                if not over.any():
                    break
                excess = float((w_new[over] - max_w).sum())
                w_new[over] = max_w
                under = ~over
                under_sum = float(w_new[under].sum())
                if under_sum > 0:
                    w_new[under] += excess * w_new[under] / under_sum
                else:
                    break
        w = w_new
    return w


def w_minvar(_mcaps, cov, _mu):
    return _solve_qp_box(cov, np.zeros(TOP_K), obj="minvar")


def w_maxsharpe(_mcaps, cov, mu):
    return _solve_qp_box(cov, mu, obj="maxsharpe")


# ----------- Covariance estimation -----------

def ledoit_wolf_shrink(cov: np.ndarray, intensity: float = 0.30) -> np.ndarray:
    """Shrink toward a constant-correlation target. Simpler than full LW."""
    K = cov.shape[0]
    vols = np.sqrt(np.diag(cov))
    corr = cov / np.outer(vols, vols).clip(min=EPS)
    # Average off-diagonal correlation
    iu = np.triu_indices(K, k=1)
    avg_corr = float(np.mean(corr[iu]))
    target_corr = np.full((K, K), avg_corr)
    np.fill_diagonal(target_corr, 1.0)
    shrunk_corr = (1 - intensity) * corr + intensity * target_corr
    shrunk_cov = shrunk_corr * np.outer(vols, vols)
    return shrunk_cov


def estimate_returns_cov(daily_ret_subset: pd.DataFrame, K: int = TOP_K) -> tuple[np.ndarray, np.ndarray]:
    """Given a wide (date × permno) daily returns matrix, estimate (mu, cov).
    Returns annualized values. Handles missing columns / short history defensively."""
    # Defensive: ensure 2D, fill NaN with 0 (no-return day for that stock)
    if isinstance(daily_ret_subset, pd.Series):
        daily_ret_subset = daily_ret_subset.to_frame()
    rets = daily_ret_subset.fillna(0.0).to_numpy(dtype=np.float64)
    if rets.ndim == 1:
        rets = rets.reshape(-1, 1)

    n_obs, n_assets = rets.shape

    if n_obs < 30 or n_assets == 0:
        # Not enough data → fallback: zero mean, identity cov (low magnitude)
        mu = np.zeros(K)
        cov = np.eye(K) * (0.20 ** 2 / 252)  # ~20% vol baseline
        cov = cov * 252
        return mu, cov

    mu_partial = rets.mean(axis=0) * 252
    cov_d = np.cov(rets, rowvar=False, ddof=1)
    if cov_d.ndim == 0:  # single asset → scalar
        cov_d = np.array([[float(cov_d)]])
    cov_a = cov_d * 252
    cov_shrunk = ledoit_wolf_shrink(cov_a, intensity=0.30)

    # Pad if returned dimension < K (defensive — caller might use full K-length vectors)
    if n_assets < K:
        mu_full = np.zeros(K)
        cov_full = np.eye(K) * (0.20 ** 2)  # 20% vol baseline (annualized)
        mu_full[:n_assets] = mu_partial
        cov_full[:n_assets, :n_assets] = cov_shrunk
        return mu_full, cov_full

    return mu_partial, cov_shrunk


# ----------- Daily return panel for cov estimation -----------

_daily_cache: dict[int, pd.DataFrame] = {}

def load_daily_panel(year_start: int, year_end: int) -> pd.DataFrame:
    """Cache + concat panel year shards for (year_start..year_end), return
    (permno, date, ret) frame."""
    key = (year_start, year_end)
    if key in _daily_cache:
        return _daily_cache[key]
    frames = []
    for y in range(year_start, year_end + 1):
        for p in sorted((PANEL_DIR / f"year={y}").glob("*.parquet")):
            df = pd.read_parquet(p, columns=["permno", "date", "ret"])
            df["date"] = pd.to_datetime(df["date"])
            df["permno"] = df["permno"].astype("int64")
            frames.append(df)
    daily = pd.concat(frames, ignore_index=True).sort_values(["permno", "date"]).reset_index(drop=True)
    _daily_cache[key] = daily
    return daily


# ----------- Backtest driver -----------

def backtest_walk(walk_id: int, scheme_name: str, w_fn) -> pd.DataFrame:
    """Run one weighting scheme on walk_id's test year scoreboard."""
    test_year = 2008 + walk_id
    sb_path = FACTOR_SB / f"walk-{walk_id:03d}" / "scoreboard.parquet"
    sb = pd.read_parquet(sb_path)
    sb["date"] = pd.to_datetime(sb["date"])
    sb["permno"] = sb["permno"].astype("int64")
    sb_test = sb[(sb["date"] >= f"{test_year}-01-01") &
                 (sb["date"] <= f"{test_year}-12-31")].copy().reset_index(drop=True)

    # Daily returns lookup (need ~252 days back from earliest test date for cov)
    daily = load_daily_panel(test_year - 1, test_year)
    daily_pivot = daily.pivot_table(index="date", columns="permno", values="ret", aggfunc="first")

    by_date = {d: g.reset_index(drop=True) for d, g in sb_test.groupby("date")}
    dates = sorted(by_date.keys())
    rows = []
    for d in dates:
        cur = by_date[d]
        permnos = cur["permno"].to_numpy()[:TOP_K]
        mcaps = cur["mcap"].to_numpy(dtype=np.float64)[:TOP_K]

        # Trailing daily returns window for cov: from t-VOL_LOOKBACK to t-1 (exclusive of t)
        end_idx = daily_pivot.index.searchsorted(d)
        start_idx = max(0, end_idx - VOL_LOOKBACK)
        # Defensive: drop permnos not in pivot's columns (may happen for delisted)
        avail = [p for p in permnos if p in daily_pivot.columns]
        if end_idx > 0 and avail:
            window = daily_pivot.iloc[start_idx:end_idx][avail]
            # If we have fewer columns than TOP_K, pad columns with 0 returns
            if len(avail) < TOP_K:
                missing = [p for p in permnos if p not in avail]
                missing_df = pd.DataFrame(0.0, index=window.index, columns=missing)
                window = pd.concat([window, missing_df], axis=1)[list(permnos)]
            else:
                window = window[list(permnos)]
        else:
            window = pd.DataFrame(0.0, index=[], columns=list(permnos))
        mu, cov = estimate_returns_cov(window, K=TOP_K)

        w = w_fn(mcaps, cov, mu)
        # Clamp + renormalize defensively
        w = np.clip(w, 0.0, MAX_WEIGHT)
        if w.sum() <= 0:
            w = np.full(TOP_K, 1.0 / TOP_K)
        else:
            w = w / w.sum()
        fwd = cur["fwd_ret_5d"].to_numpy(dtype=np.float64)[:TOP_K]
        fwd = np.where(np.isnan(fwd), 0.0, fwd)
        rows.append({"date": d, "weekly_ret": float(np.dot(w, fwd)),
                     "turnover": float(np.sum(np.abs(w - w_fn(mcaps, cov, mu) if False else w))),  # placeholder
                     "max_w": float(w.max())})
    return pd.DataFrame(rows)


def metrics(rets: np.ndarray) -> dict:
    rets = np.asarray(rets, dtype=float)
    if len(rets) < 2:
        return {}
    cum = float(np.prod(1.0 + rets) - 1.0)
    ann_ret = (1.0 + cum) ** (52.0 / len(rets)) - 1.0
    ann_vol = float(np.std(rets, ddof=1) * np.sqrt(52.0))
    sh = ann_ret / ann_vol if ann_vol > 0 else 0.0
    eq = np.cumprod(1.0 + rets); peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    calmar = ann_ret / abs(mdd) if mdd < 0 else 0.0
    hit = float((rets > 0).mean())
    return {"n_weeks": int(len(rets)), "total_ret": cum, "ann_ret": ann_ret,
            "ann_vol": ann_vol, "sharpe": sh, "mdd": mdd, "calmar": calmar, "hit_rate": hit}


def build_spy_weekly(dates_idx: pd.DatetimeIndex) -> pd.Series:
    spy = pd.read_parquet(SPY_PATH).reset_index()
    spy = spy[["Date", "close"]].rename(columns={"Date": "date"})
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").set_index("date")["close"]
    union = spy.index.union(dates_idx)
    closes = spy.reindex(union).sort_index().ffill().reindex(dates_idx)
    rets = closes.pct_change().fillna(0.0).to_numpy()
    return pd.Series(rets, index=dates_idx, name="spy_return")


def main():
    configure_logging()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    schemes = [
        ("mcap", w_mcap),
        ("equal", w_equal),
        ("invvol", w_invvol),
        ("minvar", w_minvar),
        ("maxsharpe", w_maxsharpe),
    ]

    # Run each scheme across all 17 walks
    all_rets = {name: [] for name, _ in schemes}
    for walk_id in range(1, 18):
        t0 = time.time()
        for name, fn in schemes:
            df = backtest_walk(walk_id, name, fn)
            df["scheme"] = name
            df["walk_id"] = walk_id
            all_rets[name].append(df)
        log.info("walk %2d done [%.1fs]", walk_id, time.time() - t0)

    # Concatenate per-scheme weekly returns
    concat: dict[str, pd.DataFrame] = {}
    for name in all_rets:
        concat[name] = pd.concat(all_rets[name], ignore_index=True).sort_values("date").reset_index(drop=True)
        concat[name].to_parquet(OUT_DIR / f"weekly_factor_v1_{name}.parquet",
                                compression="zstd", index=False)

    # Build SPY weekly aligned to (any) scheme's dates (they should be identical)
    dates = pd.DatetimeIndex(concat["mcap"]["date"])
    spy = build_spy_weekly(dates)

    # Also load the prior best (mcap-baseline on LightGBM ranker scoreboards)
    prior_path = REPO_ROOT / "artifacts" / "backtest_046_cap10" / "weekly_mcap_baseline_cap10.parquet"
    prior = pd.read_parquet(prior_path) if prior_path.exists() else None
    if prior is not None:
        prior["date"] = pd.to_datetime(prior["date"])
        prior = prior.set_index("date").reindex(dates)["mcap_return_gross"]

    # Print tables for the 3 windows
    years = pd.to_datetime(dates).year
    lines = []
    for label, mask in [
        ("2009-2025 (full)",        np.ones(len(dates), dtype=bool)),
        ("2010-2024",               (years >= 2010) & (years <= 2024)),
        ("2010-2025 (BAR vs SPY)",  years >= 2010),
    ]:
        sub_spy = spy[mask].to_numpy()
        spy_m = metrics(sub_spy)
        block = f"\n=== {label} ===\n"
        block += f"{'strategy':<26} {'weeks':>6} {'totret':>10} {'annret':>10} {'vol':>10} {'sharpe':>9} {'mdd':>9} {'calmar':>9}\n"
        for name, _ in schemes:
            sub = concat[name].set_index("date").reindex(dates)["weekly_ret"].to_numpy()[mask]
            m = metrics(sub)
            block += (f"factor_v1+{name:<14} {m['n_weeks']:>6} {m['total_ret']:>10.2%} "
                     f"{m['ann_ret']:>10.2%} {m['ann_vol']:>10.2%} {m['sharpe']:>9.3f} "
                     f"{m['mdd']:>9.2%} {m['calmar']:>9.3f}\n")
        if prior is not None:
            psub = prior.to_numpy()[mask]
            pm = metrics(psub)
            block += (f"{'PRIOR mcap (LightGBM)':<26} {pm['n_weeks']:>6} {pm['total_ret']:>10.2%} "
                     f"{pm['ann_ret']:>10.2%} {pm['ann_vol']:>10.2%} {pm['sharpe']:>9.3f} "
                     f"{pm['mdd']:>9.2%} {pm['calmar']:>9.3f}\n")
        block += (f"{'SPY (benchmark)':<26} {spy_m['n_weeks']:>6} {spy_m['total_ret']:>10.2%} "
                 f"{spy_m['ann_ret']:>10.2%} {spy_m['ann_vol']:>10.2%} {spy_m['sharpe']:>9.3f} "
                 f"{spy_m['mdd']:>9.2%} {spy_m['calmar']:>9.3f}\n")
        print(block)
        lines.append(block)

    # Persist consolidated report
    (REPO_ROOT / "reports" / "factor_v1_vs_spy.md").parent.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "reports" / "factor_v1_vs_spy.md").write_text(
        "# Factor screen v1 vs SPY (deterministic backtests)\n\n```\n" + "\n".join(lines) + "\n```\n"
    )


if __name__ == "__main__":
    main()
