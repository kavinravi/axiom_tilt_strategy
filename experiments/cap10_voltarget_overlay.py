"""Vol-targeting overlay on cap10 weekly returns.

Idea: scale cap10's equity exposure so trailing-realized portfolio vol = target
(default SPY's ~16.5%). Cash buffer absorbs the de-scaling. No PPO retrain.

For week t, scaling factor based on info available *before* week t opens:
    s_t = min(MAX_LEVERAGE, target_vol / realized_vol[t-LOOKBACK : t-1])

Cash earns either 0% (conservative) or DGS3MO weekly rate (more honest), per
flag. Default = 0% for the clean version your dad asked for.

Writes overlay weekly returns + 2010-2025 metrics vs SPY to
    artifacts/backtest_046_cap10/weekly_046_cap10_voltarget.parquet
    reports/cap10_voltarget_vs_spy.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import processed_dir, repo_root

REPO_ROOT = repo_root()
CAP10_WEEKLY = REPO_ROOT / "artifacts" / "backtest_046_cap10" / "weekly_046_ppo_tilt_ep104_cap10.parquet"
SPY_DAILY = Path("/home/kavin-ravi/CodingStuff/axiom_tilt_strategy/artifacts/benchmarks/spy_daily.parquet")
MACRO_PATH = processed_dir() / "macro.parquet"
OUT_WEEKLY = REPO_ROOT / "artifacts" / "backtest_046_cap10" / "weekly_046_cap10_voltarget.parquet"
OUT_REPORT = REPO_ROOT / "reports" / "cap10_voltarget_vs_spy.md"


def metrics(name: str, rets: np.ndarray) -> dict:
    rets = np.asarray(rets, dtype=float)
    if len(rets) == 0:
        return {"name": name, "n_weeks": 0}
    cum = float(np.prod(1.0 + rets) - 1.0)
    ann_ret = (1.0 + cum) ** (52.0 / len(rets)) - 1.0
    ann_vol = float(np.std(rets, ddof=1) * np.sqrt(52.0))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    equity = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    mdd = float(dd.min())
    calmar = ann_ret / abs(mdd) if mdd < 0 else 0.0
    hit = float((rets > 0).mean())
    sortino = ann_ret / (float(np.std(rets[rets < 0], ddof=1) * np.sqrt(52.0)) if (rets < 0).any() else 1.0)
    return {"name": name, "n_weeks": len(rets), "total_ret": cum,
            "ann_ret": ann_ret, "ann_vol": ann_vol, "sharpe": sharpe,
            "sortino": sortino, "mdd": mdd, "calmar": calmar, "hit_rate": hit}


def fmt(m: dict) -> str:
    return (f"{m['name']:<22} {m['n_weeks']:>6} {m['total_ret']:>10.2%} "
            f"{m['ann_ret']:>10.2%} {m['ann_vol']:>10.2%} {m['sharpe']:>9.3f} "
            f"{m['sortino']:>9.3f} {m['mdd']:>9.2%} {m['calmar']:>9.3f} {m['hit_rate']:>9.2%}")


def build_spy_weekly(ppo_dates: pd.DatetimeIndex) -> pd.DataFrame:
    spy = pd.read_parquet(SPY_DAILY).reset_index()
    spy = spy[["Date", "close"]].rename(columns={"Date": "date"})
    spy["date"] = pd.to_datetime(spy["date"])
    spy = spy.sort_values("date").set_index("date")
    union = spy.index.union(ppo_dates)
    closes = spy["close"].reindex(union).sort_index().ffill().reindex(ppo_dates)
    rets = []
    for i in range(len(closes) - 1):
        c0, c1 = closes.iloc[i], closes.iloc[i + 1]
        rets.append(c1 / c0 - 1.0 if pd.notna(c0) and pd.notna(c1) and c0 > 0 else 0.0)
    rets.append(0.0)
    return pd.DataFrame({"date": ppo_dates, "spy_return": rets})


def load_weekly_cash_rate(dates: pd.DatetimeIndex) -> np.ndarray:
    """Convert annualized DGS3MO (% units) to weekly returns, aligned to dates."""
    m = pd.read_parquet(MACRO_PATH)
    m = m[m["series"] == "DGS3MO"].copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m.sort_values("date").set_index("date")
    yields = m["value"].reindex(m.index.union(dates)).sort_index().ffill().reindex(dates) / 100.0
    return ((1.0 + yields) ** (1.0 / 52.0) - 1.0).fillna(0.0).to_numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-vol", type=float, default=0.165, help="annualized vol target")
    parser.add_argument("--lookback", type=int, default=16, help="trailing weeks for vol estimate")
    parser.add_argument("--max-leverage", type=float, default=1.0, help="cap on scaling factor")
    parser.add_argument("--cash", choices=["zero", "tbill"], default="zero")
    args = parser.parse_args()

    df = pd.read_parquet(CAP10_WEEKLY)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    ppo = df["ppo_return_gross"].to_numpy()
    dates = pd.DatetimeIndex(df["date"])

    cash = (np.zeros(len(df)) if args.cash == "zero"
            else load_weekly_cash_rate(dates))

    # Trailing vol from PRIOR weeks only (PIT). Use ddof=1.
    vol_series = pd.Series(ppo).rolling(args.lookback, min_periods=args.lookback).std(ddof=1).shift(1)
    realized_ann_vol = vol_series * np.sqrt(52.0)
    scale = (args.target_vol / realized_ann_vol).clip(upper=args.max_leverage)
    # Before we have a full lookback window, default to scale=1 (full exposure).
    scale = scale.fillna(1.0).to_numpy()

    overlay_ret = scale * ppo + (1.0 - scale) * cash
    df_out = df.copy()
    df_out["scale"] = scale
    df_out["cash_ret_weekly"] = cash
    df_out["overlay_return"] = overlay_ret
    df_out.to_parquet(OUT_WEEKLY, compression="zstd", index=False)

    # Build SPY for alignment
    spy_w = build_spy_weekly(dates).set_index("date")["spy_return"].reindex(dates).fillna(0.0).to_numpy()
    df_out["spy_return"] = spy_w
    df_out["year"] = dates.year

    print(f"\nvol-target overlay (target={args.target_vol:.1%}, lookback={args.lookback}w, "
          f"cash={args.cash})\n")
    header = f"{'strat':<22} {'weeks':>6} {'totret':>10} {'annret':>10} {'vol':>10} {'sharpe':>9} {'sortino':>9} {'mdd':>9} {'calmar':>9} {'hit':>9}"
    blocks = []
    for label, sub in [
        ("2009-2025 (full)",  df_out),
        ("2010-2024",         df_out[(df_out["year"] >= 2010) & (df_out["year"] <= 2024)]),
        ("2010-2025",         df_out[df_out["year"] >= 2010]),
    ]:
        rows = [
            metrics("cap10 (no overlay)", sub["ppo_return_gross"]),
            metrics("cap10 + voltarget", sub["overlay_return"]),
            metrics("SPY", sub["spy_return"]),
        ]
        block = f"=== {label} ===\n{header}\n" + "\n".join(fmt(r) for r in rows)
        blocks.append(block)
        print(block + "\n")

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text(
        f"# cap10 + vol-targeting overlay vs SPY\n\n"
        f"target={args.target_vol:.1%}, lookback={args.lookback}w, "
        f"max_leverage={args.max_leverage}, cash={args.cash}\n\n"
        f"```\n" + "\n\n".join(blocks) + "\n```\n"
    )
    print(f"wrote -> {OUT_REPORT.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
