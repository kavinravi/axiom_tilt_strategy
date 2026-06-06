"""Walk-forward backtest: new band select-K vs old ensemble blend vs SPY vs
static K=30 band, reported over three windows (2009-25, 2010-25, 2025-only).

Run from repo root:  python -m experiments.min_weight_band.run_backtest
Writes results.md + weekly_returns.parquet into this directory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.min_weight_band.allocator import band_per_k_weights_and_returns
from experiments.min_weight_band.backtest_lib import (
    metrics, net_returns, turnover_series, walk_forward_proba, window_mask,
)
from src.strategy.allocate import topk_mcap_weights
from src.strategy.constants import MAX_WEIGHT
from src.strategy.historical import (
    build_k_labels, load_data, load_spy_at, macro_by_date, per_k_weights_and_returns,
)
from src.strategy.k_selector import build_regime_features

OLD_K = [10, 20, 30, 50]
NEW_K = [10, 20, 30, 40, 50]
OUT_DIR = Path(__file__).resolve().parent
WINDOWS = [("full 2009-2025", 2009, 2025), ("2010-2025", 2010, 2025), ("2025 only", 2025, 2025)]


def _old_per_k(df):
    w, r = {}, {}
    for K in OLD_K:
        wk, rk = per_k_weights_and_returns(df, K, max_weight=MAX_WEIGHT)
        w[K], r[K] = wk, rk
    return w, r


def _new_per_k(df):
    w, r = {}, {}
    for K in NEW_K:
        wk, rk = band_per_k_weights_and_returns(df, K)
        w[K], r[K] = wk, rk
    return w, r


def _weights_by_date(wdf: pd.DataFrame) -> dict:
    """{date -> {permno -> weight}} from a [date,permno,weight] frame."""
    out = {}
    for d, g in wdf.groupby("date"):
        out[d] = dict(zip(g["permno"].astype(int), g["weight"].astype(float)))
    return out


def _ensemble_series(proba: pd.DataFrame, k_list, k_weights, k_returns):
    """Old path: convex blend over K. Returns (dates, gross, weight_dicts)."""
    wbd = {K: _weights_by_date(k_weights[K]) for K in k_list}
    dates, gross, wdicts = [], [], []
    for d, prow in proba.iterrows():
        p = {K: float(prow[f"c{j}"]) for j, K in enumerate(k_list)}
        combined: dict = {}
        for K in k_list:
            for permno, wt in wbd[K].get(d, {}).items():
                combined[permno] = combined.get(permno, 0.0) + p[K] * wt
        gross.append(sum(p[K] * float(k_returns[K].get(d, 0.0)) for K in k_list))
        wdicts.append(combined)
        dates.append(d)
    return pd.DatetimeIndex(dates), np.asarray(gross), wdicts


def _select_k_series(proba: pd.DataFrame, k_list, k_weights, k_returns):
    """New path: pick modal K = argmax proba. Returns (dates, gross, weight_dicts)."""
    wbd = {K: _weights_by_date(k_weights[K]) for K in k_list}
    dates, gross, wdicts = [], [], []
    for d, prow in proba.iterrows():
        j = int(np.argmax([prow[f"c{i}"] for i in range(len(k_list))]))
        Kstar = k_list[j]
        gross.append(float(k_returns[Kstar].get(d, 0.0)))
        wdicts.append(wbd[Kstar].get(d, {}))
        dates.append(d)
    return pd.DatetimeIndex(dates), np.asarray(gross), wdicts


def _static_series(k_weights, k_returns, K, oos_dates):
    wbd = _weights_by_date(k_weights[K])
    gross = np.asarray([float(k_returns[K].get(d, 0.0)) for d in oos_dates])
    wdicts = [wbd.get(d, {}) for d in oos_dates]
    return gross, wdicts


def _row(name, net, wdicts, oos_dates):
    n = np.asarray([len(w) for w in wdicts])
    mins = np.asarray([min(w.values()) if w else np.nan for w in wdicts])
    avg_tu = float(turnover_series(wdicts).mean())
    m = metrics(net)
    return {"strategy": name, "ann": m["ann"], "vol": m["vol"], "sharpe": m["sharpe"],
            "sortino": m["sortino"], "mdd": m["mdd"], "turnover": avg_tu,
            "avg_n": float(n.mean()), "avg_min_wt": float(np.nanmean(mins))}


def _fmt_table(rows):
    head = f"| {'strategy':<26} | {'ann':>7} | {'vol':>7} | {'sharpe':>7} | {'sortino':>7} | {'mdd':>7} | {'turn':>6} | {'avgN':>5} | {'minWt':>6} |"
    sep = "|" + "|".join(["-" * (len(c) + 2) for c in head.split("|")[1:-1]]) + "|"
    lines = [head, sep]
    for r in rows:
        lines.append(
            f"| {r['strategy']:<26} | {r['ann']:>6.1%} | {r['vol']:>6.1%} | "
            f"{r['sharpe']:>7.2f} | {r['sortino']:>7.2f} | {r['mdd']:>6.1%} | "
            f"{r['turnover']:>6.2f} | {r['avg_n']:>5.1f} | {r['avg_min_wt']:>6.2%} |"
        )
    return "\n".join(lines)


def main():
    print("Loading panel ...")
    df = load_data()

    print("Building per-K returns (old cap-only + new band) ...")
    old_w, old_r = _old_per_k(df)
    new_w, new_r = _new_per_k(df)

    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in new_r.values()])))
    spy_at = load_spy_at(all_dates)
    regime = build_regime_features(all_dates, spy_at, macro_by_date(df, all_dates))

    print("Walk-forward: old 4-class ...")
    old_labels, _ = build_k_labels(old_r, all_dates, OLD_K)
    old_proba = walk_forward_proba(regime, old_labels, all_dates, num_class=len(OLD_K))

    print("Walk-forward: new 5-class ...")
    new_labels, _ = build_k_labels(new_r, all_dates, NEW_K)
    new_proba = walk_forward_proba(regime, new_labels, all_dates, num_class=len(NEW_K))

    # Common OOS dates (both walks share the same scheme; intersect to be safe).
    oos = old_proba.index.intersection(new_proba.index)
    old_proba, new_proba = old_proba.loc[oos], new_proba.loc[oos]

    # Strategy gross + weights on the common OOS dates.
    od, old_g, old_wd = _ensemble_series(old_proba, OLD_K, old_w, old_r)
    nd, new_g, new_wd = _select_k_series(new_proba, NEW_K, new_w, new_r)
    k30_g, k30_wd = _static_series(new_w, new_r, 30, oos)
    spy_g = spy_at.reindex(oos).pct_change().fillna(0.0).to_numpy()
    spy_wd = [{"SPY": 1.0} for _ in oos]  # buy-hold proxy: ~0 turnover after entry

    # Net of cost.
    series = {
        "new band select-K": net_returns(new_g, turnover_series(new_wd)),
        "old ensemble blend": net_returns(old_g, turnover_series(old_wd)),
        "static K=30 band": net_returns(k30_g, turnover_series(k30_wd)),
        "SPY": net_returns(spy_g, turnover_series(spy_wd)),
    }
    wdicts = {"new band select-K": new_wd, "old ensemble blend": old_wd,
              "static K=30 band": k30_wd, "SPY": spy_wd}

    # Three-window report.
    out_md = ["# Min-weight band — backtest report",
              "", f"OOS Fridays: {len(oos)} ({oos.min().date()} .. {oos.max().date()}), "
              "net of 5 bps x one-way turnover.", ""]
    for title, y0, y1 in WINDOWS:
        mask = window_mask(oos, y0, y1)
        rows = [_row(name, series[name][mask],
                     [w for w, m in zip(wdicts[name], mask) if m], oos[mask])
                for name in series]
        out_md += [f"## {title}  ({int(mask.sum())} weeks)", "", _fmt_table(rows), ""]
        print(f"\n=== {title} ({int(mask.sum())} weeks) ===")
        print(_fmt_table(rows))

    (OUT_DIR / "results.md").write_text("\n".join(out_md))

    long = []
    for name, net in series.items():
        for d, r in zip(oos, net):
            long.append({"date": d, "strategy": name, "ret": float(r)})
    pd.DataFrame(long).to_parquet(OUT_DIR / "weekly_returns.parquet", index=False)
    print(f"\nWrote {OUT_DIR / 'results.md'} and weekly_returns.parquet")


if __name__ == "__main__":
    main()
