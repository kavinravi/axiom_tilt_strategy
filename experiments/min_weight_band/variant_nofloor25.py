"""Variant sweep, year-by-year 2010-2025, all strategies vs SPY.

Adds a NEW variant: no floor (0%), 10% cap, modal-K select over K in {10,15,20,25}
(cap forces K>=10; book capped at 25 names). Compared head-to-head, per calendar
year, against: new band select-K (floor 2%, K up to 50), old ensemble blend,
static K=30 band, and SPY.

Run from repo root:  python -m experiments.min_weight_band.variant_nofloor25
Writes variant_nofloor25.md + variant_nofloor25_detail.parquet into this dir.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.min_weight_band.allocator import band_per_k_weights_and_returns
from experiments.min_weight_band.backtest_lib import (
    net_returns, turnover_series, walk_forward_proba,
)
from experiments.min_weight_band.run_backtest import (
    NEW_K, OLD_K, _ensemble_series, _new_per_k, _old_per_k, _select_k_series,
    _static_series, _weights_by_date,
)
from src.strategy.historical import (
    build_k_labels, load_data, load_spy_at, macro_by_date,
)
from src.strategy.k_selector import build_regime_features

OUT_DIR = Path(__file__).resolve().parent
NF25_K = [10, 15, 20, 25]            # no-floor, 10% cap: K>=10 feasibility, <=25 cap
Y0, Y1 = 2010, 2025
ORDER = ["new nofloor cap25", "new band select-K", "old ensemble blend",
         "static K=30 band", "SPY"]


def _nf_per_k(df):
    w, r = {}, {}
    for K in NF25_K:
        wk, rk = band_per_k_weights_and_returns(df, K, floor=0.0, cap=0.10)
        w[K], r[K] = wk, rk
    return w, r


def _year_metrics(net: np.ndarray) -> dict:
    """Calendar-year stats: actual cumulative return (not annualized), annualized
    vol, within-year max drawdown."""
    r = np.asarray(net, dtype=np.float64)
    ret = float(np.prod(1.0 + r) - 1.0)
    vol = float(np.std(r, ddof=1) * np.sqrt(52)) if len(r) > 1 else 0.0
    eq = np.cumprod(1.0 + r)
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min()) if len(r) else 0.0
    return {"ret": ret, "vol": vol, "mdd": mdd}


def main():
    print("Loading panel ...")
    df = load_data()

    print("Building per-K returns (old cap-only, new band, nofloor-25) ...")
    old_w, old_r = _old_per_k(df)
    new_w, new_r = _new_per_k(df)
    nf_w, nf_r = _nf_per_k(df)

    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in new_r.values()])))
    spy_at = load_spy_at(all_dates)
    regime = build_regime_features(all_dates, spy_at, macro_by_date(df, all_dates))

    print("Walk-forward: old 4-class ...")
    old_labels, _ = build_k_labels(old_r, all_dates, OLD_K)
    old_proba = walk_forward_proba(regime, old_labels, all_dates, num_class=len(OLD_K))
    print("Walk-forward: new band 5-class ...")
    new_labels, _ = build_k_labels(new_r, all_dates, NEW_K)
    new_proba = walk_forward_proba(regime, new_labels, all_dates, num_class=len(NEW_K))
    print("Walk-forward: nofloor-25 4-class ...")
    nf_labels, _ = build_k_labels(nf_r, all_dates, NF25_K)
    nf_proba = walk_forward_proba(regime, nf_labels, all_dates, num_class=len(NF25_K))

    oos = old_proba.index.intersection(new_proba.index).intersection(nf_proba.index)
    old_proba, new_proba, nf_proba = old_proba.loc[oos], new_proba.loc[oos], nf_proba.loc[oos]

    _, old_g, old_wd = _ensemble_series(old_proba, OLD_K, old_w, old_r)
    _, new_g, new_wd = _select_k_series(new_proba, NEW_K, new_w, new_r)
    _, nf_g, nf_wd = _select_k_series(nf_proba, NF25_K, nf_w, nf_r)
    k30_g, k30_wd = _static_series(new_w, new_r, 30, oos)
    spy_g = spy_at.reindex(oos).pct_change().fillna(0.0).to_numpy()
    spy_wd = [{"SPY": 1.0} for _ in oos]

    series = {
        "new nofloor cap25": net_returns(nf_g, turnover_series(nf_wd)),
        "new band select-K": net_returns(new_g, turnover_series(new_wd)),
        "old ensemble blend": net_returns(old_g, turnover_series(old_wd)),
        "static K=30 band": net_returns(k30_g, turnover_series(k30_wd)),
        "SPY": net_returns(spy_g, turnover_series(spy_wd)),
    }
    wdicts = {"new nofloor cap25": nf_wd, "new band select-K": new_wd,
              "old ensemble blend": old_wd, "static K=30 band": k30_wd, "SPY": spy_wd}

    years = pd.DatetimeIndex(oos).year.to_numpy()
    keep = (years >= Y0) & (years <= Y1)

    detail = []
    for y in sorted(set(years[keep])):
        mask = years == y
        for name in ORDER:
            wd = [w for w, m in zip(wdicts[name], mask) if m]
            ns = np.asarray([len(w) for w in wd])
            mins = np.asarray([min(w.values()) if w else np.nan for w in wd])
            tu = float(turnover_series(wd).mean()) if wd else 0.0
            m = _year_metrics(series[name][mask])
            detail.append({"year": int(y), "strategy": name, "weeks": int(mask.sum()),
                           "ret": m["ret"], "vol": m["vol"], "mdd": m["mdd"],
                           "turn": tu, "avg_n": float(ns.mean()),
                           "avg_min_wt": float(np.nanmean(mins))})

    # 2010-2025 aggregate per strategy (full-window metrics over the masked series).
    agg = {}
    for name in ORDER:
        net = series[name][keep]
        cum = float(np.prod(1.0 + net) - 1.0)
        ann = (1.0 + cum) ** (52 / len(net)) - 1.0
        vol = float(np.std(net, ddof=1) * np.sqrt(52))
        eq = np.cumprod(1.0 + net)
        mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
        wd = [w for w, m in zip(wdicts[name], keep) if m]
        ns = np.mean([len(w) for w in wd])
        agg[name] = {"ann": ann, "vol": vol, "sharpe": ann / vol if vol else 0.0,
                     "mdd": mdd, "turn": float(turnover_series(wd).mean()), "avg_n": float(ns)}

    dt = pd.DataFrame(detail)

    def matrix(metric, fmt):
        piv = dt.pivot(index="year", columns="strategy", values=metric)[ORDER]
        head = "| year | " + " | ".join(ORDER) + " |"
        sep = "|------|" + "|".join(["------:"] * len(ORDER)) + "|"
        lines = [head, sep]
        for y, row in piv.iterrows():
            lines.append(f"| {y} | " + " | ".join(fmt(row[c]) for c in ORDER) + " |")
        return "\n".join(lines)

    out = ["# No-floor / 10% cap / max 25 stocks — variant sweep (2010-2025)", "",
           f"New variant: no floor, 10% cap, modal-K select over K in {NF25_K}. "
           "Net of 5 bps x one-way turnover.", "",
           "## Annual return", "", matrix("ret", lambda v: f"{v:>6.1%}"), "",
           "## Annual vol", "", matrix("vol", lambda v: f"{v:>5.1%}"), "",
           "## Within-year max drawdown", "", matrix("mdd", lambda v: f"{v:>6.1%}"), "",
           "## Avg # holdings (N)", "", matrix("avg_n", lambda v: f"{v:>4.1f}"), "",
           "## 2010-2025 aggregate", "",
           "| strategy | ann | vol | sharpe | mdd | turn | avgN |",
           "|----------|----:|----:|-------:|----:|-----:|-----:|"]
    for name in ORDER:
        a = agg[name]
        out.append(f"| {name} | {a['ann']:>6.1%} | {a['vol']:>5.1%} | {a['sharpe']:>5.2f} "
                   f"| {a['mdd']:>6.1%} | {a['turn']:>5.2f} | {a['avg_n']:>4.1f} |")
    out.append("")

    md = "\n".join(out)
    (OUT_DIR / "variant_nofloor25.md").write_text(md)
    dt.to_parquet(OUT_DIR / "variant_nofloor25_detail.parquet", index=False)
    print("\n" + md)
    print(f"\nWrote {OUT_DIR / 'variant_nofloor25.md'} and variant_nofloor25_detail.parquet")


if __name__ == "__main__":
    main()
