"""Year-by-year breakdown (return / vol / maxDD / avg N / avg min-wt) for the
new band select-K vs old ensemble blend (+ SPY reference).

Reuses run_backtest's pipeline; re-runs the walk-forward once to recover the
per-week holding counts (N) that results.md only aggregates over 3 windows.

Run from repo root:  python -m experiments.min_weight_band.year_by_year
Writes year_by_year.md + year_by_year_detail.parquet into this directory.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.min_weight_band.backtest_lib import (
    net_returns, turnover_series, walk_forward_proba,
)
from experiments.min_weight_band.run_backtest import (
    NEW_K, OLD_K, _ensemble_series, _new_per_k, _old_per_k, _select_k_series,
)
from src.strategy.historical import (
    build_k_labels, load_data, load_spy_at, macro_by_date,
)
from src.strategy.k_selector import build_regime_features

OUT_DIR = Path(__file__).resolve().parent


def _year_metrics(net: np.ndarray) -> dict:
    """Calendar-year stats: ACTUAL cumulative return (not annualized, so partial
    years aren't inflated), annualized vol, and within-year max drawdown."""
    r = np.asarray(net, dtype=np.float64)
    ret = float(np.prod(1.0 + r) - 1.0)
    vol = float(np.std(r, ddof=1) * np.sqrt(52)) if len(r) > 1 else 0.0
    eq = np.cumprod(1.0 + r)
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min()) if len(r) else 0.0
    return {"ret": ret, "vol": vol, "mdd": mdd}


def _build():
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

    oos = old_proba.index.intersection(new_proba.index)
    old_proba, new_proba = old_proba.loc[oos], new_proba.loc[oos]

    _, old_g, old_wd = _ensemble_series(old_proba, OLD_K, old_w, old_r)
    _, new_g, new_wd = _select_k_series(new_proba, NEW_K, new_w, new_r)
    spy_g = spy_at.reindex(oos).pct_change().fillna(0.0).to_numpy()
    spy_wd = [{"SPY": 1.0} for _ in oos]

    series = {
        "new band select-K": net_returns(new_g, turnover_series(new_wd)),
        "old ensemble blend": net_returns(old_g, turnover_series(old_wd)),
        "SPY": net_returns(spy_g, turnover_series(spy_wd)),
    }
    wdicts = {"new band select-K": new_wd, "old ensemble blend": old_wd, "SPY": spy_wd}
    return oos, series, wdicts


def main():
    oos, series, wdicts = _build()
    years = pd.DatetimeIndex(oos).year.to_numpy()

    detail, table = [], []
    for y in sorted(set(years)):
        mask = years == y
        for name in series:
            wd = [w for w, m in zip(wdicts[name], mask) if m]
            ns = np.asarray([len(w) for w in wd])
            mins = np.asarray([min(w.values()) if w else np.nan for w in wd])
            m = _year_metrics(series[name][mask])
            row = {"year": int(y), "strategy": name, "weeks": int(mask.sum()),
                   "ret": m["ret"], "vol": m["vol"], "mdd": m["mdd"],
                   "avg_n": float(ns.mean()), "avg_min_wt": float(np.nanmean(mins))}
            table.append(row)
            detail.append(row)

    # Markdown: one block per strategy, year rows.
    out = ["# Year-by-year — new band vs old blend (+ SPY)", "",
           "Net of 5 bps x one-way turnover. `ret` = actual calendar-year return; "
           "`vol` annualized; `mdd` within-year; `avgN` mean weekly holdings.", ""]
    order = ["new band select-K", "old ensemble blend", "SPY"]
    for name in order:
        out += [f"## {name}", "",
                f"| year | weeks | ret | vol | mdd | avgN | minWt |",
                f"|------|------:|------:|------:|------:|-----:|------:|"]
        for r in [t for t in table if t["strategy"] == name]:
            out.append(f"| {r['year']} | {r['weeks']} | {r['ret']:>6.1%} | "
                       f"{r['vol']:>5.1%} | {r['mdd']:>6.1%} | {r['avg_n']:>4.1f} | "
                       f"{r['avg_min_wt']:>5.1%} |")
        out.append("")

    md = "\n".join(out)
    (OUT_DIR / "year_by_year.md").write_text(md)
    pd.DataFrame(detail).to_parquet(OUT_DIR / "year_by_year_detail.parquet", index=False)
    print("\n" + md)
    print(f"Wrote {OUT_DIR / 'year_by_year.md'} and year_by_year_detail.parquet")


if __name__ == "__main__":
    main()
