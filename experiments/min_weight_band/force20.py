"""Force-20 experiment: fixed 20-name books vs the dynamic candidates + SPY,
year-by-year 2010-2025. Addresses the overconcentration worry directly.

Two forced-20 variants (both ALWAYS hold exactly 20 names, no model K-selection):
  - static K=20 band     : 2% floor / 10% cap, mcap-tilted (every name in [2%,10%])
  - static K=20 nofloor  : 10% cap only, no floor (tilt can run small)
vs new band select-K (dynamic), old ensemble blend (current prod), SPY.

Run from repo root:  python -m experiments.min_weight_band.force20
Writes force20.md + force20_detail.parquet into this dir.
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
    _static_series,
)
from src.strategy.historical import (
    build_k_labels, load_data, load_spy_at, macro_by_date,
)
from src.strategy.k_selector import build_regime_features

OUT_DIR = Path(__file__).resolve().parent
Y0, Y1 = 2010, 2025
ORDER = ["static K=20 band", "static K=20 nofloor", "new band select-K",
         "old ensemble blend", "SPY"]


def _year_metrics(net: np.ndarray) -> dict:
    r = np.asarray(net, dtype=np.float64)
    ret = float(np.prod(1.0 + r) - 1.0)
    vol = float(np.std(r, ddof=1) * np.sqrt(52)) if len(r) > 1 else 0.0
    eq = np.cumprod(1.0 + r)
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min()) if len(r) else 0.0
    return {"ret": ret, "vol": vol, "mdd": mdd}


def main():
    print("Loading panel ...")
    df = load_data()

    print("Building per-K returns (band + cap-only) ...")
    old_w, old_r = _old_per_k(df)
    new_w, new_r = _new_per_k(df)                                  # band 2%/10%, has K=20
    cap20_w, cap20_r = band_per_k_weights_and_returns(df, 20, floor=0.0, cap=0.10)

    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in new_r.values()])))
    spy_at = load_spy_at(all_dates)
    regime = build_regime_features(all_dates, spy_at, macro_by_date(df, all_dates))

    print("Walk-forward: old 4-class + new band 5-class (for the dynamic columns) ...")
    old_labels, _ = build_k_labels(old_r, all_dates, OLD_K)
    old_proba = walk_forward_proba(regime, old_labels, all_dates, num_class=len(OLD_K))
    new_labels, _ = build_k_labels(new_r, all_dates, NEW_K)
    new_proba = walk_forward_proba(regime, new_labels, all_dates, num_class=len(NEW_K))

    oos = old_proba.index.intersection(new_proba.index)
    old_proba, new_proba = old_proba.loc[oos], new_proba.loc[oos]

    # Dynamic columns.
    _, old_g, old_wd = _ensemble_series(old_proba, OLD_K, old_w, old_r)
    _, new_g, new_wd = _select_k_series(new_proba, NEW_K, new_w, new_r)
    # Forced-20 static columns (no K-selection).
    k20b_g, k20b_wd = _static_series(new_w, new_r, 20, oos)        # band 2%/10% at K=20
    k20c_g, k20c_wd = _static_series({20: cap20_w}, {20: cap20_r}, 20, oos)  # cap-only at K=20
    spy_g = spy_at.reindex(oos).pct_change().fillna(0.0).to_numpy()
    spy_wd = [{"SPY": 1.0} for _ in oos]

    series = {
        "static K=20 band": net_returns(k20b_g, turnover_series(k20b_wd)),
        "static K=20 nofloor": net_returns(k20c_g, turnover_series(k20c_wd)),
        "new band select-K": net_returns(new_g, turnover_series(new_wd)),
        "old ensemble blend": net_returns(old_g, turnover_series(old_wd)),
        "SPY": net_returns(spy_g, turnover_series(spy_wd)),
    }
    wdicts = {"static K=20 band": k20b_wd, "static K=20 nofloor": k20c_wd,
              "new band select-K": new_wd, "old ensemble blend": old_wd, "SPY": spy_wd}

    years = pd.DatetimeIndex(oos).year.to_numpy()
    keep = (years >= Y0) & (years <= Y1)

    detail = []
    for y in sorted(set(years[keep])):
        mask = years == y
        for name in ORDER:
            wd = [w for w, m in zip(wdicts[name], mask) if m]
            ns = np.asarray([len(w) for w in wd])
            mins = np.asarray([min(w.values()) if w else np.nan for w in wd])
            m = _year_metrics(series[name][mask])
            detail.append({"year": int(y), "strategy": name, "weeks": int(mask.sum()),
                           "ret": m["ret"], "vol": m["vol"], "mdd": m["mdd"],
                           "avg_n": float(ns.mean()), "avg_min_wt": float(np.nanmean(mins))})

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
        mins = np.nanmean([min(w.values()) if w else np.nan for w in wd])
        agg[name] = {"ann": ann, "vol": vol, "sharpe": ann / vol if vol else 0.0,
                     "mdd": mdd, "turn": float(turnover_series(wd).mean()),
                     "avg_n": float(ns), "min_wt": float(mins)}

    dt = pd.DataFrame(detail)

    def matrix(metric, fmt):
        piv = dt.pivot(index="year", columns="strategy", values=metric)[ORDER]
        head = "| year | " + " | ".join(ORDER) + " |"
        sep = "|------|" + "|".join(["------:"] * len(ORDER)) + "|"
        lines = [head, sep]
        for y, row in piv.iterrows():
            lines.append(f"| {y} | " + " | ".join(fmt(row[c]) for c in ORDER) + " |")
        return "\n".join(lines)

    out = ["# Force-20 experiment (2010-2025): fixed 20-name books vs dynamic + SPY", "",
           "Two forced-20 books (always exactly 20 names): band = 2% floor/10% cap "
           "mcap-tilted; nofloor = 10% cap only. Net of 5 bps x one-way turnover.", "",
           "## Annual return", "", matrix("ret", lambda v: f"{v:>6.1%}"), "",
           "## Annual vol", "", matrix("vol", lambda v: f"{v:>5.1%}"), "",
           "## Within-year max drawdown", "", matrix("mdd", lambda v: f"{v:>6.1%}"), "",
           "## Avg # holdings (N)", "", matrix("avg_n", lambda v: f"{v:>4.1f}"), "",
           "## 2010-2025 aggregate", "",
           "| strategy | ann | vol | sharpe | mdd | turn | avgN | minWt |",
           "|----------|----:|----:|-------:|----:|-----:|-----:|------:|"]
    for name in ORDER:
        a = agg[name]
        out.append(f"| {name} | {a['ann']:>6.1%} | {a['vol']:>5.1%} | {a['sharpe']:>5.2f} "
                   f"| {a['mdd']:>6.1%} | {a['turn']:>5.2f} | {a['avg_n']:>4.1f} "
                   f"| {a['min_wt']:>5.1%} |")
    out.append("")

    md = "\n".join(out)
    (OUT_DIR / "force20.md").write_text(md)
    dt.to_parquet(OUT_DIR / "force20_detail.parquet", index=False)
    print("\n" + md)
    print(f"\nWrote {OUT_DIR / 'force20.md'} and force20_detail.parquet")


if __name__ == "__main__":
    main()
