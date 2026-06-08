"""Floored-blend experiment (2010-2025): keep the ML (regime->K LGBM ensemble
blend) but impose a minimum allocation, killing the long tail of dust positions.

The tiny weights in the old ensemble blend are NOT an LGBM artifact -- they fall
out of the convex blend w(i) = sum_K p_K * w_K(i): a name that only appears in
the top-50 sub-book gets ~p_50 * small ~ 0.1%. So the fix is post-blend, not in
the model: take the blended weights, drop anything below `floor`, then re-project
the survivors onto the band [floor, cap] summing to 1 (the blend stays the tilt).
The LGBM still picks the K-mixture every week -- the ML is fully intact.

Variants compared on the SAME walk-forward OOS as force20.md:
  - old blend (no floor)   : current prod, reproduces force20's "old ensemble blend"
  - old blend + 1% floor    : prune <1%, band-project survivors to [1%, 10%]
  - old blend + 2% floor    : prune <2%, band-project survivors to [2%, 10%]
  - static K=20 nofloor      : the no-ML reference

Run from repo root:  python -m experiments.min_weight_band.floor_blend
Writes floor_blend.md + floor_blend_detail.parquet into this dir.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.min_weight_band.allocator import (
    band_per_k_weights_and_returns, band_water_fill,
)
from experiments.min_weight_band.backtest_lib import (
    metrics, net_returns, turnover_series, walk_forward_proba, window_mask,
)
from experiments.min_weight_band.run_backtest import (
    NEW_K, OLD_K, _ensemble_series, _new_per_k, _old_per_k, _static_series,
)
from src.strategy.historical import (
    build_k_labels, load_data, load_spy_at, macro_by_date,
)
from src.strategy.k_selector import build_regime_features

OUT_DIR = Path(__file__).resolve().parent
Y0, Y1 = 2010, 2025
CAP = 0.10
FLOORS = [0.01, 0.02]


def fwd_lookup(df: pd.DataFrame) -> dict:
    """{date -> {permno -> nan_to_num(fwd_ret_5d)}} for recomputing portfolio
    returns of an arbitrary weight set (matches the per-K return convention)."""
    lut = {}
    for d, g in df.groupby("date", sort=False):
        lut[d] = dict(zip(g["permno"].astype(int).to_numpy(),
                          np.nan_to_num(g["fwd_ret_5d"].to_numpy(dtype=np.float64))))
    return lut


def floor_blend_dict(w: dict, floor: float, cap: float = CAP,
                     min_names: int = 0) -> dict:
    """Hold at least `min_names` names, each in [floor, cap], summing to 1.

    Held set = the names whose raw blend weight clears `floor`, but never fewer
    than `min_names` (topped up with the next-largest blend weights when the
    natural >=floor set is too small). The held set is then band-projected onto
    [floor, cap] via band_water_fill, which pulls any topped-up sub-floor names
    up to the floor and caps at `cap`. Needs min_names >= 1/cap for feasibility
    (20 >= 10 here)."""
    items = sorted(w.items(), key=lambda kv: kv[1], reverse=True)
    n_above = sum(1 for _, x in items if x >= floor)
    n_hold = min(max(n_above, min_names), len(items))
    if n_hold == 0:
        return {}
    held = items[:n_hold]
    names = [p for p, _ in held]
    base = np.asarray([x for _, x in held], dtype=np.float64)
    if n_hold * cap < 1.0 - 1e-12:               # too few names to reach 100% under cap
        v = np.minimum(base / base.sum(), cap)
        v = v / v.sum()
    else:
        v = band_water_fill(base, floor=floor, cap=cap)
    return {p: float(x) for p, x in zip(names, v)}


def port_ret(wdict: dict, fwd_d: dict) -> float:
    return float(sum(wt * fwd_d.get(int(p), 0.0) for p, wt in wdict.items()))


def agg_metrics(net: np.ndarray, wdicts: list) -> dict:
    m = metrics(net)
    ns = np.asarray([len(w) for w in wdicts])
    mins = np.asarray([min(w.values()) if w else np.nan for w in wdicts])
    return {**m, "turn": float(turnover_series(wdicts).mean()),
            "avg_n": float(ns.mean()), "min_wt": float(np.nanmean(mins)),
            "max_n": int(ns.max()), "min_n": int(ns.min())}


def main():
    print("Loading panel ...")
    df = load_data()
    fwd = fwd_lookup(df)

    print("Building per-K returns (old cap-only + new band) ...")
    old_w, old_r = _old_per_k(df)
    new_w, new_r = _new_per_k(df)
    cap20_w, cap20_r = band_per_k_weights_and_returns(df, 20, floor=0.0, cap=CAP)

    all_dates = pd.DatetimeIndex(sorted(set().union(*[set(s.index) for s in new_r.values()])))
    spy_at = load_spy_at(all_dates)
    regime = build_regime_features(all_dates, spy_at, macro_by_date(df, all_dates))

    print("Walk-forward: old 4-class + new 5-class ...")
    old_labels, _ = build_k_labels(old_r, all_dates, OLD_K)
    old_proba = walk_forward_proba(regime, old_labels, all_dates, num_class=len(OLD_K))
    new_labels, _ = build_k_labels(new_r, all_dates, NEW_K)
    new_proba = walk_forward_proba(regime, new_labels, all_dates, num_class=len(NEW_K))
    oos = old_proba.index.intersection(new_proba.index)
    old_proba = old_proba.loc[oos]

    # Old ensemble blend (the ML path) + its per-date blended weight dicts.
    _, old_g, old_wd = _ensemble_series(old_proba, OLD_K, old_w, old_r)

    # Sanity: recomputing the blend return from the fwd lookup must match the
    # harness's convex-blend gross (validates the lookup + return convention).
    recomputed = np.asarray([port_ret(w, fwd[d]) for w, d in zip(old_wd, oos)])
    assert np.allclose(recomputed, old_g, atol=1e-9), \
        f"return recompute mismatch: max |d|={np.abs(recomputed - old_g).max():.2e}"
    print("  return-recompute sanity OK")

    # Floored variants. (floor, min_names, label)
    variants = [
        (0.01, 0, "old blend + 1% floor"),
        (0.02, 0, "old blend + 2% floor"),
        (0.01, 20, "old blend + 1% floor, >=20"),
        (0.02, 20, "old blend + 2% floor, >=20"),
    ]
    floored = {}
    for fl, mn, label in variants:
        wd = [floor_blend_dict(w, fl, min_names=mn) for w in old_wd]
        g = np.asarray([port_ret(w, fwd[d]) for w, d in zip(wd, oos)])
        floored[label] = (g, wd)

    # Static K=20 nofloor reference (no ML).
    k20c_g, k20c_wd = _static_series({20: cap20_w}, {20: cap20_r}, 20, oos)

    series = {
        "old blend (no floor)": (old_g, old_wd),
        "old blend + 1% floor": floored["old blend + 1% floor"],
        "old blend + 1% floor, >=20": floored["old blend + 1% floor, >=20"],
        "old blend + 2% floor, >=20": floored["old blend + 2% floor, >=20"],
        "static K=20 nofloor": (k20c_g, k20c_wd),
    }

    mask = window_mask(oos, Y0, Y1)
    rows = []
    for name, (g, wd) in series.items():
        net = net_returns(g, turnover_series(wd))[mask]
        wdm = [w for w, m in zip(wd, mask) if m]
        a = agg_metrics(net, wdm)
        rows.append({"strategy": name, **a})

    out = [f"# Floored-blend experiment ({Y0}-{Y1}): keep the ML, kill the tail", "",
           "The ML (regime->K LGBM) blends top-K books every Friday; we then prune "
           "sub-floor dust and band-project survivors to [floor, 10%]. Net of 5 bps "
           "x one-way turnover. avgN/minN/maxN = holdings count; minWt = mean of the "
           "per-week smallest weight.", "",
           "| strategy | ann | vol | sharpe | sortino | mdd | turn | avgN | minN | maxN | minWt |",
           "|----------|----:|----:|-------:|--------:|----:|-----:|-----:|-----:|-----:|------:|"]
    for r in rows:
        out.append(
            f"| {r['strategy']} | {r['ann']:>6.1%} | {r['vol']:>5.1%} | {r['sharpe']:>5.2f} "
            f"| {r['sortino']:>5.2f} | {r['mdd']:>6.1%} | {r['turn']:>4.2f} | {r['avg_n']:>4.1f} "
            f"| {r['min_n']:>3d} | {r['max_n']:>3d} | {r['min_wt']:>5.2%} |")
    out.append("")

    md = "\n".join(out)
    (OUT_DIR / "floor_blend.md").write_text(md)
    pd.DataFrame(rows).to_parquet(OUT_DIR / "floor_blend_detail.parquet", index=False)
    print("\n" + md)
    print(f"\nWrote {OUT_DIR / 'floor_blend.md'} and floor_blend_detail.parquet")


if __name__ == "__main__":
    main()
