"""Compute 2010-2024 (and 2009-2024) head-to-head metrics for:
    - cap10 retrain (046_ppo_tilt_ep104_cap10)
    - uncapped baseline (038_ppo_tilt_ep104)
    - Score-Prop baselines for each
    - SPY benchmark

Reads weekly returns from artifacts/backtest_046_cap10/ and artifacts/backtest_round2/,
aligns to SPY weekly returns from axiom_tilt_strategy/artifacts/benchmarks/spy_daily.parquet.

Prints a head-to-head table; writes reports/cap10_vs_spy.md.

usage: python experiments/cap10_vs_spy_2010_2024.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import repo_root

REPO_ROOT = repo_root()
CAP10_BACKTEST = REPO_ROOT / 'artifacts' / 'backtest_046_cap10' / 'weekly_046_ppo_tilt_ep104_cap10.parquet'
UNCAP_BACKTEST = REPO_ROOT / 'artifacts' / 'backtest_round2' / 'weekly_038_ppo_tilt_ep104.parquet'
SPY_DAILY = Path('/home/kavin-ravi/CodingStuff/axiom_tilt_strategy/artifacts/benchmarks/spy_daily.parquet')
OUT_REPORT = REPO_ROOT / 'reports' / 'cap10_vs_spy.md'


def metrics(name: str, rets: np.ndarray) -> dict:
    rets = np.asarray(rets, dtype=float)
    if len(rets) == 0:
        return {'name': name, 'n': 0}
    gross_cum = float(np.prod(1.0 + rets) - 1.0)
    ann_ret = (1.0 + gross_cum) ** (52.0 / len(rets)) - 1.0
    ann_vol = float(np.std(rets, ddof=1) * np.sqrt(52.0))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    equity = np.cumprod(1.0 + rets)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    mdd = float(dd.min())
    calmar = ann_ret / abs(mdd) if mdd < 0 else 0.0
    hit = float((rets > 0).mean())
    sortino = ann_ret / (float(np.std(rets[rets < 0], ddof=1) * np.sqrt(52.0)) if (rets < 0).any() else 1.0)
    return {'name': name, 'n_weeks': len(rets), 'total_ret': gross_cum,
            'ann_ret': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe,
            'sortino': sortino, 'mdd': mdd, 'calmar': calmar, 'hit_rate': hit}


def fmt_row(m: dict) -> str:
    if not m or m.get('n_weeks', 0) == 0:
        return f"{m.get('name', '?'):<14} (no data)"
    return (f"{m['name']:<14} {m['n_weeks']:>6} {m['total_ret']:>10.2%} "
            f"{m['ann_ret']:>10.2%} {m['ann_vol']:>10.2%} {m['sharpe']:>9.3f} "
            f"{m['sortino']:>9.3f} {m['mdd']:>9.2%} {m['calmar']:>9.3f} {m['hit_rate']:>9.2%}")


def build_spy_weekly(ppo_dates: pd.DatetimeIndex) -> pd.DataFrame:
    spy = pd.read_parquet(SPY_DAILY).reset_index()
    spy = spy[['Date', 'close']].rename(columns={'Date': 'date'})
    spy['date'] = pd.to_datetime(spy['date'])
    spy = spy.sort_values('date').set_index('date')

    union_idx = spy.index.union(ppo_dates)
    spy_close_at_ppo = spy['close'].reindex(union_idx).sort_index().ffill().reindex(ppo_dates)
    rets = []
    for i in range(len(spy_close_at_ppo) - 1):
        c0 = spy_close_at_ppo.iloc[i]
        c1 = spy_close_at_ppo.iloc[i + 1]
        rets.append(c1 / c0 - 1.0 if pd.notna(c0) and pd.notna(c1) and c0 > 0 else 0.0)
    rets.append(0.0)
    return pd.DataFrame({'date': ppo_dates, 'spy_return': rets})


def main():
    if not CAP10_BACKTEST.exists():
        print(f'ERROR: missing {CAP10_BACKTEST} — run backtest_full_period.py on the cap10 config first.')
        sys.exit(1)
    if not UNCAP_BACKTEST.exists():
        print(f'ERROR: missing {UNCAP_BACKTEST}')
        sys.exit(1)
    if not SPY_DAILY.exists():
        print(f'ERROR: missing {SPY_DAILY}')
        sys.exit(1)

    cap10 = pd.read_parquet(CAP10_BACKTEST)
    cap10['date'] = pd.to_datetime(cap10['date'])
    uncap = pd.read_parquet(UNCAP_BACKTEST)
    uncap['date'] = pd.to_datetime(uncap['date'])

    # Union of dates (they should be identical, but be defensive)
    dates = pd.DatetimeIndex(sorted(set(cap10['date']) | set(uncap['date'])))
    spy_w = build_spy_weekly(dates)
    cap10 = cap10.merge(spy_w, on='date', how='left')
    uncap = uncap.merge(spy_w, on='date', how='left')

    cap10['year'] = cap10['date'].dt.year
    uncap['year'] = uncap['date'].dt.year

    blocks = []
    for label, df_cap, df_unc in [
        ('2009-2025 (full, cap10 only)', cap10, uncap[uncap['year'] <= 2024]),
        ('2010-2024 (uncap still missing 2025)', cap10[(cap10['year'] >= 2010) & (cap10['year'] <= 2024)], uncap[uncap['year'] >= 2010]),
        ('2010-2025 (cap10 only)', cap10[cap10['year'] >= 2010], uncap[(uncap['year'] >= 2010) & (uncap['year'] <= 2024)]),
    ]:
        rows = []
        rows.append(metrics('cap10 (046)', df_cap['ppo_return_gross']))
        rows.append(metrics('cap10-SP',    df_cap['sp_return_gross']))
        rows.append(metrics('uncap (038)', df_unc['ppo_return_gross']))
        rows.append(metrics('uncap-SP',    df_unc['sp_return_gross']))
        rows.append(metrics('SPY',         df_cap['spy_return']))
        header = f"{'strat':<14} {'weeks':>6} {'totret':>10} {'annret':>10} {'vol':>10} {'sharpe':>9} {'sortino':>9} {'mdd':>9} {'calmar':>9} {'hit':>9}"
        lines = [f'=== {label} ===', header] + [fmt_row(r) for r in rows]
        block = '\n'.join(lines)
        blocks.append(block)
        print(block + '\n')

    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text('# cap10 (046) vs uncapped (038) vs SPY\n\n```\n' + '\n\n'.join(blocks) + '\n```\n')
    print(f'wrote -> {OUT_REPORT.relative_to(REPO_ROOT)}')


if __name__ == '__main__':
    main()
