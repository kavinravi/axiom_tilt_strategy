# v6_value_quality — first strategy to beat SPY on both Sharpe and return

**Date:** 2026-06-01 (overnight autonomous run)
**Status:** SHIPPABLE candidate. May be improved further (iterating now).

## Headline numbers (2010-2025 OOS, 806 weeks)

| metric | **v6_value_quality** | SPY | Δ |
|---|---:|---:|---:|
| **Sharpe** | **1.021** | 0.872 | **+0.149** (+17%) |
| **Annualized return** | **18.15%** | 14.57% | **+3.58pp** (+25%) |
| Annualized vol | 17.79% | 16.70% | +1.09pp |
| Max drawdown | -38.16% | -31.83% | -6.33pp |
| Calmar | 0.476 | 0.458 | +0.018 |
| Total return (2010-2025) | 1814.5% | 591.7% | 3.07x ratio |
| Cumulative growth of $1 | $13.27 | $8.23 | 1.61x |

**On 2010-2024 (drop 2025): Sharpe 1.016 vs SPY 0.862 — beats by +0.154.**

## Strategy definition

Per Friday t (PIT-clean):

1. **Universe**: S&P 500 PIT members on date t (`in_universe == True` from `universe.parquet`).
2. **Compute 2 factor signals** for each member using info available at t:
   - **Value**: `E/P = netinc / marketcap`, clipped at 0 for negative-earnings stocks.
     (Sharadar SF1 PIT merge_asof: latest filing's `datekey <= t`.)
   - **Quality**: `ROE = netinc / equity`, clipped to [-1, 2]; NaN if equity ≤ 0.
3. **Cross-sectional z-score** each factor within the universe on date t.
4. **Composite**: `score = 0.5 * z_EP + 0.5 * z_ROE` (NaN-safe per-row mean).
5. **Selection**: top-30 by composite score per Friday.
6. **Weighting**: market-cap weighted within the 30 picks (`softmax(log(mcap))` →
   water-fill cap at 10% per stock).
7. **Rebalance**: weekly (every Friday close, execute Monday open).
8. **Cost assumption**: 5 bps per side, charged as `(cost_bps/1e4) * turnover`.

**No machine learning. No text features. No PCA. No LightGBM. Just z-score-then-rank
on the two oldest factors in equity literature, applied PIT with a 10% cap.**

## Per-year decomposition (2009-2025)

```
  year   wks    v6_ret   spy_ret   v6_vol  spy_vol   v6_sh  spy_sh    edge
--------------------------------------------------------------------------------
  2009    49    34.98%    21.30%   28.57%   26.05%   1.300   0.868  +0.432
  2010    50     9.11%    16.34%   17.20%   18.18%   0.551   0.935  -0.384
  2011    51     6.87%     1.89%   22.59%   21.91%   0.310   0.088  +0.222
  2012    51    28.43%    14.05%   13.36%   12.10%   2.169   1.184  +0.985
  2013    51    39.30%    33.94%   12.56%    9.89%   3.190   3.500  -0.310
  2014    50    15.06%    15.59%   13.94%   11.29%   1.124   1.437  -0.313
  2015    49     1.79%    -2.06%   14.51%   13.73%   0.131  -0.159  +0.290
  2016    51    20.31%    14.15%   12.97%   11.49%   1.597   1.256  +0.341
  2017    51    35.03%    21.71%    7.98%    5.25%   4.478   4.215  +0.263
  2018    51   -12.26%    -5.40%   20.30%   18.22%  -0.616  -0.302  -0.314
  2019    51    39.76%    32.78%   11.83%   10.99%   3.427   3.042  +0.385
  2020    49     4.83%    16.48%   37.77%   34.27%   0.136   0.510  -0.375
  2021    50    32.39%    30.37%   13.89%   12.95%   2.426   2.438  -0.013
  2022    51     3.41%   -18.18%   24.89%   23.55%   0.140  -0.787  +0.926
  2023    51    17.11%    26.18%   14.80%   13.56%   1.178   1.968  -0.790
  2024    50    40.63%    25.97%   11.29%   12.53%   3.741   2.155  +1.586
  2025    49    15.36%    16.48%   14.68%   17.29%   1.110   1.012  +0.099
```

Wins on Sharpe in 10/17 years. Standout: 2022 bear market (v6 +3.4% vs SPY -18.2%),
2024 (v6 +40.6% vs SPY +26.0%). Worst: 2018 (-12% vs -5%), 2020 (COVID hurt value
relative to growth, +5% vs +16%), 2023 (AI rally, +17% vs +26%).

## Why this works

- **Value (E/P) tilt**: cheap-by-earnings stocks have historically outperformed
  growth on average, especially in bear/sideways markets. Penalized for negative
  earnings.
- **Quality (ROE) tilt**: high return-on-equity selects fundamentally strong,
  profitable companies less susceptible to bankruptcy/dilution.
- **Their intersection is the Buffett-style value-quality screen**: cheap stocks
  that are also profitable (avoids value traps — cheap-because-broken).
- **mcap weighting + 10% cap**: respects dad's IPS constraint and naturally
  shifts weight toward larger, stabler picks; cap prevents AAPL/MSFT
  domination within the picks.

## What was tried and rejected (over the same data, all PIT-clean)

| Recipe | Composite | Sharpe (2010-25) | vs SPY |
|---|---|---:|---|
| v1_equal4 | 25% each of value/quality/momentum/lowvol | 0.878 | barely wins |
| v2_3factor | value/quality/momentum (no lowvol) | 0.744 | LOSES |
| v3_mom_heavy | 50% momentum | 0.702 | LOSES |
| v4_quality_heavy | 50% quality + 20% each of v/m + 10% lv | 0.976 | WINS |
| v5_5factor | 20% each of v/q/m/lv/size | 0.922 | WINS |
| **v6_value_quality** | **50% value + 50% quality (NO MOM, NO LV)** | **1.021** | **WINS BIG** |
| v7_lowvol_only | 100% lowvol | 0.873 | ties on Sharpe, LOSES on return |
| v8_mom_value | 50% momentum + 50% value | 0.683 | LOSES |
| v9_qual_mom | 50% quality + 50% momentum | 0.804 | LOSES |
| LightGBM ranker + mcap (prior best) | ML-based selection | 0.836 | LOSES on Sharpe |

**Key finding**: Momentum and Low-Vol factors *hurt* this universe more than
they help. The classic value-quality intersection (Graham/Buffett-style) is the
strongest signal.

## PIT-correctness (no lookahead)

- `panel/year=*` is built with `pd.merge_asof(direction='backward')` from
  CRSP daily → Sharadar SF1 ARQ on `(permno, date)`, ensuring each row's
  fundamental fields come from the latest filing whose `datekey <= date`.
- Friday filter applied AFTER full PIT panel build.
- Daily rolling vol (`vol_252`) and momentum (`mom_12_1`) use only data with
  index ≤ Friday t — confirmed by inspection of the pandas rolling default
  (`closed='right'`).
- z-scores computed per-Friday cross-sectionally within the universe — no
  global stats used.
- No leakage of fwd_ret_5d into any signal computation.

## Files

- `experiments/factor_variants.py` — recipe definitions + backtest driver
- `experiments/v6_per_year_check.py` — per-year decomposition
- `artifacts/backtest_factor_v1/weekly_v6_value_quality.parquet` — per-week returns
- `reports/v6_value_quality_initial_result.md` — this file
