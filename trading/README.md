# trading/ — Live Weights Pipeline

Compute this week's target portfolio weights without executing any trades
(Plan 2 of 4).  Broker integration lives in Plan 3.

## Quick start

```bash
# From the repo root:
python -m trading.run weights
```

Output: a table of ticker → weight (%), the k_probs from the regime classifier,
and a one-line sanity-check summary.

```bash
# Override the rebalance date (must be a Friday):
python -m trading.run weights --asof 2026-05-29
```

## Data sources

| Data | Source | How |
|------|--------|-----|
| S&P 500 universe | `SHARADAR/SP500` | `action=='current'` rows |
| Fundamentals (revenue, fcf, assets) | `SHARADAR/SF1` dim=ARQ | Latest `datekey` per ticker |
| Market cap | `SHARADAR/DAILY` | Latest `date` ≤ asof per ticker |
| Macro (VIX, 10Y yield, 10Y-2Y spread) | FRED via `pandas_datareader` | `VIXCLS`, `DGS10`, `T10Y2Y` |
| SPY weekly close | FRED via `pandas_datareader` | Series `SP500`, resampled weekly |

Set `NASDAQ_DATA_LINK_API_KEY` in `.env` (Sharadar).  FRED needs no key.

## Architecture

```
trading/run.py         CLI entry point
trading/weights.py     Full pipeline + freeze + validate
trading/regime.py      Build current-Friday regime feature row
trading/data/
  snapshot.py          Assemble one cross-section per ticker
  sources.py           Sharadar SF1/DAILY + FRED wrappers (with retry)
  universe.py          Current S&P 500 tickers from Sharadar
trading/config.py      Paths, table names, FRED series map, sanity bounds
trading/models/        Persisted k_selector.txt (LightGBM Booster)
trading/audit/         Frozen weight JSONs — gitignored, regeneratable
```

## Sanity checks

After every run, `validate_weights` asserts:
- `weight_sum` within 1e-6 of 1.0
- `max_weight` ≤ 10%
- `10 ≤ n_holdings ≤ 503`

The CLI exits non-zero if any check fails.

## Running tests

```bash
# Unit tests (no network):
python -m pytest tests/trading/ -v

# Live end-to-end smoke (hits Sharadar + FRED):
python -m pytest tests/trading/test_live_smoke.py -v -m slow
```
