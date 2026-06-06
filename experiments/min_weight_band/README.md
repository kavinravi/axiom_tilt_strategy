# Min-weight band allocator experiment

Backtests a 2%-floor / 10%-cap band allocator (regime-LGBM picks the modal K,
no ensemble blend) vs the current ensemble blend.

## Run (from repo root)

    pytest experiments/min_weight_band/ -v          # unit tests
    python -m experiments.min_weight_band.run_backtest   # full walk-forward + report

## Outputs (this directory)

- `results.md`            — three-window head-to-head tables (the deliverable)
- `weekly_returns.parquet`— per-strategy net weekly returns (date, strategy, ret)

Nothing is written outside this directory. Live system is untouched.
