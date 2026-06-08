# Floored-blend experiment (2010-2025): keep the ML, kill the tail

The ML (regime->K LGBM) blends top-K books every Friday; we then prune sub-floor dust and band-project survivors to [floor, 10%]. Net of 5 bps x one-way turnover. avgN/minN/maxN = holdings count; minWt = mean of the per-week smallest weight.

| strategy | ann | vol | sharpe | sortino | mdd | turn | avgN | minN | maxN | minWt |
|----------|----:|----:|-------:|--------:|----:|-----:|-----:|-----:|-----:|------:|
| old blend (no floor) |  22.5% | 18.3% |  1.23 |  1.60 | -37.5% | 0.11 | 50.0 |  50 |  50 | 0.07% |
| old blend + 1% floor |  23.1% | 18.4% |  1.26 |  1.65 | -36.8% | 0.12 | 23.3 |  17 |  31 | 1.24% |
| old blend + 1% floor, >=20 |  23.1% | 18.4% |  1.26 |  1.65 | -36.8% | 0.12 | 23.3 |  20 |  31 | 1.22% |
| old blend + 2% floor, >=20 |  23.7% | 18.5% |  1.28 |  1.71 | -36.6% | 0.12 | 20.0 |  20 |  23 | 2.06% |
| static K=20 nofloor |  22.5% | 18.4% |  1.23 |  1.55 | -39.3% | 0.11 | 20.0 |  20 |  20 | 0.66% |
