# axiom_tilt_strategy

Production-direction successor to the paper repo `axiom_tilt`. The paper repo is
preserved in its final-paper state at `~/CodingStuff/axiom_tilt`; this repo
contains the post-paper iteration line.

## Project direction (2026-06-01)

Drop the LightGBM ranker + FinBERT text features. Replace with classic
factor screens (value, quality, momentum, low-vol, size) over the S&P 500
universe. RL agent tilts over the factor-screened picks under the 10%
per-stock IPS cap, with market-cap (mcap) baseline weighting.

## Key constraints (carried from paper)

- **S&P 500 PIT membership universe** — 826 unique permnos across 2002-2025.
- **10% per-stock cap (`max_weight=0.10`)** — IPS hard constraint, not tunable.
- **Weekly Friday rebalance, 5 bps cost** — same as paper.
- **PIT (point-in-time) everything**: ranker/screen sees only info available
  at the rebalance date; test years are sealed.
- **17 expanding-window walks** — train starts 2002-01-01, test years 2009-2025.

## Status of strategies tested so far (2010-2025 window)

| strategy | Sharpe | AnnRet | Vol | MDD | Notes |
|---|---:|---:|---:|---:|---|
| SPY benchmark | 0.856 | 14.28% | 16.67% | -32% | The bar. |
| cap10 PPO (config 046, score-prop baseline) | 0.709 | 20.27% | 28.58% | -55% | Original cap-only retrain. |
| Score-Prop deterministic | 0.738 | 20.92% | 28.36% | -54% | No RL. |
| **mcap-top30 deterministic + 10% cap** | **0.836** | **21.18%** | **25.33%** | **-43%** | Current ceiling; +6.9pp return vs SPY. |
| cap10 PPO + downside_penalty (λ=20) | ~2.13 walk-1 | — | — | — | LOSS bar; over-defensive. |
| Pruned ranker (154 features) + mcap | wash (aggregate ΔIC ~0) | — | — | — | Modest improvements on some walks, regressions on others. |

The deterministic mcap-top30 with the 10% cap is the current shippable
baseline. The next direction is replacing the LightGBM ranker with
factor screens to see if structural signal selection (vs ML) generalizes
better on the recent walks (14-17) where the ranker's IC is ~0.

## Setup

```
pyenv install 3.12.3 && pyenv local 3.12.3
pip install -e .
cp .env.example .env  # fill in WRDS_USERNAME etc.
```

## Layout

```
src/
  data/    ingestion + panel builders (CRSP, Sharadar, FRED, EDGAR)
  utils/   RL env, ranker helpers, backtest metrics, io
experiments/
  configs/                  per-experiment JSON HP sets
  *.py                      one-off experiment scripts (autoresearch trials,
                            backtests, ablations, feature-pruning utilities)
  H_*.md                    pre-registered hypothesis declarations
data/processed/             PIT panel (CRSP+SF1+macro), universe, training_panel
artifacts/
  rl/walk-NNN/              per-walk scoreboards (top-30 picks, with mcap)
  benchmarks/               SPY daily prices for benchmarking
  backtest_046_cap10/       last good cap10 RL retrain results
reports/                    markdown writeups of completed experiments
```

## Related

- Paper repo: `~/CodingStuff/axiom_tilt` (preserved at paper-publication state)
- Abandoned Dow-30 detour (May 2026): archived at
  `~/CodingStuff/axiom_tilt_strategy_dow_archive/`
