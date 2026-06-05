# H_mcap_downside: walk-1 downside-penalty autoresearch on mcap baseline

**Pre-registered:** 2026-06-01

## Motivation

Walk-1 sanity for 048 (`cap10 + mcap baseline + sharpe reward + ep104 + tilt_scale=1.0`)
LOST to Score-Prop:

| metric | 046 walk-1 (Score-Prop+RL) | 048 walk-1 (mcap+RL) |
|---|---:|---:|
| ppo_sharpe | 2.741 | **2.065** |
| ppo_annret | 1.84 | 1.22 |
| ppo_vol | 0.670 | 0.592 |
| ppo_maxdd | -41.0% | -37.9% |
| ppo_turnover | 0.425 | **0.731** |

Vol and MDD improved (mcap baseline working) BUT return dropped 33% and turnover
spiked 72% — PPO over-deviated from the mcap baseline. Search a stricter
drawdown/vol penalty reward to constrain PPO toward defensive tilts.

## Search box

Fixed across all trials:
- `max_weight = 0.10`, `baseline_type = 'mcap'`, `baseline_anchor = true`
- `episode_length = 104`, `learning_rate = 1e-4`, `total_timesteps = 1_000_000`
- `tilt_scale = 1.0`, `action_high = 5.0`, `n_envs = 4`
- All other HPs from config 046

Varied:
- `reward_type = 'downside_penalty'`
- `downside_lambda ∈ {5, 10, 20, 50}` — 4 trials

(Reward = `excess_return - downside_lambda * max(0, -portfolio_return) - cost`,
where `excess_return = portfolio_return - equal_weight_top30_return`.)

## Acceptance bars (all must hold)

For each trial's walk-1 backtest (test 2009 vs Score-Prop deterministic):

| bar | threshold | rationale |
|---|---:|---|
| `ppo_sharpe` | ≥ 2.38 | must at least tie Score-Prop's 2.38 |
| `ppo_maxdd` | ≥ -0.45 | stricter than 046's -0.41 (catch overfit) |
| `ppo_vol` | ≤ 0.65 | lower than 046's 0.67 (mcap effect must persist) |
| `ppo_turnover` | ≤ 0.50 | catch the 048 churn pathology (was 0.73) |

## Decision rule

1. Winner = trial with highest `excess_sharpe` among those passing ALL 4 bars.
2. If no trial passes, **ship deterministic mcap-top30** (Sharpe 0.836 vs SPY 0.856).
3. No retroactive bar relaxation (no-lookahead-bias rule).

## Subsequent test deployment

If a winner emerges:
- 17-walk training under the winning config (~3.5h solo)
- Full backtest, compute 2010-2025 head-to-head vs SPY
- Ship if 2010-2025 Sharpe ≥ deterministic mcap's 0.836
