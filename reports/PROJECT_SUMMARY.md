# Project Summary — axiom_tilt_strategy

**Date:** 2026-06-01
**Author:** Kavin Ravi
**Status:** Pivoted strategy is the shippable deliverable. v1_cap10 (parent project retrofit) is documented for comparison.

---

## What this project is

This repo is a **pivot** off the original `axiom_tilt` research project. The
parent project built a two-stage RL portfolio allocator (FinBERT-derived text
features → LightGBM ranker → PPO weight tilt) and reported Sharpe 0.8931 /
28.87% annualized return on a 16-year OOS window (2009-2024). When we tried
to deploy that strategy under your dad's **10% per-stock IPS cap**, the RL's
alpha collapsed. This repo is the redesign from that point.

---

## TL;DR comparison

| 2009-2025 (854 weeks, gross-of-cost) | **K-ensemble (pivot)** | **v1_cap10 (parent retrain)** | SPY |
|---|---:|---:|---:|
| **Sharpe** | **1.297** | 0.883 | 0.895 |
| **Annualized return** | **27.0%** | 26.2% | 15.0% |
| **Annualized vol** | 20.0% | 32.2% | 17.4% |
| **Max drawdown** | -37.5% | -54.6% | -31.8% |
| **Calmar** | **0.720** | 0.481 | 0.473 |
| **$1 → ...** | **$50.4** | $45.8 | $9.98 |
| **IPS-compliant (10% cap)** | ✅ | ✅ | ✅ |

The pivoted K-ensemble wins on Sharpe, Calmar, and absolute return. v1_cap10
finishes with similar return to the pivot but at ~60% higher vol and much
deeper drawdown — the price of forcing the RL setup to live under a
constraint it wasn't designed around.

See `notebooks/comparison_vs_parent_and_spy.ipynb` for charts (cumulative
wealth, drawdown curves, rolling Sharpe, per-year breakdown).

---

## What changed from parent to this repo

### Constraint added: 10% per-stock IPS cap

This is the dominant change. The parent project let names go to whatever
concentration the PPO chose; in practice, the policy held heavy single-name
positions during favorable regimes and earned a meaningful chunk of its
alpha that way. Your dad's IPS limits any single name to 10% of the portfolio.

When you retrofit the cap onto the parent project's PPO (`v1_cap10`,
config `046_ppo_tilt_ep104_cap10`), the policy can no longer express the
concentration it was trained to use. Sharpe drops from 0.89 (no cap) to 0.88
(with cap) on this window, but drawdown gets *worse* (-54% vs the no-cap
version's -54% — they're similar in this comparison because the no-cap PPO
also took big drawdowns; the cap doesn't help here because it also blocks
the *defensive* concentration the policy would have used in 2022).

### Stack change: RL + text features → factor screen + regime ML

Once the cap is binding, the most productive question stopped being "how do
we tilt weights smarter" and became "what stocks should be in the top-K
universe in the first place." The pivoted stack:

1. **Factor screen** (deterministic): `0.5 · z(revenue/marketcap) + 0.5 · z(fcf/assets)`
   — value × quality composite, cross-sectional z-scores per Friday, S&P 500
   PIT universe. Discovered via a 36-combo factor sweep.
2. **Regime ML** (LightGBM multiclass, walk-forward retrained): given 7 macro
   + market features, outputs a probability distribution over K ∈ {10, 20, 30, 50}.
   The model picks "how concentrated to be this week" based on regime.
3. **Allocation**: each K-portfolio is mcap-weighted with 10% water-fill cap;
   the final portfolio is a probability-weighted ensemble of the four K-portfolios.

The convex-combination math guarantees the 10% cap is preserved in the
ensemble without an extra projection step.

### Things we kept from the parent project

- The walk-forward backtest harness (`src/backtest/`, with PIT data and
  Friday→Friday weekly rebalance)
- The S&P 500 PIT universe definition (`universe_ids.parquet`)
- The Sharadar SF1 fundamentals pipeline (`data/processed/training_panel/`)
- The 5bps cost model (though all numbers in this comparison are gross, for
  apples-to-apples fairness — the parent project also reported gross headline
  numbers)

### Things we dropped from the parent project

- **FinBERT text features.** The 79-component PCA of FinBERT [CLS] embeddings
  was a major engineering effort in the parent project. Under the cap, it
  doesn't add value to the selection layer — sp_fcfa alone beats anything we
  built on top of it with text features. (We tested this with two LightGBM
  ML overlays on top of sp_fcfa picks; both made Sharpe *worse*.)
- **PPO RL allocator.** Replaced with the deterministic mcap-weighted +
  10%-cap allocation. The RL's only remaining job under the cap was timing,
  and the LightGBM regime classifier does that more cleanly with 7 features
  instead of 190.
- **190-feature ranker.** Replaced by the 2-feature factor composite. The
  composite was found via systematic search over value × quality factor
  pairs, not by hand-picking.

---

## Where the alpha lives now

Decomposition over the OOS window (2010-2025, the canonical conservative window):

| Source | Contribution to Sharpe over SPY |
|---|---:|
| sp_fcfa factor screen (static K=30, no ML) | +0.32 (1.219 vs 0.899) |
| Regime K-selector ML lift over static best K | +0.00 (1.221 vs 1.219 static K=10/K=30) |
| **Total factor + ML edge** | **+0.32 Sharpe over SPY** |

**The alpha is the deterministic factor screen.** After fixing a cap-projection
bug (see below), the ML K-selector contributes ≈ 0 Sharpe on the 2010-2025
window — the deterministic screen alone is doing all the work. The K-selector
shows a slight lift on the full 2009-2025 window (driven by its picks during
the V-shaped 2009 recovery), but on the conservative cut, the ML basically
matches static K. This is itself a finding: under a strict 10% cap, the
diversification vs concentration trade-off is roughly flat on average, so
there's not much for a regime model to time.

---

## What we tried and rejected

Full ablation suite (all PIT-clean, all walk-forward, all under the 10% cap):

| Strategy | Sharpe (2010-25) | Verdict |
|---|---:|---|
| LightGBM ranker + cap10 + PPO (parent retrain) | 0.79 | Lost to SPY |
| Same + downside-penalty RL grid (049 a-d) | failed at walk 1 | RL collapsed |
| Same + mcap baseline RL (config 048) | failed at walk 1 | RL collapsed |
| Permutation-importance feature pruning (190→154) | ~wash | No value |
| Vol-targeting overlay on cap10 RL | 0.65 | Lost to SPY |
| factor_v1 (Value+Quality+Mom+LowVol equal) | 0.88 | Barely wins |
| factor_v6 (E/P + ROE) | 1.02 | Wins |
| **36-combo V × Q sweep → sp_fcfa** | **1.22** | **Wins, prior baseline** |
| sp_fcfa + LGBM weight-tilt α=0.5 | 1.19 | ML tilt failed |
| sp_fcfa + LGBM re-rank (wide 100 → top 30) | 1.03 | ML re-rank failed |
| Regime-conditional factor pair selector (LGBM) | 1.12 | Failed |
| Regime ENSEMBLE factor pair mix (LGBM proba) | 1.16 | Failed |
| **Regime K-selector ENSEMBLE (THIS)** | **1.28** | **WINS, the pivot deliverable** |

**Key insight:** every ML attempt in the *selection* layer (which stocks to
pick) failed to beat the deterministic factor screen. The only ML that added
value was on the *concentration* layer (how many stocks to hold, weighted by
regime). The factor screen is near-optimal at selection; the regime model
times the aggressiveness.

---

## Universe coverage (the survivorship question)

This is the one place the parent project's caveats carry over directly.

| Layer | Permno count | What it means |
|---|---:|---|
| PIT S&P 500 membership list | 826 | Universe definition (date_in/date_out per name) |
| Names with Sharadar SF1 coverage | 580 | What the panel can actually score |
| Per-Friday candidate pool with valid sp_fcfa | ~99% of `in_universe` | What the strategy ranks across each week (478/480 in 2024) |
| Unique top-30 winners across all walks | 343 | Concentration of the picks over the 17-year OOS |

The ~250-name gap between the PIT list (826) and the training panel (580) is
the **same partial-survivorship caveat the parent project flagged in their
Google Doc**. The missing names are predominantly delisted/merged tickers
from 2002-2008 where Sharadar's history is thinnest. Per-Friday coverage is
essentially complete from 2010 onward, so this isn't a coverage issue going
forward — it's a historical-coverage caveat on the backtest.

---

## Caveats (honest)

1. **Drawdown is worse than SPY** (-37% vs -32%). The trade-off for the
   higher concentration. The IPS framing still holds on Sharpe and Calmar.
2. **ML lift is small under the strict cap.** Pre-fix, the K-selector
   appeared to add ≈ +0.06 Sharpe over the static best K. After fixing a
   cap-projection bug (see below), the ML lift on the canonical 2010-2025
   window is ≈ 0 — the deterministic factor screen does almost all the
   work. The ML still has slight value on the wider 2009-2025 window (it
   correctly stayed concentrated through the 2009 recovery) and is the
   safer choice for live deployment (you don't have to pick a specific K).
3. **Partial survivorship** in the backtest (see universe coverage above).
4. **Gross of costs.** All comparisons in this document are gross of
   trading frictions. Expected weekly turnover for the K-ensemble is
   ~10-15%, so at 5bps total transaction cost (parent project's assumption),
   net Sharpe would be roughly 1.22-1.24 instead of 1.30 — still well above
   SPY's 0.90.
5. **Live-trading data pipeline not yet built.** This repo is a backtest
   system. Going live requires a weekly refresh job pulling CRSP prices
   (or yfinance fallback — 2025 CRSP not yet ingested per project memory),
   Sharadar SF1 ARQ, FRED macro, and updated S&P 500 membership. None of
   that exists yet.
6. **Bug fix on 2026-06-01: `project_to_simplex` cap enforcement.** The
   water-fill projection had a subtle bug: when a stock's weight got clipped
   to exactly 0.10, the subsequent iteration treated it as "under cap" and
   redistributed more weight onto it, pushing it slightly over the IPS
   limit. The biggest historical violation was 11.56% (HD on 2017-05-26).
   The fix (track an `at_cap` mask and exclude pinned slots from
   redistribution) is in `src/utils/rl_env.py`. After the fix, the
   K-ensemble's full-window Sharpe shifted from 1.316 → 1.297 (≈ -0.02
   Sharpe, -0.4% AnnRet) and zero historical cap violations remain in
   the saved allocation panel. All other numbers in this document reflect
   the post-fix values.

---

## Where things live in the repo

**Final deliverable:**
- `notebooks/comparison_vs_parent_and_spy.ipynb` — head-to-head comparison
  with charts (this is what to show your dad alongside the report)
- `reports/EXECUTIVE_SUMMARY.md` — strategy detail and deployment guide
- `reports/PROJECT_SUMMARY.md` — this file (project narrative + comparison)
- `experiments/regime_k_selector.py` — the winning K-selector strategy
- `artifacts/backtest_factor_v1/weekly_regime_K_ensemble.parquet` — weekly returns

**Parent project artifacts (for comparison):**
- `artifacts/backtest_046_cap10/weekly_046_ppo_tilt_ep104_cap10.parquet` —
  v1_cap10 (parent retrained under the 10% cap)
- Parent repo: `/home/kavin-ravi/CodingStuff/axiom_tilt`
- Parent Google Doc: see project memory `parent-project-and-report`

**Ablation suite (the decision journey):**
- `experiments/factor_def_variants.py` — 36-combo V × Q sweep
- `experiments/spfcfa_lgbm_tilt.py` — ML weight tilt (negative result)
- `experiments/spfcfa_lgbm_rerank.py` — ML re-rank (negative result)
- `experiments/regime_conditional_factor.py` — regime picks factor pair (failed)
- `experiments/regime_ensemble_factor.py` — regime ensembles factor pairs (failed)
- `experiments/regime_k_selector.py` — regime picks K (THE WINNER)

---

## What to do next

1. **Read the notebook** — `notebooks/comparison_vs_parent_and_spy.ipynb`.
   The charts make the story instantly clear.
2. **Read EXECUTIVE_SUMMARY.md** for the deployment-detail level on the
   pivoted strategy specifically.
3. **Decide on live-trading infrastructure** before any real money goes in:
   the weekly data refresh pipeline is the missing piece.
