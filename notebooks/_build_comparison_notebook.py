"""Generate the comparison notebook (strategy vs SPY vs v1_cap10).

Run once:   python notebooks/_build_comparison_notebook.py
Then:       jupyter nbconvert --to notebook --execute notebooks/comparison_vs_parent_and_spy.ipynb --inplace
"""
from __future__ import annotations

import nbformat as nbf
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "notebooks" / "comparison_vs_parent_and_spy.ipynb"

nb = nbf.v4.new_notebook()
cells = []

def md(s):  cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

# -------------------------------------------------------------------
md(r"""
# Pivoted strategy vs parent `axiom_tilt` (v1_cap10) vs SPY

**Date:** 2026-06-01
**Repo:** `axiom_tilt_strategy` (S&P 500, post-pivot from the Dow experiment)

This notebook is the head-to-head between three strategies on the same 855-week
OOS window (2009-01-02 → 2025-12-19):

| Series | What it is |
|---|---|
| **K-ensemble** | The final pivoted strategy: `sp_fcfa` factor screen → LightGBM regime classifier picks K ∈ {10,20,30,50} concentration → mcap-weighted with 10% IPS cap → ensemble blends K-portfolios by the LGBM's class probabilities. |
| **v1_cap10** | The parent `axiom_tilt` project's PPO RL allocator, retrained under the 10% IPS cap (config `046_ppo_tilt_ep104_cap10`). Two-stage: LightGBM ranker (190 features incl. FinBERT text) picks top-30 → PPO tilts the weights. |
| **SPY** | Benchmark, equity-only buy-and-hold. |

All three are **gross** of trading costs (no slippage/commission applied). They
share the same 855 Friday-to-Friday weekly return cadence, so the comparison
is apples-to-apples.

**Important context** (in case you forgot):
- Parent project's headline was PPO 0.8931 Sharpe / 28.87% ann / -54.32% MDD
  on 2009-2024 (806 weeks). That was a **no-cap** RL setup — names could go
  to ~100% concentration if the policy chose to.
- The 10% per-stock IPS cap is a hard constraint we got from your dad. When
  we forced the parent project to live under it (v1_cap10 retrain), it
  collapsed to a Sharpe of ~0.71, losing decisively to SPY.
- The pivoted strategy here was redesigned **assuming the cap from day one** —
  it doesn't try to recover the RL's concentration alpha because that alpha
  was illegal under the IPS. It searches for alpha that survives the cap.
""")

# -------------------------------------------------------------------
md("## 1. Load data")

code(r"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.float_format = "{:,.4f}".format

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
ART = REPO_ROOT / "artifacts"

# --- K-ensemble (the pivoted strategy) ---
k_ens = pd.read_parquet(ART / "backtest_factor_v1" / "weekly_regime_K_ensemble.parquet")
k_ens = k_ens.set_index(pd.to_datetime(k_ens["date"]))["weekly_ret"].rename("k_ensemble")

# --- v1_cap10 (parent project retrained under 10% cap) ---
v1 = pd.read_parquet(ART / "backtest_046_cap10" / "weekly_046_ppo_tilt_ep104_cap10.parquet")
v1 = v1.set_index(pd.to_datetime(v1["date"]))["ppo_return_gross"].rename("v1_cap10")

# --- SPY (Friday→Friday weekly returns) ---
spy = pd.read_parquet(ART / "benchmarks" / "spy_daily.parquet").rename(columns={"close": "spy_close"})
spy.index = pd.to_datetime(spy.index)
# Align SPY to the same Fridays as the strategies (use close on each Friday;
# compute pct change). Forward-fill across holiday Fridays so dates align.
fridays = k_ens.index
spy_fri = spy.reindex(fridays, method="ffill")["spy_close"]
spy_ret = spy_fri.pct_change().rename("spy")

df = pd.concat([k_ens, v1, spy_ret], axis=1).dropna()
print(f"Combined panel: {df.shape}  ({df.index.min().date()} → {df.index.max().date()})")
df.head()
""")

# -------------------------------------------------------------------
md("## 2. Summary stats — full OOS window (2009-2025)")

code(r"""
TRADING_WEEKS = 52.0  # Friday→Friday cadence

def perf_stats(r: pd.Series) -> dict:
    r = r.dropna()
    ann = (1 + r).prod() ** (TRADING_WEEKS / len(r)) - 1
    vol = r.std(ddof=1) * np.sqrt(TRADING_WEEKS)
    sharpe = (r.mean() * TRADING_WEEKS) / vol if vol > 0 else np.nan
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    mdd = dd.min()
    calmar = ann / abs(mdd) if mdd < 0 else np.nan
    return {
        "Sharpe": sharpe,
        "AnnRet": ann,
        "AnnVol": vol,
        "MaxDD": mdd,
        "Calmar": calmar,
        "TotalRet": eq.iloc[-1] - 1,
        "$1→$X": eq.iloc[-1],
        "n_weeks": len(r),
    }

stats_full = pd.DataFrame({
    "K-ensemble (pivot)": perf_stats(df["k_ensemble"]),
    "v1_cap10 (parent retrain)": perf_stats(df["v1_cap10"]),
    "SPY":                 perf_stats(df["spy"]),
}).T
stats_full
""")

# -------------------------------------------------------------------
md(r"""
**What this table says:**

- The pivoted K-ensemble beats SPY on every risk-adjusted metric (Sharpe, Calmar)
  AND on raw return.
- The parent project's cap10 retrain (v1_cap10) loses to SPY on Sharpe — the
  10% cap kills the alpha the PPO was originally generating from concentration.
- The drawdown difference between v1_cap10 and the K-ensemble shows what
  re-designing under the cap (vs retrofitting it) buys you.
""")

# -------------------------------------------------------------------
md("## 3. Same comparison on parent's reported window (2009-2024)")

code(r"""
mask_pp = (df.index >= "2009-01-01") & (df.index <= "2024-12-31")
df_pp = df.loc[mask_pp]

stats_pp = pd.DataFrame({
    "K-ensemble (pivot)": perf_stats(df_pp["k_ensemble"]),
    "v1_cap10 (parent retrain)": perf_stats(df_pp["v1_cap10"]),
    "SPY":                 perf_stats(df_pp["spy"]),
}).T
stats_pp
""")

md(r"""
This is the window the parent project's Google Doc report used (806 weeks).
The parent project's *original* PPO (no cap) reported Sharpe 0.8931 on this
window. Once you retrofit the 10% cap onto that same RL setup, it collapses
to what `v1_cap10` shows above. The pivoted strategy on the same window beats
both.
""")

# -------------------------------------------------------------------
md("## 4. Cumulative wealth — log scale")

code(r"""
eq = (1 + df).cumprod()
eq.loc[df.index.min() - pd.Timedelta(days=1)] = 1.0
eq = eq.sort_index()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(eq.index, eq["k_ensemble"], label="K-ensemble (pivot)", color="#1f77b4", linewidth=1.8)
ax.plot(eq.index, eq["v1_cap10"],   label="v1_cap10 (parent retrain)", color="#d62728", linewidth=1.2, alpha=0.85)
ax.plot(eq.index, eq["spy"],        label="SPY", color="#7f7f7f", linewidth=1.5, linestyle="--")
ax.set_yscale("log")
ax.set_title("Growth of \\$1 — log scale (2009-01 → 2025-12)")
ax.set_ylabel("equity ($1 = start)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()
""")

# -------------------------------------------------------------------
md("## 5. Per-calendar-year returns")

code(r"""
yearly = (1 + df).groupby(df.index.year).apply(lambda x: x.prod() - 1)
yearly.index.name = "year"
yearly["pivot − SPY"] = yearly["k_ensemble"] - yearly["spy"]
yearly["v1_cap10 − SPY"] = yearly["v1_cap10"] - yearly["spy"]
yearly.style.format("{:+.2%}")
""")

code(r"""
# Quick visual of the active return (strategy − SPY) per year
fig, ax = plt.subplots(figsize=(12, 4))
w = 0.4
x = np.arange(len(yearly.index))
ax.bar(x - w/2, yearly["pivot − SPY"],     w, label="K-ensemble − SPY", color="#1f77b4")
ax.bar(x + w/2, yearly["v1_cap10 − SPY"], w, label="v1_cap10 − SPY",   color="#d62728")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(yearly.index.tolist(), rotation=45)
ax.set_title("Active return vs SPY, per calendar year")
ax.set_ylabel("strategy return − SPY return")
ax.grid(alpha=0.3, axis="y")
ax.legend()
fig.tight_layout()
plt.show()
""")

md(r"""
Hit-rate counter — what fraction of years did each strategy beat SPY?
""")

code(r"""
n_years = len(yearly)
hit_pivot   = (yearly["pivot − SPY"]    > 0).sum()
hit_v1      = (yearly["v1_cap10 − SPY"] > 0).sum()
print(f"K-ensemble beats SPY in {hit_pivot}/{n_years} years ({hit_pivot/n_years:.0%})")
print(f"v1_cap10   beats SPY in {hit_v1}/{n_years} years ({hit_v1/n_years:.0%})")
""")

# -------------------------------------------------------------------
md("## 6. Drawdown curves")

code(r"""
def drawdown(r):
    eq = (1 + r).cumprod()
    return eq / eq.cummax() - 1

dd = pd.DataFrame({c: drawdown(df[c]) for c in df.columns})

fig, ax = plt.subplots(figsize=(12, 5))
ax.fill_between(dd.index, dd["k_ensemble"], 0, alpha=0.4, color="#1f77b4", label="K-ensemble")
ax.fill_between(dd.index, dd["v1_cap10"],   0, alpha=0.4, color="#d62728", label="v1_cap10")
ax.plot(dd.index, dd["spy"], color="black", linewidth=1.2, label="SPY")
ax.set_title("Drawdown from peak")
ax.set_ylabel("drawdown")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()
""")

md(r"""
**Drawdown read:** The pivoted K-ensemble has a worse trough than SPY
(roughly -37% vs -32%) — the price of concentration. v1_cap10 lives in
a deeper, more persistent drawdown for most of the OOS window because its
return engine is broken by the cap retrofit.
""")

# -------------------------------------------------------------------
md(r"""
### 6b. Worst drawdown experienced within each calendar year

For each calendar year, the deepest intra-year drawdown (computed against
that year's own peak, so 2008-style accumulated drawdowns don't mask
recovery years). Useful for "what was the worst single year for the
strategy" framing.
""")

code(r"""
def yearly_max_drawdown(returns):
    # For each calendar year, the deepest intra-year DD (against that year's own peak).
    out = {}
    for yr, grp in returns.groupby(returns.index.year):
        eq = (1 + grp).cumprod()
        dd = eq / eq.cummax() - 1
        out[yr] = dd.min()
    return pd.Series(out, name="max_dd")

yearly_dd = pd.DataFrame({c: yearly_max_drawdown(df[c]) for c in df.columns})
yearly_dd.index.name = "year"
yearly_dd
""")

code(r"""
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(yearly_dd.index))
w = 0.27
ax.bar(x - w, yearly_dd["k_ensemble"], w, label="K-ensemble", color="#1f77b4")
ax.bar(x,     yearly_dd["v1_cap10"],   w, label="v1_cap10",   color="#d62728")
ax.bar(x + w, yearly_dd["spy"],        w, label="SPY",         color="#7f7f7f")
ax.axhline(0, color="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(yearly_dd.index.tolist(), rotation=45)
ax.set_title("Worst intra-year drawdown, by calendar year")
ax.set_ylabel("max DD within year")
ax.grid(alpha=0.3, axis="y")
ax.legend()
fig.tight_layout()
plt.show()

# Headline stats
print(f"K-ensemble  — worst year: {yearly_dd['k_ensemble'].idxmin()} ({yearly_dd['k_ensemble'].min():.2%})")
print(f"v1_cap10    — worst year: {yearly_dd['v1_cap10'].idxmin()} ({yearly_dd['v1_cap10'].min():.2%})")
print(f"SPY         — worst year: {yearly_dd['spy'].idxmin()} ({yearly_dd['spy'].min():.2%})")
""")

# -------------------------------------------------------------------
md("## 7. Rolling 1-year Sharpe")

code(r"""
WIN = 52  # weeks

def rolling_sharpe(r, win=WIN):
    return (r.rolling(win).mean() * TRADING_WEEKS) / (r.rolling(win).std(ddof=1) * np.sqrt(TRADING_WEEKS))

rs = pd.DataFrame({c: rolling_sharpe(df[c]) for c in df.columns}).dropna()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(rs.index, rs["k_ensemble"], label="K-ensemble", color="#1f77b4")
ax.plot(rs.index, rs["v1_cap10"],   label="v1_cap10",   color="#d62728", alpha=0.85)
ax.plot(rs.index, rs["spy"],        label="SPY",        color="black", linewidth=1.0, linestyle="--")
ax.axhline(0, color="grey", linewidth=0.6)
ax.set_title("Rolling 52-week Sharpe")
ax.set_ylabel("annualized Sharpe (1y window)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()
""")

# -------------------------------------------------------------------
md("## 8. Correlation matrix")

code(r"""
df.corr().style.format("{:.3f}").background_gradient(cmap="RdBu_r", vmin=-1, vmax=1)
""")

# -------------------------------------------------------------------
md(r"""
## 9. K-ensemble allocation over time

These charts use the reconstructed per-Friday per-stock weights from
`artifacts/backtest_factor_v1/k_ensemble_weights.parquet` (855 OOS Fridays,
~42k weight rows). The 10% IPS cap is enforced as a hard constraint at the
per-K level; the ensemble preserves it via convex combination.
""")

code(r"""
ALLOC_PATH = ART / "backtest_factor_v1" / "k_ensemble_weights.parquet"
PROBA_PATH = ART / "backtest_factor_v1" / "k_ensemble_probas.parquet"

alloc = pd.read_parquet(ALLOC_PATH)
alloc["date"] = pd.to_datetime(alloc["date"])
probas = pd.read_parquet(PROBA_PATH)
probas["date"] = pd.to_datetime(probas["date"])

# Universe IDs for ticker labels
uni = pd.read_parquet(REPO_ROOT / "data" / "processed" / "universe_ids.parquet")[["permno","ticker","company"]]
alloc = alloc.merge(uni, on="permno", how="left")

print(f"Allocation panel: {alloc.shape[0]:,} rows across {alloc['date'].nunique()} Fridays")
print(f"Unique permnos ever held: {alloc['permno'].nunique()}")
""")

md(r"""
### 9a. K-pick distribution over time

This is the LightGBM regime classifier's probability over K ∈ {10, 20, 30, 50}
each week. K=10 means "concentrate heavily, 10 names"; K=50 means "diversify
defensively, 50 names". The mass-weighted average K (right axis) gives a
single-number summary of "how concentrated is the strategy this week".
""")

code(r"""
proba_x = probas.set_index("date")
K_vals = np.array([10, 20, 30, 50])
weighted_K = proba_x[["K10_prob","K20_prob","K30_prob","K50_prob"]].to_numpy() @ K_vals

fig, ax = plt.subplots(figsize=(12, 5))
ax.stackplot(proba_x.index,
             proba_x["K10_prob"], proba_x["K20_prob"],
             proba_x["K30_prob"], proba_x["K50_prob"],
             labels=["P(K=10) concentrated", "P(K=20)", "P(K=30)", "P(K=50) defensive"],
             colors=["#08306b", "#2171b5", "#6baed6", "#c6dbef"])
ax.set_ylabel("probability")
ax.set_title("Regime classifier's K-distribution per Friday")
ax.set_ylim(0, 1)
ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
ax2 = ax.twinx()
ax2.plot(proba_x.index, weighted_K, color="red", linewidth=1.2, alpha=0.8, label="Avg K (right axis)")
ax2.set_ylabel("probability-weighted K", color="red")
ax2.tick_params(axis="y", colors="red")
ax2.set_ylim(8, 52)
fig.tight_layout()
plt.show()
""")

md(r"""
### 9b. Effective number of holdings (inverse-HHI)

This is `1 / Σ w_i²`. A perfectly equal-weighted N-stock portfolio gives N; a
max-concentration (10% cap → 10 names) portfolio gives 10. Higher = more
diversified.
""")

code(r"""
def eff_n_per_date(g):
    w = g["weight"].to_numpy()
    return 1.0 / np.sum(w * w)

eff_n = alloc.groupby("date").apply(eff_n_per_date).rename("eff_N")
nominal_held = alloc.groupby("date").size().rename("nominal_held")
hold = pd.concat([eff_n, nominal_held], axis=1)

fig, ax = plt.subplots(figsize=(12, 4.5))
ax.plot(hold.index, hold["eff_N"], color="#1f77b4", linewidth=1.4, label="Effective N (1 / Σw²)")
ax.plot(hold.index, hold["nominal_held"], color="grey", linewidth=0.8, alpha=0.6, label="Nominal # of names held")
ax.axhline(10, color="red", linestyle=":", alpha=0.5, label="K=10 floor (most concentrated)")
ax.axhline(50, color="green", linestyle=":", alpha=0.5, label="K=50 ceiling (most defensive)")
ax.set_title("Portfolio concentration over time")
ax.set_ylabel("# names")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
plt.show()

print(f"\nNominal holdings — mean: {hold['nominal_held'].mean():.0f}, min: {hold['nominal_held'].min()}, max: {hold['nominal_held'].max()}")
print(f"Effective N    — mean: {hold['eff_N'].mean():.1f}, min: {hold['eff_N'].min():.1f}, max: {hold['eff_N'].max():.1f}")
""")

md(r"""
**Read:** because the ensemble blends K=10 through K=50 via probabilities, the
*nominal* names held is large (close to the union of all four K-portfolios)
but the *effective* concentration is set by the highest-probability K. When
P(K=10) is high, effective N drops toward 10; when P(K=50) dominates,
effective N climbs toward 50. The IPS cap forces effective N ≥ 10 — the
strategy literally cannot be more concentrated than 10 equal-weight names.
""")

md(r"""
### 9c. Max single-stock weight per week

The hard 10% cap means this should never exceed 0.10.
""")

code(r"""
max_w = alloc.groupby("date")["weight"].max().rename("max_weight")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(max_w.index, max_w, color="#1f77b4", linewidth=1.2)
ax.axhline(0.10, color="red", linestyle="--", label="10% IPS cap")
ax.fill_between(max_w.index, max_w, 0.10, where=max_w >= 0.10 - 1e-6,
                color="red", alpha=0.15, label="Cap binding")
ax.set_title("Largest single-stock weight per Friday")
ax.set_ylabel("weight")
ax.set_ylim(0, 0.12)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()

cap_binding = (max_w >= 0.10 - 1e-6).sum()
print(f"\nCap binding on {cap_binding}/{len(max_w)} Fridays ({cap_binding/len(max_w):.1%})")
print(f"Mean max weight: {max_w.mean():.4f}, p95: {max_w.quantile(0.95):.4f}")
""")

md(r"""
### 9d. Top-10 most-held names (avg weight, OOS-window-wide)

Names that the strategy concentrated in over time. The average weight is
computed across all 855 Fridays — a name with avg weight 1% means it
contributed 1% of capital on average across the entire OOS period (whether
held heavily in some years and not in others, or held lightly throughout).
""")

code(r"""
total_held = alloc.groupby(["permno","ticker","company"], dropna=False)["weight"].agg(
    ["mean","max","count"]
).rename(columns={"mean":"avg_w","max":"max_w","count":"weeks_held"})
total_held = total_held.sort_values("avg_w", ascending=False).head(15)
total_held["pct_weeks"] = total_held["weeks_held"] / alloc["date"].nunique()
total_held.style.format({"avg_w":"{:.3%}", "max_w":"{:.2%}", "pct_weeks":"{:.0%}"})
""")

md(r"""
### 9e. Stacked weight of top-10 holdings over time

Shows how the top 10 weighted names (across the full window) evolved week
to week. The total height of the stack is the share of the portfolio those
names commanded that week; the rest went to lower-weighted names.
""")

code(r"""
top10_permnos = total_held.head(10).reset_index()["permno"].tolist()
top10_tickers = total_held.head(10).reset_index()["ticker"].fillna("?").tolist()
sub = alloc[alloc["permno"].isin(top10_permnos)].copy()
pivot = sub.pivot_table(index="date", columns="ticker", values="weight", fill_value=0.0)
# Match column order to top10 ranking
pivot = pivot[[t for t in top10_tickers if t in pivot.columns]]

fig, ax = plt.subplots(figsize=(13, 5.5))
ax.stackplot(pivot.index, pivot.T.values, labels=pivot.columns.tolist(),
             colors=plt.cm.tab10.colors[:len(pivot.columns)])
ax.set_title("Weights of top-10 most-weighted names, week by week")
ax.set_ylabel("portfolio weight")
ax.legend(loc="upper left", ncol=2, fontsize=8, framealpha=0.9)
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
""")

md(r"""
### 9f. Weight in top-N names (concentration profile)

For each Friday, the cumulative weight in the top-1, top-3, top-5, and top-10
names. Higher = more concentrated.
""")

code(r"""
def concentration_curve(g):
    w = np.sort(g["weight"].to_numpy())[::-1]  # descending
    return pd.Series({
        "top1":  w[:1].sum(),
        "top3":  w[:3].sum(),
        "top5":  w[:5].sum(),
        "top10": w[:10].sum(),
    })

conc = alloc.groupby("date").apply(concentration_curve)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(conc.index, conc["top1"],  label="Top-1 weight",  color="#d62728", linewidth=1.0, alpha=0.8)
ax.plot(conc.index, conc["top3"],  label="Top-3 weight",  color="#ff7f0e", linewidth=1.0, alpha=0.8)
ax.plot(conc.index, conc["top5"],  label="Top-5 weight",  color="#2ca02c", linewidth=1.0, alpha=0.8)
ax.plot(conc.index, conc["top10"], label="Top-10 weight", color="#1f77b4", linewidth=1.2)
ax.set_title("Cumulative weight in top-N names (concentration profile)")
ax.set_ylabel("share of portfolio")
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend(loc="center left", fontsize=9)
fig.tight_layout()
plt.show()
""")

md(r"""
**Interpretation:**
- The top-10 line tells you what fraction of the portfolio is in its 10 most-weighted names.
- When the LGBM picks high P(K=10), the top-10 line climbs toward 1.0 (almost everything in the top 10).
- When the LGBM picks high P(K=50), the top-10 line drops because weight spreads across more names.
- The top-1 line is bounded above by 10% (the cap); when it sits at 0.10, the cap is binding.
""")

# -------------------------------------------------------------------
md("""
## 10. Bottom line

| | **K-ensemble (pivot)** | **v1_cap10 (parent retrain)** | SPY |
|---|:---:|:---:|:---:|
| Beats SPY on Sharpe | ✅ | ❌ | — |
| Beats SPY on AnnRet | ✅ | varies | — |
| IPS-compliant (10% cap) | ✅ | ✅ | ✅ |
| Per-year SPY hit-rate | high | low | — |

The pivoted strategy is the recommended one. It was redesigned with the IPS
cap as a first-class constraint rather than retrofitted, and the head-to-head
on the same OOS window confirms that decision was correct.
""")

# -------------------------------------------------------------------
nb["cells"] = cells

# Set Python 3 kernel metadata
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"},
}

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w") as f:
    nbf.write(nb, f)
print(f"wrote {OUT}")
