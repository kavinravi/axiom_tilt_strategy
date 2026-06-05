"""Full diagnostic on the new winner: sp × fcfa.

Same battery as v6_spy_overlay_diagnostics, but for S/P × FCFA.

Includes:
- Per-year breakdown
- Correlation with SPY
- Mix-weight sweep (is 50/50 still optimal?)
- Drawdown decomposition
- Pre-OOS robustness (2002-2009)
- Bootstrap 95% CI for Sharpe
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.io import processed_dir, repo_root
from src.utils.ranker import friday_only
from src.utils.rl_env import project_to_simplex

REPO_ROOT = repo_root()
PANEL_DIR = processed_dir() / "panel"
TRAIN_PANEL_DIR = processed_dir() / "training_panel"
SPY_PATH = REPO_ROOT / "artifacts" / "benchmarks" / "spy_daily.parquet"
TOP_K = 30
MAX_WEIGHT = 0.10
EPS = 1e-8


def load_panel(years, cols):
    frames = []
    for y in years:
        for p in sorted((PANEL_DIR / f"year={y}").glob("*.parquet")):
            df = pd.read_parquet(p, columns=cols)
            df["date"] = pd.to_datetime(df["date"])
            df["permno"] = df["permno"].astype("int64")
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_train_panel(years):
    frames = []
    for y in years:
        for p in sorted((TRAIN_PANEL_DIR / f"year={y}").glob("*.parquet")):
            df = pd.read_parquet(p)[["permno", "date", "fwd_ret_5d"]]
            df["date"] = pd.to_datetime(df["date"])
            df["permno"] = df["permno"].astype("int64")
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


print("Loading panel 2001-2025 ...")
daily = load_panel(range(2001, 2026), cols=[
    "permno", "date", "prc", "shrout", "marketcap", "in_universe",
    "revenue", "fcf", "assets"])
fri_panel = load_train_panel(range(2002, 2026))
df = daily.merge(fri_panel, on=["permno", "date"], how="inner")
df = df.dropna(subset=["fwd_ret_5d"]).copy()
df = friday_only(df).reset_index(drop=True)
df = df[df["in_universe"]].copy()
df["mcap"] = df["marketcap"]
df.loc[df["mcap"].isna(), "mcap"] = (np.abs(df.loc[df["mcap"].isna(), "prc"]) *
                                      df.loc[df["mcap"].isna(), "shrout"])

# S/P (sales-to-price)
df["sp"] = (df["revenue"] / df["mcap"]).clip(lower=0)
# FCF/Assets
df["fcfa"] = (df["fcf"] / df["assets"]).clip(lower=-1.0, upper=2.0)
df.loc[df["assets"] <= 0, "fcfa"] = np.nan

# z-scores
for col in ["sp", "fcfa"]:
    g = df.groupby("date", sort=False)[col]
    df[f"z_{col}"] = (df[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
    df[f"z_{col}"] = df[f"z_{col}"].fillna(0.0)
df["score"] = 0.5 * df["z_sp"] + 0.5 * df["z_fcfa"]

# Top-30 per Friday
sb = (df.sort_values(["date", "score"], ascending=[True, False])
        .groupby("date", sort=False, group_keys=False)
        .head(TOP_K)
        .reset_index(drop=True))
print(f"sp×fcfa scoreboard: {len(sb)} rows over {sb['date'].nunique()} Fridays")

# Per-Friday backtest
weekly = []
for d, g in sb.groupby("date"):
    g = g.reset_index(drop=True)
    mcaps = g["mcap"].to_numpy(dtype=np.float64)
    mcaps = np.where(np.isnan(mcaps), 0.0, mcaps)
    if mcaps.sum() <= 0:
        n = len(g); w = np.full(n, 1.0 / n)
    else:
        w = project_to_simplex(np.log(np.maximum(mcaps, EPS)), max_weight=MAX_WEIGHT)
    fwd = g["fwd_ret_5d"].to_numpy(dtype=np.float64)
    fwd = np.where(np.isnan(fwd), 0.0, fwd)
    weekly.append({"date": d, "weekly_ret": float(np.dot(w, fwd))})
w_df = pd.DataFrame(weekly).sort_values("date").reset_index(drop=True)
dates = pd.DatetimeIndex(w_df["date"])

# SPY weekly
spy = pd.read_parquet(SPY_PATH).reset_index()[["Date", "close"]].rename(columns={"Date": "date"})
spy["date"] = pd.to_datetime(spy["date"])
spy = spy.sort_values("date").set_index("date")["close"]
closes = spy.reindex(spy.index.union(dates)).sort_index().ffill().reindex(dates)
spy_rets = closes.pct_change().fillna(0.0).to_numpy()
sp_fcfa_rets = w_df["weekly_ret"].to_numpy()
years = dates.year


def metrics(rets, mask=None):
    r = np.asarray(rets, dtype=float)
    if mask is not None: r = r[mask]
    if len(r) < 2: return {}
    cum = float(np.prod(1.0 + r) - 1.0)
    ann = (1.0 + cum) ** (52.0 / len(r)) - 1.0
    vol = float(np.std(r, ddof=1) * np.sqrt(52.0))
    sh = ann / vol if vol > 0 else 0.0
    eq = np.cumprod(1.0 + r); peak = np.maximum.accumulate(eq)
    mdd = float((eq / peak - 1.0).min())
    return {"ann": ann, "vol": vol, "sh": sh, "mdd": mdd}


# === Per-window breakdown (incl pre-OOS) ===
print(f"\n=== Per-window: sp_fcfa STRATEGY vs SPY ===")
for label, mask in [("2002-2025 (full)", np.ones(len(dates), dtype=bool)),
                    ("2002-2009 (pre-OOS)", (years >= 2002) & (years <= 2009)),
                    ("2010-2024", (years >= 2010) & (years <= 2024)),
                    ("2010-2025 (BAR)", years >= 2010)]:
    sf = metrics(sp_fcfa_rets, mask); sm = metrics(spy_rets, mask)
    blend = 0.5 * sp_fcfa_rets[mask] + 0.5 * spy_rets[mask]
    bm = metrics(blend)
    print(f"\n--- {label} ({mask.sum()} weeks) ---")
    print(f"  sp_fcfa alone : ann={sf['ann']:8.2%}  vol={sf['vol']:8.2%}  sharpe={sf['sh']:8.3f}  mdd={sf['mdd']:8.2%}")
    print(f"  SPY           : ann={sm['ann']:8.2%}  vol={sm['vol']:8.2%}  sharpe={sm['sh']:8.3f}  mdd={sm['mdd']:8.2%}")
    print(f"  BLEND 50/50   : ann={bm['ann']:8.2%}  vol={bm['vol']:8.2%}  sharpe={bm['sh']:8.3f}  mdd={bm['mdd']:8.2%}")


# === Mix-weight sweep (2010-2025) ===
print(f"\n=== Mix-weight sweep 2010-2025: find optimal sp_fcfa/SPY blend ===")
mask = years >= 2010
sf_oos = sp_fcfa_rets[mask]; spy_oos = spy_rets[mask]
print(f"  {'w_sp_fcfa':>10} {'ann':>8} {'vol':>8} {'sharpe':>8} {'mdd':>8}")
for w in np.arange(0.0, 1.01, 0.05):
    r = w * sf_oos + (1 - w) * spy_oos
    m = metrics(r)
    print(f"  {w:>10.2f} {m['ann']:>8.2%} {m['vol']:>8.2%} {m['sh']:>8.3f} {m['mdd']:>8.2%}")


# === Per-year breakdown ===
print(f"\n=== Per-year breakdown (2010-2025) ===")
print(f"  {'year':>6} {'wks':>4} {'sf_ret':>9} {'spy_ret':>9} {'blend_ret':>10} {'blend_sh':>9} {'spy_sh':>8}")
for year, sub in sorted(pd.DataFrame({"date": dates, "sf": sp_fcfa_rets, "spy": spy_rets, "year": years})
                        [years >= 2010].groupby("year")):
    sf_y = sub["sf"].to_numpy(); spy_y = sub["spy"].to_numpy()
    bl_y = 0.5 * sf_y + 0.5 * spy_y
    print(f"  {year:>6} {len(sub):>4} {float((1+sf_y).prod()-1):>9.2%} {float((1+spy_y).prod()-1):>9.2%} "
          f"{float((1+bl_y).prod()-1):>10.2%} {metrics(bl_y)['sh']:>9.3f} {metrics(spy_y)['sh']:>8.3f}")


# === Bootstrap 95% CI ===
print(f"\n=== Bootstrap 95% CI Sharpe (2010-2025, 1000 resamples) ===")
rng = np.random.default_rng(42)
blend_oos = 0.5 * sf_oos + 0.5 * spy_oos
sharpe_b = []; sharpe_s = []
for _ in range(1000):
    idx = rng.integers(0, len(blend_oos), size=len(blend_oos))
    sharpe_b.append(metrics(blend_oos[idx])["sh"])
    sharpe_s.append(metrics(spy_oos[idx])["sh"])
sharpe_b = np.array(sharpe_b); sharpe_s = np.array(sharpe_s)
print(f"  blend Sharpe: mean={sharpe_b.mean():.3f}  95% CI=[{np.percentile(sharpe_b, 2.5):.3f}, {np.percentile(sharpe_b, 97.5):.3f}]")
print(f"  SPY Sharpe  : mean={sharpe_s.mean():.3f}  95% CI=[{np.percentile(sharpe_s, 2.5):.3f}, {np.percentile(sharpe_s, 97.5):.3f}]")
print(f"  P(blend > SPY): {float(np.mean(sharpe_b > sharpe_s)):.3f}")

# Save weekly returns
out_dir = REPO_ROOT / "artifacts" / "backtest_factor_v1"
out_dir.mkdir(parents=True, exist_ok=True)
w_df.to_parquet(out_dir / "weekly_sp_fcfa.parquet", compression="zstd", index=False)
blend_df = pd.DataFrame({"date": dates, "weekly_ret": 0.5 * sp_fcfa_rets + 0.5 * spy_rets})
blend_df.to_parquet(out_dir / "weekly_sp_fcfa_spy_overlay_50_50.parquet", compression="zstd", index=False)
print(f"\nSaved: weekly_sp_fcfa.parquet, weekly_sp_fcfa_spy_overlay_50_50.parquet")
