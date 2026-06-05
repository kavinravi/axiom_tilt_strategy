# Dashboard Publisher (Plan 1 of 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the read-only **publisher** that snapshots the live IBKR portfolio + audit files into a Supabase (Postgres) datastore, so a remote frontend (Plan 2) can render it.

**Architecture:** A new `trading/publish/` package. Pure metric functions (`metrics.py`) compute holdings/returns/risk/turnover/execution-quality from plain dicts; a thin `store.py` writes them to Supabase via an injectable client; `publish.py` orchestrates (broker → audit files → metrics → store) behind a CLI a systemd timer runs every ~20 min during market hours; `backfill.py` seeds the weekly-portfolio + execution history from existing audit files. The live-money VPS only ever makes **outbound** HTTPS calls to Supabase — nothing connects in.

**Tech Stack:** Python 3.11+, `supabase` (Python client), the existing `IBKRBroker`/`DryRunBroker`, `pytest`. SQL schema applied once in the Supabase console.

**Spec:** `docs/superpowers/specs/2026-06-03-remote-dashboard-design.md` (this plan implements §4 Publisher, §5 Datastore, §8 Backfill, §9 Testing-backend).

---

### Task 1: Dependencies, config, and the Supabase schema

**Files:**
- Modify: `requirements.txt`
- Modify: `trading/config.py:55` (append publisher config block)
- Create: `trading/publish/__init__.py`
- Create: `trading/publish/schema.sql`

- [ ] **Step 1: Add the Supabase client dependency**

Append to `requirements.txt` (after the `duckdb` line):

```
# supabase — Postgres datastore client for the read-only dashboard publisher (Plan: dashboard)
supabase>=2.0
```

- [ ] **Step 2: Install it**

Run: `pip install "supabase>=2.0"`
Expected: installs `supabase` and its deps (`postgrest`, `gotrue`, …) without error.

- [ ] **Step 3: Add publisher config to `trading/config.py`**

Append at the end of `trading/config.py`:

```python
# ---------------------------------------------------------------------------
# Dashboard publisher (read-only push to Supabase). Outbound-only; the VPS opens
# no inbound ports. SUPABASE_SERVICE_KEY (write) lives ONLY in the VPS .env.
# ---------------------------------------------------------------------------
SUPABASE_URL = get_env("SUPABASE_URL", default="")
SUPABASE_SERVICE_KEY = get_env("SUPABASE_SERVICE_KEY", default="")

# Publish only during US market hours (the systemd timer also gates, this is a guard).
PUBLISH_MARKET_OPEN = "09:30"   # America/New_York
PUBLISH_MARKET_CLOSE = "16:00"  # America/New_York
```

- [ ] **Step 4: Create the package marker**

Create `trading/publish/__init__.py`:

```python
"""Read-only dashboard publisher: snapshot live portfolio + audit files to Supabase."""
```

- [ ] **Step 5: Create the Supabase schema**

Create `trading/publish/schema.sql` (run once in the Supabase SQL editor):

```sql
-- Dashboard datastore schema. Apply once in the Supabase SQL editor.
-- All tables are written by the VPS publisher (service-role key) and read by the
-- Vercel frontend (read-only key). Data volume is tiny; no indexes beyond PKs needed.

create table if not exists snapshot (
  id              int primary key default 1,
  asof            timestamptz not null,
  nav             double precision not null,
  day_pnl         double precision,
  day_pnl_pct     double precision,
  total_return    double precision,
  spy_return      double precision,
  n_positions     int,
  invested_pct    double precision,
  k_probs         jsonb,
  regime_features jsonb,
  risk            jsonb
);

create table if not exists equity_curve (
  date       date primary key,
  nav        double precision not null,
  spy_close  double precision
);

create table if not exists holdings (
  asof          timestamptz not null,
  ticker        text not null,
  shares        double precision not null,
  price         double precision,
  market_value  double precision,
  weight_actual double precision,
  weight_target double precision
);

create table if not exists weekly_portfolio (
  asof_friday   date not null,
  ticker        text not null,
  target_weight double precision not null,
  k_probs       jsonb,
  primary key (asof_friday, ticker)
);

create table if not exists executions (
  asof           date not null,
  ticker         text not null,
  side           text,
  qty            double precision,
  realized_price double precision,
  midpoint       double precision,
  slippage_bps   double precision
);
```

- [ ] **Step 6: Commit**

```bash
git add requirements.txt trading/config.py trading/publish/__init__.py trading/publish/schema.sql
git commit -m "feat(publish): add supabase dep, publisher config, datastore schema"
```

---

### Task 2: `metrics.py` — holdings table

**Files:**
- Create: `trading/publish/metrics.py`
- Test: `tests/trading/test_publish_metrics.py`

- [ ] **Step 1: Write the failing test**

Create `tests/trading/test_publish_metrics.py`:

```python
"""Tests for trading/publish/metrics.py — pure functions, no network."""
from __future__ import annotations

import math

from trading.publish.metrics import compute_holdings


def test_compute_holdings_sorts_by_actual_weight_and_skips_zero_shares():
    positions = {"AAA": 100.0, "BBB": 50.0, "CCC": 0.0}
    prices = {"AAA": 10.0, "BBB": 40.0, "CCC": 99.0}
    target = {"AAA": 0.30, "BBB": 0.70}
    nav = 3000.0

    rows = compute_holdings(positions, prices, target, nav)

    # CCC dropped (0 shares); BBB (mv 2000) before AAA (mv 1000)
    assert [r["ticker"] for r in rows] == ["BBB", "AAA"]
    assert math.isclose(rows[0]["market_value"], 2000.0)
    assert math.isclose(rows[0]["weight_actual"], 2000.0 / 3000.0)
    assert math.isclose(rows[0]["weight_target"], 0.70)
    assert math.isclose(rows[1]["weight_actual"], 1000.0 / 3000.0)


def test_compute_holdings_missing_price_is_zero_value():
    rows = compute_holdings({"AAA": 10.0}, {}, {}, nav=1000.0)
    assert rows[0]["price"] == 0.0
    assert rows[0]["market_value"] == 0.0
    assert rows[0]["weight_target"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/trading/test_publish_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading.publish.metrics'`

- [ ] **Step 3: Write minimal implementation**

Create `trading/publish/metrics.py`:

```python
"""Pure metric functions for the dashboard publisher.

No network, no Supabase, no broker — every function takes plain dicts/lists and
returns plain dicts/lists, so they are trivially unit-testable.
"""
from __future__ import annotations

import math


def compute_holdings(
    positions: dict[str, float],
    prices: dict[str, float],
    target_weights: dict[str, float],
    nav: float,
) -> list[dict]:
    """Build the per-holding table for currently-held names, sorted by actual weight.

    Skips zero-share positions. weight_actual = shares*price / nav.
    """
    rows: list[dict] = []
    for ticker, shares in positions.items():
        shares = float(shares)
        if shares == 0.0:
            continue
        price = float(prices.get(ticker, 0.0))
        market_value = shares * price
        rows.append(
            {
                "ticker": ticker,
                "shares": shares,
                "price": price,
                "market_value": market_value,
                "weight_actual": (market_value / nav) if nav > 0 else 0.0,
                "weight_target": float(target_weights.get(ticker, 0.0)),
            }
        )
    rows.sort(key=lambda r: r["weight_actual"], reverse=True)
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/trading/test_publish_metrics.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add trading/publish/metrics.py tests/trading/test_publish_metrics.py
git commit -m "feat(publish): compute_holdings (actual vs target weights)"
```

---

### Task 3: `metrics.py` — day P&L and returns

**Files:**
- Modify: `trading/publish/metrics.py`
- Test: `tests/trading/test_publish_metrics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/trading/test_publish_metrics.py`:

```python
from trading.publish.metrics import compute_day_pnl, pct_change


def test_compute_day_pnl_normal():
    pnl, pct = compute_day_pnl(nav=101_000.0, prev_nav=100_000.0)
    assert math.isclose(pnl, 1000.0)
    assert math.isclose(pct, 0.01)


def test_compute_day_pnl_no_prior_returns_none():
    assert compute_day_pnl(100_000.0, None) == (None, None)
    assert compute_day_pnl(100_000.0, 0.0) == (None, None)


def test_pct_change():
    assert math.isclose(pct_change(110.0, 100.0), 0.10)
    assert pct_change(110.0, None) is None
    assert pct_change(110.0, 0.0) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/trading/test_publish_metrics.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_day_pnl'`

- [ ] **Step 3: Write minimal implementation**

Append to `trading/publish/metrics.py`:

```python
def pct_change(now: float | None, base: float | None) -> float | None:
    """Return now/base - 1, or None if base is missing/non-positive."""
    if now is None or base is None or base <= 0:
        return None
    return now / base - 1.0


def compute_day_pnl(nav: float, prev_nav: float | None) -> tuple[float | None, float | None]:
    """Portfolio-level P&L vs the prior NAV point. (None, None) when no prior."""
    if prev_nav is None or prev_nav <= 0:
        return None, None
    pnl = nav - prev_nav
    return pnl, pnl / prev_nav
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/trading/test_publish_metrics.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add trading/publish/metrics.py tests/trading/test_publish_metrics.py
git commit -m "feat(publish): compute_day_pnl + pct_change helpers"
```

---

### Task 4: `metrics.py` — risk stats from the equity curve

**Files:**
- Modify: `trading/publish/metrics.py`
- Test: `tests/trading/test_publish_metrics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/trading/test_publish_metrics.py`:

```python
from trading.publish.metrics import compute_risk


def test_compute_risk_too_short_is_all_none():
    out = compute_risk([100.0])
    assert out == {"current_drawdown": None, "max_drawdown": None,
                   "sharpe": None, "ann_vol": None}


def test_compute_risk_drawdown():
    # peak 120 then down to 90 → max dd = 90/120 - 1 = -0.25; current dd vs all-time peak 120
    navs = [100.0, 120.0, 90.0, 108.0]
    out = compute_risk(navs)
    assert math.isclose(out["max_drawdown"], 90.0 / 120.0 - 1.0)
    assert math.isclose(out["current_drawdown"], 108.0 / 120.0 - 1.0)
    assert out["ann_vol"] is not None and out["ann_vol"] > 0
    assert out["sharpe"] is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/trading/test_publish_metrics.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_risk'`

- [ ] **Step 3: Write minimal implementation**

Append to `trading/publish/metrics.py`:

```python
_TRADING_DAYS = 252


def compute_risk(navs: list[float]) -> dict:
    """Drawdown / Sharpe / annualized vol from a chronological daily NAV series."""
    out: dict = {"current_drawdown": None, "max_drawdown": None,
                 "sharpe": None, "ann_vol": None}
    if len(navs) < 2:
        return out

    # Drawdowns
    peak = navs[0]
    max_dd = 0.0
    for v in navs:
        peak = max(peak, v)
        max_dd = min(max_dd, v / peak - 1.0)
    out["max_drawdown"] = max_dd
    out["current_drawdown"] = navs[-1] / max(navs) - 1.0

    # Daily simple returns
    rets = [navs[i] / navs[i - 1] - 1.0 for i in range(1, len(navs))]
    mean = sum(rets) / len(rets)
    if len(rets) > 1:
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    else:
        var = 0.0
    sd = math.sqrt(var)
    out["ann_vol"] = sd * math.sqrt(_TRADING_DAYS)
    if sd > 0:
        out["sharpe"] = (mean / sd) * math.sqrt(_TRADING_DAYS)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/trading/test_publish_metrics.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add trading/publish/metrics.py tests/trading/test_publish_metrics.py
git commit -m "feat(publish): compute_risk (drawdown/sharpe/vol)"
```

---

### Task 5: `metrics.py` — turnover and execution quality

**Files:**
- Modify: `trading/publish/metrics.py`
- Test: `tests/trading/test_publish_metrics.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/trading/test_publish_metrics.py`:

```python
from trading.publish.metrics import compute_turnover, compute_execution_quality


def test_compute_turnover():
    last = {"AAA": 0.5, "BBB": 0.5}
    this = {"AAA": 0.4, "CCC": 0.6}
    out = compute_turnover(this, last)
    assert out["added"] == ["CCC"]
    assert out["dropped"] == ["BBB"]
    # 0.5*(|0.4-0.5| + |0-0.5| + |0.6-0|) = 0.5*(0.1+0.5+0.6) = 0.6
    assert math.isclose(out["turnover_frac"], 0.6)


def test_compute_execution_quality_buy_above_mid_is_positive_slippage():
    audit = {
        "fills": [{"ticker": "AAA", "side": "BUY"}],
        "ladder_stages": [
            {"ticker": "AAA", "qty_filled": 100.0, "realized_price": 101.0,
             "midpoint_at_fill": 100.0},
            {"ticker": "AAA", "qty_filled": 0.0, "realized_price": None,
             "midpoint_at_fill": None},
        ],
    }
    rows = compute_execution_quality(audit)
    assert len(rows) == 1
    r = rows[0]
    assert r["ticker"] == "AAA" and r["side"] == "BUY"
    assert math.isclose(r["realized_price"], 101.0)
    assert math.isclose(r["midpoint"], 100.0)
    # BUY paid 1.0 above a 100.0 mid → +100 bps cost
    assert math.isclose(r["slippage_bps"], 100.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/trading/test_publish_metrics.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_turnover'`

- [ ] **Step 3: Write minimal implementation**

Append to `trading/publish/metrics.py`:

```python
def compute_turnover(this_weights: dict[str, float], last_weights: dict[str, float]) -> dict:
    """Names added/dropped and one-way turnover fraction (0.5 * sum|Δw|)."""
    this_set, last_set = set(this_weights), set(last_weights)
    tickers = this_set | last_set
    turnover = 0.5 * sum(
        abs(this_weights.get(t, 0.0) - last_weights.get(t, 0.0)) for t in tickers
    )
    return {
        "added": sorted(this_set - last_set),
        "dropped": sorted(last_set - this_set),
        "turnover_frac": turnover,
    }


def compute_execution_quality(orders_audit: dict) -> list[dict]:
    """Per-ticker realized fill vs NBBO midpoint, from a rebalance orders-audit dict.

    slippage_bps is signed so that POSITIVE = worse than midpoint (a cost): for a
    BUY, paying above the mid is positive; for a SELL, selling below the mid is positive.
    """
    side_by_ticker = {f["ticker"]: f.get("side", "BUY") for f in orders_audit.get("fills", [])}
    agg: dict[str, dict] = {}
    for s in orders_audit.get("ladder_stages", []):
        qty = float(s.get("qty_filled") or 0.0)
        rp = s.get("realized_price")
        mp = s.get("midpoint_at_fill")
        if qty <= 0.0 or rp is None or mp is None:
            continue
        d = agg.setdefault(s["ticker"], {"qty": 0.0, "rp_q": 0.0, "mp_q": 0.0})
        d["qty"] += qty
        d["rp_q"] += float(rp) * qty
        d["mp_q"] += float(mp) * qty

    rows: list[dict] = []
    for ticker, d in agg.items():
        realized = d["rp_q"] / d["qty"]
        mid = d["mp_q"] / d["qty"]
        side = side_by_ticker.get(ticker, "BUY")
        raw = (realized - mid) / mid if mid else 0.0
        slippage_bps = (raw if side == "BUY" else -raw) * 1e4
        rows.append(
            {
                "ticker": ticker,
                "side": side,
                "qty": d["qty"],
                "realized_price": realized,
                "midpoint": mid,
                "slippage_bps": slippage_bps,
            }
        )
    rows.sort(key=lambda r: r["ticker"])
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/trading/test_publish_metrics.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add trading/publish/metrics.py tests/trading/test_publish_metrics.py
git commit -m "feat(publish): compute_turnover + compute_execution_quality"
```

---

### Task 6: `store.py` — the Supabase writer (injectable client)

**Files:**
- Create: `trading/publish/store.py`
- Test: `tests/trading/test_publish_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/trading/test_publish_store.py`:

```python
"""Tests for trading/publish/store.py — uses a fake client; no network."""
from __future__ import annotations

from trading.publish.store import SupabaseStore


class _FakeQuery:
    """Records the chained call it represents and returns canned data on execute()."""

    def __init__(self, log, table, data=None):
        self._log = log
        self._table = table
        self._data = data or []

    def upsert(self, row, on_conflict=None):
        self._log.append(("upsert", self._table, row, on_conflict))
        return self

    def insert(self, rows):
        self._log.append(("insert", self._table, rows))
        return self

    def delete(self):
        self._log.append(("delete", self._table))
        return self

    def select(self, cols):
        self._log.append(("select", self._table, cols))
        return self

    def order(self, col):
        return self

    def eq(self, col, val):
        self._log.append(("eq", self._table, col, val))
        return self

    def neq(self, col, val):
        return self

    def execute(self):
        return type("Res", (), {"data": self._data})()


class _FakeClient:
    def __init__(self, equity_rows=None):
        self.log = []
        self._equity_rows = equity_rows or []

    def table(self, name):
        data = self._equity_rows if name == "equity_curve" else []
        return _FakeQuery(self.log, name, data)


def test_upsert_snapshot_sets_singleton_id_and_conflict():
    c = _FakeClient()
    SupabaseStore(c).upsert_snapshot({"nav": 100.0})
    assert ("upsert", "snapshot", {"nav": 100.0, "id": 1}, "id") in c.log


def test_upsert_equity_point_conflicts_on_date():
    c = _FakeClient()
    SupabaseStore(c).upsert_equity_point("2026-06-03", 100.0, 5000.0)
    assert ("upsert", "equity_curve",
            {"date": "2026-06-03", "nav": 100.0, "spy_close": 5000.0}, "date") in c.log


def test_replace_holdings_deletes_then_inserts():
    c = _FakeClient()
    SupabaseStore(c).replace_holdings([{"ticker": "AAA"}])
    kinds = [e[0] for e in c.log]
    assert kinds == ["delete", "insert"]


def test_read_equity_curve_returns_client_data():
    rows = [{"date": "2026-06-01", "nav": 100.0, "spy_close": 5000.0}]
    store = SupabaseStore(_FakeClient(equity_rows=rows))
    assert store.read_equity_curve() == rows
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/trading/test_publish_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading.publish.store'`

- [ ] **Step 3: Write minimal implementation**

Create `trading/publish/store.py`:

```python
"""Supabase writer for the dashboard publisher.

SupabaseStore wraps an injectable client (the real one in production, a fake in
tests). Delete-before-insert on the per-period tables makes re-runs idempotent.
"""
from __future__ import annotations

from typing import Any


class SupabaseStore:
    def __init__(self, client: Any) -> None:
        self._c = client

    def upsert_snapshot(self, row: dict) -> None:
        self._c.table("snapshot").upsert({**row, "id": 1}, on_conflict="id").execute()

    def upsert_equity_point(self, date: str, nav: float, spy_close: float | None) -> None:
        self._c.table("equity_curve").upsert(
            {"date": date, "nav": nav, "spy_close": spy_close}, on_conflict="date"
        ).execute()

    def replace_holdings(self, rows: list[dict]) -> None:
        self._c.table("holdings").delete().neq("ticker", "").execute()
        if rows:
            self._c.table("holdings").insert(rows).execute()

    def insert_weekly_portfolio(self, asof_friday: str, rows: list[dict]) -> None:
        self._c.table("weekly_portfolio").delete().eq("asof_friday", asof_friday).execute()
        if rows:
            self._c.table("weekly_portfolio").insert(rows).execute()

    def insert_executions(self, asof: str, rows: list[dict]) -> None:
        self._c.table("executions").delete().eq("asof", asof).execute()
        if rows:
            self._c.table("executions").insert(rows).execute()

    def read_equity_curve(self) -> list[dict]:
        res = self._c.table("equity_curve").select("*").order("date").execute()
        return res.data or []


def make_client(url: str, key: str):
    """Build a real Supabase client (imported lazily so tests need no network deps)."""
    from supabase import create_client  # noqa: PLC0415

    return create_client(url, key)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/trading/test_publish_store.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add trading/publish/store.py tests/trading/test_publish_store.py
git commit -m "feat(publish): SupabaseStore (idempotent writes, injectable client)"
```

---

### Task 7: `publish.py` — the orchestrator (`publish_once`)

**Files:**
- Create: `trading/publish/publish.py`
- Test: `tests/trading/test_publish_orchestrator.py`

- [ ] **Step 1: Write the failing test**

Create `tests/trading/test_publish_orchestrator.py`:

```python
"""End-to-end publisher test against DryRunBroker + fixture audit files + fake store."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from trading.broker.dryrun import DryRunBroker
from trading.publish.publish import publish_once


class _RecordingStore:
    """Captures every write; serves a canned equity curve for read."""

    def __init__(self, equity=None):
        self.snapshot = None
        self.equity_point = None
        self.holdings = None
        self.weekly = None
        self.executions = None
        self._equity = equity or []

    def read_equity_curve(self):
        return list(self._equity)

    def upsert_equity_point(self, date, nav, spy_close):
        self.equity_point = {"date": date, "nav": nav, "spy_close": spy_close}

    def upsert_snapshot(self, row):
        self.snapshot = row

    def replace_holdings(self, rows):
        self.holdings = rows

    def insert_weekly_portfolio(self, asof_friday, rows):
        self.weekly = {"asof_friday": asof_friday, "rows": rows}

    def insert_executions(self, asof, rows):
        self.executions = {"asof": asof, "rows": rows}


def _write_weights(weights_dir: Path, asof: str):
    weights_dir.mkdir(parents=True, exist_ok=True)
    payload = {"asof": asof, "k_probs": {"10": 0.6, "20": 0.4},
               "weights": {"AAA": 0.6, "BBB": 0.4}}
    (weights_dir / f"{asof}.json").write_text(json.dumps(payload))


def test_publish_once_writes_all_products(tmp_path):
    asof = "2026-05-29"
    today = pd.Timestamp("2026-06-01")
    _write_weights(tmp_path / "weights", asof)

    broker = DryRunBroker(
        positions={"AAA": 60.0, "BBB": 40.0},
        nav=10_000.0,
        quotes={"AAA": (99.5, 100.5), "BBB": (99.5, 100.5)},
    )
    # Prior equity point so day P&L + inception return are computed.
    store = _RecordingStore(equity=[{"date": "2026-05-29", "nav": 9_000.0, "spy_close": 5000.0}])

    summary = publish_once(
        broker, store,
        weights_dir=tmp_path / "weights",
        orders_dir=tmp_path / "orders",   # no orders file → executions skipped
        asof=asof, today=today, spy_close=5100.0,
    )

    assert summary["asof"] == asof
    assert store.equity_point == {"date": "2026-06-01", "nav": 10_000.0, "spy_close": 5100.0}
    assert store.snapshot["nav"] == 10_000.0
    assert store.snapshot["day_pnl"] == 1000.0          # 10000 - 9000
    assert store.snapshot["k_probs"] == {"10": 0.6, "20": 0.4}
    assert {h["ticker"] for h in store.holdings} == {"AAA", "BBB"}
    assert all(h["asof"] == "2026-06-01" for h in store.holdings)
    assert store.weekly["asof_friday"] == asof
    assert len(store.weekly["rows"]) == 2
    assert store.executions is None                      # no orders file


def test_publish_once_includes_executions_when_orders_file_present(tmp_path):
    asof = "2026-05-29"
    _write_weights(tmp_path / "weights", asof)
    orders_dir = tmp_path / "orders"
    orders_dir.mkdir(parents=True)
    (orders_dir / f"{asof}.json").write_text(json.dumps({
        "fills": [{"ticker": "AAA", "side": "BUY"}],
        "ladder_stages": [{"ticker": "AAA", "qty_filled": 10.0,
                           "realized_price": 100.5, "midpoint_at_fill": 100.0}],
    }))
    broker = DryRunBroker(positions={"AAA": 60.0}, nav=10_000.0,
                          quotes={"AAA": (99.5, 100.5)})
    store = _RecordingStore()

    publish_once(broker, store, weights_dir=tmp_path / "weights",
                 orders_dir=orders_dir, asof=asof,
                 today=pd.Timestamp("2026-06-01"), spy_close=5100.0)

    assert store.executions["asof"] == asof
    assert store.executions["rows"][0]["ticker"] == "AAA"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/trading/test_publish_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading.publish.publish'`

- [ ] **Step 3: Write minimal implementation**

Create `trading/publish/publish.py`:

```python
"""Orchestrate one publish: broker + audit files -> metrics -> Supabase.

publish_once is fully injectable (broker, store, dirs, dates, spy_close) so tests
run against DryRunBroker + a fake store with no network. main() wires the real
IBKRBroker + SupabaseStore for the systemd timer.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from trading.publish.metrics import (
    compute_day_pnl,
    compute_execution_quality,
    compute_holdings,
    compute_risk,
    pct_change,
)

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _prev_nav(curve: list[dict], today: pd.Timestamp) -> float | None:
    """Most recent NAV strictly before `today` (the prior close)."""
    today_str = str(today.date())
    prior = [p for p in curve if str(p["date"]) < today_str]
    return float(prior[-1]["nav"]) if prior else None


def publish_once(broker, store, *, weights_dir, orders_dir, asof, today, spy_close):
    """Compute and write one snapshot. Returns a small summary dict."""
    asof = str(pd.Timestamp(asof).date())
    today = pd.Timestamp(today).normalize()

    # 1. Live account state (connect/disconnect bracket).
    broker.connect()
    try:
        positions = broker.get_positions()
        nav = float(broker.get_nav())
        prices: dict[str, float] = {}
        for ticker in positions:
            try:
                bid, ask = broker.get_quote(ticker)
                prices[ticker] = (bid + ask) / 2.0
            except Exception as exc:  # noqa: BLE001
                logger.warning("publish: no quote for %s: %s", ticker, exc)
    finally:
        broker.disconnect()

    # 2. Frozen weights for this Friday.
    weights_payload = _load_json(Path(weights_dir) / f"{asof}.json")
    target_weights = {str(k): float(v) for k, v in (weights_payload.get("weights") or {}).items()}
    k_probs = weights_payload.get("k_probs") or {}
    regime_features = weights_payload.get("regime_features")  # None until weights pipeline adds it

    # 3. Equity history (for prior NAV, inception baselines, risk series).
    curve = store.read_equity_curve()
    prev_nav = _prev_nav(curve, today)
    inception_nav = float(curve[0]["nav"]) if curve else nav
    inception_spy = next(
        (p["spy_close"] for p in curve if p.get("spy_close") is not None), spy_close
    )
    today_str = str(today.date())
    navs = [float(p["nav"]) for p in curve if str(p["date"]) < today_str] + [nav]

    # 4. Metrics.
    holdings = compute_holdings(positions, prices, target_weights, nav)
    day_pnl, day_pnl_pct = compute_day_pnl(nav, prev_nav)
    risk = compute_risk(navs)
    invested = sum(h["market_value"] for h in holdings)

    # 5. Writes (equity point first so it is present for the next run).
    store.upsert_equity_point(today_str, nav, spy_close)
    store.upsert_snapshot(
        {
            "asof": today_str,
            "nav": nav,
            "day_pnl": day_pnl,
            "day_pnl_pct": day_pnl_pct,
            "total_return": pct_change(nav, inception_nav),
            "spy_return": pct_change(spy_close, inception_spy),
            "n_positions": len(holdings),
            "invested_pct": (invested / nav) if nav > 0 else None,
            "k_probs": k_probs,
            "regime_features": regime_features,
            "risk": risk,
        }
    )
    store.replace_holdings([{**h, "asof": today_str} for h in holdings])
    store.insert_weekly_portfolio(
        asof,
        [
            {"asof_friday": asof, "ticker": t, "target_weight": w, "k_probs": k_probs}
            for t, w in target_weights.items()
        ],
    )

    orders_path = Path(orders_dir) / f"{asof}.json"
    if orders_path.exists():
        exec_rows = compute_execution_quality(_load_json(orders_path))
        store.insert_executions(asof, [{**r, "asof": asof} for r in exec_rows])

    return {"asof": asof, "nav": nav, "n_holdings": len(holdings)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/trading/test_publish_orchestrator.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add trading/publish/publish.py tests/trading/test_publish_orchestrator.py
git commit -m "feat(publish): publish_once orchestrator (broker+audit -> store)"
```

---

### Task 8: `publish.py` — SPY fetch + market-hours guard + CLI `main()`

**Files:**
- Modify: `trading/publish/publish.py`
- Test: `tests/trading/test_publish_orchestrator.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/trading/test_publish_orchestrator.py`:

```python
from trading.publish.publish import is_market_hours


def test_is_market_hours_weekday_midday_true():
    # Monday 2026-06-01 12:00 ET
    assert is_market_hours(pd.Timestamp("2026-06-01 12:00", tz="America/New_York")) is True


def test_is_market_hours_weekend_false():
    # Saturday 2026-05-30 12:00 ET
    assert is_market_hours(pd.Timestamp("2026-05-30 12:00", tz="America/New_York")) is False


def test_is_market_hours_after_close_false():
    assert is_market_hours(pd.Timestamp("2026-06-01 16:30", tz="America/New_York")) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/trading/test_publish_orchestrator.py -v`
Expected: FAIL — `ImportError: cannot import name 'is_market_hours'`

- [ ] **Step 3: Write minimal implementation**

Append to `trading/publish/publish.py`:

```python
def is_market_hours(now=None, open_str: str = "09:30", close_str: str = "16:00") -> bool:
    """True if `now` is a weekday within [open, close] America/New_York."""
    if now is None:
        now = pd.Timestamp.now(tz="America/New_York")
    now = pd.Timestamp(now)
    if now.tz is None:
        now = now.tz_localize("America/New_York")
    else:
        now = now.tz_convert("America/New_York")
    if now.dayofweek >= 5:  # Sat/Sun
        return False
    open_t = pd.Timestamp(f"{now.date()} {open_str}", tz="America/New_York")
    close_t = pd.Timestamp(f"{now.date()} {close_str}", tz="America/New_York")
    return open_t <= now <= close_t


def fetch_spy_close() -> float | None:
    """Last SPY close via yfinance (already a dep). Returns None on failure."""
    try:
        import yfinance as yf  # noqa: PLC0415

        hist = yf.Ticker("SPY").history(period="5d")
        if hist.empty:
            return None
        return float(hist["Close"].iloc[-1])
    except Exception as exc:  # noqa: BLE001
        logger.warning("publish: SPY fetch failed: %s", exc)
        return None


def main() -> int:
    """CLI entrypoint for the systemd timer: `python -m trading.publish`."""
    import trading.config as config  # noqa: PLC0415
    from trading.broker.ibkr import IBKRBroker  # noqa: PLC0415
    from trading.data.snapshot import most_recent_friday  # noqa: PLC0415
    from trading.publish.store import SupabaseStore, make_client  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)

    now_et = pd.Timestamp.now(tz="America/New_York")
    if not is_market_hours(now_et, config.PUBLISH_MARKET_OPEN, config.PUBLISH_MARKET_CLOSE):
        logger.info("publish: outside market hours (%s) — skipping", now_et)
        return 0

    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_KEY:
        logger.error("publish: SUPABASE_URL / SUPABASE_SERVICE_KEY not set — aborting")
        return 1

    broker = IBKRBroker(host=config.IBKR_HOST, port=config.IBKR_PORT,
                        client_id=config.IBKR_CLIENT_ID)
    store = SupabaseStore(make_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY))
    summary = publish_once(
        broker, store,
        weights_dir=config.WEIGHTS_DIR, orders_dir=config.ORDERS_DIR,
        asof=most_recent_friday(), today=now_et.normalize().tz_localize(None),
        spy_close=fetch_spy_close(),
    )
    logger.info("publish: done — %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> **Note on `IBKRBroker(...)` constructor args:** confirm the parameter names against `trading/broker/ibkr.py` when implementing this step. If its signature differs (e.g. it reads host/port from config internally), call it the way that file expects — the rest of `main()` is unaffected.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/trading/test_publish_orchestrator.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add trading/publish/publish.py tests/trading/test_publish_orchestrator.py
git commit -m "feat(publish): market-hours guard, SPY fetch, CLI main()"
```

---

### Task 9: `backfill.py` — seed weekly portfolios + executions from existing audit files

**Files:**
- Create: `trading/publish/backfill.py`
- Test: `tests/trading/test_publish_backfill.py`

- [ ] **Step 1: Write the failing test**

Create `tests/trading/test_publish_backfill.py`:

```python
"""Backfill reads existing audit files into weekly_portfolio + executions."""
from __future__ import annotations

import json
from pathlib import Path

from trading.publish.backfill import backfill


class _RecordingStore:
    def __init__(self):
        self.weekly = []
        self.executions = []

    def insert_weekly_portfolio(self, asof_friday, rows):
        self.weekly.append((asof_friday, rows))

    def insert_executions(self, asof, rows):
        self.executions.append((asof, rows))


def test_backfill_loads_all_weeks_and_orders(tmp_path):
    wdir = tmp_path / "weights"
    odir = tmp_path / "orders"
    wdir.mkdir()
    odir.mkdir()
    for asof in ("2026-05-22", "2026-05-29"):
        (wdir / f"{asof}.json").write_text(json.dumps(
            {"asof": asof, "k_probs": {"10": 1.0}, "weights": {"AAA": 1.0}}))
    (odir / "2026-05-29.json").write_text(json.dumps({
        "fills": [{"ticker": "AAA", "side": "BUY"}],
        "ladder_stages": [{"ticker": "AAA", "qty_filled": 5.0,
                           "realized_price": 100.0, "midpoint_at_fill": 100.0}],
    }))

    store = _RecordingStore()
    n = backfill(store, weights_dir=wdir, orders_dir=odir)

    assert [w[0] for w in store.weekly] == ["2026-05-22", "2026-05-29"]
    assert store.executions[0][0] == "2026-05-29"
    assert store.executions[0][1][0]["asof"] == "2026-05-29"
    assert n == {"weeks": 2, "order_files": 1}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/trading/test_publish_backfill.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading.publish.backfill'`

- [ ] **Step 3: Write minimal implementation**

Create `trading/publish/backfill.py`:

```python
"""One-shot backfill: seed weekly_portfolio + executions from existing audit files.

The daily equity_curve cannot be backfilled (no historical NAV) — it builds forward
from go-live. Run once: `python -m trading.publish.backfill`.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from trading.publish.metrics import compute_execution_quality

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def backfill(store, *, weights_dir, orders_dir) -> dict:
    """Load every weights + orders audit file into the store. Returns counts."""
    weeks = 0
    for wp in sorted(Path(weights_dir).glob("*.json")):
        asof = wp.stem
        payload = _load_json(wp)
        target_weights = {str(k): float(v) for k, v in (payload.get("weights") or {}).items()}
        k_probs = payload.get("k_probs") or {}
        store.insert_weekly_portfolio(
            asof,
            [
                {"asof_friday": asof, "ticker": t, "target_weight": w, "k_probs": k_probs}
                for t, w in target_weights.items()
            ],
        )
        weeks += 1

    order_files = 0
    for op in sorted(Path(orders_dir).glob("*.json")):
        asof = op.stem
        rows = compute_execution_quality(_load_json(op))
        store.insert_executions(asof, [{**r, "asof": asof} for r in rows])
        order_files += 1

    logger.info("backfill: %d weeks, %d order files", weeks, order_files)
    return {"weeks": weeks, "order_files": order_files}


def main() -> int:
    import trading.config as config  # noqa: PLC0415
    from trading.publish.store import SupabaseStore, make_client  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)
    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_KEY:
        logger.error("backfill: SUPABASE_URL / SUPABASE_SERVICE_KEY not set — aborting")
        return 1
    store = SupabaseStore(make_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY))
    backfill(store, weights_dir=config.WEIGHTS_DIR, orders_dir=config.ORDERS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/trading/test_publish_backfill.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add trading/publish/backfill.py tests/trading/test_publish_backfill.py
git commit -m "feat(publish): backfill weekly portfolios + executions from audit files"
```

---

### Task 10: Full suite green + README + ops (systemd timer)

**Files:**
- Create: `trading/publish/README.md`
- Test: (run the whole trading suite)

- [ ] **Step 1: Run the entire trading test suite**

Run: `pytest tests/trading/ -q`
Expected: PASS — all existing trading tests plus the 4 new `test_publish_*` files green. If any pre-existing test fails, it is unrelated to this plan; note it but do not fix here.

- [ ] **Step 2: Lint the new package**

Run: `ruff check trading/publish/`
Expected: no errors. Fix any reported issues (typically import order / unused imports).

- [ ] **Step 3: Write the publisher README**

Create `trading/publish/README.md`:

````markdown
# Dashboard publisher (read-only push → Supabase)

Snapshots the live IBKR portfolio + audit files into Supabase so the Vercel
frontend (Plan 2) can render it. **Outbound-only** — the VPS opens no inbound ports.

## Modules
- `metrics.py` — pure functions (holdings, day P&L, returns, risk, turnover, execution quality).
- `store.py` — `SupabaseStore`: idempotent writes over an injectable client.
- `publish.py` — `publish_once` orchestrator + `main()` CLI (market-hours-guarded).
- `backfill.py` — one-shot seed of weekly portfolios + executions from existing audit files.
- `schema.sql` — apply once in the Supabase SQL editor.

## One-time setup
1. Create a Supabase project; run `schema.sql` in its SQL editor.
2. Put the service-role key in the VPS `.env` (NEVER in the frontend):
   ```
   SUPABASE_URL=https://<project>.supabase.co
   SUPABASE_SERVICE_KEY=<service-role key>
   ```
3. Seed history once: `python -m trading.publish.backfill`

## Run a publish
`python -m trading.publish` — skips quietly outside US market hours; needs IB Gateway up.

## systemd timer (on the VPS)
`/etc/systemd/system/dashboard-publish.service`:
```ini
[Unit]
Description=Dashboard publish (read-only snapshot to Supabase)
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/<user>/axiom_tilt_strategy
EnvironmentFile=/home/<user>/axiom_tilt_strategy/.env
ExecStart=/home/<user>/axiom_tilt_strategy/.venv/bin/python -m trading.publish
```
`/etc/systemd/system/dashboard-publish.timer`:
```ini
[Unit]
Description=Run dashboard publish every 20 min on weekdays

[Timer]
OnCalendar=Mon..Fri *-*-* 09,10,11,12,13,14,15,16:00/20 America/New_York
Persistent=false

[Install]
WantedBy=timers.target
```
Enable: `sudo systemctl enable --now dashboard-publish.timer`
(The `main()` market-hours guard is the backstop; the timer is the primary gate.)
````

- [ ] **Step 4: Commit**

```bash
git add trading/publish/README.md
git commit -m "docs(publish): README + systemd timer for the dashboard publisher"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** §4 Publisher → Tasks 2–8 (metrics + orchestrator + CLI). §5 Datastore → Task 1 (`schema.sql`) + Task 6 (`store.py`). §8 Backfill → Task 9. §9 Testing-backend → Tasks 2–9 (DryRunBroker + fake store + fixtures, idempotency covered by delete-before-insert + equity upsert-on-date). §10 ops (systemd) → Task 10.
- **Deliberate deferrals (documented in spec §4 / plan intro):** per-holding day_change (needs a prior-price store), regime *features* (not persisted yet — `regime_features` read-through is wired but null until a small weights-pipeline follow-up). These are NOT gaps; they are scoped out.
- **Type consistency:** `compute_execution_quality` emits `slippage_bps`; `schema.sql` `executions.slippage_bps` and `store.insert_executions` rows match. `SupabaseStore` method names (`upsert_snapshot`, `upsert_equity_point`, `replace_holdings`, `insert_weekly_portfolio`, `insert_executions`, `read_equity_curve`) are identical across `store.py`, `publish.py`, `backfill.py`, and the recording stores in tests.
- **Out of scope for this plan (Plan 2):** the Next.js/Vercel frontend, auth, and Playwright UI smoke test.
