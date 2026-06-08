# Audit-Trail Dashboard Publisher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Populate the dashboard's holdings, equity curve, and snapshot from the order-audit trail + yfinance prices, with no IBKR Gateway connection in the scheduled publish path.

**Architecture:** A pure `reconstruct.py` module derives current holdings (latest `post_positions`), residual cash (inception NAV − Σ fills×avg_price), and a full daily NAV curve from the audit files + an injected daily-close price frame. A new `publish_from_audit` orchestrator wires those into the existing `metrics.py` helpers and `SupabaseStore`, reusing everything the broker path already used except the broker. The systemd timer drops to once daily after close.

**Tech Stack:** Python 3.12, pandas, yfinance (existing dep), Supabase, pytest.

**Spec:** `docs/superpowers/specs/2026-06-08-audit-trail-publisher-design.md`

---

## File Structure

- **Create** `trading/publish/reconstruct.py` — pure functions: `load_history`, `current_holdings`, `inception_date`, `cash_after`, `reconstruct_curve`. No network/broker/Supabase.
- **Modify** `trading/data/sources.py` — add `fetch_close_history` (batched yfinance daily closes, forward-filled).
- **Modify** `trading/publish/store.py` — add `replace_equity_curve`.
- **Modify** `trading/publish/publish.py` — add `publish_from_audit`; repoint `main()` to it (drop broker + market-hours guard from the scheduled path). `publish_once`/`is_market_hours` stay for tests + future broker truth-up.
- **Modify** `deploy/systemd/axiom-publish.timer` + `axiom-publish.service` — daily after US close.
- **Create** `tests/trading/test_reconstruct.py`.
- **Modify** `tests/trading/test_publish_store.py`, `tests/trading/test_sources.py`.
- **Create** `tests/trading/test_publish_from_audit.py`.

Run tests from repo root with the project venv: `python -m pytest <path> -v`.

---

## Task 1: `store.replace_equity_curve`

**Files:**
- Modify: `trading/publish/store.py`
- Test: `tests/trading/test_publish_store.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/trading/test_publish_store.py`:

```python
def test_replace_equity_curve_deletes_then_inserts():
    calls = []

    class _Tbl:
        def __init__(self, name): self.name = name
        def delete(self): calls.append((self.name, "delete")); return self
        def neq(self, *a): calls.append((self.name, "neq", a)); return self
        def insert(self, rows): calls.append((self.name, "insert", rows)); return self
        def execute(self): return None

    class _Client:
        def table(self, name): return _Tbl(name)

    from trading.publish.store import SupabaseStore
    store = SupabaseStore(_Client())
    rows = [{"date": "2026-06-05", "nav": 100.0, "spy_close": 500.0}]
    store.replace_equity_curve(rows)

    assert ("equity_curve", "delete") in calls
    assert ("equity_curve", "insert", rows) in calls


def test_replace_equity_curve_empty_skips_insert():
    calls = []

    class _Tbl:
        def __init__(self, name): self.name = name
        def delete(self): calls.append((self.name, "delete")); return self
        def neq(self, *a): return self
        def insert(self, rows): calls.append((self.name, "insert")); return self
        def execute(self): return None

    class _Client:
        def table(self, name): return _Tbl(name)

    from trading.publish.store import SupabaseStore
    SupabaseStore(_Client()).replace_equity_curve([])
    assert ("equity_curve", "delete") in calls
    assert ("equity_curve", "insert") not in calls
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/trading/test_publish_store.py::test_replace_equity_curve_deletes_then_inserts -v`
Expected: FAIL — `AttributeError: 'SupabaseStore' object has no attribute 'replace_equity_curve'`.

- [ ] **Step 3: Add the method**

In `trading/publish/store.py`, after `replace_holdings` (mirrors its delete-then-insert shape; `neq("date", "")` matches all rows):

```python
    def replace_equity_curve(self, rows: list[dict]) -> None:
        self._c.table("equity_curve").delete().neq("date", "").execute()
        if rows:
            self._c.table("equity_curve").insert(rows).execute()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/trading/test_publish_store.py -v`
Expected: PASS (all, including the two new tests).

- [ ] **Step 5: Commit**

```bash
git add trading/publish/store.py tests/trading/test_publish_store.py
git commit -m "store: add replace_equity_curve (delete-all + insert)"
```

---

## Task 2: `reconstruct.load_history` + `current_holdings`

**Files:**
- Create: `trading/publish/reconstruct.py`
- Test: `tests/trading/test_reconstruct.py`

- [ ] **Step 1: Write the failing test**

Create `tests/trading/test_reconstruct.py`:

```python
"""Pure reconstruction of holdings/cash/NAV from the order audit."""
from __future__ import annotations

import json

import pandas as pd

from trading.publish import reconstruct


def _write(orders_dir, asof, record):
    orders_dir.mkdir(parents=True, exist_ok=True)
    (orders_dir / f"{asof}.json").write_text(json.dumps({"asof": asof, **record}))


def test_load_history_sorted_by_asof(tmp_path):
    od = tmp_path / "orders"
    _write(od, "2026-06-12", {"post_positions": {"AAA": 1.0}})
    _write(od, "2026-06-05", {"post_positions": {"AAA": 2.0}})
    hist = reconstruct.load_history(od)
    assert [r["asof"] for r in hist] == ["2026-06-05", "2026-06-12"]


def test_current_holdings_uses_latest_post_positions(tmp_path):
    od = tmp_path / "orders"
    _write(od, "2026-06-05", {"post_positions": {"AAA": 10.0, "BBB": 5.0}})
    _write(od, "2026-06-12", {"post_positions": {"AAA": 8.0, "CCC": 3.0}})
    hist = reconstruct.load_history(od)
    assert reconstruct.current_holdings(hist) == {"AAA": 8.0, "CCC": 3.0}


def test_current_holdings_drops_zero_shares(tmp_path):
    od = tmp_path / "orders"
    _write(od, "2026-06-05", {"post_positions": {"AAA": 10.0, "BBB": 0.0}})
    hist = reconstruct.load_history(od)
    assert reconstruct.current_holdings(hist) == {"AAA": 10.0}


def test_current_holdings_empty_history():
    assert reconstruct.current_holdings([]) == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/trading/test_reconstruct.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'trading.publish.reconstruct'`.

- [ ] **Step 3: Create the module with these two functions**

Create `trading/publish/reconstruct.py`:

```python
"""Pure reconstruction of holdings, cash, and NAV history from the order audit.

No network, no Supabase, no broker. Daily-close prices are injected (a DataFrame
indexed by normalized date, columns = tickers) so these functions are trivially
unit-testable. The audit files live in trading/audit/orders/<asof>.json and carry
post_positions (exact post-trade holdings) and fills (with avg_price).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_history(orders_dir) -> list[dict]:
    """All order-audit records under orders_dir, ascending by 'asof'."""
    records = [json.loads(p.read_text()) for p in Path(orders_dir).glob("*.json")]
    records.sort(key=lambda r: str(r.get("asof", "")))
    return records


def current_holdings(history: list[dict]) -> dict[str, float]:
    """Latest post-trade positions, zero-share names dropped."""
    for rec in reversed(history):
        post = rec.get("post_positions")
        if post is not None:
            return {str(k): float(v) for k, v in post.items() if float(v) != 0.0}
    return {}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/trading/test_reconstruct.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add trading/publish/reconstruct.py tests/trading/test_reconstruct.py
git commit -m "reconstruct: load_history + current_holdings from order audit"
```

---

## Task 3: `reconstruct.inception_date` + `cash_after`

**Files:**
- Modify: `trading/publish/reconstruct.py`
- Test: `tests/trading/test_reconstruct.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/trading/test_reconstruct.py`:

```python
def test_cash_after_first_build_residual(tmp_path):
    od = tmp_path / "orders"
    _write(od, "2026-06-05", {
        "first_build": True, "nav": 1000.0,
        "post_positions": {"AAA": 5.0},
        "fills": [{"ticker": "AAA", "side": "BUY", "quantity": 5.0, "avg_price": 180.0}],
    })
    hist = reconstruct.load_history(od)
    # 1000 - 5*180 = 100
    assert reconstruct.cash_after(hist) == 100.0


def test_cash_after_accumulates_sells_and_buys(tmp_path):
    od = tmp_path / "orders"
    _write(od, "2026-06-05", {
        "first_build": True, "nav": 1000.0, "post_positions": {"AAA": 5.0},
        "fills": [{"ticker": "AAA", "side": "BUY", "quantity": 5.0, "avg_price": 180.0}],
    })
    _write(od, "2026-06-12", {
        "first_build": False, "nav": 0.0, "post_positions": {"AAA": 3.0},
        "fills": [{"ticker": "AAA", "side": "SELL", "quantity": 2.0, "avg_price": 200.0}],
    })
    hist = reconstruct.load_history(od)
    # 100 + (sell 2*200 adds cash) = 500
    assert reconstruct.cash_after(hist) == 500.0


def test_cash_after_no_first_build_raises(tmp_path):
    od = tmp_path / "orders"
    _write(od, "2026-06-05", {"first_build": False, "nav": 1.0,
                              "post_positions": {}, "fills": []})
    hist = reconstruct.load_history(od)
    import pytest
    with pytest.raises(ValueError):
        reconstruct.cash_after(hist)


def test_inception_date_is_first_build_asof(tmp_path):
    od = tmp_path / "orders"
    _write(od, "2026-06-05", {"first_build": True, "nav": 1000.0,
                              "post_positions": {"AAA": 5.0}, "fills": []})
    _write(od, "2026-06-12", {"first_build": False, "nav": 0.0,
                              "post_positions": {"AAA": 5.0}, "fills": []})
    hist = reconstruct.load_history(od)
    assert reconstruct.inception_date(hist) == pd.Timestamp("2026-06-05")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/trading/test_reconstruct.py::test_cash_after_first_build_residual -v`
Expected: FAIL — `AttributeError: module 'trading.publish.reconstruct' has no attribute 'cash_after'`.

- [ ] **Step 3: Add the functions**

Append to `trading/publish/reconstruct.py`:

```python
def _signed_qty(fill: dict) -> float:
    qty = float(fill.get("quantity", 0.0))
    return qty if fill.get("side", "BUY") == "BUY" else -qty


def _first_build(history: list[dict]) -> dict:
    anchor = next((r for r in history if r.get("first_build")), None)
    if anchor is None:
        raise ValueError("reconstruct: no first_build record in history")
    return anchor


def inception_date(history: list[dict]) -> pd.Timestamp:
    """Normalized asof of the first_build record (when the strategy went live)."""
    return pd.Timestamp(_first_build(history)["asof"]).normalize()


def cash_after(history: list[dict]) -> float:
    """Residual cash = inception NAV − Σ(signed fill qty × avg_price).

    BUY spends cash (signed +qty); SELL returns cash (signed −qty, so subtracting
    a negative adds). Inception NAV is the first_build record's all-cash 'nav'.
    """
    cash = float(_first_build(history)["nav"])
    for rec in history:
        for f in rec.get("fills", []):
            cash -= _signed_qty(f) * float(f.get("avg_price") or 0.0)
    return cash
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/trading/test_reconstruct.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add trading/publish/reconstruct.py tests/trading/test_reconstruct.py
git commit -m "reconstruct: inception_date + cash_after from fills"
```

---

## Task 4: `reconstruct.reconstruct_curve`

**Files:**
- Modify: `trading/publish/reconstruct.py`
- Test: `tests/trading/test_reconstruct.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/trading/test_reconstruct.py`:

```python
def _curve_fixture(tmp_path):
    od = tmp_path / "orders"
    _write(od, "2026-06-05", {
        "first_build": True, "nav": 1000.0, "post_positions": {"AAA": 5.0},
        "fills": [{"ticker": "AAA", "side": "BUY", "quantity": 5.0, "avg_price": 180.0}],
    })
    return reconstruct.load_history(od)


def test_reconstruct_curve_single_rebalance(tmp_path):
    hist = _curve_fixture(tmp_path)          # cash residual = 100, holds 5 AAA
    idx = pd.to_datetime(["2026-06-05", "2026-06-08"]).normalize()
    closes = pd.DataFrame({"AAA": [180.0, 200.0], "SPY": [500.0, 510.0]}, index=idx)
    curve = reconstruct.reconstruct_curve(hist, closes, closes["SPY"])
    assert [p["date"] for p in curve] == ["2026-06-05", "2026-06-08"]
    # day 1: 100 + 5*180 = 1000 ; day 2: 100 + 5*200 = 1100
    assert curve[0]["nav"] == 1000.0
    assert curve[1]["nav"] == 1100.0
    assert curve[0]["spy_close"] == 500.0


def test_reconstruct_curve_starts_at_inception(tmp_path):
    hist = _curve_fixture(tmp_path)
    idx = pd.to_datetime(["2026-06-01", "2026-06-05"]).normalize()  # one day pre-inception
    closes = pd.DataFrame({"AAA": [170.0, 180.0], "SPY": [490.0, 500.0]}, index=idx)
    curve = reconstruct.reconstruct_curve(hist, closes, closes["SPY"])
    assert [p["date"] for p in curve] == ["2026-06-05"]  # pre-inception day dropped


def test_reconstruct_curve_skips_missing_close(tmp_path):
    hist = _curve_fixture(tmp_path)
    idx = pd.to_datetime(["2026-06-05"]).normalize()
    closes = pd.DataFrame({"AAA": [float("nan")], "SPY": [500.0]}, index=idx)
    curve = reconstruct.reconstruct_curve(hist, closes, closes["SPY"])
    # AAA price missing → contributes 0 market value → nav == cash residual (100)
    assert curve[0]["nav"] == 100.0


def test_reconstruct_curve_empty_history():
    assert reconstruct.reconstruct_curve([], pd.DataFrame(), pd.Series(dtype=float)) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/trading/test_reconstruct.py::test_reconstruct_curve_single_rebalance -v`
Expected: FAIL — `AttributeError: ... has no attribute 'reconstruct_curve'`.

- [ ] **Step 3: Add the function**

Append to `trading/publish/reconstruct.py`:

```python
def reconstruct_curve(history, close_history, spy_history) -> list[dict]:
    """Daily {date, nav, spy_close} from inception through the last price date.

    close_history: DataFrame indexed by normalized date, columns = tickers, values
    = daily close. spy_history: Series indexed by the same dates. For each day d:
    holdings/cash come from the latest rebalance with asof ≤ d; nav = cash +
    Σ(shares × close). A missing/NaN close contributes zero (price comes in via the
    forward-filled frame from fetch_close_history).
    """
    if not history or close_history is None or len(close_history.index) == 0:
        return []
    start = inception_date(history)
    rows: list[dict] = []
    for raw_d in close_history.index:
        d = pd.Timestamp(raw_d).normalize()
        if d < start:
            continue
        applied = [r for r in history if pd.Timestamp(r["asof"]).normalize() <= d]
        if not applied:
            continue
        holdings = current_holdings(applied)
        cash = cash_after(applied)
        mv = 0.0
        for ticker, shares in holdings.items():
            if ticker in close_history.columns:
                px = close_history.at[raw_d, ticker]
                if not pd.isna(px):
                    mv += shares * float(px)
        spy = spy_history.get(raw_d) if spy_history is not None else None
        rows.append({
            "date": str(d.date()),
            "nav": cash + mv,
            "spy_close": (float(spy) if spy is not None and not pd.isna(spy) else None),
        })
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/trading/test_reconstruct.py -v`
Expected: PASS (12 tests).

- [ ] **Step 5: Commit**

```bash
git add trading/publish/reconstruct.py tests/trading/test_reconstruct.py
git commit -m "reconstruct: full daily NAV curve from audit + close history"
```

---

## Task 5: `sources.fetch_close_history`

**Files:**
- Modify: `trading/data/sources.py`
- Test: `tests/trading/test_sources.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/trading/test_sources.py`:

```python
def test_fetch_close_history_extracts_and_ffills():
    import pandas as pd
    from trading.data import sources

    idx = pd.to_datetime(["2026-06-05", "2026-06-08", "2026-06-09"])
    # yfinance multi-ticker shape: columns are a (field, ticker) MultiIndex.
    cols = pd.MultiIndex.from_tuples(
        [("Close", "AAA"), ("Close", "BBB"), ("Open", "AAA"), ("Open", "BBB")]
    )
    raw = pd.DataFrame(
        [[10.0, 20.0, 1, 1], [float("nan"), 21.0, 1, 1], [12.0, 22.0, 1, 1]],
        index=idx, columns=cols,
    )

    def fake_download(tickers, **kw):
        return raw

    out = sources.fetch_close_history(["BBB", "AAA"], "2026-06-05", "2026-06-10",
                                      download=fake_download)
    assert list(out.columns) == ["AAA", "BBB"]            # sorted, Close-only
    assert out.loc[pd.Timestamp("2026-06-08"), "AAA"] == 10.0   # NaN forward-filled
    assert out.index[0] == pd.Timestamp("2026-06-05")


def test_fetch_close_history_empty_tickers():
    from trading.data import sources
    out = sources.fetch_close_history([], "2026-06-05", "2026-06-10",
                                      download=lambda *a, **k: None)
    assert out.empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/trading/test_sources.py::test_fetch_close_history_extracts_and_ffills -v`
Expected: FAIL — `AttributeError: module 'trading.data.sources' has no attribute 'fetch_close_history'`.

- [ ] **Step 3: Add the function**

Add to `trading/data/sources.py` (uses the existing `_retry` helper; `download` is injected in tests, defaults to `yfinance.download` like the other yfinance functions in this file):

```python
def fetch_close_history(tickers, start, end, download=None) -> pd.DataFrame:
    """Daily closes for `tickers` over [start, end), forward-filled.

    Returns a DataFrame indexed by normalized date, one column per ticker (sorted).
    `download` is injected in tests; defaults to yfinance.download.
    """
    tickers = sorted(set(tickers))
    if not tickers:
        return pd.DataFrame()
    if download is None:
        import yfinance as yf  # noqa: PLC0415
        download = yf.download
    raw = _retry(lambda: download(
        tickers, start=str(pd.Timestamp(start).date()),
        end=str(pd.Timestamp(end).date()), interval="1d",
        progress=False, auto_adjust=False,
    ))
    # Multi-ticker yfinance returns a (field, ticker) column MultiIndex; single
    # ticker returns flat columns. Normalize to a Close-only, ticker-columned frame.
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    elif "Close" in raw.columns:
        close = raw[["Close"]]
        close.columns = [tickers[0]]
    else:
        close = raw
    close.index = pd.to_datetime(close.index).normalize()
    return close.reindex(columns=tickers).ffill()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/trading/test_sources.py -v`
Expected: PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add trading/data/sources.py tests/trading/test_sources.py
git commit -m "sources: fetch_close_history (batched yfinance daily closes, ffilled)"
```

---

## Task 6: `publish_from_audit` orchestrator

**Files:**
- Modify: `trading/publish/publish.py`
- Test: `tests/trading/test_publish_from_audit.py`

- [ ] **Step 1: Write the failing test**

Create `tests/trading/test_publish_from_audit.py`:

```python
"""publish_from_audit against fixture audit files + injected price frame + fake store."""
from __future__ import annotations

import json

import pandas as pd

from trading.publish.publish import publish_from_audit


class _RecordingStore:
    def __init__(self):
        self.snapshot = self.holdings = self.weekly = self.executions = None
        self.equity_curve = None

    def replace_equity_curve(self, rows): self.equity_curve = rows
    def upsert_snapshot(self, row): self.snapshot = row
    def replace_holdings(self, rows): self.holdings = rows
    def insert_weekly_portfolio(self, asof_friday, rows):
        self.weekly = {"asof_friday": asof_friday, "rows": rows}
    def insert_executions(self, asof, rows):
        self.executions = {"asof": asof, "rows": rows}


def _setup(tmp_path, asof="2026-06-05"):
    wdir = tmp_path / "weights"; wdir.mkdir(parents=True)
    (wdir / f"{asof}.json").write_text(json.dumps(
        {"asof": asof, "k_probs": {"10": 1.0}, "weights": {"AAA": 1.0}}))
    odir = tmp_path / "orders"; odir.mkdir(parents=True)
    (odir / f"{asof}.json").write_text(json.dumps({
        "asof": asof, "mode": "live", "nav": 1000.0, "first_build": True,
        "post_positions": {"AAA": 5.0},
        "fills": [{"ticker": "AAA", "side": "BUY", "quantity": 5.0, "avg_price": 180.0}],
        "ladder_stages": [{"ticker": "AAA", "qty_filled": 5.0,
                           "realized_price": 180.0, "midpoint_at_fill": 179.0}],
    }))
    return wdir, odir, asof


def test_publish_from_audit_writes_curve_holdings_snapshot(tmp_path):
    wdir, odir, asof = _setup(tmp_path)
    idx = pd.to_datetime(["2026-06-05", "2026-06-08"]).normalize()
    frame = pd.DataFrame({"AAA": [180.0, 200.0], "SPY": [500.0, 510.0]}, index=idx)

    def price_fetch(tickers, start, end):
        return frame.reindex(columns=sorted(set(tickers))).ffill()

    store = _RecordingStore()
    summary = publish_from_audit(
        store, weights_dir=wdir, orders_dir=odir, asof=asof,
        today=pd.Timestamp("2026-06-08"), price_fetch=price_fetch,
    )

    assert summary["n_holdings"] == 1
    assert [p["date"] for p in store.equity_curve] == ["2026-06-05", "2026-06-08"]
    assert store.equity_curve[-1]["nav"] == 1100.0      # 100 cash + 5*200
    assert store.snapshot["nav"] == 1100.0
    assert store.snapshot["n_positions"] == 1
    assert store.snapshot["total_return"] == 0.1        # 1100/1000 - 1
    assert store.holdings[0]["ticker"] == "AAA"
    assert store.holdings[0]["price"] == 200.0          # latest close
    assert all(h["asof"] == "2026-06-08" for h in store.holdings)
    assert store.weekly["asof_friday"] == asof
    assert store.executions["asof"] == asof             # orders file present


def test_publish_from_audit_no_broker_import(tmp_path, monkeypatch):
    # Guard: the audit path must never touch the IBKR broker.
    wdir, odir, asof = _setup(tmp_path)
    idx = pd.to_datetime(["2026-06-05"]).normalize()
    frame = pd.DataFrame({"AAA": [180.0], "SPY": [500.0]}, index=idx)
    import trading.broker.ibkr as ibkr

    def _boom(*a, **k):
        raise AssertionError("publish_from_audit must not construct IBKRBroker")

    monkeypatch.setattr(ibkr, "IBKRBroker", _boom)
    store = _RecordingStore()
    publish_from_audit(store, weights_dir=wdir, orders_dir=odir, asof=asof,
                       today=pd.Timestamp("2026-06-05"),
                       price_fetch=lambda t, s, e: frame.reindex(columns=sorted(set(t))).ffill())
    assert store.snapshot["nav"] == 1000.0              # 100 cash + 5*180
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/trading/test_publish_from_audit.py -v`
Expected: FAIL — `ImportError: cannot import name 'publish_from_audit'`.

- [ ] **Step 3: Add the orchestrator**

In `trading/publish/publish.py`, add the import near the top (after the existing `from trading.publish.metrics import (...)` block):

```python
from trading.publish.reconstruct import (
    current_holdings,
    inception_date,
    load_history,
    reconstruct_curve,
)
```

Then add this function after `publish_once` (it reuses `_load_json`, `_prior_weights`, and the `metrics` helpers already imported in this module):

```python
def publish_from_audit(store, *, weights_dir, orders_dir, asof, today, price_fetch,
                       fetch_metadata=None):
    """Compute and write one snapshot from the order audit + injected prices.

    `price_fetch(tickers, start, end) -> DataFrame` returns forward-filled daily
    closes indexed by normalized date, one column per ticker (see
    sources.fetch_close_history). No broker is contacted. Returns a summary dict.
    """
    asof = str(pd.Timestamp(asof).date())
    today = pd.Timestamp(today).normalize()

    history = load_history(orders_dir)
    holdings_shares = current_holdings(history)

    weights_payload = _load_json(Path(weights_dir) / f"{asof}.json")
    target_weights = {str(k): float(v) for k, v in (weights_payload.get("weights") or {}).items()}
    k_probs = weights_payload.get("k_probs") or {}
    regime_features = weights_payload.get("regime_features")
    last_weights = _prior_weights(weights_dir, asof)
    turnover = compute_turnover(target_weights, last_weights) if last_weights else None

    # Every ticker ever held (for the historical curve) + currently held + SPY.
    ever: set[str] = set()
    for rec in history:
        ever |= {str(t) for t in (rec.get("post_positions") or {})}
    tickers = sorted(ever | set(holdings_shares) | set(target_weights))
    start = inception_date(history) if history else today
    closes = price_fetch(tickers + ["SPY"], start, today + pd.Timedelta(days=1))
    spy_history = closes["SPY"] if "SPY" in closes.columns else pd.Series(dtype=float)

    curve = reconstruct_curve(history, closes, spy_history)
    navs = [p["nav"] for p in curve]
    nav = navs[-1] if navs else 0.0
    prev_nav = navs[-2] if len(navs) >= 2 else None

    latest_closes = {
        t: float(closes[t].iloc[-1]) for t in holdings_shares
        if t in closes.columns and not pd.isna(closes[t].iloc[-1])
    }

    metadata: dict[str, dict] = {}
    if fetch_metadata is not None:
        try:
            metadata = fetch_metadata(sorted(set(holdings_shares) | set(target_weights))) or {}
        except Exception as exc:  # noqa: BLE001
            logger.warning("publish: ticker metadata unavailable (%s)", exc)

    holdings = compute_holdings(holdings_shares, latest_closes, target_weights, nav, metadata=metadata)
    day_pnl, day_pnl_pct = compute_day_pnl(nav, prev_nav)
    risk = compute_risk(navs)
    invested = sum(h["market_value"] for h in holdings)
    inception_nav = navs[0] if navs else nav
    inception_spy = next((p["spy_close"] for p in curve if p["spy_close"] is not None), None)
    spy_now = curve[-1]["spy_close"] if curve else None
    today_str = str(today.date())

    store.replace_equity_curve(curve)
    store.upsert_snapshot(
        {
            "asof": today_str,
            "nav": nav,
            "day_pnl": day_pnl,
            "day_pnl_pct": day_pnl_pct,
            "total_return": pct_change(nav, inception_nav),
            "spy_return": pct_change(spy_now, inception_spy),
            "n_positions": len(holdings),
            "invested_pct": (invested / nav) if nav > 0 else None,
            "k_probs": k_probs,
            "regime_features": regime_features,
            "risk": risk,
            "turnover": turnover,
        }
    )
    store.replace_holdings([{**h, "asof": today_str} for h in holdings])
    store.insert_weekly_portfolio(
        asof,
        [
            {"asof_friday": asof, "ticker": t, "target_weight": w, "k_probs": k_probs,
             "company_name": metadata.get(t, {}).get("company_name"),
             "sector": metadata.get(t, {}).get("sector")}
            for t, w in target_weights.items()
        ],
    )
    orders_path = Path(orders_dir) / f"{asof}.json"
    if orders_path.exists():
        exec_rows = compute_execution_quality(_load_json(orders_path))
        store.insert_executions(asof, [{**r, "asof": asof} for r in exec_rows])

    return {"asof": asof, "nav": nav, "n_holdings": len(holdings)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/trading/test_publish_from_audit.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add trading/publish/publish.py tests/trading/test_publish_from_audit.py
git commit -m "publish: broker-free publish_from_audit orchestrator"
```

---

## Task 7: Repoint `main()` to the audit path

**Files:**
- Modify: `trading/publish/publish.py` (the `main()` function, lines ~180-209)

`publish_once` and `is_market_hours` stay (still used by their tests + future broker truth-up). Only `main()` changes: no broker, no market-hours guard, use yfinance close history.

- [ ] **Step 1: Replace `main()`**

Replace the body of `main()` in `trading/publish/publish.py` with:

```python
def main() -> int:
    """CLI entrypoint for the daily timer: `python -m trading.publish`.

    Broker-free: holdings, NAV, and the equity curve are reconstructed from the
    order audit + yfinance closes. No IBKR connection, no market-hours guard.
    """
    import trading.config as config  # noqa: PLC0415
    from trading.data.snapshot import most_recent_friday  # noqa: PLC0415
    from trading.data.sources import fetch_close_history, fetch_ticker_metadata  # noqa: PLC0415
    from trading.publish.store import SupabaseStore, make_client  # noqa: PLC0415

    logging.basicConfig(level=logging.INFO)

    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_KEY:
        logger.error("publish: SUPABASE_URL / SUPABASE_SERVICE_KEY not set — aborting")
        return 1

    store = SupabaseStore(make_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_KEY))
    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    summary = publish_from_audit(
        store,
        weights_dir=config.WEIGHTS_DIR,
        orders_dir=config.ORDERS_DIR,
        asof=most_recent_friday(),
        today=today,
        price_fetch=fetch_close_history,
        fetch_metadata=fetch_ticker_metadata,
    )
    logger.info("publish: done — %s", summary)
    return 0
```

- [ ] **Step 2: Run the full publisher test suite**

Run: `python -m pytest tests/trading/test_publish_from_audit.py tests/trading/test_publish_orchestrator.py tests/trading/test_publish_store.py tests/trading/test_publish_metrics.py tests/trading/test_publish_backfill.py -v`
Expected: PASS (the broker-path `publish_once` + `is_market_hours` tests still pass; new audit tests pass).

- [ ] **Step 3: Smoke-check the module imports**

Run: `python -c "import trading.publish.publish as p; print(hasattr(p,'publish_from_audit'), hasattr(p,'publish_once'), hasattr(p,'is_market_hours'))"`
Expected: `True True True`

- [ ] **Step 4: Commit**

```bash
git add trading/publish/publish.py
git commit -m "publish: scheduled main() uses audit path (no broker, no hours guard)"
```

---

## Task 8: Daily-after-close systemd schedule

**Files:**
- Modify: `deploy/systemd/axiom-publish.timer`
- Modify: `deploy/systemd/axiom-publish.service`

- [ ] **Step 1: Update the timer to once daily after close**

Replace `deploy/systemd/axiom-publish.timer` with:

```ini
[Unit]
Description=Daily after US close - publish dashboard snapshot from the audit trail

[Timer]
# 16:30 America/New_York, every day. The publisher is broker-free (reads the order
# audit + yfinance), so it does not need the IBKR Gateway and has no market-hours guard.
OnCalendar=*-*-* 16:30 America/New_York
Persistent=true

[Install]
WantedBy=timers.target
```

- [ ] **Step 2: Update the service description/comment (no Gateway dependency)**

In `deploy/systemd/axiom-publish.service`, replace the `Description=` line and the comment above `ExecStart` with:

```ini
[Unit]
Description=Axiom Tilt - publish dashboard snapshot (audit trail + yfinance, no broker)
OnFailure=axiom-alert@%n.service

[Service]
Type=oneshot
WorkingDirectory=%h/axiom_tilt_strategy
# Broker-free: reconstructs holdings/NAV/equity from trading/audit + yfinance.
ExecStart=%h/axiom_tilt_strategy/.venv/bin/python -m trading.publish
```

- [ ] **Step 3: Validate the unit files parse (if systemd available)**

Run: `systemd-analyze verify deploy/systemd/axiom-publish.timer 2>&1 || echo "systemd-analyze not available on this host — verify on the VPS"`
Expected: no errors, or the not-available note (the files are deployed/enabled on the VPS).

- [ ] **Step 4: Commit**

```bash
git add deploy/systemd/axiom-publish.timer deploy/systemd/axiom-publish.service
git commit -m "deploy: publish daily after close (broker-free), drop 20-min timer"
```

---

## Final verification

- [ ] **Run the full trading test suite**

Run: `python -m pytest tests/trading -v`
Expected: PASS (all existing + new tests).

- [ ] **End-to-end dry check (optional, writes to Supabase)**

With the repo `.env` loaded (`SUPABASE_URL`, `SUPABASE_SERVICE_KEY`), run `python -m trading.publish` and confirm it logs `publish: done — {...}` with `n_holdings` > 0, then verify in Supabase that `holdings` and `equity_curve` are now populated and the dashboard renders them. This needs no IBKR Gateway.
