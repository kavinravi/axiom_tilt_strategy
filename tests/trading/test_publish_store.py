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

    def gte(self, col, val):
        self._log.append(("gte", self._table, col, val))
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
            {"date": "2026-06-03", "nav": 100.0, "spy_close": 5000.0, "flow": 0.0},
            "date") in c.log


def test_upsert_equity_point_carries_flow():
    c = _FakeClient()
    SupabaseStore(c).upsert_equity_point("2026-06-12", 176_435.05, 5000.0, flow=75_242.19)
    assert ("upsert", "equity_curve",
            {"date": "2026-06-12", "nav": 176_435.05, "spy_close": 5000.0,
             "flow": 75_242.19}, "date") in c.log


def test_replace_holdings_deletes_then_inserts():
    c = _FakeClient()
    SupabaseStore(c).replace_holdings([{"ticker": "AAA"}])
    kinds = [e[0] for e in c.log]
    assert kinds == ["delete", "insert"]


def test_read_equity_curve_returns_client_data():
    rows = [{"date": "2026-06-01", "nav": 100.0, "spy_close": 5000.0}]
    store = SupabaseStore(_FakeClient(equity_rows=rows))
    assert store.read_equity_curve() == rows


def test_replace_equity_curve_deletes_then_inserts():
    import datetime as _dt

    c = _FakeClient()
    rows = [{"date": "2026-06-05", "nav": 100.0, "spy_close": 500.0}]
    SupabaseStore(c).replace_equity_curve(rows)
    kinds = [e[0] for e in c.log]
    assert kinds == ["delete", "gte", "insert"]
    # Regression: the all-rows delete filter must be a VALID date — Postgres rejects
    # `date != ""` ("invalid input syntax for type date"). Confirm it parses as a date.
    gte = next(e for e in c.log if e[0] == "gte")
    assert gte[2] == "date"
    _dt.date.fromisoformat(gte[3])  # raises ValueError if not an ISO date


def test_replace_equity_curve_empty_skips_insert():
    c = _FakeClient()
    SupabaseStore(c).replace_equity_curve([])
    kinds = [e[0] for e in c.log]
    assert kinds == ["delete", "gte"]
