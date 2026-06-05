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
