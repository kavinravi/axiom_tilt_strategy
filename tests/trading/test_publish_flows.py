"""Capital-flows ledger I/O (trading/publish/flows.py)."""
from __future__ import annotations

import json

from trading.publish.flows import load_flows, save_flows


def test_load_missing_file_is_empty(tmp_path):
    assert load_flows(tmp_path / "capital_flows.json") == {}


def test_load_none_path_is_empty():
    assert load_flows(None) == {}


def test_roundtrip_sorted_by_date(tmp_path):
    path = tmp_path / "capital_flows.json"
    save_flows(path, {"2026-06-18": 24_000.0, "2026-06-12": 75_242.19})
    assert load_flows(path) == {"2026-06-12": 75_242.19, "2026-06-18": 24_000.0}
    # File itself is date-sorted for clean hand edits.
    assert list(json.loads(path.read_text())) == ["2026-06-12", "2026-06-18"]


def test_save_creates_parent_dirs(tmp_path):
    path = tmp_path / "audit" / "capital_flows.json"
    save_flows(path, {"2026-06-12": 1.0})
    assert load_flows(path) == {"2026-06-12": 1.0}


def test_load_coerces_types(tmp_path):
    path = tmp_path / "capital_flows.json"
    path.write_text(json.dumps({"2026-06-12": "75242.19"}))
    assert load_flows(path) == {"2026-06-12": 75242.19}
