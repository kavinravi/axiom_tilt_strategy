"""Capital-flows ledger: external deposits/withdrawals by date.

A flat JSON map {"YYYY-MM-DD": amount} (deposit positive, withdrawal negative)
living in the gitignored audit dir — real account flows must never reach the
public repo. The publisher auto-detects flows from broker day P&L and records
them here; entries are also hand-editable to correct an amount. An existing
entry for a date always wins over detection, and detected zeros are never
written, so manual entries survive re-publishes. Past-date edits reach the
dashboard on the next ``--from-audit`` rebuild (the curve repair path also
reads this file).
"""
from __future__ import annotations

import json
from pathlib import Path


def load_flows(path: Path | str | None) -> dict[str, float]:
    """Read the ledger; {} when the path is None or the file doesn't exist."""
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    with path.open() as f:
        raw = json.load(f)
    return {str(k): float(v) for k, v in raw.items()}


def save_flows(path: Path | str, flows: dict[str, float]) -> None:
    """Write the ledger sorted by date (stable diffs for hand edits)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(dict(sorted(flows.items())), f, indent=2)
        f.write("\n")
