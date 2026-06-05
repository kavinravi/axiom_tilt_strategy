"""CLI entry point for the live trading system.

Usage
-----
    python -m trading.run weights [--asof YYYY-MM-DD]
    python -m trading.run rebalance [--asof YYYY-MM-DD] [--mode dryrun|paper|live] [--confirm]

The ``weights`` subcommand:
1. Runs the full target-weight pipeline (snapshot → scores → regime → k_probs
   → weights → audit freeze).
2. Prints a sorted holdings table (descending by weight), k_probs, asof date,
   and summary statistics.
3. Runs validate_weights and reports results; exits non-zero on any problem.

The ``rebalance`` subcommand:
1. Loads frozen weights for <asof> from the audit directory.
2. Connects to broker (dryrun by default; paper/live via IBKRBroker when Phase C lands).
3. Reconciles positions/NAV, computes orders, runs safety checks.
4. If --confirm: prints the order table and requires typed 'yes'.
5. Executes via the three-stage passive→midprice→market ladder.
6. Writes an order/fill audit JSON; prints a fill summary.
7. Exits non-zero on SafetyError or abort.

format_report(result) -> str is a PURE helper: no side effects, deterministic,
safe to call in unit tests without any network access.

format_rebalance_report(summary) -> str is also PURE: renders a rebalance
summary dict to a human-readable string.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any

import pandas as pd

from trading.weights import compute_target_weights, validate_weights


# ---------------------------------------------------------------------------
# format_report — pure, no side effects
# ---------------------------------------------------------------------------

def format_report(result: dict) -> str:
    """Render a result dict returned by compute_target_weights to a string.

    The output is deterministic and has no side effects; safe to call in unit
    tests without any network or model access.

    Args:
        result: dict with keys asof, weights, k_probs, n_holdings,
                weight_sum, max_weight.

    Returns:
        A multi-line string ready for printing.
    """
    lines: list[str] = []

    asof: pd.Timestamp = result["asof"]
    weights: dict[Any, float] = result["weights"]
    k_probs: dict[int, float] = result["k_probs"]
    n_holdings: int = result["n_holdings"]
    weight_sum: float = result["weight_sum"]
    max_weight: float = result["max_weight"]

    lines.append("=" * 56)
    lines.append(f"  Target Weights  —  asof {asof.date()}")
    lines.append("=" * 56)

    # --- Holdings table sorted by weight descending ---
    sorted_items = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
    lines.append(f"{'Ticker':<12}  {'Weight':>8}")
    lines.append("-" * 24)
    for ticker, w in sorted_items:
        lines.append(f"{str(ticker):<12}  {w:>7.2%}")

    lines.append("-" * 24)

    # --- Summary statistics ---
    lines.append(f"n_holdings : {n_holdings}")
    lines.append(f"weight_sum : {weight_sum:.6f}")
    lines.append(f"max_weight : {max_weight:.4%}")

    # --- k_probs ---
    lines.append("")
    lines.append("Regime K-probabilities (k_probs):")
    for k in sorted(k_probs):
        lines.append(f"  k={k:>2d} : {k_probs[k]:.4f}")

    lines.append("=" * 56)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# format_rebalance_report — pure, no side effects
# ---------------------------------------------------------------------------

def format_rebalance_report(summary: dict) -> str:
    """Render a rebalance summary dict to a human-readable string.

    The output is deterministic and has no side effects; safe to call in unit
    tests on a canned summary dict.

    Args:
        summary: dict returned by run_rebalance, with keys:
            asof, mode, n_orders, n_filled, fills, audit, orders_path,
            first_build.

    Returns:
        A multi-line string ready for printing.
    """
    lines: list[str] = []

    asof = summary.get("asof")
    asof_str = asof.date() if hasattr(asof, "date") else str(asof)
    mode = summary.get("mode", "unknown")
    n_orders = summary.get("n_orders", 0)
    n_filled = summary.get("n_filled", 0)
    fills = summary.get("fills", [])
    orders_path = summary.get("orders_path", "—")
    first_build = summary.get("first_build", False)

    lines.append("=" * 60)
    lines.append(f"  Rebalance Summary  —  asof {asof_str}  [{mode}]")
    if first_build:
        lines.append("  (First build from cash — turnover cap exempted)")
    lines.append("=" * 60)

    # Fill table
    lines.append(f"{'Ticker':<12}  {'Side':<6}  {'Qty':>12}  {'Avg Price':>10}  {'Status':<10}")
    lines.append("-" * 60)
    for f in sorted(fills, key=lambda x: getattr(x, "ticker", "")):
        ticker = getattr(f, "ticker", "?")
        side = getattr(f, "side", "?")
        qty = getattr(f, "quantity", 0.0)
        price = getattr(f, "avg_price", 0.0)
        status = getattr(f, "status", "?")
        lines.append(
            f"{ticker:<12}  {side:<6}  {qty:>12.4f}  {price:>10.4f}  {status:<10}"
        )
    lines.append("-" * 60)
    lines.append(f"  Orders: {n_orders}   Filled/partial: {n_filled}")
    lines.append(f"  Audit: {orders_path}")
    lines.append("=" * 60)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_weights(args: argparse.Namespace) -> int:
    """Run the weights pipeline and print the result.

    Returns 0 on success, 1 if any sanity check fails.
    """
    asof: pd.Timestamp | None = None
    if args.asof:
        asof = pd.Timestamp(args.asof).normalize()

    result = compute_target_weights(asof=asof)

    print(format_report(result))

    problems = validate_weights(result)
    if not problems:
        print("✓ sanity checks passed")
        return 0
    else:
        print("\nSanity check FAILURES:")
        for p in problems:
            print(f"  - {p}")
        return 1


def cmd_rebalance(args: argparse.Namespace) -> int:
    """Run the rebalance pipeline and print a fill summary.

    Returns 0 on success, 1 on SafetyError or user abort.
    """
    from trading.execution.rebalance import run_rebalance  # noqa: PLC0415
    from trading.execution.safety import SafetyError  # noqa: PLC0415

    asof = args.asof or None
    mode = args.mode or None
    confirm = args.confirm

    try:
        summary = run_rebalance(asof=asof, mode=mode, confirm=confirm)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except SafetyError as exc:
        print(f"SAFETY ABORT: {exc}", file=sys.stderr)
        return 1
    except NotImplementedError as exc:
        print(f"NOT IMPLEMENTED: {exc}", file=sys.stderr)
        return 1

    print(format_rebalance_report(summary))
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trading.run",
        description="Live trading CLI — compute and inspect target weights.",
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # weights subcommand
    weights_p = sub.add_parser(
        "weights",
        help="Compute this week's target weights and print a sanity-checked report.",
    )
    weights_p.add_argument(
        "--asof",
        metavar="YYYY-MM-DD",
        default=None,
        help="Override the rebalance date (must be a Friday; defaults to most recent Friday).",
    )

    # rebalance subcommand
    rebalance_p = sub.add_parser(
        "rebalance",
        help="Execute a Monday rebalance: load frozen weights, compute orders, run safety, place via ladder.",
    )
    rebalance_p.add_argument(
        "--asof",
        metavar="YYYY-MM-DD",
        default=None,
        help="Rebalance date (defaults to most recent Friday).",
    )
    rebalance_p.add_argument(
        "--mode",
        choices=["dryrun", "paper", "live"],
        default=None,
        help="Execution mode (default: config.EXECUTION_MODE = 'dryrun').",
    )
    rebalance_p.add_argument(
        "--confirm",
        action="store_true",
        default=False,
        help="Print the order table and require explicit 'yes' before placing orders.",
    )

    return parser


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "weights":
        exit_code = cmd_weights(args)
        sys.exit(exit_code)
    elif args.command == "rebalance":
        exit_code = cmd_rebalance(args)
        sys.exit(exit_code)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
