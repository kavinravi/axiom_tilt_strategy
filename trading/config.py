"""Configuration for the live trading system (single place for paths + sources)."""
from __future__ import annotations

from src.strategy.constants import K_CANDIDATES, MAX_WEIGHT, MIN_ALLOCATION
from src.utils.env import get_env
from src.utils.io import repo_root

REPO_ROOT = repo_root()
TRADING_DIR = REPO_ROOT / "trading"
MODEL_PATH = TRADING_DIR / "models" / "k_selector.txt"
AUDIT_DIR = TRADING_DIR / "audit"
WEIGHTS_DIR = AUDIT_DIR / "weights"

# Execution mode scaffold (Plan 3+ uses paper/live; Plan 2 only computes weights).
EXECUTION_MODE = "dryrun"  # one of: dryrun | paper | live

# Sharadar tables (Nasdaq Data Link)
SHARADAR_SP500 = "SHARADAR/SP500"
SHARADAR_SF1 = "SHARADAR/SF1"
SHARADAR_DAILY = "SHARADAR/DAILY"
SF1_DIMENSION = "ARQ"  # As-Reported Quarterly, matches the backtest panel

# FRED series -> snapshot column names (via pandas_datareader)
FRED_MACRO_SERIES = {"VIXCLS": "macro_vixcls", "DGS10": "macro_dgs10", "T10Y2Y": "macro_t10y2y"}
FRED_SPY_SERIES = "SP500"  # S&P 500 index level (SPY ETF not freshly available in Sharadar)

# Regime feature window: enough weekly Fridays for the 26w vol + shift(1).
REGIME_HISTORY_WEEKS = 40

# Sanity bounds for the weights output
MIN_HOLDINGS = 10
MAX_HOLDINGS = 503
WEIGHT_SUM_TOL = 1e-6

# IBKR connection (see docs/ibkr-account-and-gateway-setup.md). Env-overridable so the
# WSL->Windows host IP (e.g. IBKR_HOST=172.18.0.1) and the VPS default (127.0.0.1) need no
# code change. Verified 2026-06-03: paper Gateway reachable from WSL at 172.18.0.1:4002.
IBKR_HOST = get_env("IBKR_HOST", default="127.0.0.1")
IBKR_PORT = int(get_env("IBKR_PORT", default="4002"))   # 4002 paper / 4001 live (IB Gateway); used by the publisher
IBKR_CLIENT_ID = int(get_env("IBKR_CLIENT_ID", default="11"))
# The publisher connects read-only on its own client id so the Monday 15:00
# intraday tick can coexist with the rebalance connection.
IBKR_PUBLISH_CLIENT_ID = int(get_env("IBKR_PUBLISH_CLIENT_ID", default="12"))
# Distinct paper/live ports so a paper rebalance can NEVER reach the live gateway.
# run_rebalance picks the port by mode (see _select_broker); these are the IB Gateway defaults.
IBKR_PAPER_PORT = int(get_env("IBKR_PAPER_PORT", default="4002"))
IBKR_LIVE_PORT = int(get_env("IBKR_LIVE_PORT", default="4001"))

# Safety rails
KILL_SWITCH_FILE = TRADING_DIR / "KILL_SWITCH"   # if this file exists, abort all order placement
MAX_ORDER_FRAC_NAV = 0.12     # reject if any single order notional > 12% of NAV
# NOTE: a first-build from cash has ~100% turnover by definition and is exempted from
# this cap by run_rebalance (which passes skip_turnover_check=True when current_positions
# is empty). All subsequent rebalances (incremental) are subject to this cap.
MAX_TURNOVER_FRAC = 0.60      # reject the whole rebalance if total traded notional > 60% of NAV

# Execution ladder (Monday)
LADDER_PASSIVE_WAIT_SEC = 180        # stage 1 wait before escalating to MIDPRICE
LADDER_MIDPRICE_WAIT_SEC = 120       # stage 2 wait before terminal cross
LADDER_CANCEL_GRACE_SEC = 3          # settle wait after cancels (allow cancel ACK to propagate)
LADDER_TERMINAL_CROSS = True         # stage 3: cross the spread near close to guarantee completion
ORDERS_DIR = AUDIT_DIR / "orders"    # per-run order/fill audit logs
# External deposits/withdrawals by landing date ({"YYYY-MM-DD": amount}); the
# publisher auto-detects and records them, hand-editable for corrections.
# Gitignored with the rest of the audit dir — real account flows stay private.
CAPITAL_FLOWS_PATH = AUDIT_DIR / "capital_flows.json"

# ---------------------------------------------------------------------------
# Dashboard publisher (read-only push to Supabase). Outbound-only; the VPS opens
# no inbound ports. SUPABASE_SERVICE_KEY (write) lives ONLY in the VPS .env.
# ---------------------------------------------------------------------------
SUPABASE_URL = get_env("SUPABASE_URL", default="")
SUPABASE_SERVICE_KEY = get_env("SUPABASE_SERVICE_KEY", default="")

# Publish only during US market hours (the systemd timer also gates, this is a guard).
PUBLISH_MARKET_OPEN = "09:30"   # America/New_York
PUBLISH_MARKET_CLOSE = "16:00"  # America/New_York
