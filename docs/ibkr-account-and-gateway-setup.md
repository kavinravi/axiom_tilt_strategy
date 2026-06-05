# IBKR Account + Gateway Setup Guide

**For:** wiring the live trading system (`trading/`) to Interactive Brokers.
**Status as of 2026-06-03:** you have a **funded live IBKR account**; **no paper account in use yet**; **IB Gateway not installed**. This guide gets you from there to "the code can place orders." Do the steps in order; each is quick.

> **Why this matters:** the trading system talks to IBKR through a local **IB Gateway** (or TWS) over a socket — there is no cloud REST API. Nothing in `trading/` can read positions or place orders until a Gateway is running and logged in.

---

## Step 1 — Activate your free paper account (5 min)

Every funded IBKR account comes with a **free paper-trading account** that mirrors your real account's market data and permissions. We validate the whole pipeline there before a single real order.

1. Log in to **Client Portal** (interactivebrokers.com → Login).
2. Top-right user menu → **Settings → Account Settings → Paper Trading Account** → **Yes, create**.
3. You'll get a **separate paper username/password** (often your live username + a suffix). Save them.
4. Paper market data: in Settings, enable "share live market data subscriptions with the paper account" so the paper account sees real quotes.

You now have two logins: **live** and **paper**. We use paper first.

## Step 2 — Install IB Gateway (10 min)

IB Gateway is the lightweight, headless-friendly API endpoint (TWS works too but is heavier; Gateway is what you'll run on the VPS later).

1. Download **IB Gateway — latest** from interactivebrokers.com → Trading → Platforms → IB Gateway.
2. Install. Launch it. Choose **IB API** (not FIX).
3. Log in with the **paper** credentials first.
4. Leave it running. Gateway must be **up and logged in** whenever the system trades.

## Step 3 — Configure the API (5 min)

In IB Gateway: **Configure → Settings → API → Settings**:

- ☑ **Enable ActiveX and Socket Clients**
- ☐ **Read-Only API** — must be **unchecked** to place orders (check it now while testing reads only; uncheck when you're ready to trade).
- **Socket port** — note it; defaults:
  | | Live | Paper |
  |---|---|---|
  | **IB Gateway** | 4001 | 4002 |
  | **TWS** | 7496 | 7497 |
- ☑ **Allow connections from localhost only** — fine if the code runs on the same machine as Gateway. If the code runs elsewhere (e.g., WSL → Gateway on Windows), uncheck this and add the client IP under **Trusted IPs**.
- **Master API client ID** — leave blank; our code uses a configurable `clientId`.
- ☑ **Download open orders on connection** (recommended — lets the system see pre-existing orders).

Apply, then restart Gateway so settings take effect.

## Step 4 — WSL networking note (you're on WSL2)

Two options for where Gateway runs relative to the Python code:

- **Gateway on the Windows host, code in WSL (typical now):** WSL2 reaches the Windows host via the host IP. Get it inside WSL with `ip route show | grep -i default | awk '{print $3}'` (the gateway/host IP). Set that as `IBKR_HOST` and **uncheck "localhost only"** + add the WSL IP to **Trusted IPs** in Gateway. Windows Firewall may need an inbound allow for the port.
- **Gateway inside WSL (Linux):** workable but needs a headless display (Xvfb) — more setup. For production we'll run Gateway on an always-on VPS anyway (see `docs/vps-deployment-setup.md`), so this is only for local testing.

Quick reachability test from WSL once Gateway is up:
`nc -vz <IBKR_HOST> 4002` (paper Gateway) → should connect.

## Step 5 — Install ib_async + test the connection (5 min)

```bash
cd ~/CodingStuff/axiom_tilt_strategy
pip install ib_async        # also added to requirements.txt by Plan 3
```

Minimal read-only connection test (paper):

```python
from ib_async import IB
ib = IB()
ib.connect("127.0.0.1", 4002, clientId=11)   # use your IBKR_HOST + paper port
print("connected:", ib.isConnected())
print("accounts:", ib.managedAccounts())
print("net liq:", [v for v in ib.accountSummary() if v.tag == "NetLiquidation"])
print("positions:", ib.positions())
ib.disconnect()
```

If this prints your paper NAV and positions, the plumbing works.

## Step 6 — 2FA reality (important for automation)

- **Live** accounts require **two-factor auth** (IBKR Mobile / IB Key) at login. For **manual/supervised** runs you approve the push on your phone — fine.
- **Unattended** login (the VPS scheduler) is the hard part: IB Gateway forces a **daily session reset**, and 2FA can't be tapped by a script. The standard solutions (IBC / IBeam) automate restart + login; see `docs/vps-deployment-setup.md`. **Paper** logins are simpler (often no 2FA), which is another reason we validate on paper first.

## How this maps to the code

Plan 3 adds these to `trading/config.py`:

```python
EXECUTION_MODE = "dryrun"   # dryrun | paper | live
IBKR_HOST = "127.0.0.1"     # or the Windows-host IP from Step 4
IBKR_PORT = 4002            # 4002 paper / 4001 live (Gateway)
IBKR_CLIENT_ID = 11
```

Rollout (from the spec): **dryrun** (no connection, already works) → set `EXECUTION_MODE="paper"` once Steps 1–5 pass → **supervised manual first live run** (`live`, confirm-before-submit) → enable the automated scheduler.

## Your morning checklist
- [ ] Activate paper account (Step 1)
- [ ] Install + log into IB Gateway with **paper** creds (Step 2)
- [ ] Configure API settings, note the port (Step 3)
- [ ] Confirm WSL can reach Gateway (Step 4) — or decide to go straight to a VPS
- [ ] `pip install ib_async`, run the connection test (Step 5)
- [ ] Tell me the working `host:port` and whether you're testing local-WSL or on a VPS — I'll wire `trading/config.py` and run the paper rebalance.
