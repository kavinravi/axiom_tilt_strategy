# Operating Instructions (live money, manual)

Weekly runbook until the VPS takes over (see `deploy/README.md`). Run **every command
from the repo root**:

    /home/kavin-ravi/CodingStuff/axiom_tilt_strategy

Cadence: **decide Friday after close**, **trade Monday 12:00 PT / 15:00 ET** (1h before close).

---

## Friday — freeze the decision + publish (no Gateway needed)

Run after the US close (evening PT, once Sharadar's EOD data has posted):

```bash
python -m trading.run weights --asof 2026-06-05     # fetch + freeze target weights
python -m trading.publish.backfill                  # push target to the dashboard
```

Expect `weight_sum : 1.000000` and `✓ sanity checks passed`. Each week, change the
date to that Friday (or omit `--asof` if you run it on Friday itself).

---

## Monday — execute with REAL MONEY

**0. Point `.env` at the live account (one-time):**
```
IBKR_PORT=4001      # 4001 = live  (4002 was paper).  IBKR_HOST=172.18.0.1 stays.
```

**1. Start IB Gateway on Windows** with your **LIVE** account, log in (approve 2FA),
leave it running. Live API port is 4001 (confirm in Gateway → Configuration → API →
Settings). Live data is real-time, so the ladder prices off real bid/ask.

**2. Place the orders — live, supervised.** Be at the terminal at 12:00 PT:
```bash
python -m trading.run rebalance --asof 2026-06-05 --mode live --confirm
```
It prints the full order table (real dollar sizes against your live NAV) and waits.
**Read it, then type `yes`.** Orders run the passive → midprice → cross ladder and
finish before the close; an audit lands in `trading/audit/orders/`.

> Abort hatch: `touch trading/KILL_SWITCH` blocks all order placement. Delete it to re-enable.

**3. Publish live state to the dashboard** (Gateway up, market open):
```bash
python -m trading.publish
```

---

## Quick reference — all from repo root

| When | Command | Gateway |
|------|---------|---------|
| Fri PM | `python -m trading.run weights --asof <fri>` | no |
| Fri PM | `python -m trading.publish.backfill` | no |
| Mon 12 PT | `python -m trading.run rebalance --asof <fri> --mode live --confirm` | **live, 4001** |
| Mon (open) | `python -m trading.publish` | yes |
