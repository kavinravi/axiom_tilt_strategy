# VPS Deployment Guide (always-on autonomous trading)

**Goal:** run the trading system unattended for years on a small cloud host, so the Friday-weights and Monday-execution jobs fire even when your laptop is off (you noted school resumes ~Sep 2026 and local uptime isn't guaranteed).

**The honest hard part is not the server — it's keeping IB Gateway logged in unattended through IBKR's daily session reset and 2FA.** This guide covers both, and the failure-alerting safety net for when it hiccups.

---

## 1. Provision the host

- **Provider:** any reputable VPS (Hetzner, DigitalOcean, AWS Lightsail, Vultr). ~$6–12/mo.
- **Specs:** 2 vCPU / 2–4 GB RAM / 20 GB disk. IB Gateway wants ~1 GB; the pipeline is light (the heavy lifting — model training — is rare and can run anywhere).
- **OS:** Ubuntu 24.04 LTS.
- **Region:** US-East is sensible (closest to IBKR's US servers), but latency is irrelevant for a midpoint order an hour before close — pick what's cheap/reliable.
- **Hardening:** SSH keys only (disable password login), `ufw` firewall allowing only SSH (22). **Never expose the Gateway API port (4001/4002) to the internet** — it's localhost-only on the box.

## 2. Headless IB Gateway: IBC or IBeam

IB Gateway is a GUI app; on a headless server you automate it. Two standard options:

- **IBeam (recommended)** — a Docker container that bundles Gateway + IBC, auto-logs-in from env-var credentials, auto-restarts on the daily reset, and exposes the API port to the host. Easiest path.
  ```bash
  docker run -d --name ibeam --restart unless-stopped \
    -e IBEAM_ACCOUNT=<paper_or_live_user> \
    -e IBEAM_PASSWORD=<password> \
    -e IBEAM_TRADING_MODE=paper \      # paper first; switch to live later
    -p 127.0.0.1:4002:4002 \           # bind to localhost ONLY
    voyz/ibeam
  ```
- **IBC** — installs IBC alongside a native Gateway; you run it under a process manager (systemd) with a virtual display (`xvfb`). More moving parts than IBeam but no Docker.

Either way, the API endpoint becomes `127.0.0.1:4002` (paper) / `4001` (live) on the VPS, exactly what `trading/config.py` expects.

## 3. The 2FA problem (read this before going live)

- IBKR forces a **daily Gateway restart** (~midnight account time). IBC/IBeam handle the re-login automatically **as long as 2FA doesn't prompt**.
- **Live accounts** generally enforce 2FA (IBKR Mobile). Unattended options, roughly in order of robustness:
  1. **Run on paper indefinitely for validation** (paper usually has no 2FA) — zero-touch, but not real money.
  2. **IBKR's 2FA arrangements for API users** — IBKR offers configurations to reduce/remove repeated 2FA for automated Gateway logins; check current IBKR policy for your account tier (it changes). Some users dedicate a **second login** with appropriate settings.
  3. **Accept a weekly manual touch** — since the strategy only acts Fri/Mon, a once-a-week phone approval is tolerable if needed.
- **Plan accordingly:** the system is built to **degrade gracefully** — if Gateway is down/unauthenticated when a job fires, the job fails *loudly* and alerts you (Section 5) rather than trading on stale state.

## 4. Deploy the code

```bash
git clone <your repo> ~/axiom_tilt_strategy && cd ~/axiom_tilt_strategy
python3 -m venv .venv && source .venv/bin/activate
pip install -e .            # or pip install -r requirements.txt
cp .env .env               # put NASDAQ_DATA_LINK_API_KEY + ALPHAVANTAGE_API_KEY here
# trading/models/k_selector.txt is committed, so the model ships with the repo
python -m trading.run weights        # dryrun smoke: prints this week's target weights
```

## 5. Schedule the jobs (cron, tz-aware)

The decision/execution cadence (from the spec): **weights frozen Friday**, **execution Monday 15:00 America/New_York** (= 12:00 PT = 1h before the 16:00 ET close). Use cron with an explicit timezone so DST is automatic.

`crontab -e` (set `CRON_TZ` so the times track ET through DST):
```cron
CRON_TZ=America/New_York
# Friday 16:30 ET — compute + freeze this week's weights (after Sharadar EOD lands)
30 16 * * 5  cd ~/axiom_tilt_strategy && .venv/bin/python -m trading.run weights   >> ~/trading.log 2>&1
# Monday 15:00 ET — execute the rebalance toward the frozen weights (Plan 3/4)
0 15 * * 1   cd ~/axiom_tilt_strategy && .venv/bin/python -m trading.run rebalance  >> ~/trading.log 2>&1
```
(`rebalance` lands in Plan 3/4. The `weights` job already works today.) systemd timers are an equivalent, more observable alternative.

## 6. Failure alerting (the safety net)

Plan 4 adds `trading/alerts.py`: if a job fails — Gateway down, login lapsed, connection refused, or a safety-check abort — it sends you an email/push so a missed rebalance becomes "you get pinged to intervene," not a silent miss. Because the strategy trades only twice a week, an occasional manual intervention is acceptable. Configure the alert channel (email SMTP or a push service token) in `.env`.

## 7. Going-live order of operations

1. VPS up, IBeam running on **paper**, `python -m trading.run weights` works.
2. Validate a full **paper rebalance** (Plan 3) end-to-end: reads positions/NAV, places `MIDPRICE`/limit orders, fills, audit log.
3. Flip IBeam + `trading/config.py` to **live**; do **one supervised manual** `rebalance --confirm` (you watch it place the first real orders).
4. Only then enable the **Monday cron** on live.

## Decisions to confirm with me later
- Provider/region choice (I'll tailor exact setup commands).
- IBeam vs IBC.
- Alert channel (email vs push) + credentials.
- Whether to run an extended paper period or go live after one clean paper rebalance (you indicated: straight to live after validation).
