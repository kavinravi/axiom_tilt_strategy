# VPS Deployment — autonomous weekly trading

Authoritative deploy runbook. Goal: the Friday decision, Monday execution, and
dashboard publishing all run unattended on an always-on host, with **one weekly 2FA
tap on your phone**. (This supersedes the IBeam approach in
`docs/vps-deployment-setup.md` — see the correction below.)

## Why IBC + IB Gateway (not IBeam)

Our broker uses `ib_async`, the **socket TWS API** (ports 4001 live / 4002 paper).
IBeam automates the *Client Portal Web API* — a different REST interface, wrong stack
for us. We use **IB Gateway + IBC** via the `gnzsnz/ib-gateway-docker` image, which
exposes the 4001/4002 socket and runs IBC's auto-restart.

## The 2FA reality (read first)

- IBKR invalidates session tokens **Sunday 01:00 ET**. The first login after that
  needs 2FA. With IB Gateway **auto-restart** (IBC default), the daily restarts the
  rest of the week reuse the token with **no 2FA**. So it's **one 2FA per week**, not
  per login.
- The push goes to whichever phone holds **IBKR Mobile / IB Key for that username**.
  Today that's your dad's phone. Fix: the account holder creates a **dedicated second
  username** for the VPS (Client Portal → Settings → Users & Access Rights), then you
  register **IBKR Mobile / IB Key on YOUR phone** for that username. One-time setup;
  the weekly tap is then yours.
- Use that username ONLY on the VPS. A competing login (Client Portal / TWS / mobile)
  with the same username kills the auto-restart session.
- Zero-touch alternatives: run **paper** (no 2FA) for validation, or IBeam's
  SMS-scraping 2FA handlers (fragile — not for real money). The weekly tap is the
  robust choice, and it lands Sunday/Monday — right before the only day we trade.

## 1. Provision
- Ubuntu 24.04 LTS, 2 vCPU / 4 GB / 20 GB, any reputable provider (~$6–12/mo).
- SSH keys only; `ufw allow 22` and nothing else. Never expose 4001/4002.
- Install Docker + the compose plugin.

## 2. IB Gateway (IBC) container
```bash
cd ~/axiom_tilt_strategy/deploy/ib-gateway
cp ib-gateway.env.example ib-gateway.env    # fill in the dedicated username/password
docker compose up -d
docker compose logs -f                        # watch it log in; approve the weekly 2FA on your phone
```
The API is then on `127.0.0.1:4002` (paper) / `4001` (live).

## 3. Deploy the code
```bash
git clone <repo> ~/axiom_tilt_strategy && cd ~/axiom_tilt_strategy
python3 -m venv .venv && .venv/bin/pip install -e .
cp /path/to/.env .env      # data keys + SUPABASE_* + ALERT_* ; set IBKR_HOST=127.0.0.1
.venv/bin/python -m trading.run weights       # smoke test (no Gateway needed)
```

## 4. Schedule (systemd user timers)
```bash
loginctl enable-linger "$USER"                # let user timers run while logged out
mkdir -p ~/.config/systemd/user
cp deploy/systemd/* ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now axiom-weights.timer axiom-rebalance.timer axiom-publish.timer
systemctl --user list-timers
```
Times are ET, DST-safe: **weights Fri 18:00**, **rebalance Mon 15:00**, **publish
every 20 min during market hours**. The rebalance unit ships as `--mode paper` —
flip it to live only in step 5.

## 5. Go-live order
1. Container on **paper**, `weights` works, one full **paper rebalance** is clean.
2. Container `TRADING_MODE=live` (host `IBKR_PORT=4001`); do **one supervised**
   `rebalance --mode live --confirm` you watch by hand.
3. Edit `axiom-rebalance.service` → `--mode live`; `systemctl --user daemon-reload`.
4. Done — it trades itself; failures alert you.

## Alerts
Set `ALERT_WEBHOOK_URL` (ntfy/Slack/Discord/Telegram) and/or `ALERT_EMAIL_*` in
`.env`. Every job has `OnFailure=axiom-alert@…`, so a crash, a missed Gateway login,
or a safety abort pings you instead of failing silently.

## Status / not-yet-done
- `trading/alerts.py` is in place but still needs a unit test + a real end-to-end
  send before you rely on it for live money.
- The `gnzsnz/ib-gateway-docker` env var names and socat port mapping shift between
  image versions — verify `ib-gateway.env.example` and the compose port lines against
  the image's current README on first deploy.
