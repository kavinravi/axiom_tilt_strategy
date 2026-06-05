# Dashboard publisher (read-only push → Supabase)

Snapshots the live IBKR portfolio + audit files into Supabase so the Vercel
frontend (Plan 2) can render it. **Outbound-only** — the VPS opens no inbound ports.

## Modules
- `metrics.py` — pure functions (holdings, day P&L, returns, risk, turnover, execution quality).
- `store.py` — `SupabaseStore`: idempotent writes over an injectable client.
- `publish.py` — `publish_once` orchestrator + `main()` CLI (market-hours-guarded).
- `backfill.py` — one-shot seed of weekly portfolios + executions from existing audit files.
- `schema.sql` — apply once in the Supabase SQL editor (creates tables + enables RLS).

## One-time setup
1. Create a Supabase project; run `schema.sql` in its SQL editor.
2. Put the service-role key in the VPS `.env` (NEVER in the frontend):
   ```
   SUPABASE_URL=https://<project>.supabase.co
   SUPABASE_SERVICE_KEY=<service-role key>
   ```
3. Seed history once: `python -m trading.publish.backfill`

## Run a publish
`python -m trading.publish` — skips quietly outside US market hours; needs IB Gateway up.

## systemd timer (on the VPS)
`/etc/systemd/system/dashboard-publish.service`:
```ini
[Unit]
Description=Dashboard publish (read-only snapshot to Supabase)
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/home/<user>/axiom_tilt_strategy
EnvironmentFile=/home/<user>/axiom_tilt_strategy/.env
ExecStart=/home/<user>/axiom_tilt_strategy/.venv/bin/python -m trading.publish
```
`/etc/systemd/system/dashboard-publish.timer`:
```ini
[Unit]
Description=Run dashboard publish every 20 min on weekdays

[Timer]
OnCalendar=Mon..Fri *-*-* 09,10,11,12,13,14,15,16:00/20 America/New_York
Persistent=false

[Install]
WantedBy=timers.target
```
Enable: `sudo systemctl enable --now dashboard-publish.timer`
(The `main()` market-hours guard is the backstop; the timer is the primary gate.)
