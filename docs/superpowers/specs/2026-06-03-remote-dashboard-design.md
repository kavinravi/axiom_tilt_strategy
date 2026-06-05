# Remote monitoring dashboard — design

**Date:** 2026-06-03
**Status:** Approved (design); ready for implementation planning
**Author:** brainstorming session (kavinravi)
**Related:** `2026-06-02-live-trading-system-design.md` (the live trading system this monitors), `docs/vps-deployment-setup.md`

---

## 1. Motivation

The k-ensemble strategy is going live on Interactive Brokers, deployed unattended on
an always-on VPS (weights frozen Friday, executed Monday 15:00 ET). Once school
resumes (~Sep 2026) local uptime isn't guaranteed and the system runs hands-off.

We want a **remote, read-only status board** to answer, from any device:

1. **How is the portfolio doing *now*?** — current value, today's P&L, return vs SPY.
2. **What are the 50 holdings and at what concentrations?** — actual weights vs the
   10% IPS cap, target vs realized.
3. **What did past weekly portfolios look like?** — week-by-week archive to see which
   names persist and how concentrations drift.

It should be prettier than a raw Streamlit page and reachable from a phone.

## 2. Goals / non-goals

**Goals**
- Read-only monitoring: trades/fills, live positions & NAV, performance over time.
- Intraday freshness: a roughly-current NAV during market hours (not true real-time).
- Keep the live-money VPS with **zero inbound network exposure**.
- Cheap (target $0 on free tiers) and low-maintenance.

**Non-goals (explicitly out of scope)**
- **No control actions.** The dashboard never approves, pauses, or triggers trades.
  All execution stays on the VPS via the existing `confirm`-before-submit path. This
  is the single most important scope boundary — it lets the trading box stay sealed.
- No true real-time streaming (websockets / live tick feed).
- No multi-portfolio / multi-account support (one account, one strategy).

## 3. Architecture

Push model: the VPS publishes snapshots **outbound** to a managed datastore; a Vercel
frontend reads from the datastore. Nothing on the internet can reach into the VPS.

```
 VPS (live-money box)                      Cloud                         Client
 ┌───────────────────────┐                                          
 │ IB Gateway (localhost)│                                          
 │ trading/audit/*.json  │   outbound HTTPS    ┌─────────────┐  read  ┌──────────────┐
 │ trading/publish/  ────┼────────────────────▶│  Supabase   │◀───────│ Vercel       │
 │  (systemd timer)      │   (upsert only)     │  (Postgres) │        │ Next.js app  │
 └───────────────────────┘                     └─────────────┘        │ (behind login)│
   no inbound ports                                                   └──────┬───────┘
   Gateway API stays localhost-only                                         │ login
                                                                      📱 you, anywhere
```

Three pieces, each independently understandable and testable:

- **Publisher** (`trading/publish/`) — reads broker + audit files, computes a snapshot,
  upserts to Supabase. Pure data-out; knows nothing about the frontend.
- **Datastore** (Supabase Postgres) — the contract between publisher and frontend. A
  handful of small tables.
- **Frontend** (`dashboard/`, Next.js on Vercel) — reads the datastore, renders three
  tabs. Knows nothing about IBKR or the VPS.

## 4. Publisher (`trading/publish/`)

A new module in the existing codebase, invoked by a systemd timer / cron on the VPS.

**Inputs**
- `IBKRBroker` (reused from `trading/broker/ibkr.py`): `get_nav()`, `get_positions()`,
  `get_quote(ticker)`.
- Local audit files: `trading/audit/weights/<friday>.json` (target weights + `k_probs`),
  `trading/audit/orders/<asof>.json` (orders, fills, ladder stages, pre/post positions).

**Computes**
- **Snapshot:** NAV; today's P&L ($ and %); total return since inception and the SPY
  return over the same window; number of positions; invested %.
- **Holdings:** per ticker — shares, last price, market value, **actual weight**
  (`shares × price / NAV`), **target weight** (from the latest frozen weights), day change.
- **Regime call:** this week's `k_probs` ({10/20/30/50}) + the regime features that drove
  the model (VIX, 10Y yield, term spread, trailing SPY ret/vol).
- **Risk stats:** current & max drawdown, Sharpe-to-date, annualized vol — from the
  accumulated equity curve.
- **Turnover/churn:** names added/dropped vs the prior Friday + % turnover.
- **Execution quality:** per-rebalance realized fill price vs NBBO midpoint, from the
  ladder audit (`realized_price`, `midpoint_at_fill`).

**Side effect:** appends one `(date, nav, spy_close)` point to `equity_curve` each run
(idempotent per date — upsert on date so intraday re-runs overwrite the day's point;
the close-time run is authoritative).

**Output:** upserts to Supabase using the Supabase Python client and the **service-role
key** (stored only in the VPS `.env`). All network egress is outbound HTTPS to Supabase;
no ports are opened on the VPS.

**Cadence (systemd timer on the VPS):**
- Every ~20 min, Mon–Fri 09:30–16:00 America/New_York (intraday refresh).
- Once right after each Monday rebalance completes (fresh holdings + execution quality).
- Reuses the IB Gateway session already up for trading. If the Gateway is
  down/unauthenticated, the publish fails loudly and logs (consistent with the trading
  system's degrade-loudly posture); the dashboard simply shows a stale "as of" time
  rather than wrong data.

## 5. Datastore (Supabase Postgres)

| table | columns (sketch) | write pattern |
|---|---|---|
| `snapshot` | `id` (singleton), `asof`, `nav`, `day_pnl`, `day_pnl_pct`, `total_return`, `spy_return`, `n_positions`, `invested_pct`, `k_probs` (jsonb), `regime_features` (jsonb), `risk` (jsonb) | upsert single row |
| `equity_curve` | `date` (pk), `nav`, `spy_close` | upsert by date (append/overwrite the day) |
| `holdings` | `asof`, `ticker`, `shares`, `price`, `market_value`, `weight_actual`, `weight_target`, `day_change` | delete-all + insert each publish |
| `weekly_portfolio` | `asof_friday`, `ticker`, `target_weight`, `k_probs` (jsonb) | insert one set per rebalance Friday |
| `executions` | `asof`, `ticker`, `side`, `qty`, `realized_price`, `midpoint`, `slippage` | insert per rebalance |

Data volume is tiny (one portfolio, ~50 holdings, weekly history, daily NAV), so the
free tier is ample. SQL makes the "does NVDA persist week over week" query trivial.

**Keys / access:** the VPS holds the service-role (write) key. The frontend reads through
a restricted path (see §7). Row-level security can be permissive given the dashboard sits
behind its own login and the data is non-sensitive by the owner's assessment, but the
service-role key is never shipped to the browser.

## 6. Frontend (`dashboard/`, Next.js on Vercel)

A `dashboard/` subdirectory in this repo (Vercel "root directory" = `dashboard/`).
Reads Supabase; renders three tabs. Open tabs **poll every ~60s** and display
"as of HH:MM ET" so freshness is always honest.

- **Now** — hero stats (portfolio value, today's $/% P&L, total return vs SPY, position
  count) · equity curve (strategy vs SPY, range pills 1W/1M/3M/All) · risk stats
  (drawdown / Sharpe / vol) · **model's regime call** (k_probs bar + regime features).
- **Holdings** — the 50 names as concentration bars sorted by weight, with the **10% IPS
  cap** drawn as a reference line; each row shows target-vs-actual weight and day change;
  click to expand (shares, price, market value).
- **History** — week-by-week Friday "folders" (pick one → load that week's full portfolio)
  · **persistence heatmap** (names × weeks, color = weight, gaps = not held) · **turnover/
  churn** (added/dropped, % turnover) · **execution quality** (realized vs midpoint, to
  settle bid-first vs midpoint empirically).

A charting library (e.g. Recharts) renders the equity curve, concentration bars, and
heatmap. Styling is clean and mobile-first (the primary use is a quick phone check).

## 7. Security model

- **VPS:** outbound-only to Supabase. No dashboard ports opened. Gateway API stays
  localhost-only (unchanged from the trading deployment).
- **Secrets:** Supabase service-role (write) key only in the VPS `.env`. Vercel env holds
  a read-only Supabase key + the dashboard password. No IBKR credential ever leaves the VPS.
- **Auth:** a single password gate via Next.js middleware (one env var) — lowest friction
  for phone use. Documented drop-in upgrade: Auth.js with Google sign-in allow-listed to
  the owner's email (+ optionally his father's, who follows the strategy).
- **Blast radius:** because the dashboard is strictly read-only and the VPS accepts no
  inbound connections, a compromise of the Vercel app or Supabase exposes (private)
  portfolio data but **cannot move money or reach the trading box.**

## 8. Backfill & history semantics

- **`weekly_portfolio` and `executions` backfill immediately** from existing audit files
  (`trading/audit/weights/*.json`, `orders/*.json`). One Friday exists today (2026-05-29);
  each rebalance adds one.
- **`equity_curve` is forward-only.** There is no persisted historical NAV, so the
  performance curve builds from go-live. The UI labels it "builds forward from go-live"
  rather than faking a backfilled curve. SPY benchmark is drawn alongside from go-live.

## 9. Testing

- **Publisher:** unit-tested against `DryRunBroker` + fixture audit JSON (matches the
  existing injectable-broker test style in `tests/trading/`). The Supabase client is
  mocked to assert upsert payload shape per table. Snapshot/risk/turnover/execution-quality
  math is tested on fixtures with known expected values.
- **Frontend:** a Playwright smoke test (the `webapp-testing` skill) renders each tab
  against fixture data and asserts the key panels appear. A **local fixture mode** lets the
  frontend run off seeded JSON / a local Supabase so the UI can be developed without the
  live account.
- **Idempotency:** test that re-running the publisher for the same date overwrites (not
  duplicates) the `equity_curve` point and the `holdings`/`snapshot` rows.

## 10. Deployment & cost

- **Repo layout:** monorepo — publisher in `trading/publish/`, frontend in `dashboard/`.
- **Publisher:** systemd timer on the VPS (alongside the existing trading timers), `.env`
  carries the Supabase service key.
- **Frontend:** Vercel project with root directory `dashboard/`, env vars for the read-only
  Supabase key + dashboard password. Auto-deploys from the repo.
- **Cost:** Supabase free tier + Vercel Hobby = **$0** for personal use. (Vercel Hobby is
  intended for non-commercial use; a personal monitoring tool for one's own account
  qualifies. Vercel Pro $20/mo if ever needed.)

## 11. Future (explicitly deferred)

- A **control path** (approve/pause/trigger from the dashboard) — deliberately out of scope
  now; would require an authenticated write path into the VPS and a much larger security
  review. The read-only push architecture can coexist with one later without rework.
- Google-allow-list auth upgrade.
- Optional alerting (push/email) on rebalance completion or publish failure — the trading
  system already owns failure alerting; the dashboard could surface it but doesn't own it.
