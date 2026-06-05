# Dashboard frontend (Plan 2) — design

**Date:** 2026-06-04
**Status:** Approved (design); ready for implementation planning
**Author:** brainstorming session (kavinravi)
**Related:** `2026-06-03-remote-dashboard-design.md` (the whole-dashboard design; this
refines §6–§7, the frontend), `trading/publish/` (Plan 1 publisher — the read contract).

---

## 1. Context & scope

Plan 1 (the Supabase publisher) is done. This is **Plan 2: the Next.js/Vercel
frontend** that reads the publisher's tables and renders the three tabs. The
architecture (push model, read-only, Supabase-as-contract) is already approved in the
2026-06-03 design; this document fixes the concrete frontend decisions.

**Decided this session:**
- **Real data only, polished empty states.** No demo/seed data. Because the strategy
  has not gone live (one target-weights file `2026-05-29.json`, no executions, no
  equity history), the empty states ARE the first thing viewers see and must look
  intentional — not blank panels or zeros.
- **All three tabs** in v1 (Now / Holdings / History).
- **Single shared password** auth (owner texts the password to friends/family).

**Non-goals** (inherited from the 2026-06-03 design): no control path, no real-time
streaming, no multi-account. Additionally out of scope here: demo/seed data, Google
allow-list auth (documented as a drop-in upgrade, not built now).

## 2. Stack

- **Next.js (App Router) + TypeScript + Tailwind CSS**, mobile-first (primary use is a
  quick phone check).
- **Recharts** for the equity curve, concentration bars, and persistence heatmap.
- **`dashboard/` subdirectory** of this repo; Vercel "root directory" = `dashboard/`,
  auto-deploys from the repo. Vercel Hobby + Supabase free tier = $0.

## 3. Data access — server-side only

All Supabase reads happen **server-side** (Server Components / Route Handlers) using the
**service-role key**. The key never reaches the browser; the client only ever calls our
own Next.js routes. This is the refinement of 2026-06-03 §7: no anon/read key is shipped
to the client, so the RLS-enabled-no-policies tables stay fully sealed and there is no
public read path. (Supersedes the "Vercel env holds a read-only Supabase key" line in
the older spec.)

**Swappable data source.** A `DataSource` interface with two implementations:
- `SupabaseSource` — production reads via the service-role key.
- `FixtureSource` — reads seeded JSON from `dashboard/fixtures/`.

An env flag `DASHBOARD_DATA_SOURCE=supabase|fixture` selects one. This lets the entire
UI — every populated **and** empty state — be developed and Playwright-tested without the
live account, and matches the "local fixture mode" called for in 2026-06-03 §9.

**Freshness.** Open tabs poll their route every ~60s and render "as of HH:MM ET" from
`snapshot.asof`. A missing or stale snapshot is stated honestly in the header rather than
rendered as zeros.

## 4. Tabs (each maps to publisher tables)

### Now ← `snapshot` + `equity_curve`
- Hero stats: portfolio value (NAV), today's $/% P&L, total return vs SPY, position count.
- Equity curve: strategy vs SPY, range pills 1W / 1M / 3M / All.
- Risk: current & max drawdown, Sharpe-to-date, annualized vol (from `snapshot.risk`).
- Regime call: `k_probs` bar ({10/20/30/50}) + `regime_features` (null-tolerant — features
  are published as null until the weights pipeline emits them; `k_probs` is present).
- **Empty state:** "Performance builds forward from go-live"; show the regime call if a
  snapshot/weekly row carries `k_probs`, otherwise a clean "awaiting go-live" hero.

### Holdings ← `holdings`
- 50 names as concentration bars sorted by actual weight, with the **10% IPS cap** drawn
  as a reference line. Each row: target-vs-actual weight; expand for shares / price /
  market value.
- A **"no quote" flag**: a held name whose intraday quote failed publishes a zero/short
  bar; surface a small badge so an understated `invested_pct` reads as degraded-not-wrong
  (the known follow-up from Plan 1).
- **Empty state:** "No live positions yet — the target portfolio is shown under History."

### History ← `weekly_portfolio` + `executions`
- Friday picker → that week's full target portfolio (target weight + `k_probs`).
- Persistence heatmap: names × weeks, color = weight, gaps = not held.
- Turnover/churn: names added/dropped + % turnover (when ≥2 weeks exist).
- Execution quality: realized fill vs NBBO midpoint, `slippage_bps` signed
  (positive = worse than mid).
- **This is the one tab with real content now** (the backfilled 2026-05-29 week, after
  the user runs `python -m trading.publish.backfill`). Turnover/heatmap/executions
  degrade gracefully to "needs ≥2 weeks" / "no fills yet" until more data accrues.

## 5. Auth & security

- **Single shared password via Next.js middleware.** Unauthenticated requests redirect to
  `/login`; a correct password POST sets a **signed, httpOnly cookie**; middleware checks
  it on every route and on the data Route Handlers. One env var, `DASHBOARD_PASSWORD`.
- **Vercel env vars:** `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `DASHBOARD_PASSWORD`,
  `DASHBOARD_DATA_SOURCE`. No IBKR credential and no service key ever reach the browser.
- **Blast radius** (unchanged): the dashboard is strictly read-only and the VPS accepts no
  inbound connections, so a compromise of Vercel/Supabase exposes portfolio data but
  cannot move money or reach the trading box.

## 6. Project structure (sketch)

```
dashboard/
  app/
    login/            password form -> sets cookie
    (tabs)/now/       Now tab
    (tabs)/holdings/  Holdings tab
    (tabs)/history/   History tab
    api/              Route Handlers (server-side reads, polled by client)
  lib/
    datasource/       DataSource interface + SupabaseSource + FixtureSource
    format.ts         money / % / "as of ET" helpers
  components/         StatCard, EquityChart, ConcentrationBars, Heatmap, RegimeBar, ...
  middleware.ts       password gate
  fixtures/           seed JSON: populated + empty scenarios
  tests/              Playwright smoke tests
```

## 7. Testing

- **Playwright smoke test** (webapp-testing skill): with `DASHBOARD_DATA_SOURCE=fixture`,
  render each tab and assert key panels appear; include an **empty-state fixture** so the
  "builds from go-live" / "no positions yet" states are explicitly covered (they are what
  ships first).
- **Auth test:** unauthenticated request redirects to `/login`; correct password sets the
  cookie and grants access; wrong password is rejected.
- **DataSource unit tests:** `FixtureSource` returns the expected shapes; `SupabaseSource`
  is tested against a mocked client asserting it reads the right tables/columns.

## 8. Deployment

- Vercel project, root directory `dashboard/`, env vars per §5, auto-deploy from repo.
- The publisher (Plan 1) is unchanged. Before friends see real numbers the user must
  (from the Plan 1 follow-ups): re-run `schema.sql`, put `SUPABASE_URL` +
  service-role key in the VPS `.env`, and run the one-time backfill.

## 9. Future (deferred, unchanged from 2026-06-03)

- Google allow-list auth upgrade (Auth.js).
- Demo/seed data mode (the FixtureSource makes this a thin addition if ever wanted).
- Per-holding intraday day-change (needs a prior-price store).
- Control path — explicitly out of scope; would need a much larger security review.
