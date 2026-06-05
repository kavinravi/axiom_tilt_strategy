# Axiom Tilt — dashboard

Read-only Next.js monitor for the live k-ensemble strategy. Reads the Supabase
tables written by `trading/publish/` (the publisher). Server-side reads only;
the service key never reaches the browser. Single shared-password gate.

## Local dev (fixtures — no live account needed)

```bash
cd dashboard
cp .env.example .env.local   # DASHBOARD_DATA_SOURCE=fixture, set DASHBOARD_PASSWORD
npm install
npm run dev                  # http://localhost:3000
```

Append `?scenario=empty` to any tab to preview the pre-go-live empty states.

## Tests

```bash
npm run test       # vitest (lib + datasource + fixtures)
npm run test:e2e   # playwright (tab + auth smoke, fixture data)
```

## Deploy to Vercel

1. New Vercel project from this repo; **Root Directory = `dashboard/`**.
2. Environment variables:
   - `DASHBOARD_DATA_SOURCE=supabase`
   - `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` (service-role key)
   - `DASHBOARD_PASSWORD` (share with viewers)
3. Deploy. Vercel auto-deploys on push.

Cost: Vercel Hobby + Supabase free tier = $0.
