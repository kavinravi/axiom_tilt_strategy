# Dashboard Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `dashboard/` Next.js app that reads the Supabase publisher tables and renders the Now / Holdings / History tabs behind a single shared-password gate, deployable to Vercel.

**Architecture:** App Router app in `dashboard/`. All Supabase reads happen server-side (Route Handlers) via a swappable `DataSource` (Supabase in prod, in-memory fixtures in dev/test) using the service-role key — never shipped to the browser. Client tab pages poll their Route Handler every 60s and show "as of … ET". A password middleware gates every route. Real-data-only with polished empty states.

**Tech Stack:** Next.js (App Router) + TypeScript + Tailwind CSS + Recharts; `@supabase/supabase-js`; Vitest (lib unit tests) + Playwright (tab smoke tests).

---

## File Structure

```
dashboard/
  package.json, tsconfig.json, next.config.mjs, postcss.config.mjs,
  tailwind.config.ts, vitest.config.ts, playwright.config.ts, .env.example, README.md
  middleware.ts                         password gate (edge)
  app/
    globals.css
    layout.tsx                          root layout (Tailwind, font)
    page.tsx                            redirect "/" -> "/now"
    login/page.tsx                      password form
    (dash)/layout.tsx                   tab nav + shell
    (dash)/now/page.tsx                 Now tab (client, polls /api/now)
    (dash)/holdings/page.tsx            Holdings tab (polls /api/holdings)
    (dash)/history/page.tsx             History tab (polls /api/history)
    api/login/route.ts                  POST password -> set cookie
    api/now/route.ts                    snapshot + equity curve JSON
    api/holdings/route.ts               holdings JSON
    api/history/route.ts                weekly + executions JSON
  lib/
    types.ts                            row types shared with publisher contract
    auth.ts                             HMAC token sign/verify (edge-safe)
    format.ts                           money / pct / bps / "as of ET"
    usePolling.ts                       client hook: fetch + 60s poll
    datasource/
      index.ts                          DataSource interface + getDataSource()
      fixture.ts                        FixtureSource
      supabase.ts                       SupabaseSource
  fixtures/
    populated.ts                        full sample dataset
    empty.ts                            pre-go-live dataset (one weekly only)
  components/
    StatCard.tsx, EquityChart.tsx, RiskCard.tsx, RegimeBar.tsx,
    ConcentrationBars.tsx, HoldingRow.tsx, WeekPicker.tsx,
    PersistenceHeatmap.tsx, TurnoverCard.tsx, ExecQualityTable.tsx,
    Empty.tsx, AsOf.tsx
  tests/
    now.spec.ts, holdings.spec.ts, history.spec.ts, auth.spec.ts
```

All commands below are run from `dashboard/` unless noted.

---

### Task 1: Scaffold the Next.js app + toolchain

**Files:**
- Create: `dashboard/package.json`, `dashboard/tsconfig.json`, `dashboard/next.config.mjs`, `dashboard/postcss.config.mjs`, `dashboard/tailwind.config.ts`, `dashboard/app/globals.css`, `dashboard/app/layout.tsx`, `dashboard/app/page.tsx`, `dashboard/.gitignore`, `dashboard/.env.example`

- [ ] **Step 1: Create `dashboard/package.json`**

```json
{
  "name": "axiom-dashboard",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "test": "vitest run",
    "test:e2e": "playwright test"
  },
  "dependencies": {
    "next": "15.3.0",
    "react": "19.1.0",
    "react-dom": "19.1.0",
    "recharts": "2.15.0",
    "@supabase/supabase-js": "2.45.0"
  },
  "devDependencies": {
    "typescript": "5.6.3",
    "@types/node": "22.7.0",
    "@types/react": "19.1.0",
    "@types/react-dom": "19.1.0",
    "tailwindcss": "3.4.13",
    "postcss": "8.4.47",
    "autoprefixer": "10.4.20",
    "vitest": "2.1.2",
    "@playwright/test": "1.48.0",
    "eslint": "8.57.1",
    "eslint-config-next": "15.3.0"
  }
}
```

- [ ] **Step 2: Create config files**

`dashboard/tsconfig.json`:
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["dom", "dom.iterable", "ES2022"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": { "@/*": ["./*"] }
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

`dashboard/next.config.mjs`:
```js
/** @type {import('next').NextConfig} */
const nextConfig = {};
export default nextConfig;
```

`dashboard/postcss.config.mjs`:
```js
export default { plugins: { tailwindcss: {}, autoprefixer: {} } };
```

`dashboard/tailwind.config.ts`:
```ts
import type { Config } from "tailwindcss";
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: { extend: {} },
  plugins: [],
};
export default config;
```

`dashboard/.gitignore`:
```
node_modules
.next
.env
.env.local
playwright-report
test-results
```

`dashboard/.env.example`:
```
# Pick the data source: "fixture" (dev/demo) or "supabase" (prod)
DASHBOARD_DATA_SOURCE=fixture
# Required when DASHBOARD_DATA_SOURCE=supabase
SUPABASE_URL=
SUPABASE_SERVICE_KEY=
# Shared password for the gate
DASHBOARD_PASSWORD=changeme
```

- [ ] **Step 3: Create the root shell**

`dashboard/app/globals.css`:
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
:root { color-scheme: dark; }
body { @apply bg-neutral-950 text-neutral-100 antialiased; }
```

`dashboard/app/layout.tsx`:
```tsx
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = { title: "Axiom Tilt", description: "Strategy monitor" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
```

`dashboard/app/page.tsx`:
```tsx
import { redirect } from "next/navigation";
export default function Home() { redirect("/now"); }
```

- [ ] **Step 4: Install and verify the app boots**

Run (from `dashboard/`):
```bash
npm install
npx playwright install chromium
npm run build
```
Expected: `npm run build` completes with "Compiled successfully" (the `/` route and a 404 for `/now` until Task 6 — build still succeeds).

- [ ] **Step 5: Commit**

```bash
git add dashboard/
git commit -m "feat(dashboard): scaffold Next.js app + Tailwind + toolchain"
```

---

### Task 2: Row types + dataset shape

**Files:**
- Create: `dashboard/lib/types.ts`
- Test: `dashboard/lib/types.test.ts`

- [ ] **Step 1: Write the failing test** (`dashboard/lib/types.test.ts`)

```ts
import { describe, it, expect } from "vitest";
import type { Dataset } from "./types";
import { emptyDataset } from "./types";

describe("emptyDataset", () => {
  it("is a fully-shaped, empty Dataset", () => {
    const d: Dataset = emptyDataset();
    expect(d.snapshot).toBeNull();
    expect(d.equityCurve).toEqual([]);
    expect(d.holdings).toEqual([]);
    expect(d.weekly).toEqual([]);
    expect(d.executions).toEqual([]);
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run lib/types.test.ts`
Expected: FAIL — cannot find module `./types`.

- [ ] **Step 3: Create `dashboard/lib/types.ts`**

```ts
// Mirrors the publisher's Supabase tables (trading/publish/schema.sql).
export interface Risk {
  current_drawdown: number | null;
  max_drawdown: number | null;
  sharpe: number | null;
  ann_vol: number | null;
}
export interface Turnover {
  added: string[];
  dropped: string[];
  turnover_frac: number;
}
export interface Snapshot {
  asof: string;
  nav: number;
  day_pnl: number | null;
  day_pnl_pct: number | null;
  total_return: number | null;
  spy_return: number | null;
  n_positions: number | null;
  invested_pct: number | null;
  k_probs: Record<string, number> | null;
  regime_features: Record<string, number> | null;
  risk: Risk | null;
  turnover: Turnover | null;
}
export interface EquityPoint {
  date: string;
  nav: number;
  spy_close: number | null;
}
export interface Holding {
  ticker: string;
  shares: number;
  price: number | null;
  market_value: number | null;
  weight_actual: number | null;
  weight_target: number | null;
}
export interface WeeklyRow {
  asof_friday: string;
  ticker: string;
  target_weight: number;
  k_probs: Record<string, number> | null;
}
export interface Execution {
  asof: string;
  ticker: string;
  side: string | null;
  qty: number | null;
  realized_price: number | null;
  midpoint: number | null;
  slippage_bps: number | null;
}
export interface Dataset {
  snapshot: Snapshot | null;
  equityCurve: EquityPoint[];
  holdings: Holding[];
  weekly: WeeklyRow[];
  executions: Execution[];
}
export function emptyDataset(): Dataset {
  return { snapshot: null, equityCurve: [], holdings: [], weekly: [], executions: [] };
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `npx vitest run lib/types.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/lib/types.ts dashboard/lib/types.test.ts
git commit -m "feat(dashboard): publisher-contract row types + Dataset"
```

---

### Task 3: Fixtures (populated + empty)

**Files:**
- Create: `dashboard/fixtures/populated.ts`, `dashboard/fixtures/empty.ts`
- Test: `dashboard/fixtures/fixtures.test.ts`

- [ ] **Step 1: Write the failing test** (`dashboard/fixtures/fixtures.test.ts`)

```ts
import { describe, it, expect } from "vitest";
import { populated } from "./populated";
import { empty } from "./empty";

describe("fixtures", () => {
  it("populated has a snapshot, equity history, holdings, and >=2 weeks", () => {
    expect(populated.snapshot).not.toBeNull();
    expect(populated.equityCurve.length).toBeGreaterThan(1);
    expect(populated.holdings.length).toBeGreaterThan(0);
    const fridays = new Set(populated.weekly.map((w) => w.asof_friday));
    expect(fridays.size).toBeGreaterThanOrEqual(2);
  });
  it("empty mirrors pre-go-live: no snapshot/holdings/equity, exactly one week", () => {
    expect(empty.snapshot).toBeNull();
    expect(empty.equityCurve).toEqual([]);
    expect(empty.holdings).toEqual([]);
    expect(empty.executions).toEqual([]);
    const fridays = new Set(empty.weekly.map((w) => w.asof_friday));
    expect(fridays.size).toBe(1);
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run fixtures/fixtures.test.ts`
Expected: FAIL — cannot find module `./populated`.

- [ ] **Step 3: Create `dashboard/fixtures/populated.ts`**

```ts
import type { Dataset } from "@/lib/types";

const KPROBS = { "10": 0.15, "20": 0.25, "30": 0.35, "50": 0.25 };

export const populated: Dataset = {
  snapshot: {
    asof: "2026-06-08T19:00:00+00:00",
    nav: 104230.55,
    day_pnl: 612.4,
    day_pnl_pct: 0.0059,
    total_return: 0.0423,
    spy_return: 0.0311,
    n_positions: 6,
    invested_pct: 0.97,
    k_probs: KPROBS,
    regime_features: { vix: 14.2, y10: 0.0431, term_spread: 0.0052, spy_ret: 0.012, spy_vol: 0.009 },
    risk: { current_drawdown: -0.008, max_drawdown: -0.021, sharpe: 1.42, ann_vol: 0.11 },
    turnover: { added: ["AVGO"], dropped: ["AMD"], turnover_frac: 0.18 },
  },
  equityCurve: [
    { date: "2026-06-01", nav: 100000, spy_close: 540.1 },
    { date: "2026-06-02", nav: 100850, spy_close: 542.0 },
    { date: "2026-06-03", nav: 101400, spy_close: 543.3 },
    { date: "2026-06-04", nav: 100920, spy_close: 541.7 },
    { date: "2026-06-05", nav: 103618, spy_close: 549.2 },
    { date: "2026-06-08", nav: 104230.55, spy_close: 550.9 },
  ],
  holdings: [
    { ticker: "NVDA", shares: 70, price: 132.4, market_value: 9268, weight_actual: 0.0889, weight_target: 0.10 },
    { ticker: "AAPL", shares: 40, price: 212.1, market_value: 8484, weight_actual: 0.0814, weight_target: 0.08 },
    { ticker: "MSFT", shares: 18, price: 441.0, market_value: 7938, weight_actual: 0.0762, weight_target: 0.075 },
    { ticker: "AVGO", shares: 45, price: 161.2, market_value: 7254, weight_actual: 0.0696, weight_target: 0.07 },
    { ticker: "GOOGL", shares: 38, price: 178.5, market_value: 6783, weight_actual: 0.0651, weight_target: 0.065 },
    { ticker: "ZZZQ", shares: 10, price: 0, market_value: 0, weight_actual: 0, weight_target: 0.04 },
  ],
  weekly: [
    { asof_friday: "2026-05-29", ticker: "NVDA", target_weight: 0.10, k_probs: KPROBS },
    { asof_friday: "2026-05-29", ticker: "AMD", target_weight: 0.08, k_probs: KPROBS },
    { asof_friday: "2026-05-29", ticker: "AAPL", target_weight: 0.08, k_probs: KPROBS },
    { asof_friday: "2026-06-05", ticker: "NVDA", target_weight: 0.10, k_probs: KPROBS },
    { asof_friday: "2026-06-05", ticker: "AVGO", target_weight: 0.07, k_probs: KPROBS },
    { asof_friday: "2026-06-05", ticker: "AAPL", target_weight: 0.08, k_probs: KPROBS },
  ],
  executions: [
    { asof: "2026-06-01", ticker: "NVDA", side: "BUY", qty: 70, realized_price: 132.45, midpoint: 132.40, slippage_bps: 3.8 },
    { asof: "2026-06-01", ticker: "AMD", side: "SELL", qty: 30, realized_price: 168.10, midpoint: 168.25, slippage_bps: 8.9 },
  ],
};
```

- [ ] **Step 4: Create `dashboard/fixtures/empty.ts`**

```ts
import type { Dataset } from "@/lib/types";

// Mirrors today's real state: one backfilled target week, nothing live yet.
const KPROBS = { "10": 0.2, "20": 0.3, "30": 0.3, "50": 0.2 };

export const empty: Dataset = {
  snapshot: null,
  equityCurve: [],
  holdings: [],
  weekly: [
    { asof_friday: "2026-05-29", ticker: "NVDA", target_weight: 0.10, k_probs: KPROBS },
    { asof_friday: "2026-05-29", ticker: "AAPL", target_weight: 0.08, k_probs: KPROBS },
    { asof_friday: "2026-05-29", ticker: "MSFT", target_weight: 0.075, k_probs: KPROBS },
  ],
  executions: [],
};
```

- [ ] **Step 5: Run it to verify it passes**

Run: `npx vitest run fixtures/fixtures.test.ts`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add dashboard/fixtures/
git commit -m "feat(dashboard): populated + empty fixtures"
```

---

### Task 4: DataSource interface, FixtureSource, SupabaseSource, factory

**Files:**
- Create: `dashboard/lib/datasource/index.ts`, `dashboard/lib/datasource/fixture.ts`, `dashboard/lib/datasource/supabase.ts`
- Test: `dashboard/lib/datasource/fixture.test.ts`, `dashboard/lib/datasource/supabase.test.ts`

- [ ] **Step 1: Write the failing test** (`dashboard/lib/datasource/fixture.test.ts`)

```ts
import { describe, it, expect } from "vitest";
import { FixtureSource } from "./fixture";
import { populated } from "@/fixtures/populated";

const src = new FixtureSource(populated);

describe("FixtureSource", () => {
  it("returns the snapshot and equity curve", async () => {
    expect((await src.getSnapshot())?.nav).toBe(104230.55);
    expect((await src.getEquityCurve()).length).toBe(6);
  });
  it("lists distinct fridays newest-first", async () => {
    expect(await src.getWeeklyFridays()).toEqual(["2026-06-05", "2026-05-29"]);
  });
  it("returns one week's portfolio and its executions", async () => {
    expect((await src.getWeeklyPortfolio("2026-05-29")).length).toBe(3);
    expect((await src.getExecutions("2026-06-01")).length).toBe(2);
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run lib/datasource/fixture.test.ts`
Expected: FAIL — cannot find module `./fixture`.

- [ ] **Step 3: Create `dashboard/lib/datasource/index.ts`**

```ts
import type { Dataset, EquityPoint, Execution, Holding, Snapshot, WeeklyRow } from "@/lib/types";

export interface DataSource {
  getSnapshot(): Promise<Snapshot | null>;
  getEquityCurve(): Promise<EquityPoint[]>;
  getHoldings(): Promise<Holding[]>;
  getWeeklyFridays(): Promise<string[]>;
  getWeeklyPortfolio(friday: string): Promise<WeeklyRow[]>;
  getAllWeekly(): Promise<WeeklyRow[]>;
  getExecutions(friday: string): Promise<Execution[]>;
}

// Lazy imports so a Supabase prod build never bundles fixtures and vice-versa.
export async function getDataSource(scenario?: string): Promise<DataSource> {
  if (process.env.DASHBOARD_DATA_SOURCE === "supabase") {
    const { SupabaseSource } = await import("./supabase");
    return new SupabaseSource();
  }
  const { FixtureSource } = await import("./fixture");
  if (scenario === "empty") {
    const { empty } = await import("@/fixtures/empty");
    return new FixtureSource(empty);
  }
  const { populated } = await import("@/fixtures/populated");
  return new FixtureSource(populated as Dataset);
}
```

- [ ] **Step 4: Create `dashboard/lib/datasource/fixture.ts`**

```ts
import type { Dataset, EquityPoint, Execution, Holding, Snapshot, WeeklyRow } from "@/lib/types";
import type { DataSource } from "./index";

export class FixtureSource implements DataSource {
  constructor(private readonly data: Dataset) {}
  async getSnapshot(): Promise<Snapshot | null> { return this.data.snapshot; }
  async getEquityCurve(): Promise<EquityPoint[]> { return this.data.equityCurve; }
  async getHoldings(): Promise<Holding[]> { return this.data.holdings; }
  async getWeeklyFridays(): Promise<string[]> {
    return [...new Set(this.data.weekly.map((w) => w.asof_friday))].sort().reverse();
  }
  async getWeeklyPortfolio(friday: string): Promise<WeeklyRow[]> {
    return this.data.weekly
      .filter((w) => w.asof_friday === friday)
      .sort((a, b) => b.target_weight - a.target_weight);
  }
  async getAllWeekly(): Promise<WeeklyRow[]> { return this.data.weekly; }
  async getExecutions(friday: string): Promise<Execution[]> {
    return this.data.executions.filter((e) => e.asof === friday);
  }
}
```

- [ ] **Step 5: Run the fixture test to verify it passes**

Run: `npx vitest run lib/datasource/fixture.test.ts`
Expected: PASS.

- [ ] **Step 6: Write the SupabaseSource test** (`dashboard/lib/datasource/supabase.test.ts`)

```ts
import { describe, it, expect, vi } from "vitest";
import { SupabaseSource } from "./supabase";

// A chainable fake mimicking the supabase-js query builder.
function fakeClient(tables: Record<string, any[]>) {
  return {
    from(table: string) {
      let rows = [...(tables[table] ?? [])];
      const api: any = {
        select: () => api,
        order: () => api,
        eq: (col: string, val: any) => { rows = rows.filter((r) => r[col] === val); return api; },
        limit: () => api,
        maybeSingle: async () => ({ data: rows[0] ?? null, error: null }),
        then: (res: any) => Promise.resolve({ data: rows, error: null }).then(res),
      };
      return api;
    },
  };
}

describe("SupabaseSource", () => {
  it("reads the snapshot row", async () => {
    const c = fakeClient({ snapshot: [{ id: 1, nav: 999 }] });
    const src = new SupabaseSource(c as any);
    expect((await src.getSnapshot())?.nav).toBe(999);
  });
  it("filters weekly by friday", async () => {
    const c = fakeClient({ weekly_portfolio: [
      { asof_friday: "2026-05-29", ticker: "NVDA", target_weight: 0.1 },
      { asof_friday: "2026-06-05", ticker: "AAPL", target_weight: 0.08 },
    ] });
    const src = new SupabaseSource(c as any);
    const rows = await src.getWeeklyPortfolio("2026-05-29");
    expect(rows.map((r) => r.ticker)).toEqual(["NVDA"]);
  });
});
```

- [ ] **Step 7: Run it to verify it fails**

Run: `npx vitest run lib/datasource/supabase.test.ts`
Expected: FAIL — cannot find module `./supabase`.

- [ ] **Step 8: Create `dashboard/lib/datasource/supabase.ts`**

```ts
import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import type { EquityPoint, Execution, Holding, Snapshot, WeeklyRow } from "@/lib/types";
import type { DataSource } from "./index";

function realClient(): SupabaseClient {
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_KEY;
  if (!url || !key) throw new Error("SUPABASE_URL / SUPABASE_SERVICE_KEY not set");
  return createClient(url, key, { auth: { persistSession: false } });
}

export class SupabaseSource implements DataSource {
  private c: SupabaseClient;
  constructor(client?: SupabaseClient) { this.c = client ?? realClient(); }

  async getSnapshot(): Promise<Snapshot | null> {
    const { data, error } = await this.c.from("snapshot").select("*").eq("id", 1).maybeSingle();
    if (error) throw error;
    return (data as Snapshot) ?? null;
  }
  async getEquityCurve(): Promise<EquityPoint[]> {
    const { data, error } = await this.c.from("equity_curve").select("*").order("date");
    if (error) throw error;
    return (data as EquityPoint[]) ?? [];
  }
  async getHoldings(): Promise<Holding[]> {
    const { data, error } = await this.c.from("holdings").select("*").order("weight_actual", { ascending: false });
    if (error) throw error;
    return (data as Holding[]) ?? [];
  }
  async getAllWeekly(): Promise<WeeklyRow[]> {
    const { data, error } = await this.c.from("weekly_portfolio").select("*").order("asof_friday");
    if (error) throw error;
    return (data as WeeklyRow[]) ?? [];
  }
  async getWeeklyFridays(): Promise<string[]> {
    const all = await this.getAllWeekly();
    return [...new Set(all.map((w) => w.asof_friday))].sort().reverse();
  }
  async getWeeklyPortfolio(friday: string): Promise<WeeklyRow[]> {
    const { data, error } = await this.c
      .from("weekly_portfolio").select("*").eq("asof_friday", friday).order("target_weight", { ascending: false });
    if (error) throw error;
    return (data as WeeklyRow[]) ?? [];
  }
  async getExecutions(friday: string): Promise<Execution[]> {
    const { data, error } = await this.c.from("executions").select("*").eq("asof", friday).order("ticker");
    if (error) throw error;
    return (data as Execution[]) ?? [];
  }
}
```

- [ ] **Step 9: Run both datasource tests to verify they pass**

Run: `npx vitest run lib/datasource/`
Expected: PASS (both files).

- [ ] **Step 10: Commit**

```bash
git add dashboard/lib/datasource/
git commit -m "feat(dashboard): DataSource interface + Fixture/Supabase impls"
```

---

### Task 5: Format helpers

**Files:**
- Create: `dashboard/lib/format.ts`
- Test: `dashboard/lib/format.test.ts`

- [ ] **Step 1: Write the failing test** (`dashboard/lib/format.test.ts`)

```ts
import { describe, it, expect } from "vitest";
import { fmtMoney, fmtPct, fmtSignedPct, fmtBps, asOfET } from "./format";

describe("format", () => {
  it("money", () => { expect(fmtMoney(104230.55)).toBe("$104,230.55"); });
  it("null -> dash", () => { expect(fmtMoney(null)).toBe("—"); expect(fmtPct(null)).toBe("—"); });
  it("pct from fraction", () => { expect(fmtPct(0.0423)).toBe("4.23%"); });
  it("signed pct", () => { expect(fmtSignedPct(0.0059)).toBe("+0.59%"); expect(fmtSignedPct(-0.01)).toBe("-1.00%"); });
  it("bps", () => { expect(fmtBps(3.8)).toBe("3.8 bps"); });
  it("asOfET renders an ET clock label", () => {
    expect(asOfET("2026-06-08T19:00:00+00:00")).toBe("as of 3:00 PM ET");
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run lib/format.test.ts`
Expected: FAIL — cannot find module `./format`.

- [ ] **Step 3: Create `dashboard/lib/format.ts`**

```ts
const DASH = "—";

export function fmtMoney(n: number | null | undefined): string {
  if (n === null || n === undefined) return DASH;
  return n.toLocaleString("en-US", { style: "currency", currency: "USD" });
}
export function fmtPct(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined) return DASH;
  return `${(n * 100).toFixed(digits)}%`;
}
export function fmtSignedPct(n: number | null | undefined, digits = 2): string {
  if (n === null || n === undefined) return DASH;
  const s = (n * 100).toFixed(digits);
  return `${n >= 0 ? "+" : ""}${s}%`;
}
export function fmtBps(n: number | null | undefined, digits = 1): string {
  if (n === null || n === undefined) return DASH;
  return `${n.toFixed(digits)} bps`;
}
export function asOfET(iso: string | null | undefined): string {
  if (!iso) return "no data yet";
  const t = new Date(iso).toLocaleString("en-US", {
    timeZone: "America/New_York", hour: "numeric", minute: "2-digit", hour12: true,
  });
  return `as of ${t} ET`;
}
```

- [ ] **Step 4: Run it to verify it passes**

Run: `npx vitest run lib/format.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dashboard/lib/format.ts dashboard/lib/format.test.ts
git commit -m "feat(dashboard): money/pct/bps/as-of-ET formatters"
```

---

### Task 6: Auth — token, middleware, login, route guard

**Files:**
- Create: `dashboard/lib/auth.ts`, `dashboard/middleware.ts`, `dashboard/app/login/page.tsx`, `dashboard/app/api/login/route.ts`
- Test: `dashboard/lib/auth.test.ts`, `dashboard/tests/auth.spec.ts`

- [ ] **Step 1: Write the failing unit test** (`dashboard/lib/auth.test.ts`)

```ts
import { describe, it, expect, beforeEach } from "vitest";
import { tokenFor, verifyToken, COOKIE } from "./auth";

beforeEach(() => { process.env.DASHBOARD_PASSWORD = "hunter2"; });

describe("auth token", () => {
  it("verifies a token minted from the current password", async () => {
    const tok = await tokenFor();
    expect(await verifyToken(tok)).toBe(true);
  });
  it("rejects a bad token", async () => {
    expect(await verifyToken("nope")).toBe(false);
    expect(await verifyToken(null)).toBe(false);
  });
  it("exposes a stable cookie name", () => { expect(COOKIE).toBe("dash_auth"); });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run lib/auth.test.ts`
Expected: FAIL — cannot find module `./auth`.

- [ ] **Step 3: Create `dashboard/lib/auth.ts`** (Web Crypto — works in the edge middleware runtime)

```ts
export const COOKIE = "dash_auth";
const MSG = "axiom-dash-v1";

async function hmac(password: string): Promise<string> {
  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw", enc.encode(password), { name: "HMAC", hash: "SHA-256" }, false, ["sign"],
  );
  const sig = await crypto.subtle.sign("HMAC", key, enc.encode(MSG));
  return btoa(String.fromCharCode(...new Uint8Array(sig)));
}

export async function tokenFor(): Promise<string> {
  const pw = process.env.DASHBOARD_PASSWORD ?? "";
  return hmac(pw);
}
export async function verifyToken(token: string | null | undefined): Promise<boolean> {
  if (!token) return false;
  const expected = await tokenFor();
  // constant-time-ish compare
  if (token.length !== expected.length) return false;
  let diff = 0;
  for (let i = 0; i < token.length; i++) diff |= token.charCodeAt(i) ^ expected.charCodeAt(i);
  return diff === 0;
}
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `npx vitest run lib/auth.test.ts`
Expected: PASS.

- [ ] **Step 5: Create `dashboard/middleware.ts`**

```ts
import { NextRequest, NextResponse } from "next/server";
import { COOKIE, verifyToken } from "@/lib/auth";

export const config = { matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"] };

export async function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;
  if (pathname === "/login" || pathname === "/api/login") return NextResponse.next();

  const ok = await verifyToken(req.cookies.get(COOKIE)?.value);
  if (ok) return NextResponse.next();

  if (pathname.startsWith("/api/")) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  const url = req.nextUrl.clone();
  url.pathname = "/login";
  return NextResponse.redirect(url);
}
```

- [ ] **Step 6: Create `dashboard/app/api/login/route.ts`**

```ts
import { NextRequest, NextResponse } from "next/server";
import { COOKIE, tokenFor } from "@/lib/auth";

export async function POST(req: NextRequest) {
  const { password } = await req.json().catch(() => ({ password: "" }));
  if (!password || password !== process.env.DASHBOARD_PASSWORD) {
    return NextResponse.json({ error: "wrong password" }, { status: 401 });
  }
  const res = NextResponse.json({ ok: true });
  res.cookies.set(COOKIE, await tokenFor(), {
    httpOnly: true, secure: true, sameSite: "lax", path: "/", maxAge: 60 * 60 * 24 * 30,
  });
  return res;
}
```

- [ ] **Step 7: Create `dashboard/app/login/page.tsx`**

```tsx
"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";

export default function Login() {
  const router = useRouter();
  const [pw, setPw] = useState("");
  const [err, setErr] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr(false);
    const res = await fetch("/api/login", {
      method: "POST", headers: { "content-type": "application/json" },
      body: JSON.stringify({ password: pw }),
    });
    if (res.ok) router.replace("/now");
    else setErr(true);
  }

  return (
    <main className="flex min-h-screen items-center justify-center p-6">
      <form onSubmit={submit} className="w-full max-w-xs space-y-4">
        <h1 className="text-xl font-semibold">Axiom Tilt</h1>
        <input
          type="password" value={pw} onChange={(e) => setPw(e.target.value)}
          placeholder="Password" aria-label="Password"
          className="w-full rounded-md bg-neutral-900 px-3 py-2 outline-none ring-1 ring-neutral-700 focus:ring-neutral-400"
        />
        {err && <p className="text-sm text-red-400">Wrong password.</p>}
        <button className="w-full rounded-md bg-neutral-100 px-3 py-2 font-medium text-neutral-900">
          Enter
        </button>
      </form>
    </main>
  );
}
```

- [ ] **Step 8: Write the auth E2E test** (`dashboard/tests/auth.spec.ts`)

```ts
import { test, expect } from "@playwright/test";

test("unauthenticated visit redirects to login", async ({ page }) => {
  await page.goto("/now");
  await expect(page).toHaveURL(/\/login$/);
  await expect(page.getByRole("heading", { name: "Axiom Tilt" })).toBeVisible();
});

test("wrong password is rejected, correct password enters", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Password").fill("wrong");
  await page.getByRole("button", { name: "Enter" }).click();
  await expect(page.getByText("Wrong password.")).toBeVisible();

  await page.getByLabel("Password").fill("testpass");
  await page.getByRole("button", { name: "Enter" }).click();
  await expect(page).toHaveURL(/\/now$/);
});
```

- [ ] **Step 9: Create `dashboard/playwright.config.ts`** (boots the app in fixture mode with a known password)

```ts
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  use: { baseURL: "http://localhost:3100" },
  webServer: {
    command: "npm run build && npm run start -- -p 3100",
    url: "http://localhost:3100/login",
    timeout: 120_000,
    reuseExistingServer: false,
    env: { DASHBOARD_DATA_SOURCE: "fixture", DASHBOARD_PASSWORD: "testpass" },
  },
});
```

- [ ] **Step 10: Run the auth E2E to verify it passes**

Run: `npx playwright test tests/auth.spec.ts`
Expected: PASS (2 tests). Login pages and the gate work; `/now` itself is built in Task 7 but the redirect-to-`/now` after login resolves once Task 7 lands. If running before Task 7, assert only the redirect-to-login and wrong-password cases; re-run the full file after Task 7.

- [ ] **Step 11: Commit**

```bash
git add dashboard/lib/auth.ts dashboard/lib/auth.test.ts dashboard/middleware.ts \
        dashboard/app/login dashboard/app/api/login dashboard/playwright.config.ts dashboard/tests/auth.spec.ts
git commit -m "feat(dashboard): shared-password gate (middleware + login + token)"
```

---

### Task 7: Tab shell, polling hook, shared UI atoms

**Files:**
- Create: `dashboard/lib/usePolling.ts`, `dashboard/app/(dash)/layout.tsx`, `dashboard/components/AsOf.tsx`, `dashboard/components/Empty.tsx`, `dashboard/components/StatCard.tsx`

- [ ] **Step 1: Create `dashboard/lib/usePolling.ts`**

```ts
"use client";
import { useEffect, useState, useCallback } from "react";

export function usePolling<T>(url: string, intervalMs = 60_000) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setData((await res.json()) as T);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "error");
    } finally {
      setLoading(false);
    }
  }, [url]);

  useEffect(() => {
    load();
    const id = setInterval(load, intervalMs);
    return () => clearInterval(id);
  }, [load, intervalMs]);

  return { data, error, loading };
}
```

- [ ] **Step 2: Create `dashboard/components/AsOf.tsx`**

```tsx
import { asOfET } from "@/lib/format";
export function AsOf({ iso }: { iso: string | null }) {
  return <span className="text-xs text-neutral-400">{asOfET(iso)}</span>;
}
```

- [ ] **Step 3: Create `dashboard/components/Empty.tsx`**

```tsx
export function Empty({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-dashed border-neutral-700 p-8 text-center">
      <p className="font-medium text-neutral-200">{title}</p>
      {hint && <p className="mt-1 text-sm text-neutral-400">{hint}</p>}
    </div>
  );
}
```

- [ ] **Step 4: Create `dashboard/components/StatCard.tsx`**

```tsx
export function StatCard({ label, value, sub, tone }: {
  label: string; value: string; sub?: string; tone?: "up" | "down" | "flat";
}) {
  const toneClass = tone === "up" ? "text-emerald-400" : tone === "down" ? "text-red-400" : "text-neutral-100";
  return (
    <div className="rounded-lg bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <p className="text-xs uppercase tracking-wide text-neutral-400">{label}</p>
      <p className={`mt-1 text-2xl font-semibold ${toneClass}`}>{value}</p>
      {sub && <p className="mt-0.5 text-sm text-neutral-400">{sub}</p>}
    </div>
  );
}
```

- [ ] **Step 5: Create `dashboard/app/(dash)/layout.tsx`** (tab nav shell)

```tsx
"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  { href: "/now", label: "Now" },
  { href: "/holdings", label: "Holdings" },
  { href: "/history", label: "History" },
];

export default function DashLayout({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  return (
    <div className="mx-auto max-w-3xl px-4 pb-16">
      <header className="flex items-center justify-between py-4">
        <span className="font-semibold">Axiom Tilt</span>
      </header>
      <nav className="mb-4 flex gap-1 rounded-lg bg-neutral-900 p-1 ring-1 ring-neutral-800">
        {TABS.map((t) => (
          <Link
            key={t.href} href={t.href}
            className={`flex-1 rounded-md px-3 py-2 text-center text-sm ${
              path === t.href ? "bg-neutral-100 text-neutral-900 font-medium" : "text-neutral-300"
            }`}
          >
            {t.label}
          </Link>
        ))}
      </nav>
      {children}
    </div>
  );
}
```

- [ ] **Step 6: Verify it builds**

Run: `npm run build`
Expected: "Compiled successfully" (routes for `(dash)` group exist once Task 8 adds pages; the layout alone builds clean).

- [ ] **Step 7: Commit**

```bash
git add dashboard/lib/usePolling.ts "dashboard/app/(dash)/layout.tsx" dashboard/components/AsOf.tsx dashboard/components/Empty.tsx dashboard/components/StatCard.tsx
git commit -m "feat(dashboard): tab shell, polling hook, shared UI atoms"
```

---

### Task 8: Now tab

**Files:**
- Create: `dashboard/app/api/now/route.ts`, `dashboard/app/(dash)/now/page.tsx`, `dashboard/components/EquityChart.tsx`, `dashboard/components/RiskCard.tsx`, `dashboard/components/RegimeBar.tsx`
- Test: `dashboard/tests/now.spec.ts`

- [ ] **Step 1: Write the failing E2E test** (`dashboard/tests/now.spec.ts`)

```ts
import { test, expect } from "@playwright/test";

async function login(page: any) {
  await page.goto("/login");
  await page.getByLabel("Password").fill("testpass");
  await page.getByRole("button", { name: "Enter" }).click();
  await expect(page).toHaveURL(/\/now$/);
}

test("Now tab shows hero stats for populated data", async ({ page }) => {
  await login(page);
  await expect(page.getByText("Portfolio Value")).toBeVisible();
  await expect(page.getByText("$104,230.55")).toBeVisible();
  await expect(page.getByText("Regime Call")).toBeVisible();
});

test("Now tab shows the go-live empty state when there is no snapshot", async ({ page }) => {
  await login(page);
  await page.goto("/now?scenario=empty");
  await expect(page.getByText(/builds forward from go-live/i)).toBeVisible();
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx playwright test tests/now.spec.ts`
Expected: FAIL — `/now` not found / texts absent.

- [ ] **Step 3: Create `dashboard/app/api/now/route.ts`**

```ts
import { NextRequest, NextResponse } from "next/server";
import { getDataSource } from "@/lib/datasource";

export async function GET(req: NextRequest) {
  const scenario = req.nextUrl.searchParams.get("scenario") ?? undefined;
  const ds = await getDataSource(scenario);
  const [snapshot, equityCurve] = await Promise.all([ds.getSnapshot(), ds.getEquityCurve()]);
  return NextResponse.json({ snapshot, equityCurve });
}
```

- [ ] **Step 4: Create `dashboard/components/EquityChart.tsx`**

```tsx
"use client";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import type { EquityPoint } from "@/lib/types";

export function EquityChart({ points }: { points: EquityPoint[] }) {
  const base = points[0];
  const data = points.map((p) => ({
    date: p.date,
    Strategy: base ? p.nav / base.nav - 1 : 0,
    SPY: base?.spy_close && p.spy_close ? p.spy_close / base.spy_close - 1 : null,
  }));
  return (
    <div className="h-56 w-full rounded-lg bg-neutral-900 p-3 ring-1 ring-neutral-800">
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <XAxis dataKey="date" tick={{ fill: "#a3a3a3", fontSize: 11 }} minTickGap={24} />
          <YAxis tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} tick={{ fill: "#a3a3a3", fontSize: 11 }} width={36} />
          <Tooltip formatter={(v: number) => `${(v * 100).toFixed(2)}%`}
            contentStyle={{ background: "#171717", border: "1px solid #404040" }} />
          <Line type="monotone" dataKey="Strategy" stroke="#34d399" dot={false} strokeWidth={2} />
          <Line type="monotone" dataKey="SPY" stroke="#a3a3a3" dot={false} strokeWidth={1.5} connectNulls />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
```

- [ ] **Step 5: Create `dashboard/components/RiskCard.tsx`**

```tsx
import type { Risk } from "@/lib/types";
import { fmtPct } from "@/lib/format";

export function RiskCard({ risk }: { risk: Risk | null }) {
  const rows: [string, string][] = [
    ["Current drawdown", fmtPct(risk?.current_drawdown ?? null)],
    ["Max drawdown", fmtPct(risk?.max_drawdown ?? null)],
    ["Sharpe (to date)", risk?.sharpe != null ? risk.sharpe.toFixed(2) : "—"],
    ["Annualized vol", fmtPct(risk?.ann_vol ?? null)],
  ];
  return (
    <div className="rounded-lg bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <p className="mb-2 text-sm font-medium">Risk</p>
      <dl className="grid grid-cols-2 gap-2 text-sm">
        {rows.map(([k, v]) => (
          <div key={k} className="flex justify-between gap-2">
            <dt className="text-neutral-400">{k}</dt><dd>{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
```

- [ ] **Step 6: Create `dashboard/components/RegimeBar.tsx`**

```tsx
import { fmtPct } from "@/lib/format";

const SLEEVES = ["10", "20", "30", "50"] as const;
const COLORS: Record<string, string> = { "10": "#60a5fa", "20": "#34d399", "30": "#fbbf24", "50": "#f87171" };

export function RegimeBar({ kProbs, features }: {
  kProbs: Record<string, number> | null; features: Record<string, number> | null;
}) {
  return (
    <div className="rounded-lg bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <p className="mb-2 text-sm font-medium">Regime Call</p>
      {kProbs ? (
        <>
          <div className="flex h-4 w-full overflow-hidden rounded">
            {SLEEVES.map((k) => (
              <div key={k} style={{ width: `${(kProbs[k] ?? 0) * 100}%`, background: COLORS[k] }} title={`k=${k}: ${fmtPct(kProbs[k] ?? 0)}`} />
            ))}
          </div>
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-neutral-400">
            {SLEEVES.map((k) => <span key={k}>k={k}: {fmtPct(kProbs[k] ?? 0)}</span>)}
          </div>
        </>
      ) : <p className="text-sm text-neutral-400">No regime call yet.</p>}
      {features && (
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-neutral-500">
          {Object.entries(features).map(([k, v]) => <span key={k}>{k}: {v}</span>)}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 7: Create `dashboard/app/(dash)/now/page.tsx`**

```tsx
"use client";
import { usePolling } from "@/lib/usePolling";
import type { EquityPoint, Snapshot } from "@/lib/types";
import { fmtMoney, fmtPct, fmtSignedPct } from "@/lib/format";
import { useSearchParams } from "next/navigation";
import { StatCard } from "@/components/StatCard";
import { EquityChart } from "@/components/EquityChart";
import { RiskCard } from "@/components/RiskCard";
import { RegimeBar } from "@/components/RegimeBar";
import { Empty } from "@/components/Empty";
import { AsOf } from "@/components/AsOf";

type Payload = { snapshot: Snapshot | null; equityCurve: EquityPoint[] };

export default function NowPage() {
  const scenario = useSearchParams().get("scenario");
  const url = scenario ? `/api/now?scenario=${scenario}` : "/api/now";
  const { data, loading } = usePolling<Payload>(url);

  if (loading && !data) return <p className="text-sm text-neutral-400">Loading…</p>;
  const s = data?.snapshot ?? null;
  const curve = data?.equityCurve ?? [];

  if (!s) {
    return (
      <div className="space-y-4">
        <Empty title="Performance builds forward from go-live"
               hint="No live snapshot yet. The target portfolio is visible under History." />
        <RegimeBar kProbs={null} features={null} />
      </div>
    );
  }

  const dayTone = (s.day_pnl ?? 0) > 0 ? "up" : (s.day_pnl ?? 0) < 0 ? "down" : "flat";
  const totTone = (s.total_return ?? 0) >= (s.spy_return ?? 0) ? "up" : "down";

  return (
    <div className="space-y-4">
      <div className="flex justify-end"><AsOf iso={s.asof} /></div>
      <div className="grid grid-cols-2 gap-3">
        <StatCard label="Portfolio Value" value={fmtMoney(s.nav)} sub={`${s.n_positions ?? 0} positions`} />
        <StatCard label="Today" value={fmtMoney(s.day_pnl)} sub={fmtSignedPct(s.day_pnl_pct)} tone={dayTone} />
        <StatCard label="Total Return" value={fmtPct(s.total_return)} sub={`SPY ${fmtPct(s.spy_return)}`} tone={totTone} />
        <StatCard label="Invested" value={fmtPct(s.invested_pct)} />
      </div>
      {curve.length > 1
        ? <EquityChart points={curve} />
        : <Empty title="Equity curve builds forward from go-live" />}
      <RiskCard risk={s.risk} />
      <RegimeBar kProbs={s.k_probs} features={s.regime_features} />
    </div>
  );
}
```

- [ ] **Step 8: Run the Now E2E to verify it passes**

Run: `npx playwright test tests/now.spec.ts`
Expected: PASS (2 tests).

- [ ] **Step 9: Commit**

```bash
git add dashboard/app/api/now "dashboard/app/(dash)/now" dashboard/components/EquityChart.tsx dashboard/components/RiskCard.tsx dashboard/components/RegimeBar.tsx dashboard/tests/now.spec.ts
git commit -m "feat(dashboard): Now tab (hero stats, equity curve, risk, regime)"
```

---

### Task 9: Holdings tab

**Files:**
- Create: `dashboard/app/api/holdings/route.ts`, `dashboard/app/(dash)/holdings/page.tsx`, `dashboard/components/ConcentrationBars.tsx`, `dashboard/components/HoldingRow.tsx`
- Test: `dashboard/tests/holdings.spec.ts`

- [ ] **Step 1: Write the failing E2E test** (`dashboard/tests/holdings.spec.ts`)

```ts
import { test, expect } from "@playwright/test";

async function login(page: any) {
  await page.goto("/login");
  await page.getByLabel("Password").fill("testpass");
  await page.getByRole("button", { name: "Enter" }).click();
  await expect(page).toHaveURL(/\/now$/);
}

test("Holdings shows names, the 10% cap line, and a no-quote flag", async ({ page }) => {
  await login(page);
  await page.goto("/holdings");
  await expect(page.getByText("NVDA")).toBeVisible();
  await expect(page.getByText("10% cap")).toBeVisible();
  await expect(page.getByText("no quote")).toBeVisible(); // the ZZZQ row has price 0
});

test("Holdings empty state before go-live", async ({ page }) => {
  await login(page);
  await page.goto("/holdings?scenario=empty");
  await expect(page.getByText(/No live positions yet/i)).toBeVisible();
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx playwright test tests/holdings.spec.ts`
Expected: FAIL — `/holdings` not found.

- [ ] **Step 3: Create `dashboard/app/api/holdings/route.ts`**

```ts
import { NextRequest, NextResponse } from "next/server";
import { getDataSource } from "@/lib/datasource";

export async function GET(req: NextRequest) {
  const scenario = req.nextUrl.searchParams.get("scenario") ?? undefined;
  const ds = await getDataSource(scenario);
  const [holdings, snapshot] = await Promise.all([ds.getHoldings(), ds.getSnapshot()]);
  return NextResponse.json({ holdings, asof: snapshot?.asof ?? null });
}
```

- [ ] **Step 4: Create `dashboard/components/HoldingRow.tsx`**

```tsx
"use client";
import { useState } from "react";
import type { Holding } from "@/lib/types";
import { fmtMoney, fmtPct } from "@/lib/format";

const CAP = 0.10;

export function HoldingRow({ h }: { h: Holding }) {
  const [open, setOpen] = useState(false);
  const w = h.weight_actual ?? 0;
  const noQuote = (h.price ?? 0) === 0 && h.shares > 0;
  const pctOfCap = Math.min(w / CAP, 1) * 100;
  return (
    <div className="rounded-md bg-neutral-900 px-3 py-2 ring-1 ring-neutral-800">
      <button onClick={() => setOpen(!open)} className="flex w-full items-center gap-3 text-left">
        <span className="w-16 font-medium">{h.ticker}</span>
        <span className="relative h-3 flex-1 overflow-hidden rounded bg-neutral-800">
          <span className="absolute inset-y-0 left-0 bg-emerald-500" style={{ width: `${pctOfCap}%` }} />
        </span>
        <span className="w-14 text-right text-sm tabular-nums">{fmtPct(w)}</span>
        {noQuote && <span className="rounded bg-amber-500/20 px-1.5 py-0.5 text-[10px] text-amber-300">no quote</span>}
      </button>
      {open && (
        <div className="mt-2 grid grid-cols-2 gap-1 pl-16 text-xs text-neutral-400">
          <span>Target: {fmtPct(h.weight_target ?? null)}</span>
          <span>Shares: {h.shares}</span>
          <span>Price: {fmtMoney(h.price)}</span>
          <span>Value: {fmtMoney(h.market_value)}</span>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 5: Create `dashboard/components/ConcentrationBars.tsx`**

```tsx
import type { Holding } from "@/lib/types";
import { HoldingRow } from "./HoldingRow";

export function ConcentrationBars({ holdings }: { holdings: Holding[] }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-3 px-3 text-xs text-neutral-500">
        <span className="w-16">Ticker</span>
        <span className="flex-1 text-right">10% cap →</span>
        <span className="w-14 text-right">Weight</span>
      </div>
      <p className="px-3 text-[11px] text-neutral-500">Bars fill toward the <span className="text-amber-300">10% cap</span>.</p>
      {holdings.map((h) => <HoldingRow key={h.ticker} h={h} />)}
    </div>
  );
}
```

- [ ] **Step 6: Create `dashboard/app/(dash)/holdings/page.tsx`**

```tsx
"use client";
import { usePolling } from "@/lib/usePolling";
import { useSearchParams } from "next/navigation";
import type { Holding } from "@/lib/types";
import { ConcentrationBars } from "@/components/ConcentrationBars";
import { Empty } from "@/components/Empty";
import { AsOf } from "@/components/AsOf";

type Payload = { holdings: Holding[]; asof: string | null };

export default function HoldingsPage() {
  const scenario = useSearchParams().get("scenario");
  const url = scenario ? `/api/holdings?scenario=${scenario}` : "/api/holdings";
  const { data, loading } = usePolling<Payload>(url);

  if (loading && !data) return <p className="text-sm text-neutral-400">Loading…</p>;
  const holdings = data?.holdings ?? [];

  if (holdings.length === 0) {
    return <Empty title="No live positions yet"
                  hint="The target portfolio is visible under History until the strategy trades." />;
  }
  return (
    <div className="space-y-3">
      <div className="flex justify-end"><AsOf iso={data?.asof ?? null} /></div>
      <ConcentrationBars holdings={holdings} />
    </div>
  );
}
```

- [ ] **Step 7: Run the Holdings E2E to verify it passes**

Run: `npx playwright test tests/holdings.spec.ts`
Expected: PASS (2 tests).

- [ ] **Step 8: Commit**

```bash
git add dashboard/app/api/holdings "dashboard/app/(dash)/holdings" dashboard/components/ConcentrationBars.tsx dashboard/components/HoldingRow.tsx dashboard/tests/holdings.spec.ts
git commit -m "feat(dashboard): Holdings tab (concentration bars, 10% cap, no-quote flag)"
```

---

### Task 10: History tab

**Files:**
- Create: `dashboard/app/api/history/route.ts`, `dashboard/app/(dash)/history/page.tsx`, `dashboard/components/WeekPicker.tsx`, `dashboard/components/PersistenceHeatmap.tsx`, `dashboard/components/TurnoverCard.tsx`, `dashboard/components/ExecQualityTable.tsx`
- Test: `dashboard/tests/history.spec.ts`

- [ ] **Step 1: Write the failing E2E test** (`dashboard/tests/history.spec.ts`)

```ts
import { test, expect } from "@playwright/test";

async function login(page: any) {
  await page.goto("/login");
  await page.getByLabel("Password").fill("testpass");
  await page.getByRole("button", { name: "Enter" }).click();
  await expect(page).toHaveURL(/\/now$/);
}

test("History lists fridays and shows the selected week's portfolio", async ({ page }) => {
  await login(page);
  await page.goto("/history");
  await expect(page.getByRole("button", { name: "2026-05-29" })).toBeVisible();
  await expect(page.getByText("Persistence")).toBeVisible();
  await page.getByRole("button", { name: "2026-05-29" }).click();
  await expect(page.getByText("NVDA")).toBeVisible();
});

test("History single-week state hides turnover until >=2 weeks", async ({ page }) => {
  await login(page);
  await page.goto("/history?scenario=empty");
  await expect(page.getByText(/needs at least two weeks/i)).toBeVisible();
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx playwright test tests/history.spec.ts`
Expected: FAIL — `/history` not found.

- [ ] **Step 3: Create `dashboard/app/api/history/route.ts`**

```ts
import { NextRequest, NextResponse } from "next/server";
import { getDataSource } from "@/lib/datasource";
import { computeTurnover } from "@/lib/turnover";

export async function GET(req: NextRequest) {
  const ds = await getDataSource(req.nextUrl.searchParams.get("scenario") ?? undefined);
  const fridays = await ds.getWeeklyFridays();
  const selected = req.nextUrl.searchParams.get("friday") ?? fridays[0] ?? null;
  const [weekly, allWeekly, executions] = await Promise.all([
    selected ? ds.getWeeklyPortfolio(selected) : Promise.resolve([]),
    ds.getAllWeekly(),
    selected ? ds.getExecutions(selected) : Promise.resolve([]),
  ]);
  const turnover = computeTurnover(allWeekly, fridays, selected);
  return NextResponse.json({ fridays, selected, weekly, allWeekly, executions, turnover });
}
```

- [ ] **Step 4: Write a unit test for the turnover helper** (`dashboard/lib/turnover.test.ts`)

```ts
import { describe, it, expect } from "vitest";
import { computeTurnover } from "./turnover";
import type { WeeklyRow } from "./types";

const rows: WeeklyRow[] = [
  { asof_friday: "2026-05-29", ticker: "NVDA", target_weight: 0.1, k_probs: null },
  { asof_friday: "2026-05-29", ticker: "AMD", target_weight: 0.08, k_probs: null },
  { asof_friday: "2026-06-05", ticker: "NVDA", target_weight: 0.1, k_probs: null },
  { asof_friday: "2026-06-05", ticker: "AVGO", target_weight: 0.07, k_probs: null },
];

describe("computeTurnover", () => {
  it("returns null with fewer than two weeks", () => {
    expect(computeTurnover(rows, ["2026-05-29"], "2026-05-29")).toBeNull();
  });
  it("diffs the selected week against the prior week", () => {
    const t = computeTurnover(rows, ["2026-06-05", "2026-05-29"], "2026-06-05");
    expect(t).not.toBeNull();
    expect(t!.added).toEqual(["AVGO"]);
    expect(t!.dropped).toEqual(["AMD"]);
  });
});
```

- [ ] **Step 5: Run it to verify it fails**

Run: `npx vitest run lib/turnover.test.ts`
Expected: FAIL — cannot find module `./turnover`.

- [ ] **Step 6: Create `dashboard/lib/turnover.ts`**

```ts
import type { Turnover, WeeklyRow } from "./types";

export function computeTurnover(
  all: WeeklyRow[], fridaysDesc: string[], selected: string | null,
): Turnover | null {
  if (!selected) return null;
  const idx = fridaysDesc.indexOf(selected);
  const prior = fridaysDesc[idx + 1];
  if (!prior) return null;
  const wOf = (f: string) =>
    new Map(all.filter((r) => r.asof_friday === f).map((r) => [r.ticker, r.target_weight]));
  const cur = wOf(selected);
  const prev = wOf(prior);
  const tickers = new Set([...cur.keys(), ...prev.keys()]);
  let turnover = 0;
  for (const t of tickers) turnover += Math.abs((cur.get(t) ?? 0) - (prev.get(t) ?? 0));
  return {
    added: [...cur.keys()].filter((t) => !prev.has(t)).sort(),
    dropped: [...prev.keys()].filter((t) => !cur.has(t)).sort(),
    turnover_frac: 0.5 * turnover,
  };
}
```

- [ ] **Step 7: Run the turnover unit test to verify it passes**

Run: `npx vitest run lib/turnover.test.ts`
Expected: PASS.

- [ ] **Step 8: Create `dashboard/components/WeekPicker.tsx`**

```tsx
"use client";
export function WeekPicker({ fridays, selected, onPick }: {
  fridays: string[]; selected: string | null; onPick: (f: string) => void;
}) {
  return (
    <div className="flex flex-wrap gap-2">
      {fridays.map((f) => (
        <button key={f} onClick={() => onPick(f)}
          className={`rounded-md px-2.5 py-1 text-sm ring-1 ${
            f === selected ? "bg-neutral-100 text-neutral-900 ring-neutral-100" : "bg-neutral-900 text-neutral-300 ring-neutral-700"
          }`}>
          {f}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 9: Create `dashboard/components/PersistenceHeatmap.tsx`**

```tsx
import type { WeeklyRow } from "@/lib/types";

function color(w: number): string {
  if (w <= 0) return "#171717";
  const a = Math.min(w / 0.10, 1);
  return `rgba(52, 211, 153, ${0.2 + a * 0.8})`;
}

export function PersistenceHeatmap({ all }: { all: WeeklyRow[] }) {
  const weeks = [...new Set(all.map((r) => r.asof_friday))].sort();
  const tickers = [...new Set(all.map((r) => r.ticker))].sort();
  const wOf = new Map(all.map((r) => [`${r.asof_friday}|${r.ticker}`, r.target_weight]));
  return (
    <div className="rounded-lg bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <p className="mb-2 text-sm font-medium">Persistence</p>
      <div className="overflow-x-auto">
        <table className="border-separate border-spacing-1 text-xs">
          <thead>
            <tr><th className="text-left text-neutral-500"></th>
              {weeks.map((w) => <th key={w} className="px-1 text-neutral-500">{w.slice(5)}</th>)}</tr>
          </thead>
          <tbody>
            {tickers.map((t) => (
              <tr key={t}>
                <td className="pr-2 text-right text-neutral-300">{t}</td>
                {weeks.map((w) => (
                  <td key={w}>
                    <div className="h-5 w-8 rounded-sm" style={{ background: color(wOf.get(`${w}|${t}`) ?? 0) }} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
```

- [ ] **Step 10: Create `dashboard/components/TurnoverCard.tsx`**

```tsx
import type { Turnover } from "@/lib/types";
import { fmtPct } from "@/lib/format";

export function TurnoverCard({ turnover }: { turnover: Turnover | null }) {
  return (
    <div className="rounded-lg bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <p className="mb-2 text-sm font-medium">Turnover vs prior week</p>
      {turnover ? (
        <div className="space-y-1 text-sm">
          <p>One-way turnover: <span className="tabular-nums">{fmtPct(turnover.turnover_frac)}</span></p>
          <p className="text-emerald-400">Added: {turnover.added.join(", ") || "—"}</p>
          <p className="text-red-400">Dropped: {turnover.dropped.join(", ") || "—"}</p>
        </div>
      ) : <p className="text-sm text-neutral-400">Turnover needs at least two weeks of history.</p>}
    </div>
  );
}
```

- [ ] **Step 11: Create `dashboard/components/ExecQualityTable.tsx`**

```tsx
import type { Execution } from "@/lib/types";
import { fmtBps, fmtMoney } from "@/lib/format";

export function ExecQualityTable({ executions }: { executions: Execution[] }) {
  return (
    <div className="rounded-lg bg-neutral-900 p-4 ring-1 ring-neutral-800">
      <p className="mb-2 text-sm font-medium">Execution quality</p>
      {executions.length ? (
        <table className="w-full text-sm">
          <thead><tr className="text-left text-neutral-500">
            <th>Ticker</th><th>Side</th><th className="text-right">Fill</th><th className="text-right">Mid</th><th className="text-right">Slippage</th>
          </tr></thead>
          <tbody>
            {executions.map((e) => (
              <tr key={e.ticker} className="border-t border-neutral-800">
                <td>{e.ticker}</td><td>{e.side}</td>
                <td className="text-right">{fmtMoney(e.realized_price)}</td>
                <td className="text-right">{fmtMoney(e.midpoint)}</td>
                <td className={`text-right ${(e.slippage_bps ?? 0) > 0 ? "text-red-400" : "text-emerald-400"}`}>{fmtBps(e.slippage_bps)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : <p className="text-sm text-neutral-400">No fills for this week yet.</p>}
    </div>
  );
}
```

- [ ] **Step 12: Create `dashboard/app/(dash)/history/page.tsx`**

```tsx
"use client";
import { useState } from "react";
import { usePolling } from "@/lib/usePolling";
import { useSearchParams } from "next/navigation";
import type { Execution, Turnover, WeeklyRow } from "@/lib/types";
import { fmtPct } from "@/lib/format";
import { WeekPicker } from "@/components/WeekPicker";
import { PersistenceHeatmap } from "@/components/PersistenceHeatmap";
import { TurnoverCard } from "@/components/TurnoverCard";
import { ExecQualityTable } from "@/components/ExecQualityTable";
import { Empty } from "@/components/Empty";

type Payload = {
  fridays: string[]; selected: string | null;
  weekly: WeeklyRow[]; allWeekly: WeeklyRow[]; executions: Execution[]; turnover: Turnover | null;
};

export default function HistoryPage() {
  const scenario = useSearchParams().get("scenario");
  const [friday, setFriday] = useState<string | null>(null);
  const q = new URLSearchParams();
  if (scenario) q.set("scenario", scenario);
  if (friday) q.set("friday", friday);
  const url = `/api/history${q.toString() ? `?${q}` : ""}`;
  const { data, loading } = usePolling<Payload>(url);

  if (loading && !data) return <p className="text-sm text-neutral-400">Loading…</p>;
  if (!data || data.fridays.length === 0) return <Empty title="No weekly portfolios yet" />;

  return (
    <div className="space-y-4">
      <WeekPicker fridays={data.fridays} selected={data.selected} onPick={setFriday} />
      <div className="rounded-lg bg-neutral-900 p-4 ring-1 ring-neutral-800">
        <p className="mb-2 text-sm font-medium">Target portfolio — {data.selected}</p>
        <table className="w-full text-sm">
          <tbody>
            {data.weekly.map((r) => (
              <tr key={r.ticker} className="border-t border-neutral-800">
                <td>{r.ticker}</td>
                <td className="text-right tabular-nums">{fmtPct(r.target_weight)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <TurnoverCard turnover={data.turnover} />
      <PersistenceHeatmap all={data.allWeekly} />
      <ExecQualityTable executions={data.executions} />
    </div>
  );
}
```

- [ ] **Step 13: Run the History E2E + the full vitest suite to verify they pass**

Run: `npx vitest run && npx playwright test tests/history.spec.ts`
Expected: PASS (all vitest files; 2 history E2E tests).

- [ ] **Step 14: Commit**

```bash
git add dashboard/app/api/history "dashboard/app/(dash)/history" dashboard/lib/turnover.ts dashboard/lib/turnover.test.ts \
        dashboard/components/WeekPicker.tsx dashboard/components/PersistenceHeatmap.tsx dashboard/components/TurnoverCard.tsx dashboard/components/ExecQualityTable.tsx dashboard/tests/history.spec.ts
git commit -m "feat(dashboard): History tab (weeks, turnover, heatmap, exec quality)"
```

---

### Task 11: Full suite + README + deploy notes

**Files:**
- Create: `dashboard/README.md`

- [ ] **Step 1: Run the entire test suite**

Run (from `dashboard/`):
```bash
npx vitest run && npx playwright test
```
Expected: all vitest files PASS; all Playwright specs (auth, now, holdings, history) PASS.

- [ ] **Step 2: Create `dashboard/README.md`**

````markdown
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
````

- [ ] **Step 3: Manual smoke against fixtures (verification-before-completion)**

Run: `npm run dev`, open `http://localhost:3000`, log in with the password from `.env.local`, click through Now / Holdings / History, then visit `/now?scenario=empty` and `/holdings?scenario=empty` to confirm the empty states render.
Expected: all three tabs render with populated data; empty scenarios show the go-live messaging.

- [ ] **Step 4: Commit**

```bash
git add dashboard/README.md
git commit -m "docs(dashboard): README + Vercel deploy notes"
```

---

## Self-Review

**Spec coverage** (against `2026-06-04-dashboard-frontend-design.md`):
- §2 stack (Next App Router + TS + Tailwind + Recharts) → Task 1. ✓
- §3 server-side service-key reads → Task 4 (`SupabaseSource`, route handlers). ✓
- §3 swappable data source + `DASHBOARD_DATA_SOURCE` → Task 4 `getDataSource`. ✓
- §3 fixture mode (populated + empty) → Tasks 3, 8–10 `?scenario=`. ✓
- §3 ~60s polling + "as of ET" → Task 7 `usePolling`, `AsOf`. ✓
- §4 Now (hero/equity/risk/regime + empty) → Task 8. ✓
- §4 Holdings (bars, 10% cap, no-quote flag, expand + empty) → Task 9. ✓
- §4 History (week picker, target table, heatmap, turnover, exec quality + degraded) → Task 10. ✓
- §5 password middleware + login + cookie → Task 6. ✓
- §6 structure → matches the File Structure section. ✓
- §7 testing (Playwright per tab + auth + empty-state fixtures; datasource unit tests) → Tasks 4, 6, 8–10. ✓
- §8 deploy (Vercel root `dashboard/`, env vars) → Task 11 README. ✓

**Placeholder scan:** no TBD/TODO; every code step shows complete code; fixtures are compact but real (not placeholders).

**Type consistency:** `DataSource` method names (`getSnapshot`, `getEquityCurve`, `getHoldings`, `getWeeklyFridays`, `getWeeklyPortfolio`, `getAllWeekly`, `getExecutions`) are identical across the interface (Task 4), both implementations (Task 4), and all route handlers (Tasks 8–10). Row types (`Snapshot`, `EquityPoint`, `Holding`, `WeeklyRow`, `Execution`, `Risk`, `Turnover`, `Dataset`) defined once in Task 2 and imported everywhere. `COOKIE`/`tokenFor`/`verifyToken` consistent across Task 6. `computeTurnover(all, fridaysDesc, selected)` signature matches between Task 10 helper, its test, and the route handler.

**Note on ordering:** the `auth.spec.ts` assertion that login redirects to `/now` fully passes only once Task 8 lands; Task 6 Step 10 calls this out. Run `npx playwright test` whole-suite at Task 11 for the authoritative green.
