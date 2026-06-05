import { describe, it, expect } from "vitest";
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
