import type { EquityPoint, Execution, Holding, Snapshot, WeeklyRow } from "@/lib/types";

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
  return new FixtureSource(populated);
}
