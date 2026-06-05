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
