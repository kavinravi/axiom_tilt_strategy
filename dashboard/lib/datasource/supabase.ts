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
