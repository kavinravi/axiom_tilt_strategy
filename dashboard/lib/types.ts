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
  company_name: string | null;
  sector: string | null;
  shares: number;
  price: number | null;
  market_value: number | null;
  weight_actual: number | null;
  weight_target: number | null;
}
export interface WeeklyRow {
  asof_friday: string;
  ticker: string;
  company_name: string | null;
  sector: string | null;
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
