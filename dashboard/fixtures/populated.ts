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
    invested_pct: 0.38,
    k_probs: KPROBS,
    regime_features: { vix: 14.2, y10: 0.0431, term_spread: 0.0052, spy_ret: 0.012, spy_vol: 0.009 },
    risk: { current_drawdown: -0.008, max_drawdown: -0.021, sharpe: 1.42, ann_vol: 0.11 },
    turnover: { added: ["AVGO"], dropped: ["AMD"], turnover_frac: 0.18 },
    week_vs_spy: {
      baseline_date: "2026-06-05",
      portfolio_return: 0.0059,
      spy_return: 0.0031,
      excess_return: 0.0028,
    },
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
    { ticker: "NVDA", company_name: "NVIDIA Corp", sector: "Technology", shares: 70, price: 132.4, market_value: 9268, weight_actual: 0.0889, weight_target: 0.10, avg_cost: 128.2, unrealized_pnl: 294.0, daily_pnl: 87.5 },
    { ticker: "AAPL", company_name: "Apple Inc", sector: "Technology", shares: 40, price: 212.1, market_value: 8484, weight_actual: 0.0814, weight_target: 0.08, avg_cost: 215.4, unrealized_pnl: -132.0, daily_pnl: -41.2 },
    { ticker: "MSFT", company_name: "Microsoft Corp", sector: "Technology", shares: 18, price: 441.0, market_value: 7938, weight_actual: 0.0762, weight_target: 0.075, avg_cost: 432.8, unrealized_pnl: 147.6, daily_pnl: 23.4 },
    { ticker: "AVGO", company_name: "Broadcom Inc", sector: "Technology", shares: 45, price: 161.2, market_value: 7254, weight_actual: 0.0696, weight_target: 0.07, avg_cost: 158.9, unrealized_pnl: 103.5, daily_pnl: 65.7 },
    { ticker: "GOOGL", company_name: "Alphabet Inc", sector: "Communication Services", shares: 38, price: 178.5, market_value: 6783, weight_actual: 0.0651, weight_target: 0.065, avg_cost: 176.0, unrealized_pnl: 95.0, daily_pnl: -12.3 },
    { ticker: "ZZZQ", company_name: null, sector: null, shares: 10, price: 0, market_value: 0, weight_actual: 0, weight_target: 0.04, avg_cost: null, unrealized_pnl: null, daily_pnl: null },
  ],
  weekly: [
    { asof_friday: "2026-05-29", ticker: "NVDA", company_name: "NVIDIA Corp", sector: "Technology", target_weight: 0.10, k_probs: KPROBS },
    { asof_friday: "2026-05-29", ticker: "AMD", company_name: "Advanced Micro Devices Inc", sector: "Technology", target_weight: 0.08, k_probs: KPROBS },
    { asof_friday: "2026-05-29", ticker: "AAPL", company_name: "Apple Inc", sector: "Technology", target_weight: 0.08, k_probs: KPROBS },
    { asof_friday: "2026-06-05", ticker: "NVDA", company_name: "NVIDIA Corp", sector: "Technology", target_weight: 0.10, k_probs: KPROBS },
    { asof_friday: "2026-06-05", ticker: "AVGO", company_name: "Broadcom Inc", sector: "Technology", target_weight: 0.07, k_probs: KPROBS },
    { asof_friday: "2026-06-05", ticker: "AAPL", company_name: "Apple Inc", sector: "Technology", target_weight: 0.08, k_probs: KPROBS },
  ],
  executions: [
    { asof: "2026-06-05", ticker: "NVDA", side: "BUY", qty: 70, realized_price: 132.45, midpoint: 132.40, slippage_bps: 3.8 },
    { asof: "2026-06-05", ticker: "AMD", side: "SELL", qty: 30, realized_price: 168.10, midpoint: 168.25, slippage_bps: 8.9 },
  ],
};
