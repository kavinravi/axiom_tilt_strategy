import type { Dataset, Holding } from "@/lib/types";

const KPROBS = { "10": 0.15, "20": 0.25, "30": 0.35, "50": 0.25 };

// Holdings are derived from shares × price so market_value / weight_actual /
// invested_pct stay mutually consistent (weight_actual is vs total NAV, as the
// publisher writes it; the UI re-bases onto the invested book).
const NAV = 104230.55;
const h = (
  ticker: string, company_name: string | null, sector: string | null,
  shares: number, price: number, weight_target: number,
  avg_cost: number | null, daily_pnl: number | null,
): Holding => ({
  ticker, company_name, sector, shares, price,
  market_value: shares * price,
  weight_actual: (shares * price) / NAV,
  weight_target, avg_cost,
  unrealized_pnl: avg_cost == null ? null : (price - avg_cost) * shares,
  daily_pnl,
});

// ~56% invested (a deposit waits in cash), NVDA drifted just above the 10% cap.
const HOLDINGS: Holding[] = [
  h("NVDA", "NVIDIA Corp", "Technology", 47, 131.64, 0.10, 128.2, 87.5),
  h("AAPL", "Apple Inc", "Technology", 27, 211.86, 0.098, 215.4, -41.2),
  h("MSFT", "Microsoft Corp", "Technology", 12, 457.30, 0.094, 432.8, 23.4),
  h("AVGO", "Broadcom Inc", "Technology", 21, 250.10, 0.09, 245.3, 65.7),
  h("GOOGL", "Alphabet Inc", "Communication Services", 28, 179.30, 0.086, 176.0, -12.3),
  h("AMZN", "Amazon.com Inc", "Consumer Cyclical", 22, 217.60, 0.082, 210.0, 31.0),
  h("META", "Meta Platforms Inc", "Communication Services", 7, 658.90, 0.079, 640.0, -22.6),
  h("LLY", "Eli Lilly & Co", "Healthcare", 6, 739.40, 0.076, 750.0, 12.4),
  h("JPM", "JPMorgan Chase & Co", "Financial Services", 16, 266.40, 0.073, 259.1, 18.9),
  h("XOM", "Exxon Mobil Corp", "Energy", 38, 110.60, 0.072, 113.0, -15.2),
  h("UNH", "UnitedHealth Group Inc", "Healthcare", 14, 300.20, 0.072, 296.5, 27.3),
  h("COST", "Costco Wholesale Corp", "Consumer Defensive", 4, 1050.60, 0.072, 1021.0, 9.1),
  h("ZZZQ", null, null, 10, 0, 0.04, null, null), // no-quote row
];
const INVESTED = HOLDINGS.reduce((s, x) => s + (x.market_value ?? 0), 0);

export const populated: Dataset = {
  snapshot: {
    asof: "2026-06-08T19:00:00+00:00",
    nav: NAV,
    day_pnl: 612.4,
    day_pnl_pct: 0.0059,
    total_return: 0.0423,
    spy_return: 0.0311,
    n_positions: HOLDINGS.length - 1,
    invested_pct: INVESTED / NAV,
    k_probs: KPROBS,
    regime_features: { vix: 14.2, y10: 0.0431, term_spread: 0.0052, spy_ret: 0.012, spy_vol: 0.009 },
    risk: { current_drawdown: -0.008, max_drawdown: -0.021, sharpe: 1.42, ann_vol: 0.11 },
    turnover: { added: ["AVGO"], dropped: ["AMD"], turnover_frac: 0.18 },
    week_vs_spy: {
      baseline_date: "2026-06-05",
      portfolio_return: 0.0123,
      spy_return: 0.0046,
      excess_return: 0.0077,
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
  holdings: HOLDINGS,
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
