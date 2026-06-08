import type { Dataset } from "@/lib/types";

// Mirrors today's real state: one backfilled target week, nothing live yet.
const KPROBS = { "10": 0.2, "20": 0.3, "30": 0.3, "50": 0.2 };

export const empty: Dataset = {
  snapshot: null,
  equityCurve: [],
  holdings: [],
  weekly: [
    { asof_friday: "2026-05-29", ticker: "NVDA", company_name: "NVIDIA Corp", sector: "Technology", target_weight: 0.10, k_probs: KPROBS },
    { asof_friday: "2026-05-29", ticker: "AAPL", company_name: "Apple Inc", sector: "Technology", target_weight: 0.08, k_probs: KPROBS },
    { asof_friday: "2026-05-29", ticker: "MSFT", company_name: "Microsoft Corp", sector: "Technology", target_weight: 0.075, k_probs: KPROBS },
  ],
  executions: [],
};
