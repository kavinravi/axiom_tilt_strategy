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
    expect(t!.turnover_frac).toBeCloseTo(0.075);
  });

  it("returns null when the selected week is not in the list", () => {
    expect(computeTurnover(rows, ["2026-06-05", "2026-05-29"], "2099-01-01")).toBeNull();
  });
});
