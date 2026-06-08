import { describe, it, expect } from "vitest";
import { aggregateSectors, SP500_SECTORS } from "./sectors";

describe("aggregateSectors", () => {
  it("sums weights per sector", () => {
    const items = [
      { sector: "Technology", weight_actual: 0.08 },
      { sector: "Technology", weight_actual: 0.07 },
      { sector: "Healthcare", weight_actual: 0.05 },
    ];
    const result = aggregateSectors(items, "weight_actual");
    const tech = result.find((r) => r.sector === "Technology");
    const health = result.find((r) => r.sector === "Healthcare");
    expect(tech?.weight).toBeCloseTo(0.15);
    expect(health?.weight).toBeCloseTo(0.05);
  });

  it("sorts by weight descending", () => {
    const items = [
      { sector: "Healthcare", weight_actual: 0.05 },
      { sector: "Technology", weight_actual: 0.15 },
    ];
    const result = aggregateSectors(items, "weight_actual");
    expect(result[0].sector).toBe("Technology");
    expect(result[1].sector).toBe("Healthcare");
  });

  it("buckets null sector as Unknown", () => {
    const items = [
      { sector: null, weight_actual: 0.03 },
      { sector: undefined, weight_actual: 0.02 },
      { sector: "", weight_actual: 0.01 },
    ];
    const result = aggregateSectors(items, "weight_actual");
    expect(result).toHaveLength(1);
    expect(result[0].sector).toBe("Unknown");
    expect(result[0].weight).toBeCloseTo(0.06);
  });

  it("skips zero and negative weights", () => {
    const items = [
      { sector: "Technology", weight_actual: 0 },
      { sector: "Healthcare", weight_actual: -0.01 },
      { sector: "Industrials", weight_actual: 0.04 },
    ];
    const result = aggregateSectors(items, "weight_actual");
    expect(result).toHaveLength(1);
    expect(result[0].sector).toBe("Industrials");
  });

  it("returns empty array for empty input", () => {
    expect(aggregateSectors([], "weight_actual")).toEqual([]);
  });

  it("works with target_weight key (WeeklyRow style)", () => {
    const items = [
      { sector: "Technology", target_weight: 0.10 },
      { sector: "Communication Services", target_weight: 0.06 },
    ];
    const result = aggregateSectors(items, "target_weight");
    expect(result[0].sector).toBe("Technology");
    expect(result[0].weight).toBeCloseTo(0.10);
  });
});

describe("SP500_SECTORS", () => {
  it("weights normalize to ~1.0", () => {
    const total = SP500_SECTORS.reduce((sum, e) => sum + e.weight, 0);
    expect(total).toBeCloseTo(1.0, 5);
  });

  it("is sorted descending by weight", () => {
    for (let i = 1; i < SP500_SECTORS.length; i++) {
      expect(SP500_SECTORS[i - 1].weight).toBeGreaterThanOrEqual(SP500_SECTORS[i].weight);
    }
  });

  it("contains Technology as the largest sector", () => {
    expect(SP500_SECTORS[0].sector).toBe("Technology");
  });

  it("has 11 sectors", () => {
    expect(SP500_SECTORS).toHaveLength(11);
  });
});
