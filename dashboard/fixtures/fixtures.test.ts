import { describe, it, expect } from "vitest";
import { populated } from "./populated";
import { empty } from "./empty";

describe("fixtures", () => {
  it("populated has a snapshot, equity history, holdings, and >=2 weeks", () => {
    expect(populated.snapshot).not.toBeNull();
    expect(populated.equityCurve.length).toBeGreaterThan(1);
    expect(populated.holdings.length).toBeGreaterThan(0);
    const fridays = new Set(populated.weekly.map((w) => w.asof_friday));
    expect(fridays.size).toBeGreaterThanOrEqual(2);
  });
  it("empty mirrors pre-go-live: no snapshot/holdings/equity, exactly one week", () => {
    expect(empty.snapshot).toBeNull();
    expect(empty.equityCurve).toEqual([]);
    expect(empty.holdings).toEqual([]);
    expect(empty.executions).toEqual([]);
    const fridays = new Set(empty.weekly.map((w) => w.asof_friday));
    expect(fridays.size).toBe(1);
  });
});
