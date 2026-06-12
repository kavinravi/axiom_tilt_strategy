import { describe, expect, it } from "vitest";
import { twrReturns } from "./twr";
import type { EquityPoint } from "./types";

const p = (date: string, nav: number, flow?: number | null): EquityPoint => ({
  date,
  nav,
  spy_close: null,
  flow,
});

describe("twrReturns", () => {
  it("matches plain NAV growth when there are no flows", () => {
    const out = twrReturns([p("d1", 100), p("d2", 110), p("d3", 99)]);
    expect(out[0]).toBe(0);
    expect(out[1]).toBeCloseTo(0.1, 10);
    expect(out[2]).toBeCloseTo(-0.01, 10);
  });

  it("a deposit day contributes zero growth", () => {
    // 100k → 176k on a 75k deposit with 1k of real P&L → +1%, not +76%.
    const out = twrReturns([p("d1", 100_000), p("d2", 176_000, 75_000)]);
    expect(out[1]).toBeCloseTo(0.01, 10);
  });

  it("a withdrawal is added back (no fake loss)", () => {
    const out = twrReturns([p("d1", 100_000), p("d2", 90_000, -10_000)]);
    expect(out[1]).toBeCloseTo(0, 10);
  });

  it("null/undefined flow is treated as zero", () => {
    const out = twrReturns([p("d1", 100), p("d2", 105, null), p("d3", 105)]);
    expect(out[1]).toBeCloseTo(0.05, 10);
    expect(out[2]).toBeCloseTo(0.05, 10);
  });

  it("carries flat across a non-positive prior NAV", () => {
    const out = twrReturns([p("d1", 100), p("d2", 0), p("d3", 50)]);
    expect(out[1]).toBe(-1); // 0/100: a measurable total loss
    expect(out[2]).toBe(-1); // prior nav 0 → unmeasurable → flat
  });

  it("empty input yields empty output", () => {
    expect(twrReturns([])).toEqual([]);
  });
});
