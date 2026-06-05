import { describe, it, expect } from "vitest";
import { FixtureSource } from "./fixture";
import { populated } from "@/fixtures/populated";

const src = new FixtureSource(populated);

describe("FixtureSource", () => {
  it("returns the snapshot and equity curve", async () => {
    expect((await src.getSnapshot())?.nav).toBe(104230.55);
    expect((await src.getEquityCurve()).length).toBe(6);
  });
  it("lists distinct fridays newest-first", async () => {
    expect(await src.getWeeklyFridays()).toEqual(["2026-06-05", "2026-05-29"]);
  });
  it("returns one week's portfolio and its executions", async () => {
    expect((await src.getWeeklyPortfolio("2026-05-29")).length).toBe(3);
    expect((await src.getExecutions("2026-06-05")).length).toBe(2);
  });
});
