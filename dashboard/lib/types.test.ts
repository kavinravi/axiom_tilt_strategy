import { describe, it, expect } from "vitest";
import type { Dataset } from "./types";
import { emptyDataset } from "./types";

describe("emptyDataset", () => {
  it("is a fully-shaped, empty Dataset", () => {
    const d: Dataset = emptyDataset();
    expect(d.snapshot).toBeNull();
    expect(d.equityCurve).toEqual([]);
    expect(d.holdings).toEqual([]);
    expect(d.weekly).toEqual([]);
    expect(d.executions).toEqual([]);
  });
});
