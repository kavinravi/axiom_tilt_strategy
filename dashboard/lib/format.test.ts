import { describe, it, expect } from "vitest";
import { fmtMoney, fmtPct, fmtSignedPct, fmtBps, asOfET } from "./format";

describe("format", () => {
  it("money", () => { expect(fmtMoney(104230.55)).toBe("$104,230.55"); });
  it("null -> dash", () => { expect(fmtMoney(null)).toBe("—"); expect(fmtPct(null)).toBe("—"); });
  it("pct from fraction", () => { expect(fmtPct(0.0423)).toBe("4.23%"); });
  it("signed pct", () => { expect(fmtSignedPct(0.0059)).toBe("+0.59%"); expect(fmtSignedPct(-0.01)).toBe("-1.00%"); });
  it("bps", () => { expect(fmtBps(3.8)).toBe("3.8 bps"); });
  it("asOfET renders an ET clock label", () => {
    expect(asOfET("2026-06-08T19:00:00+00:00")).toBe("as of 3:00 PM ET");
  });
});
