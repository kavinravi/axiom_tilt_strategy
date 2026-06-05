import { describe, it, expect, beforeEach } from "vitest";
import { tokenFor, verifyToken, COOKIE } from "./auth";

beforeEach(() => { process.env.DASHBOARD_PASSWORD = "hunter2"; });

describe("auth token", () => {
  it("verifies a token minted from the current password", async () => {
    const tok = await tokenFor();
    expect(await verifyToken(tok)).toBe(true);
  });
  it("rejects a bad token", async () => {
    expect(await verifyToken("nope")).toBe(false);
    expect(await verifyToken(null)).toBe(false);
  });
  it("exposes a stable cookie name", () => { expect(COOKIE).toBe("dash_auth"); });
  it("rejects a token from a different password", async () => {
    const tok = await tokenFor();           // minted under "hunter2" (set by beforeEach)
    process.env.DASHBOARD_PASSWORD = "different";
    expect(await verifyToken(tok)).toBe(false);
  });
});
