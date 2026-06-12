import { test, expect, type Page } from "@playwright/test";

async function login(page: Page) {
  await page.goto("/login");
  await page.getByLabel("Password").fill("testpass");
  await page.getByRole("button", { name: "Enter" }).click();
  await expect(page).toHaveURL(/\/now$/);
}

test("Holdings shows names, the 10% cap line, and a no-quote flag", async ({ page }) => {
  await login(page);
  await page.goto("/holdings");
  await expect(page.getByText("NVDA")).toBeVisible();
  await expect(page.getByText("10% per-stock cap", { exact: true })).toBeVisible();
  await expect(page.getByText("Today's P&L", { exact: true })).toBeVisible();
  await expect(page.getByText("no quote")).toBeVisible(); // the ZZZQ row has price 0
});

test("Holdings weights are vs the invested book, with an explicit cash row", async ({ page }) => {
  await login(page);
  await page.goto("/holdings");
  // Uninvested cash is its own row so the account still sums to 100%.
  await expect(page.getByText("Cash", { exact: true })).toBeVisible();
  await expect(page.getByText("awaiting next rebalance")).toBeVisible();
  // NVDA drifted above the 10% cap within the invested book (fixture: 10.60%).
  await expect(page.getByText("10.60%")).toBeVisible();
  await page.getByText("NVDA", { exact: true }).click();
  await expect(page.getByText("Above the 10% cap — trimmed at next rebalance")).toBeVisible();
  await expect(page.getByText(/Share of whole account/)).toBeVisible();
});

test("Holdings empty state before go-live", async ({ page }) => {
  await login(page);
  await page.goto("/holdings?scenario=empty");
  await expect(page.getByText(/No live positions yet/i)).toBeVisible();
});
