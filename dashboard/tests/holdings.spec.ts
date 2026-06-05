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
  await expect(page.getByText("10% cap", { exact: true })).toBeVisible();
  await expect(page.getByText("no quote")).toBeVisible(); // the ZZZQ row has price 0
});

test("Holdings empty state before go-live", async ({ page }) => {
  await login(page);
  await page.goto("/holdings?scenario=empty");
  await expect(page.getByText(/No live positions yet/i)).toBeVisible();
});
