import { test, expect, type Page } from "@playwright/test";

async function login(page: Page) {
  await page.goto("/login");
  await page.getByLabel("Password").fill("testpass");
  await page.getByRole("button", { name: "Enter" }).click();
  await expect(page).toHaveURL(/\/now$/);
}

test("History lists fridays and shows the selected week's portfolio", async ({ page }) => {
  await login(page);
  await page.goto("/history");
  await expect(page.getByRole("button", { name: "2026-05-29" })).toBeVisible();
  await expect(page.getByText("Persistence")).toBeVisible();
  await page.getByRole("button", { name: "2026-05-29" }).click();
  await expect(page.getByRole("cell", { name: "NVDA" }).first()).toBeVisible();
});

test("History single-week state hides turnover until >=2 weeks", async ({ page }) => {
  await login(page);
  await page.goto("/history?scenario=empty");
  await expect(page.getByText(/needs at least two weeks/i)).toBeVisible();
});
