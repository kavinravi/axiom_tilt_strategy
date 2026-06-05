import { test, expect, type Page } from "@playwright/test";

async function login(page: Page) {
  await page.goto("/login");
  await page.getByLabel("Password").fill("testpass");
  await page.getByRole("button", { name: "Enter" }).click();
  await expect(page).toHaveURL(/\/now$/);
}

test("Now tab shows hero stats for populated data", async ({ page }) => {
  await login(page);
  await expect(page.getByText("Portfolio Value")).toBeVisible();
  await expect(page.getByText("$104,230.55")).toBeVisible();
  await expect(page.getByText("Regime Call")).toBeVisible();
  await expect(page.getByText("Risk")).toBeVisible();

  // range pills present and interactive
  await expect(page.getByRole("button", { name: "1W" })).toBeVisible();
  await expect(page.getByRole("button", { name: "All" })).toBeVisible();
  await page.getByRole("button", { name: "1W" }).click();
  await expect(page.getByText("Portfolio Value")).toBeVisible(); // chart still renders after range change
});

test("Now tab shows the go-live empty state when there is no snapshot", async ({ page }) => {
  await login(page);
  await page.goto("/now?scenario=empty");
  await expect(page.getByText(/builds forward from go-live/i)).toBeVisible();
});
