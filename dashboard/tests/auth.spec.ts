import { test, expect } from "@playwright/test";

test("unauthenticated visit redirects to login", async ({ page }) => {
  await page.goto("/now");
  await expect(page).toHaveURL(/\/login$/);
  await expect(page.getByRole("heading", { name: "Axiom Tilt" })).toBeVisible();
});

test("wrong password is rejected, correct password enters", async ({ page }) => {
  await page.goto("/login");
  await page.getByLabel("Password").fill("wrong");
  await page.getByRole("button", { name: "Enter" }).click();
  await expect(page.getByText("Wrong password.")).toBeVisible();

  await page.getByLabel("Password").fill("testpass");
  await page.getByRole("button", { name: "Enter" }).click();
  await expect(page).toHaveURL(/\/now$/);
});
