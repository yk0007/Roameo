import { expect, test } from "@playwright/test";

test("landing page renders the Roameo brand shell", async ({ page }) => {
  await page.goto("/");

  await expect(
    page.getByRole("heading", { name: /plan your perfect adventure/i })
  ).toBeVisible();
  await expect(page.getByRole("link", { name: /log in/i })).toBeVisible();
});

test("login page uses the single auth entry route", async ({ page }) => {
  await page.goto("/auth/login");

  await expect(page).toHaveURL(/\/auth\/login$/);
  await expect(
    page.getByRole("heading", { name: /connect supabase to get started/i })
  ).toBeVisible();
});

test("dashboard redirects unauthenticated users to login", async ({ page }) => {
  await page.goto("/dashboard");

  await page.waitForURL("**/auth/login", { timeout: 60_000 });
  await expect(page).toHaveURL(/\/auth\/login$/);
});

test("workspace routes share the same unauthenticated redirect path", async ({ page }) => {
  await page.goto("/chat?sessionId=test-session");

  await page.waitForURL("**/auth/login", { timeout: 60_000 });
  await expect(page).toHaveURL(/\/auth\/login$/);

  await page.goto("/profile");
  await page.waitForURL("**/auth/login", { timeout: 60_000 });
  await expect(page).toHaveURL(/\/auth\/login$/);
});
