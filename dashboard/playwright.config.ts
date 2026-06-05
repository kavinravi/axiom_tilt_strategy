import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  use: { baseURL: "http://localhost:3100" },
  webServer: {
    command: "npm run build && npm run start -- -p 3100",
    url: "http://localhost:3100/login",
    timeout: 120_000,
    reuseExistingServer: false,
    env: { DASHBOARD_DATA_SOURCE: "fixture", DASHBOARD_PASSWORD: "testpass" },
  },
});
