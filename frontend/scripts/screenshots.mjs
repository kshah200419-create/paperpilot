// Capture real screenshots of a running PaperPilot instance (headless Chromium via Playwright).
//   BASE_URL=http://localhost:5173 node scripts/screenshots.mjs
// Set CHROME_PATH to use a system Chrome instead of Playwright's bundled Chromium.
import { mkdirSync } from "node:fs";
import { chromium } from "playwright";

const BASE = process.env.BASE_URL ?? "http://localhost:5173";
const OUT = new URL("../../docs/screenshots/", import.meta.url).pathname;
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({ executablePath: process.env.CHROME_PATH || undefined });
const page = await browser.newPage({
  viewport: { width: 1280, height: 860 },
  deviceScaleFactor: 2,
});

async function shot(name, opts = {}) {
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${OUT}${name}.png`, ...opts });
  console.log(`saved ${name}.png`);
}

// 1. Search with highlighted hybrid results
await page.goto(
  `${BASE}/?q=${encodeURIComponent("efficient inference for mixture-of-experts models")}`,
);
await page.getByText("results for").waitFor();
await shot("search");

// 2. Ask: streamed, cited answer + citation cards
await page.goto(`${BASE}/ask`);
await page
  .getByLabel("Your question")
  .fill("How can mixture-of-experts inference be made cheaper?");
await page.getByRole("button", { name: /^Ask/ }).click();
await page.getByText(/ms · \d+ sources/).waitFor({ timeout: 60_000 });
await page.getByRole("button", { name: "Show source 1" }).first().click();
await shot("ask");

// 3. Trends dashboard
await page.goto(`${BASE}/trends`);
await page.getByText("Papers per month").waitFor();
await page.locator(".recharts-area").first().waitFor();
await page.locator(".recharts-scatter").first().waitFor();
await page.waitForTimeout(2000); // let Recharts entry animations finish
await shot("trends", { fullPage: true });

// 4. Paper detail
await page.goto(`${BASE}/papers/2609.04575`);
await page.getByText("Similar papers").waitFor();
await shot("paper");

// 5. Mobile search
await page.setViewportSize({ width: 390, height: 844 });
await page.goto(`${BASE}/?q=${encodeURIComponent("SLAM for legged robots")}`);
await page.getByText("results for").waitFor();
await shot("search-mobile");

await browser.close();
