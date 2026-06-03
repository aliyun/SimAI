import { test, expect } from '@playwright/test';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const LLD_PATH = path.resolve(__dirname, '..', '..', '..', 'lld.json');
const SHOT_DIR = path.resolve(__dirname, '..', '..', 'test-results', 'edg-visual');
fs.mkdirSync(SHOT_DIR, { recursive: true });

test('EDG visual: OXC port-level adjustment diagram', async ({ page }) => {
  test.setTimeout(90000);

  // ============ 1. Create a network from lld.json ============
  await page.goto('http://localhost:3000');
  await page.waitForLoadState('networkidle');
  await page.screenshot({ path: path.join(SHOT_DIR, '01-home.png'), fullPage: true });

  await page.click('text=新建组网');
  await expect(page.getByText('导入 lld.json（自动调用 EDG 初始化组网）')).toBeVisible({ timeout: 5000 });
  const lldInput = page.locator('input[type="file"][accept=".json"]').first();
  await lldInput.setInputFiles(LLD_PATH);
  await page.waitForTimeout(1500);
  await page.click('button:has-text("创建")');
  await page.waitForTimeout(3000);

  // ============ 2. Workload — pick a model that fits 4 servers × 8 GPU = 32 ============
  await page.goto('http://localhost:3000/deploy/workload');
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(500);
  // Pick first available model card / preset
  const modelBtn = page.locator('button, [role=button]').filter({ hasText: /LLaMA|GPT|DeepSeek/i }).first();
  if (await modelBtn.count() > 0) {
    await modelBtn.click();
    await page.waitForTimeout(500);
  }
  await page.screenshot({ path: path.join(SHOT_DIR, '02-workload.png'), fullPage: true });

  // Click 下一步 to reach EDG page
  const nextBtn = page.locator('button, a').filter({ hasText: /下一步|OXC 调节/ }).first();
  await nextBtn.click();
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(1500);

  // ============ 3. EDG page: baseline view ============
  await page.waitForSelector('.react-flow', { timeout: 10000 });
  await page.waitForTimeout(1000);
  await page.screenshot({ path: path.join(SHOT_DIR, '03-edg-baseline.png'), fullPage: true });

  // ============ 4. Trigger adjustment ============
  const adjustBtn = page.locator('button:has-text("调节网络")');
  await expect(adjustBtn).toBeVisible();
  await adjustBtn.click();
  await page.waitForTimeout(4000);  // allow the before/after to render
  await page.screenshot({ path: path.join(SHOT_DIR, '04-edg-adjusted.png'), fullPage: true });

  // Focused shot on the two panels
  const panels = page.locator('.react-flow').first();
  await panels.screenshot({ path: path.join(SHOT_DIR, '05-edg-panels.png') });

  console.log('Screenshots saved under:', SHOT_DIR);
});
