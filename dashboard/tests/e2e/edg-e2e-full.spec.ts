import { test, expect } from '@playwright/test';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const LLD_PATH = path.resolve(__dirname, '..', '..', '..', 'lld.json');

test('EDG full E2E: create network with GPU params → auto task registration on launch', async ({ page }) => {
  test.setTimeout(120000);

  // Step 1: Homepage
  await page.goto('http://localhost:3000');
  await page.waitForLoadState('networkidle');

  // Step 2: Open create network modal
  await page.click('text=新建组网');
  await expect(page.getByText('导入 lld.json（自动调用 EDG 初始化组网）')).toBeVisible({ timeout: 5000 });

  // Step 3: Upload lld.json
  const lldInput = page.locator('input[type="file"][accept=".json"]').first();
  await lldInput.setInputFiles(LLD_PATH);
  await page.waitForTimeout(1000);

  // Verify auto-fill
  const nameInput = page.locator('input[placeholder="例如: Spectrum-X-128G"]');
  await expect(nameInput).toHaveValue(/EDG-/);

  // Verify GPU params are visible in modal
  await expect(page.locator('label:has-text("GPU/Server")')).toBeVisible();
  await expect(page.locator('label:has-text("GPU 型号")')).toBeVisible();
  await expect(page.locator('label:has-text("NVLink 带宽")')).toBeVisible();
  await expect(page.locator('label:has-text("链路带宽")')).toBeVisible();

  await page.screenshot({ path: 'test-results/e2e-v2-01-create-modal.png', fullPage: true });

  // Step 4: Click create
  await page.click('button:has-text("创建")');
  await page.waitForTimeout(3000);
  await page.screenshot({ path: 'test-results/e2e-v2-02-after-create.png', fullPage: true });

  // Step 5: Go to launch page
  await page.goto('http://localhost:3000/deploy/launch');
  await page.waitForLoadState('networkidle');

  // Step 6: Select the EDG network
  const networkSelect = page.locator('select').first();
  const options = await networkSelect.locator('option').allTextContents();
  console.log('Network options:', options);
  const edgOption = options.find(o => o.includes('EDG-'));
  if (edgOption) {
    await networkSelect.selectOption({ label: edgOption });
    await page.waitForTimeout(500);
  }

  // Step 7: Verify EDG auto-scheduling indicator appears (no manual upload needed)
  await expect(page.locator('text=EDG 自动调度')).toBeVisible({ timeout: 5000 });
  await expect(page.locator('text=启动时自动根据')).toBeVisible();

  // Verify NO manual npu_match upload or task_id input
  await expect(page.locator('label:has-text("npu_match.json")')).not.toBeVisible();
  await expect(page.locator('label:has-text("Task ID")')).not.toBeVisible();

  await page.screenshot({ path: 'test-results/e2e-v2-03-launch-page.png', fullPage: true });

  // Step 8: Check monitor
  await page.goto('http://localhost:3000/monitor');
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(2000);
  await page.screenshot({ path: 'test-results/e2e-v2-04-monitor.png', fullPage: true });

  console.log('E2E v2 flow completed successfully');
});
