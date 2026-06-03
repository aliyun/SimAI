import { test, expect } from '@playwright/test';

test('EDG integration: Create network modal has lld.json upload', async ({ page }) => {
  await page.goto('http://localhost:3000');
  await page.waitForLoadState('networkidle');

  // Click "+ 新建组网" to open modal
  await page.click('text=新建组网');

  // Modal should be visible
  await expect(page.getByText('导入 lld.json（自动调用 EDG 初始化组网）')).toBeVisible({ timeout: 5000 });

  await page.screenshot({ path: 'test-results/edg-create-modal.png', fullPage: true });
});

test('EDG integration: LaunchPage has task registration', async ({ page }) => {
  await page.goto('http://localhost:3000/deploy/launch');
  await page.waitForLoadState('networkidle');

  await page.screenshot({ path: 'test-results/edg-launch-page.png', fullPage: true });

  const edgTaskSection = page.locator('text=AI 训练任务调度 (EDG)');
  await expect(edgTaskSection).toBeVisible({ timeout: 10000 });

  const taskIdInput = page.locator('input[value="T001"]');
  await expect(taskIdInput).toBeVisible();

  const npuUpload = page.getByText('npu_match.json', { exact: true });
  await expect(npuUpload).toBeVisible();
});
