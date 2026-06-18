/**
 * E2E: ProcessList fail-loud + ResultsPage shows all finished tasks
 *
 * Covers two recent fixes:
 *   1. ProcessList no longer silently swallows fetch errors; it shows a
 *      "Updated Xs ago" indicator and goes red+"Stale" if polling stalls.
 *   2. ResultsPage now lists all finished tasks (not just those with
 *      EndToEnd.csv). Tasks without EndToEnd output (e.g. NS3 OXC, whose
 *      getopt is "ht:w:g:s:n:c:" and accepts no result-prefix flag) appear
 *      with a disabled button and "无 EndToEnd 输出" hint.
 *
 * Prereq: dev frontend on :3000, Flask backend on :5001, sqlite has at least
 * one finished SimAI_simulator[_oxc] (NS3) process and one finished
 * SimAI_analytical* (with EndToEnd.csv) for username='default'.
 */

import { test, expect, type Page } from '@playwright/test';

const BACKEND = 'http://localhost:5001';
const USERNAME = 'default';

async function getToken(): Promise<string> {
  const res = await fetch(`${BACKEND}/api/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: USERNAME }),
  });
  if (!res.ok) throw new Error(`Login failed: ${res.status}`);
  const data = await res.json();
  return data.token as string;
}

async function loginAndGoto(page: Page, path: string): Promise<void> {
  const token = await getToken();
  // Hit any page first so localStorage has a real origin.
  await page.goto('/');
  await page.evaluate(
    ({ t, u }) => {
      localStorage.setItem('ocs_sim_token', t);
      localStorage.setItem('ocs_sim_username', u);
    },
    { t: token, u: USERNAME },
  );
  await page.goto(path);
}

test.describe('ProcessList fail-loud', () => {
  test('Launch page renders Recent Processes with last-updated indicator', async ({ page }) => {
    await loginAndGoto(page, '/deploy/launch');

    // Title now reads "Recent Processes (N)" — not the misleading old "Running Processes".
    const heading = page.getByText(/Recent Processes \(\d+\)/);
    await expect(heading).toBeVisible({ timeout: 15000 });

    // Updated indicator shows up after the first successful poll.
    await expect(page.getByText(/Updated\s+/)).toBeVisible({ timeout: 10000 });

    // Status badges of historical processes — at least one finished or error.
    const statusBadge = page.locator('span').filter({ hasText: /^(finished|error)$/ }).first();
    await expect(statusBadge).toBeVisible({ timeout: 10000 });
  });
});

test.describe('ResultsPage shows all finished tasks', () => {
  test('Tasks without result files are visible but disabled with hint', async ({ page }) => {
    await loginAndGoto(page, '/results');

    await expect(page.getByRole('heading', { name: '已完成的仿真任务' })).toBeVisible({
      timeout: 10000,
    });

    // Historical NS3 OXC tasks ran with cwd=PROJECT_ROOT (before the fix) so
    // their workspace has no ncclFlowModel_*.csv -> they appear with the
    // generic "no result file" hint and a disabled button.
    const noResultHint = page.getByText(/无结果文件/).first();
    await expect(noResultHint).toBeVisible({ timeout: 10000 });

    const disabledTask = page
      .locator('button', { hasText: '无结果文件' })
      .first();
    await expect(disabledTask).toBeDisabled();
  });

  test('Analytical tasks (with EndToEnd) are clickable on subsequent pages', async ({ page }) => {
    await loginAndGoto(page, '/results');

    await expect(page.getByRole('heading', { name: '已完成的仿真任务' })).toBeVisible({ timeout: 10000 });

    // First page is filled with recent NS3 OXC tasks (no EndToEnd). Page 2
    // should expose analytical tasks with EndToEnd.csv.
    const page2 = page.getByRole('button', { name: '2', exact: true });
    await expect(page2).toBeVisible({ timeout: 5000 });
    await page2.click();
    await page.waitForTimeout(300);

    const enabledTask = page
      .locator('button:not([disabled])')
      .filter({ hasText: /EndToEnd\.csv/ })
      .first();
    await expect(enabledTask).toBeVisible({ timeout: 10000 });
  });

  test('Backend returns mix of with/without-EndToEnd tasks (regression: list was filtered before)', async () => {
    const token = await getToken();
    const res = await fetch(`${BACKEND}/api/simulation/results/list-tasks`, {
      headers: { 'X-Session-Token': token },
    });
    expect(res.ok).toBe(true);
    const data = await res.json();
    const tasks: Array<{ result_files?: { endtoend?: string } }> = data.tasks ?? [];

    const noEnd = tasks.filter((t) => !t.result_files?.endtoend).length;
    const withEnd = tasks.filter((t) => !!t.result_files?.endtoend).length;

    expect(noEnd).toBeGreaterThan(0);
    expect(withEnd).toBeGreaterThan(0);
  });
});
