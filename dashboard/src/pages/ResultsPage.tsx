import { useState, useCallback, useEffect, useMemo } from 'react';
import { DashboardShell } from '../components/layout/DashboardShell';
import { Header } from '../components/layout/Header';
import { WizardLayout } from '../components/wizard/WizardLayout';
import { FileUploadZone } from '../components/wizard/FileUploadZone';
import { ValidationBanner } from '../components/wizard/ValidationBanner';
import { OverviewMetrics } from '../components/charts/OverviewMetrics';
import { LayerTimingChart } from '../components/charts/LayerTimingChart';
import { BandwidthChart } from '../components/charts/BandwidthChart';
import { ComputeCommBreakdown } from '../components/charts/ComputeCommBreakdown';
import { DimensionBreakdown } from '../components/charts/DimensionBreakdown';
import { NodeTransferChart } from '../components/charts/NodeTransferChart';
import { Pagination } from '../components/widgets/Pagination';
import { useWizardStore } from '../stores/wizard-store';
import {
  parseEndToEnd,
  parseConsoleOutput,
  fetchCompletedTasks,
  fetchProcessStatus,
  loadResultByFilepath,
  fetchProgress,
} from '../api/simulation-api';
import type { CompletedTask, SimProgress } from '../api/simulation-api';

type ResultTab = 'overview' | 'layers' | 'bandwidth' | 'nodes';

const TABS: readonly { readonly value: ResultTab; readonly label: string }[] = [
  { value: 'overview', label: 'Overview' },
  { value: 'layers', label: 'Timeline' },
  { value: 'bandwidth', label: 'Bandwidth' },
  { value: 'nodes', label: 'Node Transfer' },
];

function formatTime(ts: number): string {
  const d = new Date(ts * 1000);
  return d.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function extractBinaryName(command: string | null): string {
  if (!command) return '未知';
  const parts = command.split('/');
  const binary = parts[parts.length - 1]?.split(' ')[0] ?? '';
  return binary || '未知';
}

function extractLabel(task: CompletedTask): string {
  // Prefer the workload basename — it identifies the *training task* (e.g.
  // "DeepSeek-16B-tp4-pp1-dp8") rather than the binary's hardcoded result
  // prefix ("ncclFlowModel_") which is the same for every NS3 run.
  if (task.workload_path) {
    const base = task.workload_path.split('/').pop() ?? '';
    const stem = base.replace(/\.(txt|json)$/, '');
    if (stem) return stem;
  }
  if (task.result_prefix) {
    return task.result_prefix.replace(/-$/, '').replace(/_$/, '') || extractBinaryName(task.command);
  }
  return extractBinaryName(task.command);
}

export function ResultsPage() {
  const {
    endtoendData, setEndtoendData,
    consoleData, setConsoleData,
    completeStep,
  } = useWizardStore();

  const [activeTab, setActiveTab] = useState<ResultTab>('overview');
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [tasks, setTasks] = useState<readonly CompletedTask[]>([]);
  const [tasksLoading, setTasksLoading] = useState(true);
  const [showUpload, setShowUpload] = useState(false);
  const [taskPage, setTaskPage] = useState(1);
  const TASK_PAGE_SIZE = 10;
  const [progressPid, setProgressPid] = useState('');
  const [progress, setProgress] = useState<SimProgress | null>(null);

  // Fetch completed tasks on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const result = await fetchCompletedTasks();
        if (!cancelled) setTasks(result);
      } catch {
        // Silently fail — user can still upload manually
      } finally {
        if (!cancelled) setTasksLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Auto-load result when resultPrefix is available (persisted via sessionStorage)
  // Also polls the process if it's still running
  useEffect(() => {
    const storedPrefix = sessionStorage.getItem('simai_result_prefix');
    const storedPid = sessionStorage.getItem('simai_result_pid');
    if (!storedPrefix || endtoendData) return;
    let cancelled = false;
    let pollInterval: ReturnType<typeof setInterval> | null = null;

    async function tryLoad(): Promise<boolean> {
      // If we have a PID, check process status first
      if (storedPid) {
        const pidNum = parseInt(storedPid, 10);
        if (!isNaN(pidNum)) {
          try {
            const proc = await fetchProcessStatus(pidNum);
            if (cancelled) return false;
            if (proc?.status === 'running') return false; // still running — keep polling
          } catch {
            // Process not found — proceed to try loading anyway
          }
        }
      }

      // Try to load the result
      try {
        const freshTasks = await fetchCompletedTasks();
        if (cancelled) return false;
        setTasks(freshTasks);
        const fromProcess = freshTasks.find(
          (t) => t.result_prefix === storedPrefix && t.tracking_id != null && t.result_files?.endtoend,
        );
        const standalone = freshTasks.find(
          (t) => t.result_prefix === storedPrefix && t.tracking_id == null && t.result_files?.endtoend,
        );
        const match = fromProcess ?? standalone;
        // Fall back to the most recent result if the stored prefix is stale.
        const recent = !match ? freshTasks
          .filter(t => t.result_files?.endtoend)
          .sort((a, b) => (b.result_prefix ?? '').localeCompare(a.result_prefix ?? ''))[0]
          : null;
        const chosen = match ?? recent;
        if (chosen?.result_files?.endtoend) {
          const result = await loadResultByFilepath(chosen.result_files.endtoend);
          if (!cancelled) {
            setEndtoendData({ ...result, totals: result.totals ?? {} });
            completeStep('results');
            sessionStorage.removeItem('simai_result_prefix');
            sessionStorage.removeItem('simai_result_pid');
            return true; // loaded successfully
          }
        }
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : '自动加载结果失败');
      }
      return false;
    }

    (async () => {
      setIsLoading(true);
      setError(null);
      const loaded = await tryLoad();
      if (cancelled) return;
      if (!loaded && storedPid) {
        // Poll every 3s until result appears
        pollInterval = setInterval(async () => {
          if (cancelled) { if (pollInterval) clearInterval(pollInterval); return; }
          const done = await tryLoad();
          if (done || cancelled) { if (pollInterval) clearInterval(pollInterval); setIsLoading(false); }
        }, 3000);
      }
      setIsLoading(false);
    })();
    return () => { cancelled = true; if (pollInterval) clearInterval(pollInterval); };
  }, [endtoendData, setEndtoendData, completeStep]);

  const handleLoadTask = useCallback(async (task: CompletedTask) => {
    const endtoendPath = task.result_files?.endtoend;
    if (!endtoendPath) {
      setError('该任务没有找到 EndToEnd 结果文件');
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const result = await loadResultByFilepath(endtoendPath, task.workload_path);
      setEndtoendData({
        ...result,
        totals: result.totals ?? {},
      });
      completeStep('results');
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载结果失败');
    } finally {
      setIsLoading(false);
    }
  }, [setEndtoendData, completeStep]);

  const handleEndToEndUpload = useCallback(async (content: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const result = await parseEndToEnd(content);
      setEndtoendData(result);
      completeStep('results');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Parse failed');
    } finally {
      setIsLoading(false);
    }
  }, [setEndtoendData, completeStep]);

  const handleConsoleUpload = useCallback(async (content: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const lines = content.split('\n');
      const result = await parseConsoleOutput(lines);
      setConsoleData(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Parse failed');
    } finally {
      setIsLoading(false);
    }
  }, [setConsoleData]);

  const hasData = endtoendData || consoleData;
  const displayTasks = tasks;
  const pagedTasks = useMemo(() => {
    const start = (taskPage - 1) * TASK_PAGE_SIZE;
    return displayTasks.slice(start, start + TASK_PAGE_SIZE);
  }, [displayTasks, taskPage]);

  return (
    <DashboardShell>
      <Header title="仿真结果分析" backLink="/" />

      <WizardLayout
        title="仿真结果分析"
        description="选择已完成的仿真任务或上传结果文件，可视化性能指标"
        actions={null}
        wide
      >
        {error && <ValidationBanner type="error" message={error} />}

        <div className="space-y-6">
          {/* Source selection: task list or upload */}
          {!hasData && (
            <div className="space-y-6">
              {/* Completed tasks */}
              <div className="space-y-3">
                <h3 className="text-sm font-medium text-[var(--color-text-secondary)]">
                  {/* ---- Progress checker ---- */}
                  <div className="mb-4 p-4 rounded-lg border border-[var(--color-bg-hover)] bg-[var(--color-bg-primary)]/30">
                    <div className="flex items-center gap-2 mb-2">
                      <span className="text-sm font-medium text-[var(--color-text-primary)]">查看仿真进度</span>
                      <input
                        type="number"
                        placeholder="输入 PID"
                        value={progressPid}
                        onChange={(e) => setProgressPid(e.target.value)}
                        className="text-xs px-2 py-1 rounded border border-[var(--color-bg-hover)] bg-[var(--color-bg-primary)] text-[var(--color-text-primary)] w-28"
                      />
                      <button
                        type="button"
                        onClick={async () => {
                          const p = parseInt(progressPid, 10);
                          if (isNaN(p)) return;
                          const pro = await fetchProgress(p);
                          setProgress(pro);
                        }}
                        className="text-xs px-2 py-1 rounded bg-[var(--color-accent-blue)]/20 text-[var(--color-accent-blue)] hover:bg-[var(--color-accent-blue)]/30"
                      >
                        查询
                      </button>
                      {progress && (
                        <button type="button" onClick={() => { setProgress(null); setProgressPid(''); }} className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)]">
                          清除
                        </button>
                      )}
                    </div>
                    {progress && (
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs text-[var(--color-text-muted)]">
                          <span>层 {progress.current_layer}/{progress.total_layers} ({progress.pct}%)</span>
                          {progress.estimated_remaining_sec > 0 && (
                            <span>预计剩余 {Math.round(progress.estimated_remaining_sec / 60)} 分钟</span>
                          )}
                        </div>
                        <div className="w-full h-2 rounded-full bg-[var(--color-bg-hover)] overflow-hidden">
                          <div
                            className="h-full rounded-full bg-[var(--color-accent-blue)] transition-all duration-500"
                            style={{ width: `${Math.min(progress.pct, 100)}%` }}
                          />
                        </div>
                        <div className="text-xs text-[var(--color-text-muted)]">
                          已运行 {Math.round(progress.elapsed_sec / 60)} 分钟 · 状态: {progress.status === 'running' ? '运行中' : progress.status}
                        </div>
                      </div>
                    )}
                    {!progress && (
                      <p className="text-xs text-[var(--color-text-muted)]">输入 PID 查看仿真任务的实时进度</p>
                    )}
                  </div>

                  已完成的仿真任务
                </h3>

                {tasksLoading && (
                  <div className="text-center py-6 text-sm text-[var(--color-text-muted)]">
                    加载中...
                  </div>
                )}

                {!tasksLoading && displayTasks.length === 0 && (
                  <div className="text-center py-6 text-sm text-[var(--color-text-muted)] border border-dashed border-[var(--color-bg-hover)] rounded-lg">
                    暂无已完成的仿真任务
                  </div>
                )}

                {!tasksLoading && displayTasks.length > 0 && (
                  <div className="space-y-2">
                    {pagedTasks.map((task, idx) => {
                      const hasEndToEnd = !!task.result_files?.endtoend;
                      return (
                      <button
                        key={task.tracking_id ?? `file-${idx}`}
                        type="button"
                        onClick={() => handleLoadTask(task)}
                        disabled={isLoading || !hasEndToEnd}
                        title={task.status === 'error' ? (task.error_message ?? `异常退出 (信号 ${Math.abs(task.return_code ?? 0)})`) : hasEndToEnd ? undefined : '该任务暂无结果文件（可能仍在运行、未输出或被覆盖）'}
                        className="w-full text-left p-3 rounded-lg border border-[var(--color-bg-hover)] bg-[var(--color-bg-primary)]/50 hover:border-[var(--color-accent-blue)]/40 hover:bg-[var(--color-accent-blue)]/5 transition-colors disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:border-[var(--color-bg-hover)] disabled:hover:bg-[var(--color-bg-primary)]/50"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              {task.tracking_id != null && (
                                <span className="text-xs px-1.5 py-0.5 rounded bg-[var(--color-accent-green)]/10 text-[var(--color-accent-green)] font-mono">
                                  #{task.tracking_id}
                                </span>
                              )}
                              {task.pid != null && (
                                <span className="text-xs text-[var(--color-text-muted)] font-mono">
                                  PID {task.pid}
                                </span>
                              )}
                              {task.status === 'error' && task.return_code != null && (
                                <span className="text-xs px-1.5 py-0.5 rounded bg-red-500/10 text-red-500" title={task.error_message ?? `异常退出 (信号 ${Math.abs(task.return_code)})`}>
                                  {task.return_code === -11 ? '内存访问错误' : task.return_code === -9 ? '内存不足被终止' : task.return_code === -6 ? '程序异常终止' : task.return_code === -15 ? '超时被终止' : `异常退出(${task.return_code})`}
                                </span>
                              )}
                              <span className="text-sm font-medium text-[var(--color-text-primary)] truncate">
                                {extractLabel(task)}
                              </span>
                            </div>
                            <div className="text-xs text-[var(--color-text-muted)] mt-1 truncate">
                              {task.finished_at ? formatTime(task.finished_at) : ''}
                              {task.result_files?.endtoend && (
                                <span className="ml-2">
                                  · {task.result_files.endtoend.split('/').pop()}
                                </span>
                              )}
                              {!hasEndToEnd && (
                                <span className="ml-2 italic">
                                  · 无结果文件
                                </span>
                              )}
                            </div>
                          </div>
                          <svg className="w-4 h-4 text-[var(--color-text-muted)] flex-shrink-0" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M9 5l7 7-7 7" />
                          </svg>
                        </div>
                      </button>
                      );
                    })}
                    <Pagination
                      currentPage={taskPage}
                      totalItems={displayTasks.length}
                      pageSize={TASK_PAGE_SIZE}
                      onPageChange={setTaskPage}
                    />
                  </div>
                )}
              </div>

              {/* Toggle upload */}
              <div className="text-center">
                <button
                  type="button"
                  onClick={() => setShowUpload(!showUpload)}
                  className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-accent-blue)] transition-colors"
                >
                  {showUpload ? '收起手动上传' : '或者手动上传文件 ↓'}
                </button>
              </div>

              {/* Manual upload (collapsible) */}
              {showUpload && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium text-[var(--color-text-secondary)]">
                      EndToEnd CSV
                    </h3>
                    <FileUploadZone
                      accept=".csv"
                      label="Upload EndToEnd.csv"
                      onFileContent={handleEndToEndUpload}
                    />
                  </div>
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium text-[var(--color-text-secondary)]">
                      Console Output
                    </h3>
                    <FileUploadZone
                      accept=".txt,.log"
                      label="Upload console output (.txt/.log)"
                      onFileContent={handleConsoleUpload}
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Loading */}
          {isLoading && (
            <div className="text-center py-8 text-sm text-[var(--color-text-muted)]">
              解析结果中...
            </div>
          )}

          {/* Results visualization */}
          {hasData && (
            <>
              {/* Tab bar */}
              <div className="flex gap-1 border-b border-[var(--color-bg-hover)]">
                {TABS.map((tab) => (
                  <button
                    key={tab.value}
                    type="button"
                    onClick={() => setActiveTab(tab.value)}
                    className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
                      activeTab === tab.value
                        ? 'border-[var(--color-accent-blue)] text-[var(--color-accent-blue)]'
                        : 'border-transparent text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}

                {/* Back to task list */}
                <button
                  type="button"
                  onClick={() => { setEndtoendData(null); setConsoleData(null); }}
                  className="ml-auto text-xs text-[var(--color-text-muted)] hover:text-[var(--color-accent-red)] transition-colors px-2"
                >
                  ← 返回任务列表
                </button>
              </div>

              {/* Overview tab */}
              {activeTab === 'overview' && (
                <div className="space-y-6">
                  <OverviewMetrics
                    totalTime={endtoendData?.totals.total_time ?? null}
                    totalCompute={endtoendData?.totals.total_compute ?? null}
                    totalExposed={endtoendData?.totals.total_exposed ?? null}
                    finishTime={consoleData?.finish_time ?? null}
                    streamsInjected={consoleData?.streams_injected ?? null}
                    streamsFinished={consoleData?.streams_finished ?? null}
                  />

                  {endtoendData && endtoendData.totals.total_compute != null && endtoendData.totals.total_exposed != null && (
                    <ComputeCommBreakdown
                      totalCompute={endtoendData.totals.total_compute}
                      totalExposed={endtoendData.totals.total_exposed}
                    />
                  )}

                  {endtoendData && (
                    <DimensionBreakdown dimensions={endtoendData.dimensions} />
                  )}
                </div>
              )}

              {/* Layer timing tab */}
              {activeTab === 'layers' && endtoendData && endtoendData.layers.length > 0 && (
                <LayerTimingChart layers={endtoendData.layers} runName={endtoendData.run_name} />
              )}
              {activeTab === 'layers' && (!endtoendData || endtoendData.layers.length === 0) && (
                <div className="text-center py-12 text-sm text-[var(--color-text-muted)]">
                  暂无 Layer Timing 数据
                </div>
              )}

              {/* Bandwidth tab */}
              {activeTab === 'bandwidth' && endtoendData && endtoendData.layers.length > 0 && (
                <BandwidthChart layers={endtoendData.layers} />
              )}
              {activeTab === 'bandwidth' && (!endtoendData || endtoendData.layers.length === 0) && (
                <div className="text-center py-12 text-sm text-[var(--color-text-muted)]">
                  暂无 Bandwidth 数据
                </div>
              )}

              {/* Node transfer tab */}
              {activeTab === 'nodes' && consoleData && (
                <NodeTransferChart nodes={consoleData.nodes} />
              )}
              {activeTab === 'nodes' && !consoleData && (
                <div className="text-center py-12 text-sm text-[var(--color-text-muted)]">
                  暂无 Node Transfer 数据 — 请上传 Console Output 文件
                </div>
              )}
            </>
          )}
        </div>
      </WizardLayout>
    </DashboardShell>
  );
}
