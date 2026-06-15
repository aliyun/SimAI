import { useState, useEffect } from 'react';
import { listProcesses, killProcess, fetchProgress } from '../../api/simulation-api';

interface ProcessEntry {
  readonly pid: number;
  readonly status: string;
  readonly command: string;
  readonly started_at: string;
  readonly return_code: number | null;
  readonly error_message: string | null;
}

interface ProgressEntry {
  readonly pct: number;
  readonly current_layer: number;
  readonly total_layers: number;
  readonly estimated_remaining_sec: number;
}

const STALE_AFTER_MS = 15000;

function formatAge(ms: number): string {
  if (ms < 1000) return 'just now';
  if (ms < 60_000) return `${Math.floor(ms / 1000)}s ago`;
  if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m ago`;
  return `${Math.floor(ms / 3_600_000)}h ago`;
}

function formatETA(sec: number): string {
  if (!sec || sec <= 0) return '';
  if (sec < 60) return `${Math.round(sec)}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m ${Math.round(sec % 60)}s`;
  return `${Math.floor(sec / 3600)}h ${Math.floor((sec % 3600) / 60)}m`;
}

/** Extract workload filename from command line (the -w argument) */
function extractWorkloadName(command: string): string {
  const m = command.match(/-w\s+\S+\/([^/\s]+\.txt)/);
  if (m) return m[1];
  // Fallback: try any .txt file mentioned
  const m2 = command.match(/([^/\s]+\.txt)/);
  return m2 ? m2[1] : '';
}

function StatusBadge({ status }: { readonly status: string }) {
  const colorMap: Record<string, string> = {
    running: 'bg-[var(--color-accent-green)]/10 text-[var(--color-accent-green)]',
    finished: 'bg-[var(--color-bg-hover)] text-[var(--color-text-muted)]',
    dead: 'bg-[var(--color-bg-hover)] text-[var(--color-text-muted)]',
    killed: 'bg-[var(--color-bg-hover)] text-[var(--color-text-muted)]',
    timeout: 'bg-[var(--color-accent-yellow)]/10 text-[var(--color-accent-yellow)]',
    error: 'bg-[var(--color-accent-red)]/10 text-[var(--color-accent-red)]',
  };
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded ${colorMap[status] ?? 'bg-[var(--color-bg-hover)] text-[var(--color-text-muted)]'}`}>
      {status}
    </span>
  );
}

function ProgressBar({ pct, current, total, eta }: { pct: number; current: number; total: number; eta: string }) {
  return (
    <div className="flex items-center gap-2 min-w-0 flex-1">
      <div className="flex-1 h-2 bg-[var(--color-bg-hover)] rounded-full overflow-hidden min-w-[80px]">
        <div
          className="h-full rounded-full bg-gradient-to-r from-[var(--color-accent-cyan)] to-[var(--color-accent-blue)] transition-all duration-500"
          style={{ width: `${Math.max(pct, 2)}%` }}
        />
      </div>
      <span className="text-xs text-[var(--color-text-muted)] whitespace-nowrap tabular-nums">
        {pct.toFixed(0)}% {total > 0 ? `(${current}/${total})` : ''} {eta ? `· ${eta}` : ''}
      </span>
    </div>
  );
}

export function ProcessList() {
  const [processes, setProcesses] = useState<readonly ProcessEntry[]>([]);
  const [progress, setProgress] = useState<Record<number, ProgressEntry>>({});
  const [error, setError] = useState<string | null>(null);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<number | null>(null);
  const [now, setNow] = useState<number>(Date.now());
  const [expandedCmd, setExpandedCmd] = useState<number | null>(null);
  const [expandedErr, setExpandedErr] = useState<number | null>(null);

  const fetchProcesses = async () => {
    try {
      const list = await listProcesses();
      setProcesses(list as readonly ProcessEntry[]);
      setLastUpdatedAt(Date.now());
      setError(null);

      // Fetch progress for running processes in parallel
      const running = list.filter((p: ProcessEntry) => p.status === 'running');
      const results = await Promise.allSettled(
        running.map((p: ProcessEntry) => fetchProgress(p.pid))
      );
      const next: Record<number, ProgressEntry> = {};
      running.forEach((p: ProcessEntry, i: number) => {
        const r = results[i];
        if (r.status === 'fulfilled' && r.value) {
          next[p.pid] = {
            pct: r.value.pct ?? 0,
            current_layer: r.value.current_layer ?? 0,
            total_layers: r.value.total_layers ?? 0,
            estimated_remaining_sec: r.value.estimated_remaining_sec ?? 0,
          };
        }
      });
      setProgress(next);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Fetch failed');
    }
  };

  useEffect(() => {
    fetchProcesses();
    const timer = setInterval(fetchProcesses, 5000);
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => {
      clearInterval(timer);
      clearInterval(tick);
    };
  }, []);

  const handleKill = async (pid: number) => {
    try {
      await killProcess(pid);
      await fetchProcesses();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Kill failed');
    }
  };

  if (processes.length === 0 && !error) return null;

  const ageMs = lastUpdatedAt ? now - lastUpdatedAt : null;
  const isStale = ageMs !== null && ageMs > STALE_AFTER_MS;

  return (
    <div className="space-y-2">
      <h3 className="text-sm font-medium text-[var(--color-text-secondary)] flex items-center gap-2">
        <span>Running Processes ({processes.length})</span>
        {ageMs !== null && (
          <span
            className={`text-xs font-normal ${
              isStale
                ? 'text-[var(--color-accent-red)]'
                : 'text-[var(--color-text-muted)]'
            }`}
            title={isStale ? 'Auto-refresh stalled' : 'Auto-refreshes every 5s'}
          >
            {isStale ? `Stale · last update ${formatAge(ageMs)}` : `Updated ${formatAge(ageMs)}`}
          </span>
        )}
      </h3>
      {error && (
        <div className="text-xs text-[var(--color-accent-red)]">
          Refresh failed: {error}
          {processes.length > 0 && ' (showing cached data)'}
        </div>
      )}
      <div className="rounded-lg border border-[var(--color-bg-hover)] divide-y divide-[var(--color-bg-hover)]">
        {processes.map((proc) => {
          const isExpanded = expandedCmd === proc.pid;
          const isErrExpanded = expandedErr === proc.pid;
          const hasError = proc.status === 'error' || proc.error_message;
          const pg = progress[proc.pid];
          const isRunning = proc.status === 'running';
          const wlName = extractWorkloadName(proc.command);

          return (
            <div key={proc.pid}>
              <div className="px-4 py-2.5 text-sm">
                {/* Row 1: status + PID + workload name + buttons */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3 min-w-0">
                    <StatusBadge status={proc.status} />
                    <span className="text-[var(--color-text-primary)] font-mono text-xs shrink-0">
                      PID {proc.pid}
                    </span>
                    {wlName && (
                      <span className="text-[var(--color-text-muted)] text-xs truncate max-w-[300px]">
                        {wlName}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    {hasError && (
                      <button
                        type="button"
                        onClick={() => setExpandedErr(isErrExpanded ? null : proc.pid)}
                        className="text-xs text-[var(--color-accent-red)]/80 hover:text-[var(--color-accent-red)] transition-colors"
                      >
                        {isErrExpanded ? 'Hide error' : 'View error'}
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => setExpandedCmd(isExpanded ? null : proc.pid)}
                      className="text-xs text-[var(--color-accent-cyan)]/80 hover:text-[var(--color-accent-cyan)] transition-colors"
                    >
                      {isExpanded ? 'Hide cmd' : 'View cmd'}
                    </button>
                    {isRunning && (
                      <button
                        type="button"
                        onClick={() => handleKill(proc.pid)}
                        className="text-xs text-[var(--color-accent-red)] hover:underline"
                      >
                        Kill
                      </button>
                    )}
                  </div>
                </div>
                {/* Row 2: progress bar (running only) */}
                {isRunning && pg && pg.total_layers > 0 && (
                  <div className="mt-2">
                    <ProgressBar
                      pct={pg.pct}
                      current={pg.current_layer}
                      total={pg.total_layers}
                      eta={formatETA(pg.estimated_remaining_sec)}
                    />
                  </div>
                )}
                {isRunning && (!pg || pg.total_layers <= 0) && (
                  <div className="mt-1 text-xs text-[var(--color-text-muted)]">Starting…</div>
                )}
              </div>

              {/* Expanded command */}
              {isExpanded && (
                <div className="px-4 pb-2.5">
                  <div className="bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] rounded-lg p-3 font-mono text-xs text-[var(--color-text-primary)] break-all leading-relaxed">
                    {proc.command}
                  </div>
                </div>
              )}

              {/* Expanded error */}
              {isErrExpanded && proc.error_message && (
                <div className="px-4 pb-2.5">
                  <div className="bg-[var(--color-accent-red)]/5 border border-[var(--color-accent-red)]/20 rounded-lg p-3">
                    <div className="flex items-center gap-2 mb-1.5">
                      <span className="text-xs font-medium text-[var(--color-accent-red)]">
                        Error {proc.return_code != null && `(exit code: ${proc.return_code})`}
                      </span>
                    </div>
                    <pre className="text-xs text-[var(--color-text-primary)] whitespace-pre-wrap break-all leading-relaxed max-h-60 overflow-y-auto">
                      {proc.error_message}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
