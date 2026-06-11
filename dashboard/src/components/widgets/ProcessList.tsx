import { useState, useEffect } from 'react';
import { listProcesses, killProcess } from '../../api/simulation-api';

interface ProcessEntry {
  readonly pid: number;
  readonly status: string;
  readonly command: string;
  readonly started_at: string;
  readonly return_code: number | null;
  readonly error_message: string | null;
}

const STALE_AFTER_MS = 15000;

function formatAge(ms: number): string {
  if (ms < 1000) return 'just now';
  if (ms < 60_000) return `${Math.floor(ms / 1000)}s ago`;
  if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m ago`;
  return `${Math.floor(ms / 3_600_000)}h ago`;
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

export function ProcessList() {
  const [processes, setProcesses] = useState<readonly ProcessEntry[]>([]);
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
          return (
            <div key={proc.pid}>
              <div className="px-4 py-2.5 flex items-center justify-between text-sm">
                <div className="flex items-center gap-3 min-w-0">
                  <StatusBadge status={proc.status} />
                  <span className="text-[var(--color-text-primary)] font-mono text-xs shrink-0">
                    PID {proc.pid}
                  </span>
                  <span className="text-[var(--color-text-muted)] text-xs truncate max-w-[300px]">
                    {proc.command}
                  </span>
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
                  {proc.status === 'running' && (
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
