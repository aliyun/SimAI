import { useState, useEffect } from 'react';
import { listProcesses, killProcess } from '../../api/simulation-api';

interface ProcessEntry {
  readonly pid: number;
  readonly status: string;
  readonly command: string;
  readonly started_at: string;
}

const STALE_AFTER_MS = 15000;

function formatAge(ms: number): string {
  if (ms < 1000) return 'just now';
  if (ms < 60_000) return `${Math.floor(ms / 1000)}s ago`;
  if (ms < 3_600_000) return `${Math.floor(ms / 60_000)}m ago`;
  return `${Math.floor(ms / 3_600_000)}h ago`;
}

export function ProcessList() {
  const [processes, setProcesses] = useState<readonly ProcessEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdatedAt, setLastUpdatedAt] = useState<number | null>(null);
  const [now, setNow] = useState<number>(Date.now());

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
        <span>Recent Processes ({processes.length})</span>
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
        {processes.map((proc) => (
          <div
            key={proc.pid}
            className="px-4 py-2.5 flex items-center justify-between text-sm"
          >
            <div className="flex items-center gap-3">
              <span className={`text-xs px-1.5 py-0.5 rounded ${
                proc.status === 'running'
                  ? 'bg-[var(--color-accent-green)]/10 text-[var(--color-accent-green)]'
                  : 'bg-[var(--color-bg-hover)] text-[var(--color-text-muted)]'
              }`}>
                {proc.status}
              </span>
              <span className="text-[var(--color-text-primary)] font-mono text-xs">
                PID {proc.pid}
              </span>
              <span className="text-[var(--color-text-muted)] text-xs truncate max-w-xs">
                {proc.command}
              </span>
            </div>
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
        ))}
      </div>
    </div>
  );
}
