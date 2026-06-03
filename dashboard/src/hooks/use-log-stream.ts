import { useEffect, useRef, useCallback, useState } from 'react';
import { fetchProcessLogs } from '../api/simulation-api';

interface UseLogStreamOptions {
  readonly pid: number | null;
  readonly intervalMs?: number;
  readonly enabled?: boolean;
}

interface UseLogStreamResult {
  readonly logs: readonly string[];
  readonly status: string;
  readonly error: string | null;
  readonly refresh: () => void;
}

export function useLogStream(options: UseLogStreamOptions): UseLogStreamResult {
  const { pid, intervalMs = 2000, enabled = true } = options;
  const [logs, setLogs] = useState<readonly string[]>([]);
  const [status, setStatus] = useState('unknown');
  const [error, setError] = useState<string | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchLogs = useCallback(async () => {
    if (!pid) return;
    try {
      const data = await fetchProcessLogs(pid);
      setLogs(data.logs);
      setStatus(data.status);
      setError(null);

      // Stop polling if process is done
      if (['exited', 'killed', 'finished', 'error', 'dead', 'timeout'].includes(data.status)) {
        if (timerRef.current) {
          clearInterval(timerRef.current);
          timerRef.current = null;
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }, [pid]);

  useEffect(() => {
    if (!pid || !enabled) {
      setLogs([]);
      setStatus('unknown');
      return;
    }

    fetchLogs();
    timerRef.current = setInterval(fetchLogs, intervalMs);

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };
  }, [pid, intervalMs, enabled, fetchLogs]);

  return { logs, status, error, refresh: fetchLogs };
}
