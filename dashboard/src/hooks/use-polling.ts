import { useEffect, useRef, useCallback } from 'react';

interface UsePollingOptions {
  readonly intervalMs: number;
  readonly enabled?: boolean;
  readonly onError?: (error: Error) => void;
}

export function usePolling(
  fetchFn: () => Promise<void>,
  options: UsePollingOptions,
): { refresh: () => void } {
  const { intervalMs, enabled = true, onError } = options;
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fetchRef = useRef(fetchFn);
  fetchRef.current = fetchFn;
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;

  const refresh = useCallback(() => {
    fetchRef.current().catch((err) => {
      onErrorRef.current?.(err instanceof Error ? err : new Error(String(err)));
    });
  }, []);

  useEffect(() => {
    if (!enabled) return;
    refresh();
    timerRef.current = setInterval(refresh, intervalMs);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [intervalMs, enabled, refresh]);

  return { refresh };
}
