import { useRef, useEffect } from 'react';

interface LogViewerProps {
  readonly logs: readonly string[];
  readonly status?: string;
  readonly maxHeight?: string;
}

export function LogViewer({ logs, status, maxHeight = '400px' }: LogViewerProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="rounded-lg border border-[var(--color-bg-hover)] overflow-hidden">
      <div className="flex items-center justify-between px-3 py-1.5 bg-[var(--color-bg-tertiary)] border-b border-[var(--color-bg-hover)]">
        <span className="text-xs text-[var(--color-text-muted)]">Console Output</span>
        {status && (
          <span className={`text-xs px-1.5 py-0.5 rounded ${
            status === 'running'
              ? 'bg-[var(--color-accent-blue)]/10 text-[var(--color-accent-blue)]'
              : status === 'exited'
                ? 'bg-[var(--color-accent-green)]/10 text-[var(--color-accent-green)]'
                : 'bg-[var(--color-bg-hover)] text-[var(--color-text-muted)]'
          }`}>
            {status}
          </span>
        )}
      </div>
      <div
        ref={containerRef}
        className="p-3 bg-[var(--color-bg-primary)] overflow-y-auto font-mono text-xs text-[var(--color-text-secondary)]"
        style={{ maxHeight }}
      >
        {logs.length === 0 ? (
          <div className="text-[var(--color-text-muted)] italic">No output yet...</div>
        ) : (
          logs.map((line, i) => (
            <div key={i} className="whitespace-pre-wrap break-all leading-5">
              {line}
            </div>
          ))
        )}
      </div>
    </div>
  );
}
