interface ValidationBannerProps {
  readonly type: 'success' | 'error' | 'warning' | 'info';
  readonly message: string;
  readonly details?: readonly string[];
}

const BANNER_STYLES = {
  success: 'bg-[var(--color-accent-green)]/10 border-[var(--color-accent-green)]/30 text-[var(--color-accent-green)]',
  error: 'bg-[var(--color-accent-red)]/10 border-[var(--color-accent-red)]/30 text-[var(--color-accent-red)]',
  warning: 'bg-[var(--color-accent-yellow)]/10 border-[var(--color-accent-yellow)]/30 text-[var(--color-accent-yellow)]',
  info: 'bg-[var(--color-accent-blue)]/10 border-[var(--color-accent-blue)]/30 text-[var(--color-accent-blue)]',
} as const;

export function ValidationBanner({ type, message, details }: ValidationBannerProps) {
  return (
    <div className={`rounded-lg border p-3 ${BANNER_STYLES[type]}`}>
      <div className="text-sm font-medium">{message}</div>
      {details && details.length > 0 && (
        <ul className="mt-1.5 text-xs space-y-0.5 opacity-80">
          {details.map((d, i) => (
            <li key={i}>- {d}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
