interface ModeSelectorProps<T extends string> {
  readonly value: T;
  readonly onChange: (value: T) => void;
  readonly options: readonly { readonly value: T; readonly label: string; readonly description: string }[];
  readonly disabled?: boolean;
}

export function ModeSelector<T extends string>({ value, onChange, options, disabled = false }: ModeSelectorProps<T>) {
  return (
    <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${Math.min(options.length, 3)}, 1fr)` }}>
      {options.map((opt) => (
        <button
          key={opt.value}
          type="button"
          onClick={() => !disabled && onChange(opt.value)}
          disabled={disabled}
          className={`
            p-3 rounded-lg border text-left transition-all
            ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            ${value === opt.value
              ? 'border-[var(--color-accent-blue)] bg-[var(--color-accent-blue)]/10'
              : disabled
                ? 'border-[var(--color-bg-hover)] bg-[var(--color-bg-secondary)]'
                : 'border-[var(--color-bg-hover)] bg-[var(--color-bg-secondary)] hover:border-[var(--color-text-muted)]'
            }
          `}
        >
          <div className="text-sm font-medium text-[var(--color-text-primary)]">{opt.label}</div>
          <div className="mt-0.5 text-xs text-[var(--color-text-muted)]">{opt.description}</div>
        </button>
      ))}
    </div>
  );
}
