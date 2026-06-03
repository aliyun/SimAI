interface CodePreviewProps {
  readonly content: string;
  readonly language?: string;
  readonly maxHeight?: string;
}

export function CodePreview({ content, maxHeight = '300px' }: CodePreviewProps) {
  return (
    <div
      className="rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] overflow-hidden"
    >
      <div className="flex items-center justify-between px-3 py-1.5 bg-[var(--color-bg-tertiary)] border-b border-[var(--color-bg-hover)]">
        <span className="text-xs text-[var(--color-text-muted)]">Preview</span>
        <button
          type="button"
          onClick={() => navigator.clipboard.writeText(content)}
          className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)] transition-colors"
        >
          Copy
        </button>
      </div>
      <pre
        className="p-3 text-xs font-mono text-[var(--color-text-secondary)] overflow-auto"
        style={{ maxHeight }}
      >
        {content}
      </pre>
    </div>
  );
}
