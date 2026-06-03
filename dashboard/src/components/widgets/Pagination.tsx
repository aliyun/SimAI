import { useMemo } from 'react';

interface PaginationProps {
  readonly currentPage: number;
  readonly totalItems: number;
  readonly pageSize: number;
  readonly onPageChange: (page: number) => void;
}

export function Pagination({ currentPage, totalItems, pageSize, onPageChange }: PaginationProps) {
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));

  const pageNumbers = useMemo(() => {
    const pages: (number | '...')[] = [];
    if (totalPages <= 7) {
      for (let i = 1; i <= totalPages; i++) pages.push(i);
    } else {
      pages.push(1);
      if (currentPage > 3) pages.push('...');
      const start = Math.max(2, currentPage - 1);
      const end = Math.min(totalPages - 1, currentPage + 1);
      for (let i = start; i <= end; i++) pages.push(i);
      if (currentPage < totalPages - 2) pages.push('...');
      pages.push(totalPages);
    }
    return pages;
  }, [currentPage, totalPages]);

  if (totalPages <= 1) return null;

  const btnBase = 'px-2.5 py-1 text-xs rounded transition-colors';
  const btnActive = 'bg-[var(--color-accent-blue)] text-white';
  const btnInactive = 'text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] hover:bg-[var(--color-bg-hover)]';
  const btnDisabled = 'text-[var(--color-text-muted)]/40 cursor-not-allowed';

  return (
    <div className="flex items-center justify-between py-2">
      <span className="text-xs text-[var(--color-text-muted)]">
        {totalItems} 条 · 第 {currentPage}/{totalPages} 页
      </span>
      <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={() => onPageChange(currentPage - 1)}
          disabled={currentPage <= 1}
          className={`${btnBase} ${currentPage <= 1 ? btnDisabled : btnInactive}`}
        >
          ‹
        </button>
        {pageNumbers.map((p, i) =>
          p === '...' ? (
            <span key={`dots-${i}`} className="px-1 text-xs text-[var(--color-text-muted)]">…</span>
          ) : (
            <button
              key={p}
              type="button"
              onClick={() => onPageChange(p)}
              className={`${btnBase} ${p === currentPage ? btnActive : btnInactive}`}
            >
              {p}
            </button>
          ),
        )}
        <button
          type="button"
          onClick={() => onPageChange(currentPage + 1)}
          disabled={currentPage >= totalPages}
          className={`${btnBase} ${currentPage >= totalPages ? btnDisabled : btnInactive}`}
        >
          ›
        </button>
      </div>
    </div>
  );
}
