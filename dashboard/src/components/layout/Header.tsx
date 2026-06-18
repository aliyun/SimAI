import { Link, useLocation } from 'react-router-dom';

interface HeaderProps {
  readonly title: string;
  readonly backLink?: string;
  readonly lastUpdated?: number | null;
}

const NAV_ITEMS = [
  { path: '/', label: '首页' },
  { path: '/results', label: '结果分析' },
  { path: '/monitor', label: '监控大屏' },
] as const;

export function Header({ title, backLink, lastUpdated }: HeaderProps) {
  const { pathname } = useLocation();
  const timeStr = lastUpdated ? new Date(lastUpdated).toLocaleTimeString() : null;

  return (
    <header className="h-14 px-6 flex items-center justify-between border-b border-[var(--color-bg-hover)] bg-[var(--color-bg-secondary)] shrink-0">
      <div className="flex items-center gap-4">
        {backLink && (
          <Link
            to={backLink}
            className="text-[var(--color-text-secondary)] hover:text-white transition-colors text-sm"
          >
            &larr; 返回
          </Link>
        )}

        {!backLink && (
          <h1 className="text-lg font-semibold tracking-wide text-[var(--color-text-primary)]">
            {title}
          </h1>
        )}

        <nav className="hidden md:flex items-center gap-1 ml-4">
          {NAV_ITEMS.map((item) => (
            <Link
              key={item.path}
              to={item.path}
              className={`
                px-3 py-1.5 rounded text-sm font-medium transition-colors
                ${pathname === item.path
                  ? 'bg-[var(--color-accent-blue)]/15 text-[var(--color-accent-blue)]'
                  : 'text-[var(--color-text-secondary)] hover:text-white'
                }
              `}
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </div>

      <div className="flex items-center gap-4 text-sm text-[var(--color-text-secondary)]">
        {timeStr && <span>Last update: {timeStr}</span>}
        {timeStr && <StatusDot />}
      </div>
    </header>
  );
}

function StatusDot() {
  return (
    <span className="relative flex h-3 w-3">
      <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75 bg-[var(--color-status-healthy)]" />
      <span className="relative inline-flex rounded-full h-3 w-3 bg-[var(--color-status-healthy)]" />
    </span>
  );
}
