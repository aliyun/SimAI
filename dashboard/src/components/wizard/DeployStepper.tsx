import { Link, useLocation } from 'react-router-dom';

const DEPLOY_STEPS = [
  { path: '/deploy/workload', label: 'Workload', num: 1 },
  { path: '/deploy/edg', label: 'OXC 调节', num: 2 },
  { path: '/deploy/launch', label: '启动仿真', num: 3 },
] as const;

export function DeployStepper() {
  const { pathname } = useLocation();
  const currentIdx = DEPLOY_STEPS.findIndex((s) => s.path === pathname);

  return (
    <nav className="flex items-center gap-1">
      {DEPLOY_STEPS.map((step, idx) => {
        const isCompleted = idx < currentIdx;
        const isCurrent = idx === currentIdx;

        return (
          <div key={step.path} className="flex items-center">
            {idx > 0 && (
              <div
                className={`w-8 h-px mx-1 ${
                  isCompleted ? 'bg-[var(--color-accent-blue)]' : 'bg-[var(--color-bg-hover)]'
                }`}
              />
            )}
            <Link
              to={step.path}
              className={`
                flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium transition-colors
                ${isCurrent
                  ? 'bg-[var(--color-accent-blue)] text-white'
                  : isCompleted
                    ? 'bg-[var(--color-bg-tertiary)] text-[var(--color-accent-blue)]'
                    : 'text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]'
                }
              `}
            >
              <span
                className={`
                  flex items-center justify-center w-5 h-5 rounded-full text-[10px] font-bold
                  ${isCurrent
                    ? 'bg-white/20 text-white'
                    : isCompleted
                      ? 'bg-[var(--color-accent-blue)]/20 text-[var(--color-accent-blue)]'
                      : 'bg-[var(--color-bg-hover)] text-[var(--color-text-muted)]'
                  }
                `}
              >
                {isCompleted ? '\u2713' : step.num}
              </span>
              {step.label}
            </Link>
          </div>
        );
      })}
    </nav>
  );
}
