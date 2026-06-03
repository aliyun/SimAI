interface DashboardShellProps {
  readonly children: React.ReactNode;
}

export function DashboardShell({ children }: DashboardShellProps) {
  return (
    <div className="h-screen bg-[var(--color-bg-primary)] text-[var(--color-text-primary)] flex flex-col overflow-hidden">
      {children}
    </div>
  );
}
