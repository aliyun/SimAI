import { DeployStepper } from './DeployStepper';

interface WizardLayoutProps {
  readonly title: string;
  readonly description?: string;
  readonly children: React.ReactNode;
  readonly actions?: React.ReactNode;
  readonly showDeployStepper?: boolean;
  // Widen the content area for pages with large graphs/charts (EDG, Timeline).
  readonly wide?: boolean;
}

export function WizardLayout({
  title,
  description,
  children,
  actions,
  showDeployStepper = false,
  wide = false,
}: WizardLayoutProps) {
  const widthClass = wide ? 'max-w-[1600px]' : 'max-w-4xl';
  return (
    <div className="flex-1 flex flex-col overflow-hidden">
      {showDeployStepper && (
        <div className="px-6 py-3 border-b border-[var(--color-bg-hover)] bg-[var(--color-bg-secondary)] flex items-center justify-center">
          <DeployStepper />
        </div>
      )}
      <div className="flex-1 overflow-y-auto">
        <div className={`${widthClass} mx-auto px-6 py-8`}>
          <div className="mb-6">
            <h2 className="text-xl font-semibold text-[var(--color-text-primary)]">{title}</h2>
            {description && (
              <p className="mt-1 text-sm text-[var(--color-text-secondary)]">{description}</p>
            )}
          </div>

          {children}

          {actions && (
            <div className="mt-8 flex items-center justify-end gap-3 border-t border-[var(--color-bg-hover)] pt-6">
              {actions}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
