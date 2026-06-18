export const COLORS = {
  bg: {
    primary: '#0a0e17',
    secondary: '#111827',
    tertiary: '#1f2937',
    hover: '#374151',
  },
  text: {
    primary: '#f3f4f6',
    secondary: '#9ca3af',
    muted: '#6b7280',
  },
  accent: {
    blue: '#3b82f6',
    green: '#10b981',
    yellow: '#f59e0b',
    red: '#ef4444',
    cyan: '#06b6d4',
    purple: '#8b5cf6',
  },
  node: {
    pod: '#3b82f6',
    oxc: '#8b5cf6',
    spine: '#06b6d4',
    leaf: '#10b981',
    server: '#f59e0b',
    superNode: 'rgba(59, 130, 246, 0.1)',
  },
  status: {
    healthy: '#10b981',
    warning: '#f59e0b',
    critical: '#ef4444',
    unknown: '#6b7280',
  },
  edge: {
    normal: '#4b5563',
    active: '#3b82f6',
    highUtil: '#f59e0b',
    error: '#ef4444',
  },
} as const;

export const FONT = {
  family: "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
  sizeXs: '10px',
  sizeSm: '12px',
  sizeMd: '14px',
  sizeLg: '18px',
  sizeXl: '24px',
  sizeTitle: '32px',
} as const;
