import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import './index.css';

async function main() {
  // Enable MSW mock only in DEV mode when backend is unavailable.
  // Explicit override: VITE_USE_MOCK=true forces mock, VITE_USE_MOCK=false disables it.
  if (import.meta.env.DEV && import.meta.env.VITE_USE_MOCK !== 'false') {
    let useMock = import.meta.env.VITE_USE_MOCK === 'true';
    if (!useMock) {
      try {
        const resp = await fetch('/api/monitor/healthz', { signal: AbortSignal.timeout(2000) });
        useMock = !resp.ok;
      } catch {
        useMock = true;
      }
    }
    if (useMock) {
      const { worker } = await import('./mock/browser');
      await worker.start({ onUnhandledRequest: 'bypass' });
      console.info('[OCS-Sim] Using MSW mock data (backend unavailable)');
    } else {
      console.info('[OCS-Sim] Using real backend');
    }
  }

  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <App />
    </StrictMode>,
  );
}

main();
