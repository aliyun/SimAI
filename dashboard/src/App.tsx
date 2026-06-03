import { useEffect } from 'react';
import { BrowserRouter, Routes, Route, useNavigate } from 'react-router-dom';
import { HomePage } from './pages/HomePage';
import { NetworkDetailPage } from './pages/NetworkDetailPage';
import { PodDetailPage } from './pages/PodDetailPage';
import { WorkloadPage } from './pages/WorkloadPage';
import { EdgPage } from './pages/EdgPage';
import { LaunchPage } from './pages/LaunchPage';
import { ResultsPage } from './pages/ResultsPage';
import { OpsPage } from './pages/OpsPage';
import { useNetworkStore } from './stores/network-store';

function MonitorRouter() {
  const navigate = useNavigate();
  const { networks, activeNetworkId } = useNetworkStore();

  useEffect(() => {
    const targetId = activeNetworkId ?? networks[0]?.id;
    if (targetId) {
      navigate(`/networks/${targetId}`, { replace: true });
    } else {
      navigate('/', { replace: true });
    }
  }, [activeNetworkId, networks, navigate]);

  return null;
}

export function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/networks/:id" element={<NetworkDetailPage />} />
        <Route path="/deploy/workload" element={<WorkloadPage />} />
        <Route path="/deploy/edg" element={<EdgPage />} />
        <Route path="/deploy/launch" element={<LaunchPage />} />
        <Route path="/results" element={<ResultsPage />} />
        <Route path="/monitor" element={<MonitorRouter />} />
        <Route path="/monitor/pod/:podId" element={<PodDetailPage />} />
        <Route path="/ops" element={<OpsPage />} />
      </Routes>
    </BrowserRouter>
  );
}
// test
