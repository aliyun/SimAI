import { http, HttpResponse } from 'msw';
import { getMockPodOverview, getMockPodDetail } from './topology-data';
import { generateClusterSummary, generateNodeMetrics } from './metrics-data';

export const handlers = [
  http.get('/api/topology/overview', () => {
    return HttpResponse.json({
      data: getMockPodOverview(),
      error: null,
      timestamp: Date.now(),
    });
  }),

  http.get('/api/topology/pod/:podId', ({ params }) => {
    const podId = decodeURIComponent(params.podId as string);
    return HttpResponse.json({
      data: getMockPodDetail(podId),
      error: null,
      timestamp: Date.now(),
    });
  }),

  http.get('/api/metrics/cluster', () => {
    return HttpResponse.json({
      data: generateClusterSummary(),
      error: null,
      timestamp: Date.now(),
    });
  }),

  http.get('/api/metrics/node/:nodeId', ({ params }) => {
    const nodeId = decodeURIComponent(params.nodeId as string);
    return HttpResponse.json({
      data: generateNodeMetrics(nodeId),
      error: null,
      timestamp: Date.now(),
    });
  }),
];
