import axios from 'axios';
import type { AxiosError, InternalAxiosRequestConfig } from 'axios';

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '',
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('ocs_sim_token');
  if (token) {
    config.headers['X-Session-Token'] = token;
  }
  return config;
});

let isRefreshing = false;
let pendingQueue: Array<{
  resolve: (token: string) => void;
  reject: (err: unknown) => void;
}> = [];

function processQueue(token: string | null, error?: unknown) {
  for (const p of pendingQueue) {
    if (token) p.resolve(token);
    else p.reject(error);
  }
  pendingQueue = [];
}

async function autoLogin(): Promise<string> {
  const username = localStorage.getItem('ocs_sim_username') || 'default';
  const { data } = await axios.post(
    `${import.meta.env.VITE_API_BASE_URL || ''}/api/auth/login`,
    { username },
  );
  const token = data.token as string;
  localStorage.setItem('ocs_sim_token', token);
  localStorage.setItem('ocs_sim_username', data.username ?? username);
  return token;
}

apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const original = error.config as InternalAxiosRequestConfig & { _retry?: boolean };

    if (error.response?.status === 401 && !original._retry) {
      original._retry = true;

      if (isRefreshing) {
        const token = await new Promise<string>((resolve, reject) => {
          pendingQueue.push({ resolve, reject });
        });
        original.headers['X-Session-Token'] = token;
        return apiClient(original);
      }

      isRefreshing = true;
      try {
        const token = await autoLogin();
        processQueue(token);
        original.headers['X-Session-Token'] = token;
        return apiClient(original);
      } catch (loginErr) {
        processQueue(null, loginErr);
        return Promise.reject(loginErr);
      } finally {
        isRefreshing = false;
      }
    }

    const message =
      (error.response?.data as Record<string, string>)?.error ||
      error.message ||
      'Network error';
    return Promise.reject(new Error(message));
  },
);

export default apiClient;
