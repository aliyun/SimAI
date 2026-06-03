import { useState, useEffect, useCallback } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import apiClient from '../api/client';

interface Interaction {
  timestamp: string;
  source: string;
  type: string;
  endpoint: string;
  method: string;
  algorithm: string;
  request: Record<string, unknown>;
  response: Record<string, unknown>;
  workspace?: string;
}

interface LogEntry {
  file: string;
  line: string;
}

export function OpsPage() {
  const [searchParams] = useSearchParams();
  const initialTab = (searchParams.get('tab') === 'logs' ? 'logs' : 'interactions') as 'interactions' | 'logs';
  const [tab, setTab] = useState<'interactions' | 'logs'>(initialTab);
  const [interactions, setInteractions] = useState<Interaction[]>([]);
  const [logQuery, setLogQuery] = useState('');
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [sourceFilter, setSourceFilter] = useState('');

  const fetchInteractions = useCallback(async () => {
    setLoading(true);
    try {
      const params = sourceFilter ? `?source=${encodeURIComponent(sourceFilter)}` : '';
      const { data } = await apiClient.get(`/api/ops/interactions${params}`);
      setInteractions(data.interactions ?? []);
    } catch { /* ignore */ }
    setLoading(false);
  }, [sourceFilter]);

  const fetchLogs = useCallback(async () => {
    if (!logQuery.trim()) { setLogs([]); return; }
    setLoading(true);
    try {
      const { data } = await apiClient.get(`/api/ops/logs?q=${encodeURIComponent(logQuery)}&limit=50`);
      setLogs(data.logs ?? []);
    } catch { /* ignore */ }
    setLoading(false);
  }, [logQuery]);

  useEffect(() => {
    if (tab === 'interactions') fetchInteractions();
  }, [tab, fetchInteractions]);

  const sourceBadge = (s: string) => {
    const colors: Record<string, string> = {
      'OXC-HCCL': 'bg-purple-500/10 text-purple-400',
      'EDG': 'bg-blue-500/10 text-blue-400',
    };
    return colors[s] ?? 'bg-gray-500/10 text-gray-400';
  };

  const typeLabel = (t: string) => {
    const m: Record<string, string> = {
      init: '初始化', decision: '路由决策', oxc_call: 'OXC调用',
      oxc_result: 'OXC结果', IMPORT_FULL_TOPO: '导入拓扑',
      NOTIFY_NODE_MATRIX: '任务注册', fallback: '降级',
    };
    return m[t] ?? t;
  };

  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="flex items-center gap-3 mb-6">
        <Link
          to="/"
          className="flex items-center gap-1 text-xs text-[var(--color-text-muted)] hover:text-[var(--color-text-primary)] transition-colors px-2 py-1 rounded hover:bg-[var(--color-bg-hover)]"
        >
          <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5m0 0l7 7m-7-7l7-7" />
          </svg>
          返回首页
        </Link>
        <h1 className="text-xl font-bold text-[var(--color-text-primary)]">运维工具</h1>
        <div className="flex gap-1 bg-[var(--color-bg-hover)] rounded-lg p-1 ml-auto">
          <button
            onClick={() => setTab('interactions')}
            className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
              tab === 'interactions' ? 'bg-[var(--color-bg-primary)] text-[var(--color-text-primary)] shadow-sm' : 'text-[var(--color-text-muted)]'
            }`}
          >
            调用记录
          </button>
          <button
            onClick={() => setTab('logs')}
            className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
              tab === 'logs' ? 'bg-[var(--color-bg-primary)] text-[var(--color-text-primary)] shadow-sm' : 'text-[var(--color-text-muted)]'
            }`}
          >
            日志查询
          </button>
        </div>
      </div>

      {tab === 'interactions' && (
        <div className="space-y-3">
          <div className="flex items-center gap-3 mb-4">
            <span className="text-sm text-[var(--color-text-muted)]">来源过滤:</span>
            <select
              value={sourceFilter}
              onChange={(e) => setSourceFilter(e.target.value)}
              className="text-sm px-2 py-1 rounded border border-[var(--color-bg-hover)] bg-[var(--color-bg-primary)] text-[var(--color-text-primary)]"
            >
              <option value="">全部</option>
              <option value="OXC-HCCL">OXC-HCCL</option>
              <option value="EDG">EDG</option>
            </select>
            <button onClick={fetchInteractions} className="text-xs px-2 py-1 rounded bg-[var(--color-accent-blue)]/20 text-[var(--color-accent-blue)]">刷新</button>
            <span className="text-xs text-[var(--color-text-muted)]">{interactions.length} 条记录</span>
          </div>
          {loading && <div className="text-sm text-[var(--color-text-muted)]">加载中...</div>}
          <div className="space-y-2">
            {interactions.map((item, i) => (
              <div key={i} className="p-3 rounded-lg border border-[var(--color-bg-hover)] bg-[var(--color-bg-primary)]/50 text-sm">
                <div className="flex items-center gap-2 mb-1.5">
                  <span className={`text-xs px-1.5 py-0.5 rounded font-mono ${sourceBadge(item.source)}`}>{item.source}</span>
                  <span className="text-xs px-1.5 py-0.5 rounded bg-green-500/10 text-green-400">{typeLabel(item.type)}</span>
                  {item.algorithm && (<span className="text-xs px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 font-mono">{item.algorithm}</span>)}
                  {item.workspace && (<span className="text-xs text-[var(--color-text-muted)] ml-auto">{item.workspace}</span>)}
                </div>
                <div className="text-xs text-[var(--color-text-muted)] font-mono">
                  <span className="text-[var(--color-accent-green)]">{item.method}</span> {item.endpoint}
                </div>
                <div className="mt-1.5 grid grid-cols-2 gap-2 text-xs">
                  <div className="p-1.5 rounded bg-[var(--color-bg-hover)]/50">
                    <span className="text-[var(--color-text-muted)]">请求: </span>
                    <span className="text-[var(--color-text-primary)] font-mono">{JSON.stringify(item.request).substring(0, 200)}</span>
                  </div>
                  <div className="p-1.5 rounded bg-[var(--color-bg-hover)]/50">
                    <span className="text-[var(--color-text-muted)]">响应: </span>
                    <span className="text-[var(--color-text-primary)] font-mono">{JSON.stringify(item.response).substring(0, 200)}</span>
                  </div>
                </div>
              </div>
            ))}
            {!loading && interactions.length === 0 && (
              <div className="text-center py-8 text-sm text-[var(--color-text-muted)]">暂无调用记录。部署 OXc NS3 任务后将自动记录 EDG/oxc 交互。</div>
            )}
          </div>
        </div>
      )}

      {tab === 'logs' && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <input type="text" placeholder="搜索关键词 (如 error, OXC, EDG, SIGSEGV)..." value={logQuery}
              onChange={(e) => setLogQuery(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && fetchLogs()}
              className="flex-1 text-sm px-3 py-1.5 rounded border border-[var(--color-bg-hover)] bg-[var(--color-bg-primary)] text-[var(--color-text-primary)]" />
            <button onClick={fetchLogs} className="text-sm px-3 py-1.5 rounded bg-[var(--color-accent-blue)]/20 text-[var(--color-accent-blue)] hover:bg-[var(--color-accent-blue)]/30">搜索</button>
          </div>
          {loading && <div className="text-sm text-[var(--color-text-muted)]">搜索中...</div>}
          <div className="space-y-1">
            {logs.map((entry, i) => (
              <div key={i} className="flex gap-2 p-1.5 rounded hover:bg-[var(--color-bg-hover)]/30 text-xs font-mono">
                <span className="text-[var(--color-text-muted)] shrink-0">{entry.file}</span>
                <span className="text-[var(--color-text-primary)] truncate">{entry.line}</span>
              </div>
            ))}
          </div>
          {!loading && logQuery && logs.length === 0 && (
            <div className="text-center py-8 text-sm text-[var(--color-text-muted)]">未找到匹配 "{logQuery}" 的日志</div>
          )}
          {!logQuery && (
            <div className="text-center py-8 text-sm text-[var(--color-text-muted)]">输入关键词搜索平台运行日志</div>
          )}
        </div>
      )}
    </div>
  );
}
