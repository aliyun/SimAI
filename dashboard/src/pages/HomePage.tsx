import { useState, useCallback } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { DashboardShell } from '../components/layout/DashboardShell';
import { Header } from '../components/layout/Header';
import { useNetworkStore } from '../stores/network-store';
import { setTopologyDir } from '../api/simulation-api';
import { initNetwork } from '../api/edg-api';

/* ─── Animated network topology SVG for the hero background ─── */
function HeroNetworkGraphic() {
  return (
    <svg
      viewBox="0 0 800 200"
      className="w-full max-w-5xl mx-auto opacity-30"
      aria-hidden="true"
    >
      <defs>
        <linearGradient id="edgeGrad" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="var(--color-accent-cyan)" stopOpacity="0.6" />
          <stop offset="50%" stopColor="var(--color-accent-blue)" stopOpacity="0.8" />
          <stop offset="100%" stopColor="var(--color-accent-purple)" stopOpacity="0.6" />
        </linearGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Edges */}
      {[
        [120, 60, 280, 100], [280, 100, 400, 40], [400, 40, 520, 100],
        [520, 100, 680, 60], [120, 60, 280, 40], [280, 40, 400, 100],
        [400, 100, 520, 40], [520, 40, 680, 60], [200, 140, 400, 160],
        [400, 160, 600, 140], [120, 60, 200, 140], [680, 60, 600, 140],
        [280, 100, 400, 160], [520, 100, 400, 160],
      ].map(([x1, y1, x2, y2], i) => (
        <line
          key={`e-${i}`}
          x1={x1} y1={y1} x2={x2} y2={y2}
          stroke="url(#edgeGrad)"
          strokeWidth="1.5"
          opacity="0.5"
        >
          <animate
            attributeName="opacity"
            values="0.2;0.8;0.2"
            dur={`${3 + (i % 4)}s`}
            repeatCount="indefinite"
          />
        </line>
      ))}

      {/* Nodes */}
      {[
        { cx: 120, cy: 60, r: 6, color: 'var(--color-accent-cyan)' },
        { cx: 280, cy: 100, r: 5, color: 'var(--color-accent-blue)' },
        { cx: 280, cy: 40, r: 4, color: 'var(--color-accent-purple)' },
        { cx: 400, cy: 40, r: 7, color: 'var(--color-accent-cyan)' },
        { cx: 400, cy: 100, r: 5, color: 'var(--color-accent-blue)' },
        { cx: 520, cy: 100, r: 5, color: 'var(--color-accent-blue)' },
        { cx: 520, cy: 40, r: 4, color: 'var(--color-accent-purple)' },
        { cx: 680, cy: 60, r: 6, color: 'var(--color-accent-cyan)' },
        { cx: 200, cy: 140, r: 4, color: 'var(--color-accent-green)' },
        { cx: 400, cy: 160, r: 6, color: 'var(--color-accent-green)' },
        { cx: 600, cy: 140, r: 4, color: 'var(--color-accent-green)' },
      ].map((node, i) => (
        <circle
          key={`n-${i}`}
          cx={node.cx} cy={node.cy} r={node.r}
          fill={node.color}
          filter="url(#glow)"
        >
          <animate
            attributeName="r"
            values={`${node.r};${node.r + 4};${node.r}`}
            dur={`${2.5 + (i % 3)}s`}
            repeatCount="indefinite"
          />
        </circle>
      ))}
    </svg>
  );
}

/* ─── Constants ─── */


const QUICK_START_STEPS = [
  { step: 1, label: '创建组网', desc: '选择拓扑模板，配置网络参数', accent: 'var(--color-accent-cyan)' },
  { step: 2, label: '配置任务', desc: '设定 Workload 和 RankTable', accent: 'var(--color-accent-blue)' },
  { step: 3, label: '启动仿真', desc: '选择 Analytical 或 NS-3 模式运行', accent: 'var(--color-accent-purple)' },
  { step: 4, label: '分析结果', desc: '可视化性能指标，导出报告', accent: 'var(--color-accent-green)' },
] as const;


export function HomePage() {
  const navigate = useNavigate();
  const { networks, createNetwork, deleteNetwork, setActiveNetwork } =
    useNetworkStore();

  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newName, setNewName] = useState('');
  const [newDir, setNewDir] = useState('');
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);
  const [edgLoading, setEdgLoading] = useState(false);
  const [edgMsg, setEdgMsg] = useState<string | null>(null);
  const [lldData, setLldData] = useState<Record<string, unknown> | null>(null);
  const [lldServerIps, setLldServerIps] = useState<string[]>([]);
  const [npuPerServer, setNpuPerServer] = useState('8');
  const [npuType, setNpuType] = useState('A3');
  const [intraBw, setIntraBw] = useState('400Gbps');
  const [linkBw, setLinkBw] = useState('');
  const [detectedTypes, setDetectedTypes] = useState<Set<string>>(new Set());

  const handleLldFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setEdgMsg(null);
    try {
      const text = await file.text();
      const lld = JSON.parse(text);
      setLldData(lld);
      const topo = (lld as Record<string, Record<string, unknown[]>>).topology ?? {};
      const serverCount = topo.server_nodes?.length ?? 0;
      const leafCount = topo.leaf_nodes?.length ?? 0;
      const autoName = `EDG-${serverCount}srv-${leafCount}leaf`;
      if (!newName.trim()) {
        setNewName(autoName);
      }
      setNewDir(`topology/${newName.trim() || autoName}`);

      // Extract server IPs
      const srvIps = ((topo.server_nodes ?? []) as Array<Record<string, string>>).map(s => s.node_id).filter(Boolean);
      setLldServerIps(srvIps);

      // Auto-detect bandwidth from leaf port_id (e.g. "400GE1/0/1:1" → "400Gbps")
      const leafNodes = (topo.leaf_nodes ?? []) as Array<Record<string, unknown>>;
      for (const leaf of leafNodes) {
        const ports = (leaf.port_id_list ?? []) as Array<Record<string, string>>;
        for (const p of ports) {
          const m = p.port_id?.match(/^(\d+)GE/);
          if (m) {
            setLinkBw(`${m[1]}Gbps`);
            break;
          }
        }
        if (linkBw) break;
      }

      // Auto-detect NPU type + server-internal bandwidth from lld chassis_topo
      // Supports mixed-NPU deployments (e.g. A2 and A3 servers)
      const NPU_INTRA_MAP: Record<string, string> = { A2: '200Gbps', A3: '400Gbps', A5: '400Gbps', A6: '400Gbps', A100: '2400Gbps', H100: '2880Gbps', H800: '2880Gbps' };
      const srvNodes = (topo.server_nodes ?? []) as Array<Record<string, string>>;
      const types = new Set<string>();
      let primaryType = '';
      let primaryBw = '400Gbps';
      for (const srv of srvNodes) {
        const chassisT = (srv as any).chassis_topo;
        if (chassisT) {
          const prefix = chassisT.split('_')[0];
          types.add(prefix);
          if (!primaryType) {
            primaryType = prefix;
            primaryBw = NPU_INTRA_MAP[prefix] ?? '400Gbps';
          }
        }
      }
      setDetectedTypes(types);
      setNpuType(types.size === 1 ? primaryType : 'MIXED');
      setIntraBw(types.size === 1 ? primaryBw : `${[...types].map(t => `${t}:${NPU_INTRA_MAP[t]??'400G'}`).join(', ')}`);

      setEdgMsg(`lld.json 已加载 (${serverCount} 服务器, ${leafCount} 叶交换机, ${srvIps.length} IPs)`);
    } catch {
      setEdgMsg('lld.json 解析失败');
      setLldData(null);
    }
    e.target.value = '';
  }, [newName]);

  const handleCreateNetwork = async () => {
    if (!newName.trim() || !newDir.trim()) return;

    // Deduplicate name: append (2), (3), etc. if name already exists
    let finalName = newName.trim();
    const existingNames = new Set(networks.map(n => n.name));
    if (existingNames.has(finalName)) {
      let i = 2;
      while (existingNames.has(`${finalName} (${i})`)) i++;
      finalName = `${finalName} (${i})`;
    }
    const finalDir = `topology/${finalName}`;

    setCreating(true);
    setCreateError(null);
    try {
      // If lld.json was uploaded, call EDG init first (creates topology dir + XMLs)
      if (lldData) {
        setEdgLoading(true);
        try {
          const result = await initNetwork(lldData, finalDir);
          const msg = `EDG 初始化完成，OXC 交叉数: ${result.crosses_count}`;
          setEdgMsg(result.warning ? `${msg} (${result.warning})` : msg);
        } catch (err) {
          setCreateError(err instanceof Error ? err.message : 'EDG 初始化失败');
          setEdgLoading(false);
          setCreating(false);
          return;
        }
        setEdgLoading(false);
      }

      await setTopologyDir(finalDir);

      const net = createNetwork({
        name: finalName,
        topologyDir: finalDir,
        npuPerServer: parseInt(npuPerServer, 10) || 8,
        npuType,
        intraBw,
        bandwidth: linkBw,
        serverIps: lldServerIps,
      });
      setActiveNetwork(net.id);
      setShowCreateModal(false);
      setNewName('');
      setNewDir('');
      setLldData(null);
      setLldServerIps([]);
      setEdgMsg(null);
      navigate(`/networks/${net.id}`);
    } catch (err) {
      setCreateError(err instanceof Error ? err.message : '创建失败');
    } finally {
      setCreating(false);
    }
  };

  const handleSelectNetwork = (id: string) => {
    setActiveNetwork(id);
    navigate(`/networks/${id}`);
  };

  return (
    <DashboardShell>
      <Header title="OCS-Sim" />

      <div className="flex-1 overflow-y-auto">
        <div className="max-w-5xl mx-auto px-6 py-10">

          {/* ═══ Hero Section ═══ */}
          <div className="relative text-center mb-14">
            {/* Animated network background */}
            <div className="absolute inset-0 -top-8 -left-12 -right-12 pointer-events-none">
              <HeroNetworkGraphic />
            </div>

            {/* Glowing accent line */}
            <div className="flex justify-center mb-6">
              <div className="h-px w-32 bg-gradient-to-r from-transparent via-[var(--color-accent-cyan)] to-transparent opacity-60" />
            </div>

            <h1 className="relative text-4xl font-bold text-[var(--color-text-primary)] mb-3 tracking-tight">
              OCS-Sim{' '}
              <span className="bg-gradient-to-r from-[var(--color-accent-cyan)] to-[var(--color-accent-blue)] bg-clip-text text-transparent">
                仿真平台
              </span>
            </h1>
            <p className="relative text-[var(--color-text-secondary)] max-w-2xl mx-auto mb-2">
              面向 AI 大规模训练的高精度仿真器
            </p>
            <p className="relative text-xs text-[var(--color-text-muted)]">
              NSDI'25 Spring &middot; 全栈建模 &middot; 万卡级规模
            </p>

            {/* Glowing accent line */}
            <div className="flex justify-center mt-6">
              <div className="h-px w-32 bg-gradient-to-r from-transparent via-[var(--color-accent-blue)] to-transparent opacity-60" />
            </div>
          </div>

          {/* ═══ Three Major Sections ═══ */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-12">

            {/* Section 1: Network Management */}
            <section className="rounded-xl border border-[var(--color-bg-hover)] bg-[var(--color-bg-secondary)]/30 p-5">
              <div className="flex items-center gap-2.5 mb-4">
                <div className="w-8 h-8 rounded-lg bg-[var(--color-accent-cyan)]/10 flex items-center justify-center">
                  <svg className="w-4 h-4 text-[var(--color-accent-cyan)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">
                    组网管理
                  </h2>
                  <p className="text-[10px] text-[var(--color-text-muted)]">
                    创建和管理网络拓扑配置
                  </p>
                </div>
              </div>

              {/* Network list */}
              <div className="space-y-2 mb-4">
                {networks.length === 0 ? (
                  <div className="py-6 text-center border border-dashed border-[var(--color-bg-hover)] rounded-lg">
                    <div className="text-2xl mb-2 opacity-30">⬡</div>
                    <p className="text-xs text-[var(--color-text-muted)]">
                      暂无组网，点击下方创建
                    </p>
                  </div>
                ) : (
                  networks.map((net) => (
                    <div
                      key={net.id}
                      className="flex items-center justify-between p-3 rounded-lg border border-[var(--color-bg-hover)] bg-[var(--color-bg-primary)]/50 hover:border-[var(--color-accent-cyan)]/40 transition-colors group"
                    >
                      <button
                        onClick={() => handleSelectNetwork(net.id)}
                        className="flex-1 text-left"
                      >
                        <div className="text-sm font-medium text-[var(--color-text-primary)]">
                          {net.name}
                        </div>
                        <div className="text-xs text-[var(--color-text-muted)] mt-0.5 truncate max-w-[180px]">
                          {net.topologyDir}
                        </div>
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          deleteNetwork(net.id);
                        }}
                        className="opacity-0 group-hover:opacity-100 text-[var(--color-text-muted)] hover:text-[var(--color-accent-red)] text-xs px-2 py-1 transition-all"
                      >
                        删除
                      </button>
                    </div>
                  ))
                )}
              </div>

              <button
                onClick={() => setShowCreateModal(true)}
                className="w-full py-2.5 rounded-lg border border-dashed border-[var(--color-accent-cyan)]/40 text-[var(--color-accent-cyan)] text-xs font-medium hover:bg-[var(--color-accent-cyan)]/5 transition-colors"
              >
                + 新建组网
              </button>
            </section>

            {/* Section 2: Simulation Deployment */}
            <section className="rounded-xl border border-[var(--color-bg-hover)] bg-[var(--color-bg-secondary)]/30 p-5">
              <div className="flex items-center gap-2.5 mb-4">
                <div className="w-8 h-8 rounded-lg bg-[var(--color-accent-blue)]/10 flex items-center justify-center">
                  <svg className="w-4 h-4 text-[var(--color-accent-blue)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
                    <path d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">
                    仿真任务部署
                  </h2>
                  <p className="text-[10px] text-[var(--color-text-muted)]">
                    配置并启动仿真任务
                  </p>
                </div>
              </div>

              <div className="space-y-2">
                {[
                  { to: '/deploy/workload', label: 'Workload 配置', desc: '模型参数与通信模式', icon: '&#9632;' },
                  { to: '/deploy/edg', label: 'OXC 调节', desc: 'OXC 光交换网络拓扑调节', icon: '&#9638;' },
                  { to: '/deploy/launch', label: '启动仿真', desc: '执行 OCS-Sim 仿真任务', icon: '&#9654;' },
                ].map((item) => (
                  <Link
                    key={item.to}
                    to={item.to}
                    className="group flex items-center gap-3 p-3 rounded-lg border border-[var(--color-bg-hover)] bg-[var(--color-bg-primary)]/50 hover:border-[var(--color-accent-blue)]/40 transition-colors"
                  >
                    <div className="flex-1">
                      <div className="text-sm font-medium text-[var(--color-text-primary)] group-hover:text-[var(--color-accent-blue)] transition-colors">
                        {item.label}
                      </div>
                      <div className="text-xs text-[var(--color-text-muted)] mt-0.5">
                        {item.desc}
                      </div>
                    </div>
                    <svg className="w-4 h-4 text-[var(--color-text-muted)] group-hover:text-[var(--color-accent-blue)] group-hover:translate-x-0.5 transition-all" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M9 5l7 7-7 7" />
                    </svg>
                  </Link>
                ))}
              </div>
            </section>

            {/* Section 3: Results Analysis */}
            <section className="rounded-xl border border-[var(--color-bg-hover)] bg-[var(--color-bg-secondary)]/30 p-5">
              <div className="flex items-center gap-2.5 mb-4">
                <div className="w-8 h-8 rounded-lg bg-[var(--color-accent-green)]/10 flex items-center justify-center">
                  <svg className="w-4 h-4 text-[var(--color-accent-green)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">
                    仿真结果分析
                  </h2>
                  <p className="text-[10px] text-[var(--color-text-muted)]">
                    可视化性能指标与报告
                  </p>
                </div>
              </div>

              <div className="space-y-2">
                <Link
                  to="/results"
                  className="group flex items-center gap-3 p-3 rounded-lg border border-[var(--color-bg-hover)] bg-[var(--color-bg-primary)]/50 hover:border-[var(--color-accent-green)]/40 transition-colors"
                >
                  <div className="flex-1">
                    <div className="text-sm font-medium text-[var(--color-text-primary)] group-hover:text-[var(--color-accent-green)] transition-colors">
                      结果可视化
                    </div>
                    <div className="text-xs text-[var(--color-text-muted)] mt-0.5">
                      EndToEnd CSV日志分析
                    </div>
                  </div>
                  <svg className="w-4 h-4 text-[var(--color-text-muted)] group-hover:text-[var(--color-accent-green)] group-hover:translate-x-0.5 transition-all" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M9 5l7 7-7 7" />
                  </svg>
                </Link>

                <Link
                  to="/monitor"
                  className="group flex items-center gap-3 p-3 rounded-lg border border-[var(--color-bg-hover)] bg-[var(--color-bg-primary)]/50 hover:border-[var(--color-accent-green)]/40 transition-colors"
                >
                  <div className="flex-1">
                    <div className="text-sm font-medium text-[var(--color-text-primary)] group-hover:text-[var(--color-accent-green)] transition-colors">
                      实时监控大屏
                    </div>
                    <div className="text-xs text-[var(--color-text-muted)] mt-0.5">
                      网络拓扑与节点状态监控
                    </div>
                  </div>
                  <svg className="w-4 h-4 text-[var(--color-text-muted)] group-hover:text-[var(--color-accent-green)] group-hover:translate-x-0.5 transition-all" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M9 5l7 7-7 7" />
                  </svg>
                </Link>

              </div>

            </section>

            {/* Section 4: Ops Tools */}
            <section className="rounded-xl border border-[var(--color-bg-hover)] bg-[var(--color-bg-secondary)]/30 p-5">
              <div className="flex items-center gap-2.5 mb-4">
                <div className="w-8 h-8 rounded-lg bg-[var(--color-accent-purple)]/10 flex items-center justify-center">
                  <svg className="w-4 h-4 text-[var(--color-accent-purple)]" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M12 15a3 3 0 100-6 3 3 0 000 6z" />
                    <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 010 2.83 2 2 0 01-2.83 0l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" />
                  </svg>
                </div>
                <div>
                  <h2 className="text-sm font-semibold text-[var(--color-text-primary)]">
                    运维工具
                  </h2>
                  <p className="text-[10px] text-[var(--color-text-muted)]">
                    OXC/EDG 调用记录与平台日志查询
                  </p>
                </div>
              </div>

              <div className="space-y-2">
                <Link
                  to="/ops?tab=interactions"
                  className="group flex items-center gap-3 p-3 rounded-lg border border-[var(--color-bg-hover)] bg-[var(--color-bg-primary)]/50 hover:border-[var(--color-accent-purple)]/40 transition-colors"
                >
                  <div className="flex-1">
                    <div className="text-sm font-medium text-[var(--color-text-primary)] group-hover:text-[var(--color-accent-purple)] transition-colors">
                      调用记录
                    </div>
                    <div className="text-xs text-[var(--color-text-muted)] mt-0.5">
                      按来源分类查看 OXc-HCCL 与 EDG 的 API 交互
                    </div>
                  </div>
                  <svg className="w-4 h-4 text-[var(--color-text-muted)] group-hover:text-[var(--color-accent-purple)] group-hover:translate-x-0.5 transition-all" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M9 5l7 7-7 7" />
                  </svg>
                </Link>

                <Link
                  to="/ops?tab=logs"
                  className="group flex items-center gap-3 p-3 rounded-lg border border-[var(--color-bg-hover)] bg-[var(--color-bg-primary)]/50 hover:border-[var(--color-accent-purple)]/40 transition-colors"
                >
                  <div className="flex-1">
                    <div className="text-sm font-medium text-[var(--color-text-primary)] group-hover:text-[var(--color-accent-purple)] transition-colors">
                      日志查询
                    </div>
                    <div className="text-xs text-[var(--color-text-muted)] mt-0.5">
                      搜索平台运行日志，快速定位错误与异常
                    </div>
                  </div>
                  <svg className="w-4 h-4 text-[var(--color-text-muted)] group-hover:text-[var(--color-accent-purple)] group-hover:translate-x-0.5 transition-all" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M9 5l7 7-7 7" />
                  </svg>
                </Link>
              </div>
            </section>
          </div>

          {/* ═══ Quick Start Guide ═══ */}
          <div className="mb-12">
            <div className="flex items-center gap-2 mb-5">
              <div className="h-px flex-1 bg-gradient-to-r from-[var(--color-bg-hover)] to-transparent" />
              <h2 className="text-sm font-semibold text-[var(--color-text-secondary)] px-3">
                快速开始
              </h2>
              <div className="h-px flex-1 bg-gradient-to-l from-[var(--color-bg-hover)] to-transparent" />
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {QUICK_START_STEPS.map((s, i) => (
                <div key={s.step} className="relative">
                  {/* Connector line */}
                  {i < QUICK_START_STEPS.length - 1 && (
                    <div className="hidden md:block absolute top-5 left-[calc(50%+24px)] w-[calc(100%-24px)] h-px bg-[var(--color-bg-hover)]" />
                  )}
                  <div className="text-center">
                    <div
                      className="inline-flex w-10 h-10 rounded-full items-center justify-center text-sm font-bold mb-3 border"
                      style={{
                        borderColor: s.accent,
                        color: s.accent,
                        backgroundColor: `color-mix(in srgb, ${s.accent} 10%, transparent)`,
                      }}
                    >
                      {s.step}
                    </div>
                    <div className="text-sm font-medium text-[var(--color-text-primary)] mb-1">
                      {s.label}
                    </div>
                    <div className="text-xs text-[var(--color-text-muted)] leading-relaxed">
                      {s.desc}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* ═══ Footer ═══ */}
          <div className="text-center pb-6">
            <div className="h-px w-full bg-[var(--color-bg-hover)] mb-6" />
            <p className="text-[10px] text-[var(--color-text-muted)]">
              OCS-Sim &copy; 2025 &middot; NSDI'25 Spring
            </p>
          </div>

        </div>
      </div>

      {/* ═══ Create Network Modal ═══ */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-[var(--color-bg-secondary)] rounded-xl border border-[var(--color-bg-hover)] w-[420px] p-6 shadow-2xl">
            <div className="flex items-center justify-between mb-5">
              <h3 className="text-base font-semibold text-[var(--color-text-primary)]">
                新建组网
              </h3>
              <button
                onClick={() => setShowCreateModal(false)}
                className="text-[var(--color-text-muted)] hover:text-white text-lg leading-none"
              >
                &times;
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="block text-xs text-[var(--color-text-secondary)] mb-1.5">
                  导入 lld.json（自动调用 EDG 初始化组网）
                </label>
                <input
                  type="file"
                  accept=".json"
                  onChange={handleLldFileSelect}
                  disabled={creating}
                  className="w-full text-xs text-[var(--color-text-secondary)] file:mr-2 file:py-1.5 file:px-3 file:rounded-lg file:border-0 file:text-xs file:font-medium file:bg-[var(--color-accent-cyan)]/10 file:text-[var(--color-accent-cyan)] hover:file:bg-[var(--color-accent-cyan)]/20 file:cursor-pointer file:transition-colors"
                />
                {edgMsg && <p className={`text-xs mt-1 ${edgMsg.includes('加载') || edgMsg.includes('完成') ? 'text-[var(--color-accent-green)]' : 'text-[var(--color-accent-red)]'}`}>{edgMsg}</p>}
              </div>
              <div>
                <label className="block text-xs text-[var(--color-text-secondary)] mb-1.5">
                  组网名称
                </label>
                <input
                  type="text"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  placeholder="例如: Spectrum-X-128G"
                  className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm focus:border-[var(--color-accent-cyan)]/60 focus:outline-none transition-colors"
                />
              </div>
              <div>
                <label className="block text-xs text-[var(--color-text-secondary)] mb-1.5">
                  拓扑目录
                </label>
                <input
                  type="text"
                  value={newDir}
                  onChange={(e) => setNewDir(e.target.value)}
                  placeholder="topology/my-network"
                  className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm focus:border-[var(--color-accent-cyan)]/60 focus:outline-none transition-colors"
                />
                <p className="text-[10px] text-[var(--color-text-muted)] mt-1">导入 lld.json 后自动生成，也可手动修改</p>
              </div>

              {/* NPU config — auto-detected from lld.json */}
              {lldData && (
                <div className="grid grid-cols-2 gap-2 px-3 py-2 rounded-lg bg-[var(--color-bg-hover)]/30 text-xs">
                  <span className="text-[var(--color-text-muted)]">NPU/Server: <strong className="text-[var(--color-text-primary)]">{npuPerServer}</strong></span>
                  <span className="text-[var(--color-text-muted)]">NPU 型号: <strong className="text-[var(--color-text-primary)]">{npuType}{detectedTypes.size > 1 ? ' (混部)' : ''}</strong></span>
                  <span className="text-[var(--color-text-muted)]">服务器内带宽: <strong className="text-[var(--color-text-primary)]">{intraBw}</strong></span>
                  <span className="text-[var(--color-text-muted)]">链路带宽: <strong className="text-[var(--color-text-primary)]">{linkBw || '自动检测'}</strong></span>
                </div>
              )}

              {createError && (
                <p className="text-xs text-[var(--color-accent-red)]">{createError}</p>
              )}

              <div className="flex gap-3 pt-2">
                <button
                  onClick={() => setShowCreateModal(false)}
                  className="flex-1 py-2 rounded-lg border border-[var(--color-bg-hover)] text-[var(--color-text-secondary)] text-sm hover:border-[var(--color-text-muted)] transition-colors"
                >
                  取消
                </button>
                <button
                  onClick={handleCreateNetwork}
                  disabled={creating || edgLoading || !newName.trim() || !newDir.trim()}
                  className="flex-1 py-2 rounded-lg bg-gradient-to-r from-[var(--color-accent-cyan)] to-[var(--color-accent-blue)] text-white text-sm font-medium hover:opacity-90 disabled:opacity-50 transition-opacity"
                >
                  {creating || edgLoading ? (lldData ? 'EDG 初始化中...' : '创建中...') : '创建'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </DashboardShell>
  );
}
