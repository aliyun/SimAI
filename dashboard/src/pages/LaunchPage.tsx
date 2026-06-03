import { useState, useCallback, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { DashboardShell } from '../components/layout/DashboardShell';
import { Header } from '../components/layout/Header';
import { WizardLayout } from '../components/wizard/WizardLayout';
import { FormField } from '../components/wizard/FormField';
import { ValidationBanner } from '../components/wizard/ValidationBanner';
import { LogViewer } from '../components/widgets/LogViewer';
import { ProcessList } from '../components/widgets/ProcessList';
import { useWizardStore } from '../stores/wizard-store';
import { useNetworkStore } from '../stores/network-store';
import { useLogStream } from '../hooks/use-log-stream';
import { launchSimulation, killProcess as apiKillProcess } from '../api/simulation-api';
import { ModeSelector } from '../components/wizard/ModeSelector';
import type { SimulationMode, CollectiveMode } from '../types/wizard';

/** 二进制选择矩阵：collectiveMode × simulationMode → binary name */
const BINARY_MAP: Record<CollectiveMode, Record<SimulationMode, string>> = {
  'nccl':     { analytical: 'SimAI_analytical',     ns3: 'SimAI_simulator'     },
  'oxc-hccl': { analytical: 'SimAI_analytical_oxc', ns3: 'SimAI_simulator_oxc' },
};

const COLLECTIVE_OPTIONS = [
  {
    value: 'nccl' as CollectiveMode,
    label: 'NCCL',
    description: '标准 NCCL 集合通信算法，适用于传统以太网 / InfiniBand 互联网络',
  },
  {
    value: 'oxc-hccl' as CollectiveMode,
    label: 'OXC-HCCL',
    description: 'OXC（光学电路交换）感知的 HCCL 算法，利用光交换网络的拓扑感知调度降低通信延迟',
  },
] as const;

const SIM_MODE_OPTIONS = [
  {
    value: 'analytical' as SimulationMode,
    label: 'Analytical（快速）',
    description: '基于 busbw 总线带宽抽象估算集合通信耗时，速度快，适合大规模扫参',
  },
  {
    value: 'ns3' as SimulationMode,
    label: 'NS3 Simulation（精细）',
    description: '全栈网络仿真，逐包建模物理传输，结果精度高，适合深度分析单配置',
  },
] as const;

export function LaunchPage() {
  const {
    launchConfig, setLaunchConfig,
    activePid, setActivePid,
    simulationStatus, setSimulationStatus,
    setResultPrefix,
    workloadSavedPath, ranktableSavedPath,
    workloadConfig,
    edgTopologyPath,
  } = useWizardStore();
  const { networks, activeNetworkId, setActiveNetwork } = useNetworkStore();

  const [isLaunching, setIsLaunching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { logs, status: processStatus } = useLogStream({
    pid: activePid,
    enabled: simulationStatus === 'running',
  });

  const activeNetwork = networks.find((n) => n.id === activeNetworkId);

  // Auto-fill paths from wizard store on mount
  useEffect(() => {
    if (workloadSavedPath && !launchConfig.workloadPath) {
      setLaunchConfig({ workloadPath: workloadSavedPath });
    }
    if (ranktableSavedPath && !launchConfig.ranktablePath) {
      setLaunchConfig({ ranktablePath: ranktableSavedPath });
    }
  }, [workloadSavedPath, ranktableSavedPath]);

  // Auto-detect completion
  useEffect(() => {
    if (['exited', 'finished', 'error', 'dead', 'timeout', 'killed'].includes(processStatus) && simulationStatus === 'running') {
      setSimulationStatus('completed');
    }
  }, [processStatus, simulationStatus, setSimulationStatus]);

  const handleNetworkChange = useCallback(
    (id: string) => {
      setActiveNetwork(id);
      const net = networks.find((n) => n.id === id);
      if (net) {
        setLaunchConfig({ topologyPath: net.topologyDir });
      }
    },
    [networks, setActiveNetwork, setLaunchConfig],
  );

  const handleLaunch = useCallback(async () => {
    if (!activeNetworkId) {
      setError('请先选择一个组网');
      return;
    }
    setIsLaunching(true);
    setError(null);
    try {
      const topoOverride = edgTopologyPath || launchConfig.topologyPath;
      const npuPerServer = activeNetwork?.npuPerServer ?? 8;

      const binary = BINARY_MAP[launchConfig.collectiveMode][launchConfig.mode];
      const extraArgs: string[] = [];

      const ts = Date.now().toString(36);
      const resultPrefix = `${workloadConfig.model_size}-tp${workloadConfig.tp_size}-pp${workloadConfig.pp_size}-dp${workloadConfig.dp_size}-ep${workloadConfig.ep_size}-g${workloadConfig.all_gpus}-${ts}-`;

      if (launchConfig.mode === 'analytical') {
        extraArgs.push(
          '-g', String(workloadConfig.all_gpus),
          '-g_p_s', String(npuPerServer),
          '-r', resultPrefix,
        );
        if (launchConfig.intraBw > 0) {
          extraArgs.push('-nv', String(launchConfig.intraBw));
        }
        if (launchConfig.nicBw > 0) {
          extraArgs.push('-nic', String(launchConfig.nicBw));
          extraArgs.push('-n_p_s', String(npuPerServer)); // only when nic specified
        }
        if (launchConfig.npuType && launchConfig.npuType !== 'A3') {
          extraArgs.push('-g_type', launchConfig.npuType);
        }
        if (launchConfig.nicType && launchConfig.nicType !== 'cx7') {
          extraArgs.push('-nic_t', launchConfig.nicType);
        }
        if (launchConfig.tpOverlap > 0) extraArgs.push('-tp_o', String(launchConfig.tpOverlap));
        if (launchConfig.dpOverlap > 0) extraArgs.push('-dp_o', String(launchConfig.dpOverlap));
        if (launchConfig.epOverlap > 0) extraArgs.push('-ep_o', String(launchConfig.epOverlap));
        if (launchConfig.ppOverlap !== 1) extraArgs.push('-pp_o', String(launchConfig.ppOverlap));
      } else {
        // NS3 OXC binary (AstraSimNetwork_oxc) only accepts: -t -w -n -c
        // -w and -n are handled by process_service; NPU info comes from topology file
        extraArgs.push(
          '-t', String(launchConfig.threads),
          '-c', 'astra-sim-alibabacloud/inputs/config/SimAI.conf',
        );
      }

      // When OXC-HCCL mode, inject OXC-enabling env vars
      const envVars: Record<string, string> = { ...launchConfig.envVars };
      if (launchConfig.collectiveMode === 'oxc-hccl') {
        envVars['AS_OXC_ENABLE'] = '1';
        if (launchConfig.oxcUrl) envVars['AS_OXC_URL'] = launchConfig.oxcUrl;
        if (launchConfig.oxcAlgo) envVars['AS_OXC_ALGO'] = launchConfig.oxcAlgo;
      }

      const result = await launchSimulation({
        binary,
        workload_file: launchConfig.workloadPath || 'workload.txt',
        ranktable_file: launchConfig.ranktablePath || ranktableSavedPath || 'ranktable.json',
        topology_file: topoOverride || undefined,
        env_vars: envVars,
        extra_args: extraArgs,
      });
      setActivePid(result.pid);
      setResultPrefix(resultPrefix);
      sessionStorage.setItem('simai_result_prefix', resultPrefix);
      sessionStorage.setItem('simai_result_pid', String(result.pid));
      setSimulationStatus('running');
    } catch (err) {
      setError(err instanceof Error ? err.message : '启动失败');
      setSimulationStatus('failed');
    } finally {
      setIsLaunching(false);
    }
  }, [activeNetworkId, launchConfig, workloadConfig, edgTopologyPath, ranktableSavedPath, setActivePid, setSimulationStatus, setResultPrefix]);

  const handleKill = useCallback(async () => {
    if (!activePid) return;
    try {
      await apiKillProcess(activePid);
      setSimulationStatus('idle');
      setActivePid(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : '停止失败');
    }
  }, [activePid, setSimulationStatus, setActivePid]);

  const isRunning = simulationStatus === 'running';

  return (
    <DashboardShell>
      <Header title="启动仿真" backLink="/" />

      <WizardLayout
        title="启动仿真"
        description="选择组网并配置仿真参数后启动 OCS-Sim 仿真任务"
        showDeployStepper
        actions={
          <Link
            to="/deploy/edg"
            className="px-4 py-2 rounded-lg border border-[var(--color-bg-hover)] text-[var(--color-text-secondary)] text-sm hover:border-[var(--color-text-muted)] transition-colors"
          >
            &larr; 上一步
          </Link>
        }
      >
        {error && <ValidationBanner type="error" message={error} />}
        {simulationStatus === 'completed' && (
          <ValidationBanner type="success" message="仿真完成！" />
        )}
        {simulationStatus === 'completed' && (
          <Link
            to="/results"
            className="block w-full text-center py-2.5 rounded-lg bg-[var(--color-accent-blue)] text-white text-sm font-medium hover:opacity-90 transition-opacity"
          >
            查看仿真结果 →
          </Link>
        )}

        <div className="space-y-6">
          {/* Network selector */}
          <FormField label="选择组网" hint="必须选择一个组网后才能启动仿真">
            <select
              value={activeNetworkId ?? ''}
              onChange={(e) => handleNetworkChange(e.target.value)}
              disabled={isRunning}
              className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm disabled:opacity-50"
            >
              <option value="">-- 请选择组网 --</option>
              {networks.map((net) => (
                <option key={net.id} value={net.id}>
                  {net.name} ({net.topologyDir})
                </option>
              ))}
            </select>
            {activeNetwork && (
              <p className="text-xs text-[var(--color-text-muted)] mt-1">
                拓扑目录: {activeNetwork.topologyDir}
              </p>
            )}
          </FormField>

          {/* Auto-filled paths from previous steps */}
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-[var(--color-text-secondary)]">已配置的文件</h3>
            <div className="grid grid-cols-2 gap-3">
              <div className={`p-3 rounded-lg border text-xs ${workloadSavedPath ? 'border-[var(--color-accent-green)]/30 bg-[var(--color-accent-green)]/5' : 'border-[var(--color-bg-hover)] bg-[var(--color-bg-secondary)]'}`}>
                <div className="text-[var(--color-text-muted)] mb-1">Workload</div>
                <div className="text-[var(--color-text-primary)] truncate">
                  {workloadSavedPath || '未配置 — 请先在 Workload 步骤保存'}
                </div>
              </div>
              <div className={`p-3 rounded-lg border text-xs ${ranktableSavedPath ? 'border-[var(--color-accent-green)]/30 bg-[var(--color-accent-green)]/5' : 'border-[var(--color-bg-hover)] bg-[var(--color-bg-secondary)]'}`}>
                <div className="text-[var(--color-text-muted)] mb-1">RankTable</div>
                <div className="text-[var(--color-text-primary)] truncate">
                  {ranktableSavedPath || '未配置 — 请先在 RankTable 步骤保存'}
                </div>
              </div>
            </div>
          </div>

          {/* Two-level mode selection */}
          <div className="space-y-4">
            {/* Level 1: Collective communication mode */}
            <FormField
              label="集合通信模式"
              hint="决定使用哪套集合通信算法库及对应的二进制"
            >
              <ModeSelector
                value={launchConfig.collectiveMode}
                onChange={(v) => setLaunchConfig({ collectiveMode: v })}
                options={COLLECTIVE_OPTIONS}
                disabled={isRunning}
              />
            </FormField>

            {/* Level 2: Simulation engine */}
            <FormField
              label="仿真引擎"
              hint={`当前将使用二进制：${BINARY_MAP[launchConfig.collectiveMode][launchConfig.mode]}`}
            >
              <ModeSelector
                value={launchConfig.mode}
                onChange={(v) => setLaunchConfig({ mode: v })}
                options={SIM_MODE_OPTIONS}
                disabled={isRunning}
              />
            </FormField>
          </div>

          {/* OXC configuration — only shown when OXC-HCCL mode is selected */}
          {launchConfig.collectiveMode === 'oxc-hccl' && (
            <div className="space-y-3 p-3 rounded-lg border border-[var(--color-accent-green)]/20 bg-[var(--color-accent-green)]/5">
              <div className="text-xs font-medium text-[var(--color-accent-green)]">OXC 配置</div>
              <div className="grid grid-cols-2 gap-3">
                <FormField label="OXC 服务地址" hint="留空则用 C++ 默认 http://localhost:8080">
                  <input
                    type="text"
                    value={launchConfig.oxcUrl}
                    onChange={(e) => setLaunchConfig({ oxcUrl: e.target.value })}
                    disabled={isRunning}
                    placeholder="留空 = http://localhost:8080"
                    className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm disabled:opacity-50"
                  />
                </FormField>
                <FormField label="调度算法" hint="留空则用 C++ 默认 ALGO_OXC_RING">
                  <select
                    value={launchConfig.oxcAlgo}
                    onChange={(e) => setLaunchConfig({ oxcAlgo: e.target.value })}
                    disabled={isRunning}
                    className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm disabled:opacity-50"
                  >
                    <option value="">默认 (ALGO_OXC_RING)</option>
                    <option value="ALGO_OXC_TREE">ALGO_OXC_TREE</option>
                    <option value="ALGO_OXC_MESH">ALGO_OXC_MESH</option>
                  </select>
                </FormField>
              </div>
            </div>
          )}

          {/* 高级参数 (analytical mode only) */}
          {launchConfig.mode === 'analytical' && (
            <details className="p-3 rounded-lg border border-[var(--color-bg-hover)] bg-[var(--color-bg-secondary)]/50">
              <summary className="text-xs font-medium text-[var(--color-text-muted)] cursor-pointer select-none">
                高级参数
              </summary>
              <div className="mt-3 space-y-3">
                <div className="text-xs text-[var(--color-text-muted)]">Overlap Ratio (计算-通信重叠，0=完全串行，1=完全隐藏)</div>
                <div className="grid grid-cols-4 gap-2">
                  {(['tpOverlap','dpOverlap','epOverlap','ppOverlap'] as const).map((k) => (
                    <FormField key={k} label={k.replace('Overlap','').toUpperCase()} hint={String(launchConfig[k])}>
                      <input type="range" min={0} max={1} step={0.1}
                        value={launchConfig[k]} disabled={isRunning}
                        onChange={(e) => setLaunchConfig({ [k]: parseFloat(e.target.value) })}
                        className="w-full" />
                    </FormField>
                  ))}
                </div>
                <div className="text-xs text-[var(--color-text-muted)]">带宽与硬件</div>
                <div className="grid grid-cols-2 gap-2">
                  <FormField label="服务器内带宽 BW (GB/s)" hint="0=自动检测">
                    <input type="number" min={0} value={launchConfig.intraBw}
                      onChange={(e) => setLaunchConfig({ intraBw: parseInt(e.target.value) || 0 })}
                      disabled={isRunning} className="w-full px-2 py-1.5 rounded bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-xs" />
                  </FormField>
                  <FormField label="NIC BW (GB/s)" hint="0=自动检测">
                    <input type="number" min={0} value={launchConfig.nicBw}
                      onChange={(e) => setLaunchConfig({ nicBw: parseInt(e.target.value) || 0 })}
                      disabled={isRunning} className="w-full px-2 py-1.5 rounded bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-xs" />
                  </FormField>
                  <FormField label="NPU 类型">
                    <select value={launchConfig.npuType}
                      onChange={(e) => setLaunchConfig({ npuType: e.target.value })}
                      disabled={isRunning} className="w-full px-2 py-1.5 rounded bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-xs">
                      <option>A3</option><option>H100</option><option>H800</option><option>A800</option><option>H20</option>
                    </select>
                  </FormField>
                  <FormField label="NIC 类型">
                    <select value={launchConfig.nicType}
                      onChange={(e) => setLaunchConfig({ nicType: e.target.value })}
                      disabled={isRunning} className="w-full px-2 py-1.5 rounded bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-xs">
                      <option>cx7</option><option>cx6</option><option>bf3</option>
                    </select>
                  </FormField>
                </div>
              </div>
            </details>
          )}

          {/* Configuration */}
          <div className="grid grid-cols-2 gap-4">
            <FormField label="线程数" hint="NS3 模式建议 8-16">
              <input
                type="number"
                min={1}
                max={64}
                value={launchConfig.threads}
                onChange={(e) => setLaunchConfig({ threads: parseInt(e.target.value) || 8 })}
                disabled={isRunning}
                className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm disabled:opacity-50"
              />
            </FormField>

            <FormField label="Workload 路径">
              <input
                type="text"
                value={launchConfig.workloadPath}
                onChange={(e) => setLaunchConfig({ workloadPath: e.target.value })}
                disabled={isRunning}
                placeholder={workloadSavedPath || 'workload.txt'}
                className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm disabled:opacity-50"
              />
            </FormField>

            <FormField label="RankTable 路径">
              <input
                type="text"
                value={launchConfig.ranktablePath}
                onChange={(e) => setLaunchConfig({ ranktablePath: e.target.value })}
                disabled={isRunning}
                placeholder={ranktableSavedPath || 'ranktable.json'}
                className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm disabled:opacity-50"
              />
            </FormField>

            <FormField label="拓扑路径" hint="NS3 模式需要">
              <input
                type="text"
                value={launchConfig.topologyPath}
                onChange={(e) => setLaunchConfig({ topologyPath: e.target.value })}
                disabled={isRunning}
                placeholder="选择组网后自动填充"
                className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm disabled:opacity-50"
              />
            </FormField>
          </div>

          {/* Launch / Kill */}
          {!isRunning ? (
            <button
              type="button"
              onClick={handleLaunch}
              disabled={isLaunching || !activeNetworkId}
              className="w-full py-2.5 rounded-lg bg-[var(--color-accent-green)] text-white text-sm font-medium hover:opacity-90 disabled:opacity-50 transition-opacity"
            >
              {isLaunching ? '启动中...' : '启动仿真'}
            </button>
          ) : (
            <button
              type="button"
              onClick={handleKill}
              className="w-full py-2.5 rounded-lg bg-[var(--color-accent-red)] text-white text-sm font-medium hover:opacity-90 transition-opacity"
            >
              停止仿真
            </button>
          )}

          {/* Process list */}
          <ProcessList />

          {/* Logs */}
          {activePid && (
            <LogViewer logs={logs} status={processStatus} />
          )}
        </div>
      </WizardLayout>
    </DashboardShell>
  );
}
