import { useState, useCallback, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { DashboardShell } from '../components/layout/DashboardShell';
import { Header } from '../components/layout/Header';
import { WizardLayout } from '../components/wizard/WizardLayout';
import { FormField } from '../components/wizard/FormField';
import { ValidationBanner } from '../components/wizard/ValidationBanner';
import { EdgDiffGraph } from '../components/edg/EdgDiffGraph';
import { useWizardStore } from '../stores/wizard-store';
import { useNetworkStore } from '../stores/network-store';
import { fetchBaselineGraph, registerTask } from '../api/edg-api';
import { generateRanktable, saveFile } from '../api/simulation-api';

export function EdgPage() {
  const navigate = useNavigate();
  const {
    workloadConfig,
    edgBaselineGraph, setEdgBaselineGraph,
    edgAdjustedGraph, setEdgAdjustedGraph,
    edgDiff, setEdgDiff,
    edgSkipped, setEdgSkipped,
    setEdgTopologyPath,
    setRanktableSavedPath,
  } = useWizardStore();
  const { networks, activeNetworkId } = useNetworkStore();

  const [loading, setLoading] = useState(false);
  const [adjusting, setAdjusting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<string | null>(null);

  const activeNetwork = networks.find((n) => n.id === activeNetworkId);
  const npuPerServer = activeNetwork?.npuPerServer ?? 8;
  const neededGpus = workloadConfig.all_gpus;
  const neededServers = Math.ceil(neededGpus / npuPerServer);
  const serverIds = useMemo(
    () => activeNetwork?.serverIds.slice(0, neededServers) ?? [],
    [activeNetwork, neededServers],
  );

  const isEdgNetwork = (activeNetwork?.serverIds.length ?? 0) > 0;
  const isAdjusted = edgAdjustedGraph !== null;
  const canProceed = isAdjusted || edgSkipped || !isEdgNetwork;

  // Fetch baseline graph on mount — show FULL network, not just participating subset
  useEffect(() => {
    if (!isEdgNetwork || !activeNetwork || edgBaselineGraph) return;
    let cancelled = false;
    setLoading(true);
    fetchBaselineGraph(activeNetwork.serverIds, npuPerServer, activeNetwork.topologyDir)
      .then((res) => {
        if (!cancelled) {
          setEdgBaselineGraph(res.graph);
          setLoading(false);
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : '加载基线拓扑失败');
          setLoading(false);
        }
      });
    return () => { cancelled = true; };
  }, [isEdgNetwork, activeNetwork, npuPerServer, edgBaselineGraph, setEdgBaselineGraph]);

  const handleAdjust = useCallback(async () => {
    if (!activeNetwork || serverIds.length === 0) return;
    setAdjusting(true);
    setError(null);
    try {
      const taskId = `T-${workloadConfig.model_size}-g${neededGpus}-${Date.now().toString(36)}`;
      const result = await registerTask(
        { server_ids: [...serverIds], npu_per_server: npuPerServer, task_id: taskId },
        taskId,
        {
          npu_per_server: String(npuPerServer),
          npu_type: activeNetwork.npuType,
          intra_bw: activeNetwork.intraBw,
          bandwidth: activeNetwork.bandwidth || undefined,
        },
        activeNetwork.topologyDir,
      );
      setEdgAdjustedGraph(result.graph);
      setEdgDiff(result.diff);
      setEdgTopologyPath(result.topology_file);
      setRanktableSavedPath(result.ranktable_path);
      const s = result.graph_stats;
      setStats(`${s.servers} 台服务器 · ${s.leaves} 个 Leaf · ${s.leaf_leaf_edges} 条 OXC 连接`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'EDG 调节失败');
    } finally {
      setAdjusting(false);
    }
  }, [activeNetwork, serverIds, npuPerServer, workloadConfig, neededGpus,
      setEdgAdjustedGraph, setEdgDiff, setEdgTopologyPath, setRanktableSavedPath]);

  const handleSkip = useCallback(async () => {
    setError(null);
    try {
      const rankCount = neededGpus;
      const res = await generateRanktable({ rank_count: rankCount, gpus_per_rack: npuPerServer, superpod_prefix: 'rack' });
      await saveFile('ranktable.json', JSON.stringify(res.ranktable, null, 2));
      setRanktableSavedPath('ranktable.json');
      setEdgSkipped(true);
      navigate('/deploy/launch');
    } catch (err) {
      setError(err instanceof Error ? err.message : '自动生成 RankTable 失败');
    }
  }, [neededGpus, npuPerServer, setRanktableSavedPath, setEdgSkipped, navigate]);

  const handleNext = useCallback(() => {
    navigate('/deploy/launch');
  }, [navigate]);

  const displayGraph = edgAdjustedGraph ?? edgBaselineGraph;
  const participatingIps = useMemo(() => new Set(serverIds), [serverIds]);

  return (
    <DashboardShell>
      <Header title="OXC 调节" backLink="/" />

      <WizardLayout
        title="OXC 组网调节"
        description="根据训练任务的 NPU 需求，调节 OXC 光交换网络的连接拓扑"
        showDeployStepper
        wide
        actions={
          <div className="flex items-center gap-3">
            <a
              href="/deploy/workload"
              className="px-4 py-2 rounded-lg border border-[var(--color-bg-hover)] text-[var(--color-text-secondary)] text-sm hover:border-[var(--color-text-muted)] transition-colors"
            >
              &larr; 上一步
            </a>
            {canProceed && (
              <button
                type="button"
                onClick={handleNext}
                className="px-4 py-2 rounded-lg bg-[var(--color-accent-blue)] text-white text-sm font-medium hover:opacity-90 transition-opacity"
              >
                下一步 &rarr;
              </button>
            )}
          </div>
        }
      >
        {error && <ValidationBanner type="error" message={error} />}

        <div className="space-y-6">
          {/* Task info */}
          <div className="grid grid-cols-3 gap-3">
            <InfoCard label="所需 GPU" value={String(neededGpus)} />
            <InfoCard label="所需服务器" value={`${neededServers} 台 (${npuPerServer} GPU/台)`} />
            <InfoCard label="组网" value={activeNetwork?.name ?? '未选择'} />
          </div>

          {/* No EDG network warning */}
          {!isEdgNetwork && (
            <ValidationBanner
              type="info"
              message="当前组网未配置服务器 IP（非 EDG 组网），将自动生成默认 RankTable。"
            />
          )}

          {/* Topology graph */}
          {loading && !displayGraph && (
            <div className="flex items-center justify-center py-16 text-sm text-[var(--color-text-muted)]">
              加载基线拓扑中...
            </div>
          )}

          {displayGraph && (
            <EdgDiffGraph
              baselineGraph={edgBaselineGraph!}
              adjustedGraph={edgAdjustedGraph}
              diff={edgDiff}
              participatingServerIds={participatingIps}
              height="380px"
            />
          )}

          {/* Stats */}
          {stats && (
            <div className="p-3 rounded-lg border border-[var(--color-accent-green)]/20 bg-[var(--color-accent-green)]/5">
              <span className="text-xs font-medium text-[var(--color-accent-green)]">调节完成</span>
              <span className="text-xs text-[var(--color-text-muted)] ml-2">{stats}</span>
            </div>
          )}

          {/* Action buttons */}
          {isEdgNetwork && !isAdjusted && (
            <div className="flex gap-3">
              <button
                type="button"
                onClick={handleAdjust}
                disabled={adjusting || !edgBaselineGraph}
                className="flex-1 py-2.5 rounded-lg bg-[var(--color-accent-cyan)] text-white text-sm font-medium hover:opacity-90 disabled:opacity-50 transition-opacity"
              >
                {adjusting ? '调节中...' : '调节网络'}
              </button>
              <button
                type="button"
                onClick={handleSkip}
                className="px-6 py-2.5 rounded-lg border border-[var(--color-bg-hover)] text-[var(--color-text-secondary)] text-sm hover:border-[var(--color-text-muted)] transition-colors"
              >
                跳过调节
              </button>
            </div>
          )}
        </div>
      </WizardLayout>
    </DashboardShell>
  );
}

function InfoCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-3 rounded-lg border border-[var(--color-bg-hover)] bg-[var(--color-bg-secondary)]">
      <div className="text-[10px] text-[var(--color-text-muted)] mb-0.5">{label}</div>
      <div className="text-sm text-[var(--color-text-primary)]">{value}</div>
    </div>
  );
}
