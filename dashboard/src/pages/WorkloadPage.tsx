import { useState, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { DashboardShell } from '../components/layout/DashboardShell';
import { Header } from '../components/layout/Header';
import { WizardLayout } from '../components/wizard/WizardLayout';
import { FormField } from '../components/wizard/FormField';
import { ModeSelector } from '../components/wizard/ModeSelector';
import { FileUploadZone } from '../components/wizard/FileUploadZone';
import { CodePreview } from '../components/wizard/CodePreview';
import { ValidationBanner } from '../components/wizard/ValidationBanner';
import { useWizardStore } from '../stores/wizard-store';
import { generatePresetWorkload, generateCustomWorkload, saveFile, fetchModelFamilies } from '../api/simulation-api';

const MODE_OPTIONS = [
  { value: 'preset' as const, label: 'Preset Model', description: 'Choose from GPT/LLaMA/DeepSeek/Mixtral presets' },
  { value: 'custom' as const, label: 'Custom Layers', description: 'Define custom layer configurations' },
  { value: 'upload' as const, label: 'Upload File', description: 'Upload an existing workload file' },
];

interface ModelInfo {
  readonly id: string;
  readonly num_layers: number;
  readonly hidden_size: number;
  readonly num_experts: number;
  readonly recommended: {
    readonly tp: number;
    readonly pp: number;
    readonly ep: number;
    readonly gpus: number;
  };
}

export function WorkloadPage() {
  const navigate = useNavigate();
  const {
    workloadMode, setWorkloadMode,
    workloadConfig, setWorkloadConfig,
    workloadContent, setWorkloadContent,
    setWorkloadSaved, setWorkloadSavedPath,
  } = useWizardStore();

  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [successMsg, setSuccessMsg] = useState<string | null>(null);
  const [families, setFamilies] = useState<Record<string, readonly ModelInfo[]>>({});
  const [selectedFamily, setSelectedFamily] = useState('LLaMA');

  // World Size = TP × DP × PP × EP
  useEffect(() => {
    const worldSize = workloadConfig.tp_size * workloadConfig.dp_size * workloadConfig.pp_size * workloadConfig.ep_size;
    if (worldSize !== workloadConfig.all_gpus) {
      setWorkloadConfig({ ...workloadConfig, all_gpus: worldSize });
    }
  }, [workloadConfig.tp_size, workloadConfig.dp_size, workloadConfig.pp_size, workloadConfig.ep_size]);

  // Fetch model families on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const result = await fetchModelFamilies();
        if (!cancelled) {
          setFamilies(result);
          const currentModel = workloadConfig.model_size;
          for (const [fam, models] of Object.entries(result)) {
            if (models.some(m => m.id === currentModel)) {
              setSelectedFamily(fam);
              break;
            }
          }
        }
      } catch {
        // Fallback: use hardcoded families
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const familyNames = Object.keys(families);
  const modelsInFamily = families[selectedFamily] ?? [];

  const handleFamilyChange = useCallback((fam: string) => {
    setSelectedFamily(fam);
    const models = families[fam];
    if (models && models.length > 0) {
      const m = models[0];
      const rec = m.recommended;
      const dp = Math.max(1, rec.gpus / (rec.tp * rec.pp * (rec.ep || 1)));
      setWorkloadConfig({
        ...workloadConfig,
        model_size: m.id,
        tp_size: rec.tp, pp_size: rec.pp, ep_size: rec.ep,
        dp_size: dp,
      });
    }
  }, [families, workloadConfig, setWorkloadConfig]);

  const handleConfigChange = useCallback(
    (field: string, value: string | number) => {
      if (field === 'model_size') {
        // Auto-fill recommended parallelism when model changes
        const model = modelsInFamily.find(m => m.id === value);
        if (model) {
          const rec = model.recommended;
          const dp = Math.max(1, rec.gpus / (rec.tp * rec.pp * (rec.ep || 1)));
          setWorkloadConfig({
            ...workloadConfig,
            model_size: value as string,
            tp_size: rec.tp, pp_size: rec.pp, ep_size: rec.ep,
            dp_size: dp,
          });
          return;
        }
      }
      setWorkloadConfig({ ...workloadConfig, [field]: value });
    },
    [workloadConfig, setWorkloadConfig, modelsInFamily],
  );

  const handleNext = useCallback(async () => {
    if (workloadMode === 'upload' && !workloadContent) {
      setError('请先上传 Workload 文件');
      return;
    }
    setIsLoading(true);
    setError(null);
    setSuccessMsg(null);
    try {
      let content = workloadContent;
      if (workloadMode === 'preset') {
        const result = await generatePresetWorkload(workloadConfig);
        content = result.content;
        setWorkloadContent(result.content, result.layers);
      } else if (workloadMode === 'custom') {
        const result = await generateCustomWorkload({
          tp_size: workloadConfig.tp_size,
          dp_size: workloadConfig.dp_size,
          pp_size: workloadConfig.pp_size,
          ep_size: workloadConfig.ep_size,
          all_gpus: workloadConfig.all_gpus,
          ga: workloadConfig.ga,
          vpp: workloadConfig.vpp,
          layers_config: [],
        });
        content = result.content;
        setWorkloadContent(result.content, []);
      }

      const wlName = `${workloadConfig.model_size}-tp${workloadConfig.tp_size}-pp${workloadConfig.pp_size}-dp${workloadConfig.dp_size}-ep${workloadConfig.ep_size}-g${workloadConfig.all_gpus}-${Date.now().toString(36)}.txt`;
      const result = await saveFile(wlName, content);
      setWorkloadSaved(true, result.path);
      setWorkloadSavedPath(result.path);
      navigate('/deploy/edg');
    } catch (err) {
      setError(err instanceof Error ? err.message : '操作失败');
    } finally {
      setIsLoading(false);
    }
  }, [workloadMode, workloadConfig, workloadContent, setWorkloadContent, setWorkloadSaved, setWorkloadSavedPath]);

  const handleFileUpload = useCallback(
    (content: string) => {
      setWorkloadContent(content, []);
    },
    [setWorkloadContent],
  );

  // Find current model info for display
  const currentModelInfo = modelsInFamily.find(m => m.id === workloadConfig.model_size);

  return (
    <DashboardShell>
      <Header title="Workload 配置" backLink="/" />

      <WizardLayout
        title="Workload 配置"
        description="生成或上传 Workload 文件，定义 AI 训练仿真的通信模式"
        showDeployStepper
        actions={
          <button
            type="button"
            onClick={handleNext}
            disabled={isLoading}
            className="px-6 py-2 rounded-lg bg-[var(--color-accent-blue)] text-white text-sm font-medium hover:opacity-90 disabled:opacity-50 transition-opacity"
          >
            {isLoading ? '处理中...' : '下一步 →'}
          </button>
        }
      >
        {error && <ValidationBanner type="error" message={error} />}
        {successMsg && <ValidationBanner type="success" message={successMsg} />}

        <div className="space-y-6">
          <ModeSelector value={workloadMode} onChange={setWorkloadMode} options={MODE_OPTIONS} />

          {/* Preset mode — two-level model selection */}
          {workloadMode === 'preset' && (
            <div className="space-y-4">
              {/* Model family tabs */}
              <FormField label="Model Family">
                <div className="flex gap-2 flex-wrap">
                  {familyNames.map(fam => (
                    <button
                      key={fam}
                      type="button"
                      onClick={() => handleFamilyChange(fam)}
                      className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                        selectedFamily === fam
                          ? 'bg-[var(--color-accent-blue)] text-white'
                          : 'bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-secondary)] hover:border-[var(--color-accent-blue)]/40'
                      }`}
                    >
                      {fam}
                    </button>
                  ))}
                </div>
              </FormField>

              {/* Model size cards */}
              <FormField label="Model Size">
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                  {modelsInFamily.map(m => (
                    <button
                      key={m.id}
                      type="button"
                      onClick={() => handleConfigChange('model_size', m.id)}
                      className={`p-3 rounded-lg text-left transition-colors border ${
                        workloadConfig.model_size === m.id
                          ? 'border-[var(--color-accent-blue)] bg-[var(--color-accent-blue)]/10'
                          : 'border-[var(--color-bg-hover)] bg-[var(--color-bg-primary)] hover:border-[var(--color-accent-blue)]/40'
                      }`}
                    >
                      <div className="text-sm font-medium text-[var(--color-text-primary)]">{m.id}</div>
                      <div className="text-xs text-[var(--color-text-muted)] mt-1">
                        {m.num_layers}L · h={m.hidden_size}
                        {m.num_experts > 0 && ` · ${m.num_experts}E`}
                      </div>
                      <div className="text-xs text-[var(--color-text-muted)] mt-0.5 opacity-60">
                        TP{m.recommended.tp} PP{m.recommended.pp}
                        {m.recommended.ep > 1 && ` EP${m.recommended.ep}`}
                        {' · '}{m.recommended.gpus} NPUs
                      </div>
                    </button>
                  ))}
                </div>
              </FormField>

              {/* Model info hint */}
              {currentModelInfo && (
                <div className="text-xs text-[var(--color-text-muted)] px-1">
                  {currentModelInfo.id}: {currentModelInfo.num_layers} layers, hidden={currentModelInfo.hidden_size}
                  {currentModelInfo.num_experts > 0 && `, ${currentModelInfo.num_experts} experts (MoE)`}
                </div>
              )}

              {/* World Size = TP × DP × PP × EP, auto-computed */}
              {(() => {
                const worldSize = workloadConfig.tp_size * workloadConfig.dp_size * workloadConfig.pp_size * workloadConfig.ep_size;
                return (
                  <FormField label="World Size (TP×DP×PP×EP)" hint={`${workloadConfig.tp_size}×${workloadConfig.dp_size}×${workloadConfig.pp_size}×${workloadConfig.ep_size} = ${worldSize} NPUs`}>
                    <input type="number" value={worldSize} readOnly
                      className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-hover)]/30 border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm cursor-default" />
                  </FormField>
                );
              })()}

              {/* Parallelism config */}
              <div className="grid grid-cols-2 gap-4">
                <FormField label="TP Size" hint="Tensor Parallelism">
                  <input type="number" min={1} value={workloadConfig.tp_size}
                    onChange={(e) => handleConfigChange('tp_size', parseInt(e.target.value) || 1)}
                    className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm" />
                </FormField>
                <FormField label="PP Size" hint="Pipeline Parallelism">
                  <input type="number" min={1} value={workloadConfig.pp_size}
                    onChange={(e) => handleConfigChange('pp_size', parseInt(e.target.value) || 1)}
                    className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm" />
                </FormField>
                <FormField label="DP Size" hint="Data Parallelism">
                  <input type="number" min={1} value={workloadConfig.dp_size}
                    onChange={(e) => handleConfigChange('dp_size', parseInt(e.target.value) || 1)}
                    className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm" />
                </FormField>
                <FormField label="EP Size" hint="Expert Parallelism (MoE)">
                  <input type="number" min={1} value={workloadConfig.ep_size}
                    onChange={(e) => handleConfigChange('ep_size', parseInt(e.target.value) || 1)}
                    className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm" />
                </FormField>
                <FormField label="GA" hint="Gradient Accumulation">
                  <input type="number" min={1} value={workloadConfig.ga}
                    onChange={(e) => handleConfigChange('ga', parseInt(e.target.value) || 8)}
                    className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm" />
                </FormField>
                <FormField label="VPP" hint="Virtual Pipeline Stages">
                  <input type="number" min={1} value={workloadConfig.vpp}
                    onChange={(e) => handleConfigChange('vpp', parseInt(e.target.value) || 1)}
                    className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm" />
                </FormField>
              </div>
            </div>
          )}

          {/* Custom mode */}
          {workloadMode === 'custom' && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <FormField label="TP Size">
                  <input type="number" min={1} value={workloadConfig.tp_size}
                    onChange={(e) => handleConfigChange('tp_size', parseInt(e.target.value) || 1)}
                    className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm" />
                </FormField>
                <FormField label="DP Size">
                  <input type="number" min={1} value={workloadConfig.dp_size}
                    onChange={(e) => handleConfigChange('dp_size', parseInt(e.target.value) || 1)}
                    className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm" />
                </FormField>
                <FormField label="PP Size">
                  <input type="number" min={1} value={workloadConfig.pp_size}
                    onChange={(e) => handleConfigChange('pp_size', parseInt(e.target.value) || 1)}
                    className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm" />
                </FormField>
                <FormField label="EP Size" hint="Expert Parallelism (MoE)">
                  <input type="number" min={1} value={workloadConfig.ep_size}
                    onChange={(e) => handleConfigChange('ep_size', parseInt(e.target.value) || 1)}
                    className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm" />
                </FormField>
              </div>
              <FormField label="World Size (TP×DP×PP×EP)" hint={`${workloadConfig.tp_size}×${workloadConfig.dp_size}×${workloadConfig.pp_size}×${workloadConfig.ep_size} = ${workloadConfig.all_gpus} NPUs`}>
                <input type="number" value={workloadConfig.all_gpus} readOnly
                  className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-hover)]/30 border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm cursor-default" />
              </FormField>
              <div className="grid grid-cols-2 gap-4">
                <FormField label="GA" hint="Gradient Accumulation">
                  <input type="number" min={1} value={workloadConfig.ga}
                    onChange={(e) => handleConfigChange('ga', parseInt(e.target.value) || 8)}
                    className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm" />
                </FormField>
                <FormField label="VPP" hint="Virtual Pipeline Parallelism">
                  <input type="number" min={1} value={workloadConfig.vpp}
                    onChange={(e) => handleConfigChange('vpp', parseInt(e.target.value) || 1)}
                    className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-bg-hover)] text-[var(--color-text-primary)] text-sm" />
                </FormField>
              </div>
            </div>
          )}

          {/* Upload mode */}
          {workloadMode === 'upload' && (
            <FileUploadZone
              accept=".txt,.workload"
              label="Drop a workload file here or click to upload (.txt)"
              onFileContent={handleFileUpload}
            />
          )}

          {/* Preview */}
          {workloadContent && (
            <div className="mt-6">
              <CodePreview content={workloadContent} maxHeight="250px" />
            </div>
          )}
        </div>
      </WizardLayout>
    </DashboardShell>
  );
}
