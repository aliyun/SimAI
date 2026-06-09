#!/usr/bin/env bash
# =============================================================================
# run_scenarios.sh — SimAI / AICB Vidur 四场景一键运行脚本
#
# 所有文件统一汇聚于 examples/vidur-ali-scenarios/ 目录:
#   examples/vidur-ali-scenarios/
#   ├── run_scenarios.sh               ← 本脚本
#   ├── logs/                          ← tee 运行日志
#   │   └── scenario_<N>_<TIMESTAMP>.log
#   └── simulator_output/              ← vidur 模拟输出 (通过 --output_dir 覆盖)
#       └── <YYYY-MM-DD_HH-MM-SS>/
#
# 用法:
#   bash examples/vidur-ali-scenarios/run_scenarios.sh --scenario <1|2|3|4>
#   bash examples/vidur-ali-scenarios/run_scenarios.sh --all
#   bash examples/vidur-ali-scenarios/run_scenarios.sh -h | --help
#
# 场景说明:
#   1  Qwen3-Next-80B  无PD分离  ws=32 (dp=32, tp=1, pp=1, ep=32)     调度: lor
#   2  Qwen3-Next-80B  PD分离    ws=8  (P=2, D=6, tp=1, pp=1)         调度: split_wise
#   3  DeepSeek-671B   PD分离    ws=8  (P=2, D=6, tp=8, pp=1, EP auto)  调度: split_wise
#   4  Qwen3-MoE-235B  PD分离    ws=8  (P=2, D=6, tp=4, pp=1, EP auto)  调度: split_wise
#
# 环境要求:
#   conda activate vidur
#   conda 路径: /root/miniconda3/envs/vidur
#   python:     /root/miniconda3/envs/vidur/bin/python
# =============================================================================

set -euo pipefail

# ===================== 路径设置 =====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDUR_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
OUTPUT_DIR="$SCRIPT_DIR/simulator_output"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

# ===================== 工具函数 =====================

cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        echo ""
        echo "[WARN] Script exited abnormally (脚本异常退出), exit_code=$exit_code"
        echo "       Log dir (日志目录): $LOG_DIR"
        echo "       Output dir (输出目录): $OUTPUT_DIR"
    fi
}
trap 'cleanup' EXIT INT TERM

validate_environment() {
    local conda_env="${CONDA_DEFAULT_ENV:-}"
    if [[ "$conda_env" != "vidur" ]]; then
        echo "[ERROR] vidur conda env not detected (未检测到 vidur conda 环境)"
        echo "        Current env (当前环境): ${conda_env:-N/A}"
        echo "        Please run (请先执行): conda activate vidur"
        echo "        conda path (路径): /root/miniconda3/envs/vidur"
        exit 1
    fi
    local python_bin
    python_bin="$(which python 2>/dev/null || true)"
    if [[ "$python_bin" != */miniconda3/envs/vidur/* ]]; then
        echo "[ERROR] python not in vidur env (python 路径不在 vidur 环境内)"
        echo "        Current python (当前 python): ${python_bin:-not found}"
        echo "        Expected path (期望路径): /root/miniconda3/envs/vidur/bin/python"
        exit 1
    fi
    echo "[INFO] Env check passed (环境检查通过): conda=$conda_env, python=$python_bin"
}

check_disk_space() {
    local required_gb=10
    local available
    available=$(df "$SCRIPT_DIR" | awk 'NR==2 {print int($4/1024/1024)}')
    if [[ "$available" -lt "$required_gb" ]]; then
        echo "[ERROR] Insufficient disk space (磁盘空间不足): need ${required_gb}GB, available ${available}GB"
        exit 1
    fi
    echo "[INFO] Disk check passed (磁盘空间检查通过): available ${available}GB, need ${required_gb}GB"
}

progress_bar() {
    local current=$1 total=$2
    local percent=$((current * 100 / total))
    local filled=$((percent / 5))
    local bar
    bar=$(printf "%${filled}s" | tr ' ' '=')
    local empty=$((20 - filled))
    local space
    space=$(printf "%${empty}s")
    printf "\n[%-20s] %d%% (%d/%d)\n" "${bar}${space}" "$percent" "$current" "$total"
}

validate_scenario_output() {
    local scenario_num=$1
    local output_dir=$2
    # Find the latest timestamped output directory
    # Use || true to prevent SIGPIPE when ls outputs multiple entries under set -eo pipefail
    local latest_dir
    latest_dir=$(ls -td "$output_dir"/*/ 2>/dev/null | head -1) || true
    if [[ -z "$latest_dir" ]]; then
        echo "[WARN] Scenario $scenario_num: no output directory found in $output_dir"
        return 1
    fi
    if [[ -f "$latest_dir/chrome_trace.json" ]]; then
        echo "[INFO] Scenario $scenario_num: validated (chrome_trace.json found)"
    else
        echo "[WARN] Scenario $scenario_num: chrome_trace.json NOT found in $latest_dir"
        return 1
    fi
}

# ===================== 公共参数 =====================
# 四个场景共用的硬件/请求生成/后端参数
COMMON_ARGS=(
    # 硬件
    --replica_config_pd_p2p_comm_bandwidth  800
    --replica_config_nvlink_bandwidth       1600
    --replica_config_rdma_bandwidth         800
    --replica_config_pd_p2p_comm_dtype      fp8
    --replica_config_network_device         h20_dgx
    --replica_config_device                 h20
    # 请求生成: Poisson QPS=100, 固定长度 prefill=100 / decode=8
    --request_generator_config_type         synthetic
    --interval_generator_config_type        poisson
    --poisson_request_interval_generator_config_qps 100
    --synthetic_request_generator_config_num_requests 4
    --length_generator_config_type          fixed
    --fixed_request_length_generator_config_prefill_tokens 100
    --fixed_request_length_generator_config_decode_tokens  8
    --trace_request_length_generator_config_trace_file \
        ./data/processed_traces/splitwise_conv.csv
    # 后端
    --random_forrest_execution_time_predictor_config_backend aicb
    # 输出目录 → examples/vidur-ali-scenarios/simulator_output/
    --metrics_config_output_dir "$OUTPUT_DIR"
)

# ===================== 场景函数 =====================

# -----------------------------------------------------------------------
# 场景 1: Qwen3-Next-80B 无PD分离
#   cluster_config_num_replicas = 32 (即 dp=32)
#   ws = tp(1) × pp(1) × dp(32) = 32，ep = ws = 32（自动）
#   调度: global=lor, replica=sarathi
# -----------------------------------------------------------------------
run_scenario_1() {
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local log_file="$LOG_DIR/scenario_1_${ts}.log"
    echo "[INFO] === Scenario 1: Qwen3-Next-80B, no PD, ws=32, lor (场景1: 无PD, ws=32, lor) ==="
    echo "[INFO] Log (日志): $log_file"
    cd "$VIDUR_ROOT"
    set +o pipefail
    python -m vidur.main \
        "${COMMON_ARGS[@]}" \
        --cluster_config_num_replicas         32 \
        --replica_config_pd_node_ratio        1 \
        --global_scheduler_config_type        lor \
        --replica_scheduler_config_type       sarathi \
        --replica_config_model_name           qwen3-next-80B \
        --replica_config_tensor_parallel_size 1 \
        --replica_config_num_pipeline_stages  1 \
        2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    set -o pipefail
    if [[ $exit_code -ne 0 ]]; then
        echo "[ERROR] Scenario 1 failed (exit_code=$exit_code), see: $log_file"
        return $exit_code
    fi
    validate_scenario_output 1 "$OUTPUT_DIR"
    echo "[INFO] Scenario 1 done (场景1 完成)"
}

# -----------------------------------------------------------------------
# 场景 2: Qwen3-Next-80B PD分离
#   总 replica=8; num_prefill_replicas=2 → prefill dp=2, decode dp=6
#   prefill: ws = tp(1) × pp(1) × dp(2) = 2，ep = 2
#   decode:  ws = tp(1) × pp(1) × dp(6) = 6，ep = 6
#   调度: global=split_wise, replica=split_wise
# -----------------------------------------------------------------------
run_scenario_2() {
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local log_file="$LOG_DIR/scenario_2_${ts}.log"
    echo "[INFO] === Scenario 2: Qwen3-Next-80B, PD, P=2 D=6, split_wise (场景2: PD分离, P=2 D=6) ==="
    echo "[INFO] Log (日志): $log_file"
    cd "$VIDUR_ROOT"
    set +o pipefail
    python -m vidur.main \
        "${COMMON_ARGS[@]}" \
        --cluster_config_num_replicas                  8 \
        --replica_config_pd_node_ratio                 0.25 \
        --replica_config_num_prefill_replicas           2 \
        --global_scheduler_config_type                 split_wise \
        --replica_scheduler_config_type                split_wise \
        --replica_config_model_name                    qwen3-next-80B \
        --replica_config_tensor_parallel_size          1 \
        --replica_config_num_pipeline_stages           1 \
        --replica_config_prefill_tensor_parallel_size  1 \
        --replica_config_prefill_num_pipeline_stages   1 \
        --replica_config_decode_tensor_parallel_size   1 \
        --replica_config_decode_num_pipeline_stages    1 \
        2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    set -o pipefail
    if [[ $exit_code -ne 0 ]]; then
        echo "[ERROR] Scenario 2 failed (exit_code=$exit_code), see: $log_file"
        return $exit_code
    fi
    validate_scenario_output 2 "$OUTPUT_DIR"
    echo "[INFO] Scenario 2 done (场景2 完成)"
}

# -----------------------------------------------------------------------
# 场景 3: DeepSeek-671B PD分离
#   总 replica=8; pd_node_ratio=0.25 → prefill dp=2, decode dp=6
#   ws = tp(8) × pp(1) × dp = 16(P)/48(D)，EP auto-set to world_size
#   调度: global=split_wise, replica=split_wise
# -----------------------------------------------------------------------
run_scenario_3() {
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local log_file="$LOG_DIR/scenario_3_${ts}.log"
    echo "[INFO] === Scenario 3: DeepSeek-671B, PD, tp=8, EP=auto, split_wise (场景3: PD分离, tp=8, EP=auto) ==="
    echo "[INFO] Log (日志): $log_file"
    cd "$VIDUR_ROOT"
    set +o pipefail
    python -m vidur.main \
        "${COMMON_ARGS[@]}" \
        --cluster_config_num_replicas                  8 \
        --replica_config_pd_node_ratio                 0.25 \
        --global_scheduler_config_type                 split_wise \
        --replica_scheduler_config_type                split_wise \
        --replica_config_model_name                    deepseek-671B \
        --replica_config_tensor_parallel_size          8 \
        --replica_config_num_pipeline_stages           1 \
        2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    set -o pipefail
    if [[ $exit_code -ne 0 ]]; then
        echo "[ERROR] Scenario 3 failed (exit_code=$exit_code), see: $log_file"
        return $exit_code
    fi
    validate_scenario_output 3 "$OUTPUT_DIR"
    echo "[INFO] Scenario 3 done (场景3 完成)"
}

# -----------------------------------------------------------------------
# 场景 4: Qwen3-MoE-235B PD分离
#   总 replica=8; pd_node_ratio=0.25 → prefill dp=2, decode dp=6
#   ws = tp(4) × pp(1) × dp = 8(P)/24(D)，EP auto-set to world_size
#   调度: global=split_wise, replica=split_wise
# -----------------------------------------------------------------------
run_scenario_4() {
    local ts
    ts="$(date +%Y%m%d_%H%M%S)"
    local log_file="$LOG_DIR/scenario_4_${ts}.log"
    echo "[INFO] === Scenario 4: Qwen3-MoE-235B, PD, tp=4, EP=auto, split_wise (场景4: PD分离, tp=4, EP=auto) ==="
    echo "[INFO] Log (日志): $log_file"
    cd "$VIDUR_ROOT"
    set +o pipefail
    python -m vidur.main \
        "${COMMON_ARGS[@]}" \
        --cluster_config_num_replicas                  8 \
        --replica_config_pd_node_ratio                 0.25 \
        --global_scheduler_config_type                 split_wise \
        --replica_scheduler_config_type                split_wise \
        --replica_config_model_name                    qwen3-moe-235B \
        --replica_config_tensor_parallel_size          4 \
        --replica_config_num_pipeline_stages           1 \
        2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}
    set -o pipefail
    if [[ $exit_code -ne 0 ]]; then
        echo "[ERROR] Scenario 4 failed (exit_code=$exit_code), see: $log_file"
        return $exit_code
    fi
    validate_scenario_output 4 "$OUTPUT_DIR"
    echo "[INFO] Scenario 4 done (场景4 完成)"
}

# ===================== 帮助信息 =====================

print_help() {
    cat <<'EOF'
Usage (用法):
  bash examples/vidur-ali-scenarios/run_scenarios.sh --scenario <N>   Run single scenario (运行单个场景, N=1~4)
  bash examples/vidur-ali-scenarios/run_scenarios.sh --all            Run all 4 scenarios (顺序运行全部四个场景)
  bash examples/vidur-ali-scenarios/run_scenarios.sh -h | --help      Print help (打印帮助)

Scenarios (场景列表):
  1  Qwen3-Next-80B  no PD (无PD分离)  ws=32             scheduler: lor
  2  Qwen3-Next-80B  PD (PD分离)      ws=8 (P=2,D=6)    scheduler: split_wise
  3  DeepSeek-671B   PD (PD分离)      tp=8, EP=auto     scheduler: split_wise
  4  Qwen3-MoE-235B  PD (PD分离)      tp=4, EP=auto     scheduler: split_wise

Output dir (输出目录): examples/vidur-ali-scenarios/simulator_output/<TIMESTAMP>/
Log dir (日志目录): examples/vidur-ali-scenarios/logs/scenario_<N>_<TIMESTAMP>.log
EOF
}

# ===================== 入口 =====================

main() {
    # --help / -h 不需要环境检查，直接处理
    case "${1:-}" in
        -h|--help|"") print_help; exit 0 ;;
    esac

    echo "============================================================"
    echo " SimAI / AICB Vidur 4-Scenario Runner (四场景运行脚本)"
    echo " Root dir (根目录): $SCRIPT_DIR"
    echo "============================================================"

    validate_environment
    check_disk_space

    case "${1:-}" in
        --scenario)
            case "${2:-}" in
                1) run_scenario_1 ;;
                2) run_scenario_2 ;;
                3) run_scenario_3 ;;
                4) run_scenario_4 ;;
                *) echo "[ERROR] Invalid scenario (无效场景编号): ${2:-}, use 1~4"; exit 1 ;;
            esac
            ;;
        --all)
            local total=4
            run_scenario_1
            progress_bar 1 $total

            run_scenario_2
            progress_bar 2 $total

            run_scenario_3
            progress_bar 3 $total

            run_scenario_4
            progress_bar 4 $total

            echo ""
            echo "[INFO] All 4 scenarios completed (全部 4 个场景运行完毕)!"
            echo "       Logs (日志): $LOG_DIR/"
            echo "       Output (输出): $OUTPUT_DIR/"
            ;;
        *)
            echo "[ERROR] Unknown argument (未知参数): $1"
            print_help
            exit 1
            ;;
    esac
}

main "$@"
