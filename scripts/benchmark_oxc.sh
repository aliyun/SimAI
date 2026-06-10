#!/bin/bash
#
# OXC Performance Benchmark Script
# 对比 SimAI 原版与 OXC 集成版本的性能
#

set -e

SCRIPT_DIR=$(dirname "$(realpath $0)")
ROOT_DIR=$(realpath "${SCRIPT_DIR:?}"/..)
BIN_DIR="${ROOT_DIR}/bin"
RESULTS_DIR="${ROOT_DIR}/results/benchmark_oxc_$(date +%Y%m%d_%H%M%S)"

# 默认参数
WORKLOAD="${ROOT_DIR}/example/workload_analytical.txt"
NUM_GPUS=16
GPUS_PER_SERVER=8
NUM_RUNS=3
MODE="analytical"  # analytical 或 ns3

# OXC 服务器配置
OXC_URL="http://localhost:8080"
OXC_ALGO="ALGO_OXC_RING"
RANKTABLE_FILE="${ROOT_DIR}/astra-sim-alibabacloud/inputs/oxc/ranktable_example.json"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function print_usage {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --mode MODE          Benchmark mode: analytical or ns3 (default: analytical)"
    echo "  -w, --workload FILE      Workload file path"
    echo "  -g, --gpus NUM           Number of GPUs (default: 16)"
    echo "  -s, --gpus-per-server N  GPUs per server (default: 8)"
    echo "  -r, --runs NUM           Number of runs for averaging (default: 3)"
    echo "  -u, --oxc-url URL        OXC server URL (default: http://localhost:8080)"
    echo "  -a, --oxc-algo ALGO      OXC algorithm (default: ALGO_OXC_RING)"
    echo "  -t, --ranktable FILE     RankTable JSON file path"
    echo "  -h, --help               Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -m analytical -g 16 -r 5"
    echo "  $0 -m ns3 -w ./example/microAllReduce.txt -g 128"
}

function log_info {
    echo -e "${GREEN}[INFO]${NC} $1"
}

function log_warn {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

function log_error {
    echo -e "${RED}[ERROR]${NC} $1"
}

function check_binary {
    local binary=$1
    if [ ! -f "$binary" ] && [ ! -L "$binary" ]; then
        log_error "Binary not found: $binary"
        log_error "Please compile first: ./scripts/build.sh -c $2"
        exit 1
    fi
}

function run_analytical_benchmark {
    local binary=$1
    local output_prefix=$2
    local use_oxc=$3

    local cmd="$binary -w $WORKLOAD -g $NUM_GPUS -g_p_s $GPUS_PER_SERVER -r ${output_prefix}"

    if [ "$use_oxc" == "true" ]; then
        # 设置 OXC 环境变量
        export AS_OXC_ENABLE=1
        export AS_OXC_URL="$OXC_URL"
        export AS_OXC_ALGO="$OXC_ALGO"
        export AS_OXC_RANKTABLE="$RANKTABLE_FILE"
        export AS_OXC_GPUS_PER_SERVER="$GPUS_PER_SERVER"
    else
        unset AS_OXC_ENABLE
        unset AS_OXC_URL
        unset AS_OXC_ALGO
        unset AS_OXC_RANKTABLE
        unset AS_OXC_GPUS_PER_SERVER
    fi

    # 运行并计时
    local start_time=$(date +%s.%N)
    $cmd > "${output_prefix}stdout.log" 2>&1
    local end_time=$(date +%s.%N)

    # 计算运行时间
    local elapsed=$(echo "$end_time - $start_time" | bc)
    echo "$elapsed"
}

function run_ns3_benchmark {
    local binary=$1
    local output_prefix=$2
    local use_oxc=$3
    local topo_dir=$4

    local cmd="$binary -t 8 -w $WORKLOAD -n $topo_dir -c ${ROOT_DIR}/astra-sim-alibabacloud/inputs/config/SimAI.conf"

    if [ "$use_oxc" == "true" ]; then
        export AS_OXC_ENABLE=1
        export AS_OXC_URL="$OXC_URL"
        export AS_OXC_ALGO="$OXC_ALGO"
        export AS_OXC_RANKTABLE="$RANKTABLE_FILE"
        export AS_OXC_GPUS_PER_SERVER="$GPUS_PER_SERVER"
    else
        unset AS_OXC_ENABLE
        unset AS_OXC_URL
        unset AS_OXC_ALGO
        unset AS_OXC_RANKTABLE
        unset AS_OXC_GPUS_PER_SERVER
    fi

    export AS_SEND_LAT=3
    export AS_NVLS_ENABLE=1

    local start_time=$(date +%s.%N)
    $cmd > "${output_prefix}stdout.log" 2>&1
    local end_time=$(date +%s.%N)

    local elapsed=$(echo "$end_time - $start_time" | bc)
    echo "$elapsed"
}

function extract_simulation_time {
    local log_file=$1
    # 从日志中提取模拟时间（根据实际输出格式调整）
    local sim_time=$(grep -oP 'total.*?time.*?(\d+\.?\d*)' "$log_file" | grep -oP '\d+\.?\d*' | tail -1)
    if [ -z "$sim_time" ]; then
        sim_time="N/A"
    fi
    echo "$sim_time"
}

function run_benchmark {
    log_info "Starting OXC Performance Benchmark"
    log_info "Mode: $MODE"
    log_info "Workload: $WORKLOAD"
    log_info "GPUs: $NUM_GPUS"
    log_info "GPUs per server: $GPUS_PER_SERVER"
    log_info "Number of runs: $NUM_RUNS"
    log_info "Results directory: $RESULTS_DIR"
    echo ""

    mkdir -p "$RESULTS_DIR"

    local baseline_binary
    local oxc_binary

    if [ "$MODE" == "analytical" ]; then
        baseline_binary="${BIN_DIR}/SimAI_analytical"
        oxc_binary="${BIN_DIR}/SimAI_analytical_oxc"
        check_binary "$baseline_binary" "analytical"
        check_binary "$oxc_binary" "analytical_oxc"
    else
        baseline_binary="${BIN_DIR}/SimAI_simulator"
        oxc_binary="${BIN_DIR}/SimAI_simulator_oxc"
        check_binary "$baseline_binary" "ns3"
        check_binary "$oxc_binary" "ns3_oxc"
    fi

    # 存储结果
    local baseline_times=()
    local oxc_times=()

    # 运行基准测试
    log_info "Running baseline (without OXC)..."
    for ((i=1; i<=NUM_RUNS; i++)); do
        log_info "  Run $i/$NUM_RUNS"
        local output_prefix="${RESULTS_DIR}/baseline_run${i}_"

        if [ "$MODE" == "analytical" ]; then
            elapsed=$(run_analytical_benchmark "$baseline_binary" "$output_prefix" "false")
        else
            elapsed=$(run_ns3_benchmark "$baseline_binary" "$output_prefix" "false" "$TOPO_DIR")
        fi

        baseline_times+=("$elapsed")
        log_info "    Elapsed: ${elapsed}s"
    done

    echo ""
    log_info "Running OXC version..."
    for ((i=1; i<=NUM_RUNS; i++)); do
        log_info "  Run $i/$NUM_RUNS"
        local output_prefix="${RESULTS_DIR}/oxc_run${i}_"

        if [ "$MODE" == "analytical" ]; then
            elapsed=$(run_analytical_benchmark "$oxc_binary" "$output_prefix" "true")
        else
            elapsed=$(run_ns3_benchmark "$oxc_binary" "$output_prefix" "true" "$TOPO_DIR")
        fi

        oxc_times+=("$elapsed")
        log_info "    Elapsed: ${elapsed}s"
    done

    # 计算统计数据
    echo ""
    log_info "=========================================="
    log_info "           BENCHMARK RESULTS"
    log_info "=========================================="

    # 计算平均值
    local baseline_sum=0
    local oxc_sum=0

    for t in "${baseline_times[@]}"; do
        baseline_sum=$(echo "$baseline_sum + $t" | bc)
    done

    for t in "${oxc_times[@]}"; do
        oxc_sum=$(echo "$oxc_sum + $t" | bc)
    done

    local baseline_avg=$(echo "scale=4; $baseline_sum / $NUM_RUNS" | bc)
    local oxc_avg=$(echo "scale=4; $oxc_sum / $NUM_RUNS" | bc)

    # 计算差异
    local diff=$(echo "scale=4; $oxc_avg - $baseline_avg" | bc)
    local diff_percent=$(echo "scale=2; ($diff / $baseline_avg) * 100" | bc)

    echo ""
    echo "Configuration:"
    echo "  Mode:            $MODE"
    echo "  Workload:        $WORKLOAD"
    echo "  GPUs:            $NUM_GPUS"
    echo "  GPUs per server: $GPUS_PER_SERVER"
    echo "  Number of runs:  $NUM_RUNS"
    echo ""
    echo "Results (wall-clock time in seconds):"
    echo "  Baseline (no OXC):"
    echo "    Runs:    ${baseline_times[*]}"
    echo "    Average: ${baseline_avg}s"
    echo ""
    echo "  OXC Version:"
    echo "    Runs:    ${oxc_times[*]}"
    echo "    Average: ${oxc_avg}s"
    echo ""
    echo "Comparison:"
    echo "  Difference: ${diff}s (${diff_percent}%)"

    if (( $(echo "$diff > 0" | bc -l) )); then
        echo "  OXC version is SLOWER by ${diff_percent}%"
    else
        local speedup=$(echo "scale=2; -1 * $diff_percent" | bc)
        echo "  OXC version is FASTER by ${speedup}%"
    fi

    # 保存结果到文件
    local report_file="${RESULTS_DIR}/benchmark_report.txt"
    {
        echo "OXC Performance Benchmark Report"
        echo "================================"
        echo "Date: $(date)"
        echo ""
        echo "Configuration:"
        echo "  Mode:            $MODE"
        echo "  Workload:        $WORKLOAD"
        echo "  GPUs:            $NUM_GPUS"
        echo "  GPUs per server: $GPUS_PER_SERVER"
        echo "  Number of runs:  $NUM_RUNS"
        echo "  OXC URL:         $OXC_URL"
        echo "  OXC Algorithm:   $OXC_ALGO"
        echo "  RankTable:       $RANKTABLE_FILE"
        echo ""
        echo "Results (wall-clock time in seconds):"
        echo "  Baseline: ${baseline_times[*]} (avg: ${baseline_avg}s)"
        echo "  OXC:      ${oxc_times[*]} (avg: ${oxc_avg}s)"
        echo ""
        echo "Difference: ${diff}s (${diff_percent}%)"
    } > "$report_file"

    # 生成 CSV 结果
    local csv_file="${RESULTS_DIR}/benchmark_results.csv"
    {
        echo "run,baseline_time,oxc_time"
        for ((i=0; i<NUM_RUNS; i++)); do
            echo "$((i+1)),${baseline_times[$i]},${oxc_times[$i]}"
        done
        echo "average,$baseline_avg,$oxc_avg"
    } > "$csv_file"

    echo ""
    log_info "Results saved to: $RESULTS_DIR"
    log_info "  Report: $report_file"
    log_info "  CSV:    $csv_file"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -w|--workload)
            WORKLOAD="$2"
            shift 2
            ;;
        -g|--gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -s|--gpus-per-server)
            GPUS_PER_SERVER="$2"
            shift 2
            ;;
        -r|--runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        -u|--oxc-url)
            OXC_URL="$2"
            shift 2
            ;;
        -a|--oxc-algo)
            OXC_ALGO="$2"
            shift 2
            ;;
        -t|--ranktable)
            RANKTABLE_FILE="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# 验证模式
if [ "$MODE" != "analytical" ] && [ "$MODE" != "ns3" ]; then
    log_error "Invalid mode: $MODE. Must be 'analytical' or 'ns3'"
    exit 1
fi

# 验证工作负载文件
if [ ! -f "$WORKLOAD" ]; then
    log_error "Workload file not found: $WORKLOAD"
    exit 1
fi

# 运行基准测试
run_benchmark
