# Result Analysis & Visualization

This guide covers how to interpret and analyze simulation outputs from all SimAI modes.

---

## SimAI-Analytical Output

### CSV Output

Running SimAI-Analytical generates a CSV file in the `results/` directory. The output contains:

- **Summary row**: Exposure time, computation time (absolute and percentage) for each communication group, and end-to-end iteration time
- **Per-layer rows**: Detailed operation timing for each layer

Key columns include per-communication-group breakdown (TP, DP, EP, PP) showing time allocation and overlap effects.

### Visualization

When running with the `-v` flag, SimAI-Analytical generates additional visualization files showing the timing breakdown across communication groups.

```bash
# Run with visualization enabled
./bin/SimAI_analytical -w example/workload_analytical.txt -g 9216 -g_p_s 8 -r test- -busbw example/busbw.yaml -v
```

---

## SimAI-Simulation Output

SimAI-Simulation (NS-3 mode) generates detailed trace data capturing fine-grained network behavior. The NS-3 backend outputs `.tr` trace files that can be analyzed using the provided analysis tools.

### Analysis Tools

Located in `ns-3-alibabacloud/analysis/`:

| Tool | Description |
|------|-------------|
| `fct_analysis.py` | Flow Completion Time analysis — reads FCT output files and produces statistics |
| `trace_reader` | Parses `.tr` trace files with filtering support |

### Using trace_reader

```bash
# Build
cd ns-3-alibabacloud/analysis
make trace_reader

# Parse trace file
./trace_reader <.tr file> [filter_expr]

# Examples:
./trace_reader output.tr "time > 2000010000"
./trace_reader output.tr "sip=0x0b000101&dip=0x0b000201"
```

### Trace Output Format

Each line in the trace output follows this format:

```
2000055540 n:338 4:3 100608 Enqu ecn:0 0b00d101 0b012301 10000 100 U 161000 0 3 1048(1000)
```

Fields: timestamp (ns), node ID, port:queue, queue length (bytes), event type, ECN flag, source IP, destination IP, source port, destination port, packet type, sequence number, TX timestamp, priority group, packet size (payload).

---

## Inference Simulation Output

### Output Directory Structure

Each inference simulation run produces:

```
<output_dir>/<YYYY-MM-DD_HH-MM-SS>/
├── request_metrics.csv     # Per-request metrics
├── chrome_trace.json       # Chrome DevTools timeline trace
├── config.json             # Configuration snapshot
└── plots/                  # Per-metric CSV/JSON files
    ├── request_e2e_time.csv
    ├── prefill_e2e_time.csv
    ├── pd_p2p_comm_time.csv
    ├── replica_N_memory_usage.json
    └── ...
```

### request_metrics.csv Columns

| Column | Meaning |
|--------|---------|
| `arrived_at` | Timestamp when the request entered the system (seconds) |
| `scheduled_at` | Timestamp when the request was first scheduled (seconds) |
| `prefill_completed_at` | Timestamp when Prefill completed and first token generated |
| `decode_arrived_at` | Timestamp when Decode phase started |
| `decode_time` | Duration of Decode phase (seconds) |
| `prefill_replica_id` | Replica ID that executed Prefill (PD mode) |
| `decode_replica_id` | Replica ID that executed Decode (PD mode) |
| `request_num_prefill_tokens` | Number of input tokens (prompt length) |
| `request_num_decode_tokens` | Number of output tokens (generation length) |
| `pd_p2p_comm_size` | P2P communication size from Prefill to Decode node (bytes) |
| `pd_p2p_comm_time` | P2P communication time (seconds) |
| `completed_at` | Request completion timestamp |
| `request_execution_time` | Total execution time excluding delays (seconds) |
| `request_preemption_time` | Wait time due to preemption/bubbles (seconds) |
| `request_scheduling_delay` | Scheduling delay: `scheduled_at - arrived_at` (seconds) |
| `request_e2e_time` | End-to-end latency: `completed_at - arrived_at` (seconds) |
| `prefill_e2e_time` | Time To First Token (TTFT): `prefill_completed_at - arrived_at` (seconds) |
| `tbt` | Time Between Tokens: `decode_time / request_num_decode_tokens` (seconds/token) |

### Chrome Trace Visualization

Open `chrome_trace.json` in Chrome DevTools for visual timeline analysis:

1. Open Chrome browser
2. Navigate to `chrome://tracing`
3. Load the `chrome_trace.json` file

### Simulation Metrics (23 metrics)

The simulator records 23 fine-grained metrics:

| Category | Metrics |
|----------|---------|
| **Request Latency** | E2E time CDF, normalized E2E CDF, execution+preemption CDF |
| **Scheduling** | Scheduling delay CDF |
| **Execution** | Execution time CDF, preemption time CDF |
| **Token-level** | Decode token execution+preemption times, inter-token delay |
| **Batch** | Batch num tokens CDF, batch sizes CDF |
| **Prefill** | Prefill E2E CDF, prefill execution+preemption CDF (normalized) |
| **Decode** | Decode execution+preemption normalized CDF |
| **Time Series** | Request/prefill/decode completions, request arrivals |
| **Per-replica** | Memory usage (weighted mean), busy time %, MFU |

For detailed metric definitions, see the [vidur metrics documentation](../components/vidur.md).

---

## AICB Physical Execution Output

### Log Output

After each communication, AICB prints:
- Communication type and group
- Message size
- Execution time
- Throughput (algbw and busbw)

### Iteration Summary

After all communications complete, a summary shows:
- Overall runtime and per-iteration timing
- Per-communication-type statistics (message sizes, frequencies, latency min/max/avg)

### CSV Output

Results are saved in `results/comm_logs/`:
- `<model>_<config>_log.csv` — Execution log with timing, phase, algbw, busbw per comm_group and comm_type
- `<model>_<config>_workload.csv` — Generated workload description

### Programmatic Analysis

```python
# Read workload log
from log_analyzer.log import Workload
workload, args = Workload.load("results/comm_logs/megatron_gpt_13B_8n_workload.csv")

# Read execution log
from log_analyzer.log import Log
log = Log.load("results/comm_logs/megatron_gpt_13B_8n_log.csv")
# log.comm_logs: List[LogItem]
# log.epoch_times: List[int]
# log.comm_log_each_epoch: List[List[LogItem]]
```

---

## See Also

- [SimAI-Analytical](simai_analytical.md) — Analytical mode usage
- [SimAI-Simulation](simai_simulation.md) — NS-3 simulation mode usage
- [Inference Simulation](inference_simulation.md) — Inference simulation guide
- [NS-3 Component](../components/ns3.md) — NS-3 analysis tools
