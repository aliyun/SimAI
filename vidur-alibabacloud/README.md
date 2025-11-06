# README


Vidur ([original](https://github.com/microsoft/vidur))  is a simulation framework for large language model (LLM) inference systems.  
**Vidur-AlibabaCloud** (this repository) is a customized version optimized for Alibaba Cloud **SimAI** scenarios.<font style="color:rgb(13, 18, 57);"> It supports advanced features such as </font>**<font style="color:rgb(13, 18, 57);">Prefill–Decode (PD) disaggregation</font>**<font style="color:rgb(13, 18, 57);"> and includes dedicated adaptations for state-of-the-art (SOTA) LLM models including </font>**<font style="color:rgb(13, 18, 57);">DeepSeek-V3-671B</font>**<font style="color:rgb(13, 18, 57);">, </font>**<font style="color:rgb(13, 18, 57);">Qwen3-MoE-235B</font>**<font style="color:rgb(13, 18, 57);">, </font>**<font style="color:rgb(13, 18, 57);">Qwen3-Next-80B</font>**<font style="color:rgb(13, 18, 57);">, and other models.</font>

---

## Key Features
+ **Prefill–Decode (PD) Separation** – Enables running the prefill and decode stages on different nodes, allowing elastic resource allocation and performance isolation.  
(Inspired by [splitwise-sim](https://github.com/Mutinifni/splitwise-sim)).
+ **Flexible Parallelism** – Supports:
    - **Data Parallel (DP)**
    - **Tensor Parallel (TP)**
    - **Pipeline Parallel (PP)**
    - **Expert Parallel (EP)** (support in progress)  
Works for both **dense** and **Mixture-of-Experts (MoE)** models (MoE support in progress).
+ **Multiple Execution-Time Prediction Backends** – Choose from:
    - **AICB/AIOB** - Partially supports computation kernels and TP, DP, PP, EP communication size for <font style="color:rgb(13, 18, 57);"> DeepSeek-V3-671B, Qwen3-Moe-235B, Qwen3-Next-80B</font>
    - **SimAi_simulation** – SimAI <font style="color:rgb(13, 18, 57);">NS-3-based</font> network simulation (supports TP)
    - **SimAi_analytical** – SimAI analytical <font style="color:rgb(13, 18, 57);">performance </font>model (supports TP)
    - **Native Vidur [original]**<font style="color:rgb(13, 18, 57);"> – Supports TP, DP, PP</font>
+ **Workload Generation & Replay** – Replay real-world traces or generate synthetic requests using fixed or Poisson distributions.
+ **Fine-Grained Metrics** – Records:
    - TTFT – Time to First Token
    - TBT / TPOT – Time Between Tokens / Time Per Output Token
    - End-to-end latency  
    - Communication cost  
    - Computation cost  
    - Scheduling delay

---

## Supported Models
+ **DeepSeek-V3-671B** (SimAI PP/EP communication、GPU memory allocation module adaptations in progress)  
+ **Qwen3-Moe-235B**, **Qwen3-Next-80B** (SimAI PP/EP communication、GPU memory allocation module <font style="color:rgb(13, 18, 57);">adaptations</font> in progress)  
+ **meta-llama/Meta-Llama-3-8B** / **Meta-Llama-3-70B**  
+ **meta-llama/Llama-2-7b-hf** / **Llama-2-70b-hf**  
+ **codellama/CodeLlama-34b-Instruct-hf**  
+ **internlm/internlm-20b**  
+ **Qwen/Qwen-72B**

---

## 📦 Environment Setup
### 1. Create Conda Environment
```bash
conda env create -p ./env -f ./environment.yml
```

### 2. (Optional) Update Dev Dependencies
```bash
conda env update -f environment-dev.yml
```

### 3. Activate Environment
```bash
conda activate vidur
```

### 4. Install Python Dependencies (Using Alibaba Cloud PyPI Mirror)
```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
pip install -r requirements-dev.txt -i https://mirrors.aliyun.com/pypi/simple/
```

---

## ▶️ Running Example
### Run <font style="color:rgb(13, 18, 57);">DeepSeek</font>-671B **<font style="color:rgb(13, 18, 57);">with</font>** AICB
**<font style="color:rgb(238, 153, 0);">Requirements: </font>**<font style="color:rgb(0, 0, 0);">SimAI and AICB Docker environment (see </font><font style="color:rgb(154, 110, 58);background-color:rgba(255, 255, 255, 0.5);">[README](../README.md)</font><font style="color:rgb(0, 0, 0);"> for setup instructions). </font>

<font style="color:rgb(0, 0, 0);">After setting up the environment, run the following commands:</font>

#### Run <font style="color:rgb(13, 18, 57);">DeepSeek</font>-671B **<font style="color:rgb(13, 18, 57);">with</font>** AICB <font style="color:rgb(13, 18, 57);"> (Fixed Length Generator)</font>
```bash
cd SimAI/vidur-alibabacloud

python -m vidur.main --replica_config_pd_p2p_comm_bandwidth 800 \
  --replica_config_nvlink_bandwidth 1600 \
  --replica_config_rdma_bandwidth 800 \
  --replica_config_pd_p2p_comm_dtype float32 \
  --poisson_request_interval_generator_config_qps 100 \
  --synthetic_request_generator_config_num_requests 5 \
  --length_generator_config_type fixed \
  --fixed_request_length_generator_config_prefill_tokens 1024 \
  --fixed_request_length_generator_config_decode_tokens 10 \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --replica_config_model_name deepseek-671B \
  --replica_config_tensor_parallel_size 2 \
  --replica_config_num_pipeline_stages 1 \
  --replica_config_expert_model_parallel_size 8 \
  --random_forrest_execution_time_predictor_config_backend aicb 
```

#### <font style="color:rgb(0, 0, 0);">Run </font><font style="color:rgb(13, 18, 57);">DeepSeek</font><font style="color:rgb(0, 0, 0);">-671B </font>**<font style="color:rgb(13, 18, 57);">with</font>**<font style="color:rgb(0, 0, 0);"> AICB </font><font style="color:rgb(13, 18, 57);"> (Trace Length Generator)</font>
```bash
cd SimAI/vidur-alibabacloud

python -m vidur.main \
  --replica_config_pd_p2p_comm_bandwidth 800 \
  --replica_config_nvlink_bandwidth 1600 \
  --replica_config_rdma_bandwidth 800 \
  --replica_config_pd_p2p_comm_dtype float32 \
  --poisson_request_interval_generator_config_qps 100 \
  --synthetic_request_generator_config_num_requests 10 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_max_tokens 1024 \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
  --interval_generator_config_type poisson \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --replica_config_model_name deepseek-671B \
  --replica_config_tensor_parallel_size 2 \
  --replica_config_num_pipeline_stages 1 \
  --replica_config_expert_model_parallel_size 8 \
  --random_forrest_execution_time_predictor_config_backend aicb
```

> ✅ Full parameter descriptions are available via `python -m vidur.main -h`.
>



### Run Llama-3-8B **<font style="color:rgb(13, 18, 57);">with</font>** simai_simulation
```bash
cd SimAI

# Compile SimAI-Simulation (ns3)
./scripts/build.sh -c ns3

# Create network topo (Spectrum-X_128g_8gps_100Gbps_A100)
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py -topo Spectrum-X -g 128 -gt A100 -bw 100Gbps -nvbw 2400Gbps


cd SimAI/vidur-alibabacloud

python -m vidur.main \
  --replica_config_pd_p2p_comm_bandwidth 800 \
  --replica_config_nvlink_bandwidth 1600 \
  --replica_config_rdma_bandwidth 800 \
  --replica_config_pd_p2p_comm_dtype float32 \
  --poisson_request_interval_generator_config_qps 100 \
  --synthetic_request_generator_config_num_requests 10 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_max_tokens 2048 \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
  --interval_generator_config_type poisson \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --replica_config_tensor_parallel_size 4 \
  --replica_config_num_pipeline_stages 1 \
  --replica_config_expert_model_parallel_size 1 \
  --random_forrest_execution_time_predictor_config_backend simai_simulation \
  --random_forrest_execution_time_predictor_config_simai_dir ../ \
  --random_forrest_execution_time_predictor_config_simai_simulation_topo ../Spectrum-X_128g_8gps_100Gbps_A100 \
  --random_forrest_execution_time_predictor_config_simai_simulation_config ../astra-sim-alibabacloud/inputs/config/SimAI.conf 
```

> 
>

### Run Llama-3-8B **<font style="color:rgb(13, 18, 57);">with</font>** simai_analytical
```bash
cd SimAI

# Compile SimAI-Analytical
$ ./scripts/build.sh -c analytical

cd SimAI/vidur-alibabacloud

python -m vidur.main \
  --replica_config_pd_p2p_comm_bandwidth 800 \
  --replica_config_nvlink_bandwidth 1600 \
  --replica_config_rdma_bandwidth 800 \
  --replica_config_pd_p2p_comm_dtype float32 \
  --poisson_request_interval_generator_config_qps 100 \
  --synthetic_request_generator_config_num_requests 10 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_max_tokens 2048 \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
  --interval_generator_config_type poisson \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --replica_config_tensor_parallel_size 4 \
  --replica_config_num_pipeline_stages 1 \
  --replica_config_expert_model_parallel_size 1 \
  --random_forrest_execution_time_predictor_config_backend simai_analytical
```

> 
>

### Run Llama-3-8B **<font style="color:rgb(13, 18, 57);">with</font>** native Vidur [original]
```bash
cd SimAI/vidur-alibabacloud

python -m vidur.main \
  --replica_config_pd_p2p_comm_bandwidth 800 \
  --replica_config_nvlink_bandwidth 1600 \
  --replica_config_rdma_bandwidth 800 \
  --replica_config_pd_p2p_comm_dtype float32 \
  --poisson_request_interval_generator_config_qps 100 \
  --synthetic_request_generator_config_num_requests 10 \
  --length_generator_config_type trace \
  --trace_request_length_generator_config_max_tokens 2048 \
  --trace_request_length_generator_config_trace_file ./data/processed_traces/splitwise_conv.csv \
  --interval_generator_config_type poisson \
  --cluster_config_num_replicas 4 \
  --replica_config_pd_node_ratio 0.5 \
  --global_scheduler_config_type split_wise \
  --replica_scheduler_config_type split_wise \
  --replica_config_model_name meta-llama/Meta-Llama-3-8B \
  --replica_config_tensor_parallel_size 4 \
  --replica_config_num_pipeline_stages 1 \
  --replica_config_expert_model_parallel_size 1 \
  --random_forrest_execution_time_predictor_config_backend vidur
```

> 
>



---

## 🔧 Key Input Parameter Reference
| Parameter | Default | Description |
| --- | --- | --- |
| `--replica_config_pd_p2p_comm_bandwidth` | 800 | Bandwidth (Gbps) for point-to-point communication between Prefill and Decode nodes in PD disaggregation |
| `--replica_config_nvlink_bandwidth` | 1600 | NVLink bandwidth (Gbps) for TP/EP communications |
| `--replica_config_rdma_bandwidth` | 800 | RDMA bandwidth (Gbps) for inter-node communication |
| `--replica_config_pd_p2p_comm_dtype` | float16 | Data type for PD communication (`float16`, `float32`, etc.) |
| `--poisson_request_interval_generator_config_qps` | 0.5 | Queries per second (QPS) for Poisson request generator |
| `--synthetic_request_generator_config_num_requests` | 128 | Number of synthetic requests to generate |
| `--length_generator_config_type` | fixed | Request length generator type (`fixed`, `trace`, etc.) |
| `--fixed_request_length_generator_config_prefill_tokens` | `2048` | Number of prefill tokens per request (only effective when `--length_generator_config_type=fixed`) |
| `--fixed_request_length_generator_config_decode_tokens` | `512` | Number of decode tokens per request (only effective when `--length_generator_config_type=fixed`) |
| `--trace_request_length_generator_config_max_tokens` | 4096 | Max tokens when using trace-based length generator |
| `--trace_request_length_generator_config_trace_file` | data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv | Path to trace file for request lengths |
| `--interval_generator_config_type` | poisson | Inter-arrival time generator type |
| `--cluster_config_num_replicas` | 1 | Total number of replicas (i.e., data parallelism degree) |
| `--replica_config_pd_node_ratio` | 0.5 | Ratio of P-nodes to  （P-nodes + D-nodes）    <font style="color:rgb(0, 0, 0);background-color:rgb(245, 242, 240);">Fraction of replicas allocated as prefill (P) nodes. The remaining replicas are used as decode (D) nodes. </font>   <font style="color:rgb(0, 0, 0);background-color:rgb(245, 242, 240);">For example, 0.5 means half of the replicas are prefill nodes and half are decode nodes (P:D = 1:1).</font> |
| `--global_scheduler_config_type` | round_robin | Global scheduler type (`split_wise`, `round_robin`, etc.) |
| `--replica_scheduler_config_type` | sarathi | Per-replica scheduler type |
| `--replica_config_model_name` | meta-llama/Llama-2-7b-hf | Model name (DeepSeek-671B,  Qwen3-Moe-235B, Qwen3-Next-80B , etc.)<br/>⚠️ **Note**:  Vidur GPU Memory management module is still under adaptation for DeepSeek-671B,  Qwen3-Moe-235B, Qwen3-Next-80B |
| `--replica_config_tensor_parallel_size` | 1 | Tensor parallelism size (TP) |
| `--replica_config_num_pipeline_stages` | 1 | Number of pipeline stages (PP) |
| `--replica_config_expert_model_parallel_size` | 1 | Expert model parallelism size (EP) |
| `--random_forrest_execution_time_predictor_config_backend` | vidur | Backend for execution time prediction <br/>('vidur', 'simai_simulation', 'simai_analytical','aicb', etc.)<br/>⚠️ **Note**: `simai_simulation` and `simai_analytical` currently only model TP communication and do not support pipeline or expert parallelism |
| `--random_forrest_execution_time_predictor_config_simai_dir` | `'../'` | Root directory of the SimAI simulator（default: `../`) <br/>（only effective when `--random_forrest_execution_time_predictor_config_backend simai_simulation`） |
| `--random_forrest_execution_time_predictor_config_simai_simulation_topo` | `'../example/topo'` | Path to SimAI topology file (e.g., `'../Spectrum-X_128g_8gps_100Gbps_A100'`)（only effective when `--random_forrest_execution_time_predictor_config_backend simai_simulation`） |
| `--random_forrest_execution_time_predictor_config_simai_simulation_config` | `'../astra-sim-alibabacloud/inputs/config/SimAI.conf'` | Path to SimAI configuration file (e.g., `'../astra-sim-alibabacloud/inputs/config/SimAI.conf'`)<br/>（only effective when `--random_forrest_execution_time_predictor_config_backend simai_simulation`） |


---

## 📊 Key Output Interpretation
Simulation results are saved to:

```plain
./simulator_output/YYYY-MM-DD_HH-MM-SS-XXXXXX/request_metrics.csv
```

### Key Columns in `request_metrics.csv`
| Column | Meaning |
| --- | --- |
| `arrived_at` / `prefill_arrived_at` | Timestamp when the request entered the system (in seconds). |
| `scheduled_at` | Timestamp when the request was first scheduled by the scheduler and began execution (in seconds). |
| `prefill_completed_at` | Timestamp when the Prefill phase completed and the first output token was generated. |
| `decode_arrived_at` | Timestamp when the Decode phase started.    In non-PD-<font style="color:rgb(0, 0, 0);">Disaggregated </font> <font style="color:rgb(0, 0, 0);">setup</font>, this typically equals `prefill_completed_at`.    In PD-<font style="color:rgb(0, 0, 0);">Disaggregated </font> <font style="color:rgb(0, 0, 0);">setup</font>, it is `prefill_completed_at + pd_p2p_comm_time`. |
| `decode_time` | Duration of the Decode phase (in seconds),    computed as `completed_at - decode_arrived_at`   (equivalently: `request_e2e_time - prefill_e2e_time`). |
| `prefill_replica_id` | Replica ID that executed the Prefill phase    (in PD-<font style="color:rgb(0, 0, 0);">Disaggregated </font> <font style="color:rgb(0, 0, 0);">setup</font>). |
| `decode_replica_id` | Replica ID that executed the Decode phase (in PD-<font style="color:rgb(0, 0, 0);">Disaggregated </font> <font style="color:rgb(0, 0, 0);">setup</font>). |
| `request_num_prefill_tokens` | Number of input tokens (i.e., prompt length). |
| `request_num_decode_tokens` | Number of output tokens (i.e., generation length). |
| `pd_p2p_comm_size` | Point-to-point communication size (in bytes) of data transferred from the Prefill node to the Decode node (KV cache, etc.) in PD-<font style="color:rgb(0, 0, 0);">Disaggregated </font> <font style="color:rgb(0, 0, 0);">setup</font>. |
| `pd_p2p_comm_time` | Point-to-point communication time (in seconds) between Prefill and Decode nodes in PD-<font style="color:rgb(0, 0, 0);">Disaggregated </font> <font style="color:rgb(0, 0, 0);">setup</font>. |
| `completed_at` | Timestamp when the request finished processing. |
| `request_execution_time` | Total actual execution time (in seconds), excluding delays due to preemption or pipeline bubbles. |
| `request_preemption_time` | Time (in seconds) spent waiting due to scheduler preemption, pipeline bubbles, or other non-execution gaps. |
| `request_scheduling_delay` | Scheduling delay before execution: `scheduled_at - arrived_at` (in seconds). |
| `request_e2e_time` | End-to-end latency: `completed_at - arrived_at` (in seconds). |
| `prefill_e2e_time` | Time To First Token (TTFT): `prefill_completed_at - arrived_at` (in seconds). |
| `tbt` | Time Between Tokens (TBT), also known as Time Per Output Token (TPOT).    Computed as:   `decode_time / request_num_decode_tokens`   or equivalently:   `(request_e2e_time - prefill_e2e_time) / request_num_decode_tokens`   (in seconds/token). |


 **Notes**:

+ All time-related fields are in **seconds (s)**, based on monotonic clock or Unix timestamps.
+ In non-PD-separated deployments, `prefill_replica_id` and `decode_replica_id` are typically identical.
+ If `request_num_decode_tokens = 0`, `tbt` is undefined (may be recorded as `NaN` or `0`).
+ **<font style="color:rgb(139, 139, 139);">TBT is not yet logged in request_metrics.csv; it can be computed manually for now.</font>**

### Sample Row (request_metrics.csv)
```plain
Request Id,request_e2e_time,...,arrived_at,prefill_arrived_at,scheduled_at,prefill_completed_at,decode_arrived_at,completed_at,...,prefill_replica_id,decode_replica_id,pd_p2p_comm_size,pd_p2p_comm_time,...
0,0.03607,...,0.0102006,0.0102006,0.0102006,0.0102265,0.0433997,0.0462744,...,0,2,3561947136,0.0331732,...
```

---

## ⚠️ Known Issue: Plotting Warning
You may see this error at exit:

```plain
RuntimeError: Kaleido requires Google Chrome to be installed.
```

This occurs because the simulator tries to generate PNG plots but lacks Chrome.  
✅ **Important**: This **does NOT affect** the generation of `request_metrics.csv`.

### Solutions:
1. **Ignore it** – CSV output is unaffected.
2. **Install Chrome**:

```bash
plotly_get_chrome
```

3. **Disable plotting** (not recommended): Comment out these lines in `vidur/simulator.py`:

```python
# self._metric_store.plot()
# logger.info("Metrics written")
```

> ⚠️ Disabling plotting will skip all visual outputs and request_metrics.csv.
>

---

## 📚 Help
View all CLI options:

```bash
python -m vidur.main -h
```

---

