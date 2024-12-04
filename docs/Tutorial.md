# 🌟 Introduction

SimAI is a comprehensive large-scale AI training simulation toolkit that provides three major simulation scenarios:

1. **SimAI-Analytical** - An analytical simulation tool that abstracts underlying network communication details. It adopts a simplified approach using busbw (bus bandwidth) to estimate communication time for collective/point-to-point communications, enabling rapid scenario validation. Key application scenarios include (but are not limited to):

    * *Performance Analysis*: Compare completion times across different models (e.g., studying the impact of Expert numbers on MoE model training performance)

    * *Framework-level Parallel Parameter Optimization*: Balance and optimize TP/EP/PP parameters to analyze end-to-end timing effects

    * *Scale-up Exploration*: Investigate parallel parameter performance across different scale-up domains for specific scenario optimization

    * *Scale-out Bandwidth Selection*: Research cost-effective bandwidth configurations for various GPU performances

> 💡 *Currently supports manual busbw.yaml configuration. Automatic busbw inference based on parallel scenarios will be open-sourced soon. Stay tuned and feel free to contact us for more details. ✨*

2. **SimAI-Simulation(NS-3)** - A high-fidelity, full-stack simulation tool that can theoretically integrate with any pure network simulator. It provides fine-grained reproduction of communication behaviors during LLM training. Currently supports NS-3 as the network backend (we encourage integration of new network simulation tools). Key research areas include:

    * *Collective Communication Algorithm Research*: Design and optimize collective communication traffic patterns for non-switch architectures and other emerging network topologies
    
    * *Network Protocol Research*: Evaluate and optimize network protocols, congestion control algorithms, routing mechanisms and other low-level network technologies across different architectures
    
    * *Novel Network Architecture Design*: Explore innovative network architectures

> 💡 We strongly encourage researchers to build upon SimAI-Simulation for innovative extensions and breakthrough research suitable for top-tier conferences. Join our community or reach out via email - we're committed to providing technical support for promising research directions! ✨

3. **SimAI-Physical(TODO)**

For additional functionalities of each component, please refer to [SimCCL](https://github.com/aliyun/SimCCL) and [ns-3-alibabacloud](https://github.com/aliyun/ns-3-alibabacloud).

# 🛠️ Environment Setup

Under normal circumstances, running SimAI requires generating a Workload file using the [AICB](https://github.com/aliyun/aicb?tab=readme-ov-file#generate-workloads-for-simulation-simai) tool. To create a precise Workload, you may need to utilize the AIOB feature to determine the timing of various computational kernels, which necessitates a GPU environment. Therefore, we recommend executing the SimAI full-stack toolkit directly within the latest **NGC image**.

> 💡 **Important Note**: SimAI-Simulation compilation requires removing ninja (which comes pre-installed in NGC images). You can remove it using:
> ```bash
> apt remove ninja-build && pip uninstall ninja
> ```

Build Instructions:

```bash
# Clone the repository
$ git clone https://github.com/aliyun/SimAI.git
$ cd ./SimAI/

# Clone submodules
$ git submodule update --init --recursive
# Make sure to use the newest commit
$ git submodule update --remote

# Compile SimAI-Analytical
$ ./scripts/build.sh -c analytical

# Compile SimAI-Simulation (ns3)
$ ./scripts/build.sh -c ns3
```

# 🌐 SimAI-Analytical Usage
## 📝 Workload Generate

To generate workloads for simulation, use the [SimAI-WorkloadGenerator](https://github.com/aliyun/aicb?tab=readme-ov-file#generate-workloads-for-simulation-simai) feature in [AICB](https://github.com/aliyun/aicb). This will produce a `.txt` file similar to [workload_analytical.txt](../example/workload_analytical.txt), which includes:

- `model_parallel_NPU_group`: Represents the size of Tensor Parallelism
- `ep`: Represents the size of Expert model parallelism
- `pp`: Represents the size of pipeline model parallelism
- `vpp`: Virtual Pipeline Parallelism (default: `--num-layers-per-virtual-pipeline-stage=1` for minimal PP bubble)

> 💡 *For more details, refer to the [AICB Workload Tutorial](https://github.com/aliyun/aicb/blob/master/training/tutorial.md#workload)*

## 🔧 Busbw Setting

SimAI-Analytical abstracts lower-level network details by directly specifying busbw to estimate collective communication times. To customize communication busbw for various scenarios, you can use a [busbw.yaml](../example/busbw.yaml) file in the following format:

```yaml
test
TP:
  allreduce,: 300      # AllReduce busbw 300GB/s in TP
  allgather,: 280
  reducescatter,: 280
  alltoall,: 230
DP:
  allreduce,: null
  allgather,: 380      # AllGather busbw 380GB/s in DP
  reducescatter,: 380
  alltoall,: null
EP:
  allreduce,: null
  allgather,: 45       # AllGather busbw 45GB/s in DP_EP
  reducescatter,: 45   # ReduceScatter busbw 45GB/s in DP_EP
  alltoall,: 80        # AlltoAll busbw 80GB/s in EP
```
> 🔍 *Interested in automated busbw calculation (considering cluster size, architecture, parallel parameters, small message adjustments, and latency)? Feel free to reach out for a discussion!* ✨

## 🖥️ Analytical Simulation

To run the analytical simulation, use the following command:

```bash
$ ./bin/SimAI_analytical -w example/workload_analytical.txt -g 9216 -g_p_s 8 -r test- -busbw example/busbw.yaml
```

### Required Parameters

| Parameter | Long Form | Description |
|:---------:|:----------|:------------|
| `-w` | `--workload` | Specifies the path to the Workload File |
| `-g` | `--gpus` | Specifies the simulation GPU scale |
| `-g_p_s` | `--gpus-per-server` | Specifies the Scale-up size |
| `-r` | `--result` | Specifies the output file path and prefix (default: `./results/`)<br>Recommended to include simulation parameters, e.g.,<br>`A100-llama405b-tp8-pp16-dp128-ga16-ep1-NVL8` |
| `-busbw` | `--bus-bandwidth` | Specifies the path to the busbw file<br>(recommend modifying `example/busbw.yaml` directly) |

### Optional Parameters

| Parameter | Long Form | Description |
|:---------:|:----------|:------------|
| `-v` | `--visual` | Specifies whether to generate visualization files |

### Communication Group Overlap Ratios

The following parameters specify the overlap ratios for communication groups (default: 0, indicating no overlap):

| Parameter | Long Form | Description | Range |
|:---------:|:----------|:------------|:------|
| `-dp_o` | `--dp-overlap-ratio` | DP overlap ratio | [0.0-1.0] |
| `-ep_o` | `--ep-overlap-ratio` | EP overlap ratio | [0.0-1.0] |
| `-tp_o` | `--tp-overlap-ratio` | TP overlap ratio | [0.0-1.0] |
| `-pp_o` | `--pp-overlap-ratio` | PP overlap ratio | [0.0-1.0] |

> 📝 *Due to the variety of overlap strategies and scenario-dependent overlap ratios, we prioritize simple and efficient methods to directly specify overlap conditions.*


## Result Analyze

### Raw Data

Running SimAI-Analytical normally will generate a CSV output as shown in the figure below.

The second row contains summary information, including the exposure time and the absolute and percentage of computational time for each communication group, as well as the end-to-end time of one iteration. Below this are details of the operation for each specific layer.

<img src="./images/simai_raw.png" alt="simai_raw" width="50%">


### Visualization

If you specify `-v` when running SimAI-Analytical, the following will be generated:

<img src="./images/simai_visual.png" alt="simai_visual" width="30%">

# SimAI-Simulation Usage
## 📝 Workload Generate

Using the same workload as SimAI-Analytical, generated by [SimAI-WorkloadGenerator](https://github.com/aliyun/aicb?tab=readme-ov-file#generate-workloads-for-simulation-simai) feature in [AICB](https://github.com/aliyun/aicb).

## 🔧 TOPO Setting

Before running SimAI-Simulator, you need to generate a `topo` file that can be recognized by `ns-3-alibabacloud`.

As shown in the figure below, the first row represents various parameters: `node_num` is the total number of nodes, `gpus_per_server` refers to the number of GPUs per server (currently, we bind each NIC to a GPU as a single node), `nvswitch_num` indicates the number of NVSwitch nodes (specifically used to implement the NVLS algorithm), `switch_num` is the number of switches, `link_num` is the total number of connections, and `gpu_type_str` describes the type of GPU.

| Abbreviation       | Description                                     |
|--------------------|-------------------------------------------------|
| `node_num`         | Total number of nodes                           |
| `gpus_per_server`  | Number of GPUs per server                       |
| `nvswitch_num`     | Number of NVSwitch nodes (for NVLS algorithm)   |
| `switch_num`       | Number of switches                              |
| `link_num`         | Total number of connections                     |
| `gpu_type_str`     | Type of GPU                                     |

<img src="./images/simai_topo.png" alt="simai_topo" width="30%">

You can choose to customize any `topo` following the format shown above. Of course, we also provide a script to directly generate a `topo` for the HPN architecture.

```bash
python3 ./astra-sim-alibabacloud/inputs/topo/gen_HPN_7.0_topo_mulgpus_one_link.py -g 128 -gt A100 -bw 100Gbps -nvbw 2400Gbps
```

| Parameter        | Description                                | Default Value |
|------------------|--------------------------------------------|---------------|
| `-l  --latency`          | NIC latency                              | 0.0005ms      |
| `-nl  --nv_latency`      | NV switch latency                        | 0.000025ms    |
| `-bw  --bandwidth`       | NIC to ASW bandwidth                     | 100Gbps       |
| `-apbw  --ap_bandwidth`  | ASW to PSW bandwidth                     | 400Gbps       |
| `-nvbw  --nvlink_bw`     | NVLink bandwidth                         | 1700Gbps      |
| `-er  --error_rate`      | Error rate                               | 0             |
| `-g  --gpu`              | Number of GPUs                           | 32            |
| `-gt  --gpu_type`        | GPU type                                 | H800          |
| `-gps  --gpu_per_server` | GPUs per server                          | 8             |
| `-psn  --psw_switch_num` | Number of PSW switches                   | 120           |
| `-nsps  --nv_switch_per_server` | NV switch per server                 | 1             |
| `--dp`                    | Enable dual plane, default single plane  | false         |
| `--st`                    | Enable DCN architecture, default HPN_7_0 | false     |

## 🖥️ SimAI-NS3 Simulation

```bash
$ AS_SEND_LAT=3 AS_NVLS_ENABLE=1 ./bin/SimAI_simulator -t 16 -w ./example/microAllReduce.txt -n ./HPN_7_0_128_gpus_8_in_one_server_with_100Gbps_A100 -c astra-sim-alibabacloud/inputs/config/SimAI.conf
```

| Environment Variable Name | Description                      | Default Value                             |
|---------------------------|----------------------------------|-------------------------------------------|
| `AS_LOG_LEVEL`            | Log level                        | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `UNKNOWN`; default is `INFO` |
| `AS_PXN_ENABLE`           | Enable PXN                       | `0/1`; default is `false`                 |
| `AS_NVLS_ENABLE`          | Enable NVLS                      | `0/1`; default is `false`                 |
| `AS_SEND_LAT`             | Set packet sending latency       | Default is `6`, unit is `us`              |
| `AS_NVLSTREE_ENABLE`      | Enable NVLSTREE                  | Default is `false`                        |

| Parameter                  | Description                              | Default Value                                                      |
|----------------------------|------------------------------------------|--------------------------------------------------------------------|
| `-t  --thread`            | Number of threads for multithreading acceleration | Default is `1`; if multithreading is enabled, control the number of threads between `8` and `16`. |
| `-w  --workload`          | Path to workload                         | `./microAllReduce.txt`                                             |
| `-n  --network-topo`      | Network topology path                    | None    

## Result Analyze

Same as SimAI-Analytical

# SimAI-Physical Usage (TODO)
