CONFIG_KEY = [
    "replica_num_pipeline_stages",
    "replica_num_tensor_parallel_workers",
    "replica_device",
    "cluster_num_replicas",
    "replica_scheduler_batch_size_cap",
    "replica_scheduler_provider",
    "sarathi_scheduler_chunk_size",
]

METRICS = {
    "request_e2e_time_normalized": "Normalized Request Latency",
    "ttft": "Time to First Token",
    "tbt": "Time Between Tokens",
}

AXIS_COLS = {
    "replica_num_pipeline_stages": "Pipeline Parallel Dim",
    "replica_num_tensor_parallel_workers": "Tensor Parallel Dim",
    "replica_scheduler_provider": "Scheduler",
    "sarathi_scheduler_chunk_size": "Sarathi Chunk Size",
    "replica_scheduler_batch_size_cap": "Batch Size",
    "replica_device": "SKU",
}

AXIS_COLS_SHORT = {
    "replica_num_pipeline_stages": "PP Dim",
    "replica_num_tensor_parallel_workers": "TP Dim",
    "replica_scheduler_provider": "Scheduler",
    "sarathi_scheduler_chunk_size": "Chunk Size",
    "replica_scheduler_batch_size_cap": "BS",
    "replica_device": "SKU",
}

AXIS_COLS_LONG_TO_SHORT = {
    "Pipeline Parallel Dim": "PP Dim",
    "Tensor Parallel Dim": "TP Dim",
    "Scheduler": "Scheduler",
    "Sarathi Chunk Size": "Chunk Size",
    "Batch Size": "BS",
    "SKU": "SKU",
}

PRETTY_NAMES = {
    "sarathi": "Sarathi-Serve",
    "orca+": "Orca+",
    "vllm": "vLLM",
    "h100": "H100",
    "a100": "A100",
    "a40": "A40",
}

# Prices extracted from https://docs.coreweave.com/welcome-to-coreweave/resource-based-pricing on 7th Feb, 2024
# Prices are in $/hr
GPU_COSTS = {
    "h100": 4.25,
    "a100": 2.21,
    "a40": 1.28,
}

CPU_MACHINE_COST = 3.36

# define colors
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]
TRANS_COLORS = [c + "95" for c in COLORS]
