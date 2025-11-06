import json
import os
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from vidur.config.base_poly_config import BasePolyConfig
from vidur.config.device_sku_config import BaseDeviceSKUConfig
from vidur.config.flat_dataclass import create_flat_dataclass
from vidur.config.model_config import BaseModelConfig
from vidur.config.node_sku_config import BaseNodeSKUConfig
from vidur.config.utils import dataclass_to_dict
from vidur.logger import init_logger
from vidur.types import (
    ExecutionTimePredictorType,
    GlobalSchedulerType,
    ReplicaSchedulerType,
    RequestGeneratorType,
    RequestIntervalGeneratorType,
    RequestLengthGeneratorType,
)

logger = init_logger(__name__)


@dataclass
class BaseRequestIntervalGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )


@dataclass
class BaseRequestLengthGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )
    max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum tokens."},
    )


@dataclass
class TraceRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/AzureFunctionsInvocationTraceForTwoWeeksJan2021Processed.csv",
        metadata={"help": "Path to the trace request interval generator file."},
    )
    start_time: str = field(
        default="1970-01-04 12:00:00",
        metadata={"help": "Start time of the trace request interval generator."},
    )
    end_time: str = field(
        default="1970-01-04 15:00:00",
        metadata={"help": "End time of the trace request interval generator."},
    )
    time_scale_factor: float = field(
        default=1.0,
        metadata={
            "help": "Time scale factor for the trace request interval generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.TRACE


@dataclass
class PoissonRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=0.5,
        metadata={"help": "Queries per second for Poisson Request Interval Generator."},
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.POISSON


@dataclass
class GammaRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    qps: float = field(
        default=0.2,
        metadata={"help": "Queries per second for Gamma Request Interval Generator."},
    )
    cv: float = field(
        default=0.5,
        metadata={
            "help": "Coefficient of variation for Gamma Request Interval Generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.GAMMA


@dataclass
class StaticRequestIntervalGeneratorConfig(BaseRequestIntervalGeneratorConfig):
    @staticmethod
    def get_type():
        return RequestIntervalGeneratorType.STATIC


@dataclass
class TraceRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv",
        metadata={"help": "Path to the trace request length generator file."},
    )
    prefill_scale_factor: float = field(
        default=1,
        metadata={
            "help": "Prefill scale factor for the trace request length generator."
        },
    )
    decode_scale_factor: float = field(
        default=1,
        metadata={
            "help": "Decode scale factor for the trace request length generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.TRACE


@dataclass
class ZipfRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    theta: float = field(
        default=0.6,
        metadata={"help": "Theta for Zipf Request Length Generator."},
    )
    scramble: bool = field(
        default=False,
        metadata={"help": "Scramble for Zipf Request Length Generator."},
    )
    min_tokens: int = field(
        default=1024,
        metadata={"help": "Minimum tokens for Zipf Request Length Generator."},
    )
    prefill_to_decode_ratio: float = field(
        default=20.0,
        metadata={"help": "Prefill to decode ratio for Zipf Request Length Generator."},
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.ZIPF


@dataclass
class UniformRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    min_tokens: int = field(
        default=1024,
        metadata={"help": "Minimum tokens for Uniform Request Length Generator."},
    )
    prefill_to_decode_ratio: float = field(
        default=20.0,
        metadata={
            "help": "Prefill to decode ratio for Uniform Request Length Generator."
        },
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.UNIFORM


@dataclass
class FixedRequestLengthGeneratorConfig(BaseRequestLengthGeneratorConfig):
    prefill_tokens: int = field(
        default=2048,
        metadata={"help": "Prefill tokens for Fixed Request Length Generator."},
    )
    decode_tokens: int = field(
        default=512,
        metadata={"help": "Decode tokens for Fixed Request Length Generator."},
    )

    @staticmethod
    def get_type():
        return RequestLengthGeneratorType.FIXED


@dataclass
class BaseRequestGeneratorConfig(BasePolyConfig):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )


@dataclass
class SyntheticRequestGeneratorConfig(BaseRequestGeneratorConfig):
    # 默认生成定长的请求，请求到来的时间满足泊松分布
    # By default, generate fixed-length requests with Poisson-distributed arrival times
    length_generator_config: BaseRequestLengthGeneratorConfig = field(
        default_factory=FixedRequestLengthGeneratorConfig,
        metadata={"help": "Length generator config for Synthetic Request Generator."},
    )
    interval_generator_config: BaseRequestIntervalGeneratorConfig = field(
        default_factory=PoissonRequestIntervalGeneratorConfig,
        metadata={"help": "Interval generator config for Synthetic Request Generator."},
    )
    num_requests: Optional[int] = field(
        default=128,
        metadata={"help": "Number of requests for Synthetic Request Generator."},
    )
    duration: Optional[float] = field(
        default=None,
        metadata={"help": "Duration of the synthetic request generator."},
    )

    def __post_init__(self):
        self.max_tokens = self.length_generator_config.max_tokens

    @staticmethod
    def get_type():
        return RequestGeneratorType.SYNTHETIC


@dataclass
class TraceRequestGeneratorConfig(BaseRequestGeneratorConfig):
    trace_file: str = field(
        default="data/processed_traces/splitwise_conv.csv",
        metadata={"help": "Path to the trace request generator file."},
    )
    prefill_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Prefill scale factor for the trace request generator."},
    )
    decode_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Decode scale factor for the trace request generator."},
    )
    time_scale_factor: float = field(
        default=1.0,
        metadata={"help": "Time scale factor for the trace request generator."},
    )
    max_tokens: int = field(
        default=4096,
        metadata={"help": "Maximum tokens for the trace request generator."},
    )

    @staticmethod
    def get_type():
        return RequestGeneratorType.TRACE_REPLAY


@dataclass
class BaseReplicaSchedulerConfig(BasePolyConfig):
    batch_size_cap: int = field(
        default=128,
        metadata={"help": "Maximum batch size cap."},
    )
    block_size: int = field(
        default=16,
        metadata={"help": "Block size."},
    )
    watermark_blocks_fraction: float = field(
        default=0.01,
        metadata={"help": "Watermark blocks fraction."},
    )
    num_blocks: Optional[int] = field(
        default=None,
        metadata={"help": "Number of blocks."},
    )


@dataclass
class VllmSchedulerConfig(BaseReplicaSchedulerConfig):
    max_tokens_in_batch: int = field(
        default=4096,
        metadata={"help": "Maximum tokens in batch for vLLM."},
    )

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.VLLM


@dataclass
class LightllmSchedulerConfig(BaseReplicaSchedulerConfig):
    max_tokens_in_batch: int = field(
        default=4096,
        metadata={"help": "Maximum tokens in batch for LightLLM."},
    )
    max_waiting_iters: int = field(
        default=10,
        metadata={"help": "Maximum waiting iterations for LightLLM."},
    )

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.LIGHTLLM


@dataclass
class OrcaSchedulerConfig(BaseReplicaSchedulerConfig):

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.ORCA



@dataclass
class FasterTransformerSchedulerConfig(BaseReplicaSchedulerConfig):

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.FASTER_TRANSFORMER


@dataclass
class SarathiSchedulerConfig(BaseReplicaSchedulerConfig):
    chunk_size: int = field(
        default=512,
        metadata={"help": "Chunk size for Sarathi."},
    )

    @staticmethod
    def get_type():
        return ReplicaSchedulerType.SARATHI

# >
@dataclass
class SplitWiseReplicaSchedulerConfig(BaseReplicaSchedulerConfig):
    max_tokens_in_batch: int = field(
        default=4096,
        metadata={"help": "Maximum tokens in batch for SplitWise."},
    )
    @staticmethod
    def get_type():
        return ReplicaSchedulerType.SPLIT_WISE


@dataclass
class MetricsConfig:
    """Metric configuration."""

    write_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to write metrics."},
    )
    write_json_trace: bool = field(
        default=False,
        metadata={"help": "Whether to write json trace."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases project name."},
    )
    wandb_group: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases group name."},
    )
    wandb_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases run name."},
    )
    wandb_sweep_id: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases sweep id."},
    )
    wandb_run_id: Optional[str] = field(
        default=None,
        metadata={"help": "Weights & Biases run id."},
    )
    enable_chrome_trace: bool = field(
        default=True,
        metadata={"help": "Enable Chrome tracing."},
    )
    save_table_to_wandb: bool = field(
        default=False,
        metadata={"help": "Whether to save table to wandb."},
    )
    store_plots: bool = field(
        default=True,
        metadata={"help": "Whether to store plots."},
    )
    store_operation_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to store operation metrics."},
    )
    store_token_completion_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to store token completion metrics."},
    )
    store_request_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store request metrics."},
    )
    store_batch_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store batch metrics."},
    )
    store_utilization_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to store utilization metrics."},
    )
    keep_individual_batch_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to keep individual batch metrics."},
    )
    subsamples: Optional[int] = field(
        default=None,
        metadata={"help": "Subsamples."},
    )
    min_batch_index: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum batch index."},
    )
    max_batch_index: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum batch index."},
    )
    output_dir: str = field(
        default="simulator_output",
        metadata={"help": "Output directory."},
    )
    cache_dir: str = field(
        default="cache",
        metadata={"help": "Cache directory."},
    )

    def __post_init__(self):
        self.output_dir = (
            f"{self.output_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"
        )
        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class ReplicaConfig:
    model_name: str = field(
        default="meta-llama/Llama-2-7b-hf",
        metadata={"help": "Model name."},
    )
    memory_margin_fraction: float = field(
        default=0.1,
        metadata={"help": "Memory margin fraction."},
    )
    num_pipeline_stages: int = field(
        default=1,
        metadata={"help": "Number of pipeline stages."},
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Tensor parallel size."},
    )
    device: str = field(
        default="a100",
        metadata={"help": "Device."},
    )
    network_device: str = field(
        default="a100_pairwise_nvlink",
        metadata={"help": "Network device."},
    )
    
    # >
    pd_p2p_comm_bandwidth: int = field(
        default=800,
        metadata={"help": "> add: pd_p2p_comm_bandwidth for pd disaggregation. (bps)"},
    )
    
    nvlink_bandwidth: int = field(
        default=1600,
        metadata={"help": "> add: nvlink_bandwidth for tp/ep communication et al. (bps)"},
    )
    
    rdma_bandwidth: int = field(
        default=800,
        metadata={"help": "> add: rdma_bandwidth for tp/ep communication et al. (bps)"},
    )
    
    pd_p2p_comm_dtype: str = field(
        default='float16',
        metadata={"help": "> add: pd_p2p_comm_dtype for pd disaggregation."},
    )
    
    pd_node_ratio: float = field(
        default=0.5,
        metadata={"help": "Number of p replicas : number of d replicas."},
    )
    
    # parser.add_argument('--expert_model_parallel_size', type=int, default=1, help='Degree of expert model parallelism.')
    expert_model_parallel_size: int = field(
        default=1,
        metadata={"help": "Degree of expert model parallelism."},
    )

    def __post_init__(self):
        self.world_size = self.num_pipeline_stages * self.tensor_parallel_size
        self.model_config: BaseModelConfig = BaseModelConfig.create_from_name(
            self.model_name
        )
        self.device_config: BaseDeviceSKUConfig = (
            BaseDeviceSKUConfig.create_from_type_string(self.device)
        )
        self.node_config: BaseNodeSKUConfig = BaseNodeSKUConfig.create_from_type_string(
            self.network_device
        )


@dataclass
class BaseGlobalSchedulerConfig(BasePolyConfig):
    pass


@dataclass
class RandomGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.RANDOM


@dataclass
class RoundRobinGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.ROUND_ROBIN


@dataclass
class LORGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.LOR
    
# > add: Add the SplitWiseGlobalSchedulerConfig configuration class
@dataclass
class SplitWiseGlobalSchedulerConfig(BaseGlobalSchedulerConfig):
    @staticmethod
    def get_type():
        return GlobalSchedulerType.SPLIT_WISE


@dataclass
class BaseExecutionTimePredictorConfig(BasePolyConfig):
    compute_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/mlp.csv",
        metadata={"help": "Path to the compute input file."},
    )
    attention_input_file: str = field(
        default="./data/profiling/compute/{DEVICE}/{MODEL}/attention.csv",
        metadata={"help": "Path to the attention input file."},
    )
    all_reduce_input_file: str = field(
        default="./data/profiling/network/{NETWORK_DEVICE}/all_reduce.csv",
        metadata={"help": "Path to the all reduce input file."},
    )
    send_recv_input_file: str = field(
        default="./data/profiling/network/{NETWORK_DEVICE}/send_recv.csv",
        metadata={"help": "Path to the send recv input file."},
    )
    # 这个现有的profiling没抓
    # This existing profiling didn't catch it.
    cpu_overhead_input_file: str = field(
        default="./data/profiling/cpu_overhead/{NETWORK_DEVICE}/{MODEL}/cpu_overheads.csv",
        metadata={"help": "Path to the cpu overhead input file."},
    )
    k_fold_cv_splits: int = field(
        default=10,
        metadata={"help": "Number of k fold cross validation splits."},
    )
    no_cache: bool = field(
        default=False,
        metadata={"help": "Whether to cache prediction models."},
    )
    kv_cache_prediction_granularity: int = field(
        default=64,
        metadata={"help": "KV cache prediction granularity."},
    )
    prediction_max_prefill_chunk_size: int = field(
        default=4096,
        metadata={"help": "Max prefill chunk size for prediction."},
    )
    prediction_max_batch_size: int = field(
        default=128,
        metadata={"help": "Max batch size for prediction."},
    )
    prediction_max_tokens_per_request: int = field(
        default=4096,
        metadata={"help": "Max tokens per request for prediction."},
    )
    attention_decode_batching_overhead_fraction: float = field(
        default=0.1,
        metadata={"help": "Attention decode batching overhead fraction."},
    )
    attention_prefill_batching_overhead_fraction: float = field(
        default=0.1,
        metadata={"help": "Attention prefill batching overhead fraction."},
    )
    nccl_cpu_launch_overhead_ms: float = field(
        default=0.02,
        metadata={"help": "NCCL CPU launch overhead in ms."},
    )
    nccl_cpu_skew_overhead_per_device_ms: float = field(
        default=0.0,
        metadata={"help": "NCCL CPU skew overhead per device in ms."},
    )
    num_training_job_threads: int = field(
        default=-1,
        metadata={"help": "Number of training job threads."},
    )
    skip_cpu_overhead_modeling: bool = field(
        default=True,
        metadata={"help": "Whether to skip CPU overhead modeling."},
    )
    backend: str = field(
        default="vidur",
        # choices=["vidur", "simai_simulation", "simai_analytical"],
        metadata={"help": "Select the backend for execution time prediction. "
         "choices=['vidur', 'simai_simulation', 'simai_analytical','aicb'],"
         "Options: 'simai_simulation' (simulation-based SimAI), "
         "'simai_analytical' (analytical SimAI), 'vidur', or 'none' to disable."},
    )
    simai_dir: str = field(
        # default=".",
        # default="../SimAI",
        default="../",
        metadata={"help": "Path to the simai simulator directory."},
    )
    simai_simulation_topo: str = field(
        # default="./SimAI/example/topo",
        # default="../SimAI/example/topo",
        default="../example/topo",
        metadata={"help": "Path to the simai topology file."},
    )
    simai_simulation_config: str = field(
        # default="./SimAI/astra-sim-alibabacloud/inputs/config/SimAI.conf",
        # default="../SimAI/astra-sim-alibabacloud/inputs/config/SimAI.conf",
        default="../astra-sim-alibabacloud/inputs/config/SimAI.conf",
        metadata={"help": "Path to the simai config file."},
    )


@dataclass
class LinearRegressionExecutionTimePredictorConfig(BaseExecutionTimePredictorConfig):
    polynomial_degree: List[int] = field(
        default_factory=lambda: list(range(1, 6)),
        metadata={"help": "Polynomial degree for linear regression."},
    )
    polynomial_include_bias: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Polynomial include bias for linear regression."},
    )
    polynomial_interaction_only: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Polynomial interaction only for linear regression."},
    )
    fit_intercept: List[bool] = field(
        default_factory=lambda: [True, False],
        metadata={"help": "Fit intercept for linear regression."},
    )

    @staticmethod
    def get_type():
        return ExecutionTimePredictorType.LINEAR_REGRESSION


@dataclass
class RandomForrestExecutionTimePredictorConfig(BaseExecutionTimePredictorConfig):
    num_estimators: List[int] = field(
        default_factory=lambda: [250, 500, 750],
        metadata={"help": "Number of estimators for random forest."},
    )
    max_depth: List[int] = field(
        default_factory=lambda: [8, 16, 32],
        metadata={"help": "Maximum depth for random forest."},
    )
    min_samples_split: List[int] = field(
        default_factory=lambda: [2, 5, 10],
        metadata={"help": "Minimum samples split for random forest."},
    )

    @staticmethod
    def get_type():
        return ExecutionTimePredictorType.RANDOM_FORREST


@dataclass
class ClusterConfig:
    # num_replicas：指定集群中副本的数量，默认为1个副本
    # Number of replicas in the cluster, defaults to 1 replica
    num_replicas: int = field(
        default=1,
        metadata={"help": "Number of replicas."},
    )
    # replica_config：定义每个副本的配置，使用 ReplicaConfig 类作为配置模板
    # Configuration for each replica using ReplicaConfig class as template
    replica_config: ReplicaConfig = field(default_factory=ReplicaConfig)
    
    # global_scheduler_config：定义全局调度器配置，使用轮询(RoundRobin)调度策略
    # Global scheduler configuration using RoundRobin scheduling strategy
    global_scheduler_config: BaseGlobalSchedulerConfig = field(
        default_factory=RoundRobinGlobalSchedulerConfig,
        metadata={"help": "Global scheduler config."},
    )
    # replica_scheduler_config：定义副本级别的调度器配置，使用 Sarathi 调度策略
    # Replica-level scheduler configuration using Sarathi scheduling strategy
    replica_scheduler_config: BaseReplicaSchedulerConfig = field(
        default_factory=SarathiSchedulerConfig,
        metadata={"help": "Replica scheduler config."},
    )

@dataclass
class SimulationConfig(ABC):
    seed: int = field(
        default=42,
        metadata={"help": "Seed for the random number generator."},
    )
    log_level: str = field(
        default="info",
        metadata={"help": "Logging level."},
    )
    time_limit: int = field(
        default=0,  # in seconds, 0 is no limit
        metadata={"help": "Time limit for simulation in seconds. 0 means no limit."},
    )
    cluster_config: ClusterConfig = field(
        default_factory=ClusterConfig,
        metadata={"help": "Cluster config."},
    )
    request_generator_config: BaseRequestGeneratorConfig = field(
        default_factory=SyntheticRequestGeneratorConfig,
        metadata={"help": "Request generator config."},
    )
    execution_time_predictor_config: BaseExecutionTimePredictorConfig = field(
        default_factory=RandomForrestExecutionTimePredictorConfig,
        metadata={"help": "Execution time predictor config."},
    )
    metrics_config: MetricsConfig = field(
        default_factory=MetricsConfig,
        metadata={"help": "Metrics config."},
    )

    def __post_init__(self):
        self.write_config_to_file()

    @classmethod
    def create_from_cli_args(cls):
        flat_config = create_flat_dataclass(cls).create_from_cli_args()
        instance = flat_config.reconstruct_original_dataclass()
        instance.__flat_config__ = flat_config
        return instance

    def to_dict(self):
        if not hasattr(self, "__flat_config__"):
            logger.warning("Flat config not found. Returning the original config.")
            return self.__dict__

        return self.__flat_config__.__dict__

    def write_config_to_file(self):
        config_dict = dataclass_to_dict(self)
        with open(f"{self.metrics_config.output_dir}/config.json", "w") as f:
            json.dump(config_dict, f, indent=4)
