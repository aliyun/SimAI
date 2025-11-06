import hashlib
from dataclasses import dataclass
from itertools import product
from typing import List, Optional


@dataclass
class ModelConfig:
    name: str
    identifier: str
    exclude_tp_dims: List[int] = None

    def get_key(self):
        return self.name

    def to_config_dict(self):
        return {
            "replica_config_model_name": self.identifier,
        }

    def is_tensor_parallel_degree_valid(self, tp_degree: int):
        return self.exclude_tp_dims is None or tp_degree not in self.exclude_tp_dims


@dataclass
class TraceConfig:
    name: str
    trace_file: str
    max_seq_len: int
    num_requests: int
    start_qps: float

    def get_key(self):
        return f"{self.name}_tk{self.max_seq_len}_rq{self.num_requests}"

    def to_config_dict(self):
        return {
            "request_generator_config_type": "synthetic",
            "length_generator_config_type": "trace",
            "interval_generator_config_type": "poisson",
            "synthetic_request_generator_config_max_tokens": self.max_seq_len,
            "trace_request_length_generator_config_max_tokens": self.max_seq_len,
            "zipf_request_length_generator_config_max_tokens": self.max_seq_len,
            "uniform_request_length_generator_config_max_tokens": self.max_seq_len,
            "fixed_request_length_generator_config_max_tokens": self.max_seq_len,
            "trace_request_generator_config_max_tokens": self.max_seq_len,
            "trace_request_length_generator_config_trace_file": self.trace_file,
            "trace_request_length_generator_config_prefill_scale_factor": 1,
            "trace_request_length_generator_config_decode_scale_factor": 1,
            "synthetic_request_generator_config_num_requests": self.num_requests,
            "vllm_scheduler_config_max_tokens_in_batch": self.max_seq_len,
        }


@dataclass
class ClusterConfig:
    device: str
    num_gpus: int
    gpus_per_node: int

    def get_key(self):
        return self.device

    def to_config_dict(self):
        return {
            "replica_config_device": self.device,
        }


@dataclass
class SchedulerConfig:
    scheduler: str
    chunk_size: Optional[int] = None

    def get_key(self):
        scheduler = self.scheduler

        if self.chunk_size is not None:
            scheduler += f"_cs{self.chunk_size}"

        return scheduler

    def to_config_dict(self):
        if self.scheduler == "vllm":
            return {
                "replica_scheduler_config_type": "vllm",
            }

        assert self.scheduler == "sarathi"
        assert self.chunk_size is not None
        return {
            "replica_scheduler_config_type": "sarathi",
            "sarathi_scheduler_config_chunk_size": self.chunk_size,
        }


class JobConfig:
    def __init__(
        self,
        model_config: ModelConfig,
        trace_config: TraceConfig,
        cluster_config: ClusterConfig,
        scheduler_config: SchedulerConfig,
        num_tensor_parallel_workers: int,
        num_pipeline_stages: int,
        batch_size: int,
    ):
        self.model_config = model_config
        self.trace_config = trace_config
        self.cluster_config = cluster_config
        self.scheduler_config = scheduler_config
        self.num_tensor_parallel_workers = num_tensor_parallel_workers
        self.num_pipeline_stages = num_pipeline_stages
        self.num_workers = self.num_tensor_parallel_workers * self.num_pipeline_stages
        self.batch_size = batch_size * num_pipeline_stages
        self.num_replicas = self.cluster_config.num_gpus // self.num_workers

        self.start_qps = self.trace_config.start_qps

    def is_valid(self):
        return (
            self.num_replicas > 0
            and self.model_config.is_tensor_parallel_degree_valid(
                self.num_tensor_parallel_workers
            )
            and self.num_tensor_parallel_workers <= self.cluster_config.gpus_per_node
        )

    def get_key(self):
        return (
            f"{self.model_config.name}_{self.trace_config.get_key()}_{self.cluster_config.get_key()}_{self.scheduler_config.get_key()}"
            f"_tp{self.num_tensor_parallel_workers}_pp{self.num_pipeline_stages}_bsz{self.batch_size}"
        )

    def get_human_readable_name(self):
        return (
            f"Model: {self.model_config.name}, Trace: {self.trace_config.name}, Cluster: {self.cluster_config.device}, "
            f"Scheduler: {self.scheduler_config.scheduler}, TP: {self.num_tensor_parallel_workers}, "
            f"PP: {self.num_pipeline_stages}, BSZ: {self.batch_size}, CS: {self.scheduler_config.chunk_size}, Hash: {self.get_hash()}"
        )

    def get_hash(self):
        return hashlib.sha1(self.get_key().encode("utf-8")).hexdigest()[:8]

    def to_config_dict(self):
        return {
            **self.model_config.to_config_dict(),
            **self.trace_config.to_config_dict(),
            **self.cluster_config.to_config_dict(),
            **self.scheduler_config.to_config_dict(),
            "replica_config_tensor_parallel_size": self.num_tensor_parallel_workers,
            "replica_config_num_pipeline_stages": self.num_pipeline_stages,
            "vllm_scheduler_config_batch_size_cap": self.batch_size,
            "lightllm_scheduler_config_batch_size_cap": self.batch_size,
            "orca_scheduler_config_batch_size_cap": self.batch_size,
            "faster_transformer_scheduler_config_batch_size_cap": self.batch_size,
            "sarathi_scheduler_config_batch_size_cap": self.batch_size,
            "cluster_config_num_replicas": self.num_replicas,
        }

    @classmethod
    def generate_job_configs(cls, config: dict):
        job_configs = []
        for (
            model_config,
            trace_config,
            cluster_config,
            scheduler_config,
            tp_dimension,
            pp_dimension,
            batch_size,
        ) in product(
            config["models"],
            config["traces"],
            config["clusters"],
            config["schedulers"],
            config["tp_dimensions"],
            config["pp_dimensions"],
            config["batch_sizes"],
        ):
            job_config = cls(
                ModelConfig(**model_config),
                TraceConfig(**trace_config),
                ClusterConfig(**cluster_config),
                SchedulerConfig(**scheduler_config),
                tp_dimension,
                pp_dimension,
                batch_size,
            )
            if not job_config.is_valid():
                continue

            job_configs.append(job_config)

        return job_configs

    @classmethod
    def generate_unique_model_job_configs(cls, config: dict, num_requests: int = 32):
        job_configs = []

        trace_config = TraceConfig(**config["traces"][0])
        trace_config.num_requests = num_requests
        scheduler_config = SchedulerConfig(**config["schedulers"][0])
        batch_size = config["batch_sizes"][0]
        # set pp_dimensions to 2 because it covers all the options
        pp_dimensions = [2]

        for model_config, cluster_config, tp_dimension, pp_dimension in product(
            config["models"],
            config["clusters"],
            config["tp_dimensions"],
            pp_dimensions,
        ):
            job_config = cls(
                ModelConfig(**model_config),
                trace_config,
                ClusterConfig(**cluster_config),
                scheduler_config,
                tp_dimension,
                pp_dimension,
                batch_size,
            )
            if not job_config.is_valid():
                continue

            job_configs.append(job_config)

        return job_configs


@dataclass
class SimulationConfig:
    output_dir: str
    cache_dir: str
    qps: float
    time_limit: int
    job_config: JobConfig

    def to_config_dict(self):
        return {
            **self.job_config.to_config_dict(),
            "metrics_config_output_dir": self.get_run_dir(),
            "metrics_config_cache_dir": self.cache_dir,
            "poisson_request_interval_generator_config_qps": self.qps,
            "gamma_request_interval_generator_config_qps": self.qps,
            "time_limit": self.time_limit * 60,  # to seconds
            "no-metrics_config_save_table_to_wandb": None,
            "no-metrics_config_store_plots": None,
            "no-metrics_config_store_operation_metrics": None,
            "no-metrics_config_store_token_completion_metrics": None,
            "no-metrics_config_enable_chrome_trace": None,
            "linear_regression_execution_time_predictor_config_skip_cpu_overhead_modeling": None,
            "random_forrest_execution_time_predictor_config_skip_cpu_overhead_modeling": None,
        }

    def to_args(self):
        args = []

        for key, value in self.to_config_dict().items():
            if value is not None:
                args.append(f"--{key} {value}")
            else:
                args.append(f"--{key}")

        return " ".join(args)

    def to_human_readable_name(self):
        return f"{self.job_config.get_human_readable_name()}, QPS: {self.qps}"

    def get_run_dir(self):
        return f"{self.output_dir}/runs/{self.job_config.get_hash()}/{self.qps}"
