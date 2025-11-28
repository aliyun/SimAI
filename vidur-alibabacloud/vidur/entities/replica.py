from math import ceil

from vidur.config import BaseRequestGeneratorConfig, ReplicaConfig
from vidur.entities.base_entity import BaseEntity
from vidur.logger import init_logger

logger = init_logger(__name__)

# >
import bisect
from enum import IntEnum  
class ReplicaType(IntEnum):  # Define task type enumeration class, inheriting from IntEnum
    MIXED = 0 # Mixed, no distinction
    PREFILL = 1  # Prompt task (prefill stage)
    DECODE = 2  # Token task (generation stage)


# Replica是一个模型实体，即一个DP单位
# Replica represents a model entity, which is a Data Parallelism (DP) unit
class Replica(BaseEntity):
    def __init__(
        self,
        replica_config: ReplicaConfig,
        generator_config: BaseRequestGeneratorConfig,
    ) -> None:
        self._id = Replica.generate_id()

        self._replica_config = replica_config
        self._model_config = replica_config.model_config
        self._device_config = replica_config.device_config
        self._generator_config = generator_config

        assert (
            self._model_config.num_layers % self._replica_config.num_pipeline_stages
            == 0
        )
        assert (
            self._model_config.embedding_dim % self._replica_config.tensor_parallel_size
            == 0
        )
        
        # > sw
        # TODO > Decouple this from replica, as vidur itself is decoupled from it
        # self._pending_requests = []
        self.pending_requests = []
        self._pending_tasks = []
        # > scheduler metadata
        # self.sched_memory = self.model.size.total_size  # Memory usage from scheduler's perspective
        self.sched_memory = self._device_config.total_memory_gb
        self.sched_pending_tokens = 0  # Number of pending tokens from scheduler's perspective
        self.sched_tag = None  # Scheduler tag
        # Separate pending queue for prompt tasks (to prioritize prompts)
        self.pending_prompt_queue = []
        # Map requests->tasks on this instance
        self.request_tasks = {}
        self.replica_type = ReplicaType.MIXED
        
        # >
        self.pd_p2p_comm_bandwidth = self._replica_config.pd_p2p_comm_bandwidth
        self.pd_p2p_comm_dtype = self._replica_config.pd_p2p_comm_dtype
        self.pd_node_ratio = self._replica_config.pd_node_ratio
        self.nvlink_bandwidth = self._replica_config.nvlink_bandwidth
        self.rdma_bandwidth = self._replica_config.rdma_bandwidth
        
    @property
    def id(self) -> int:
        return self._id

    @property
    def num_layers(self) -> int:
        return self._model_config.num_layers

    @property
    def num_q_heads(self) -> int:
        return self._model_config.num_q_heads

    @property
    def num_kv_heads(self) -> int:
        return self._model_config.num_kv_heads

    @property
    def embedding_dim(self) -> int:
        return self._model_config.embedding_dim

    @property
    def mlp_hidden_dim(self) -> int:
        return self._model_config.mlp_hidden_dim

    @property
    def use_gated_mlp(self) -> int:
        return self._model_config.use_gated_mlp

    @property
    def vocab_size(self) -> int:
        return self._model_config.vocab_size

    @property
    def num_pipeline_stages(self) -> int:
        return self._replica_config.num_pipeline_stages

    @property
    def num_layers_per_pipeline_stage(self) -> int:
        return self._model_config.num_layers // self._replica_config.num_pipeline_stages

    @property
    def attention_head_dim(self) -> int:
        return self._model_config.embedding_dim // self._model_config.num_q_heads

    @property
    def q_heads_per_tensor_parallel_worker(self) -> int:
        return (
            self._model_config.num_q_heads // self._replica_config.tensor_parallel_size
        )

    @property
    def kv_heads_per_tensor_parallel_worker(self) -> int:
        return ceil(
            self._model_config.num_kv_heads / self._replica_config.tensor_parallel_size
        )

    @property
    def num_tensor_parallel_workers(self) -> int:
        return self._replica_config.tensor_parallel_size

    @property
    def total_memory_gb(self) -> int:
        return self._device_config.total_memory_gb

    @property
    def memory_margin_fraction(self) -> float:
        return self._replica_config.memory_margin_fraction

    @property
    def max_request_tokens(self) -> int:
        return self._generator_config.max_tokens

    @property
    def per_device_flops(self) -> float:
        return self._device_config.fp16_tflops * 2**40
    
    # > sw
    # @property
    # def pending_requests(self) -> list:
    #     return self._pending_requests

    @property
    def pending_tasks(self) -> list:
        return self._pending_tasks
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "num_layers": self.num_layers,
            "num_q_heads": self.num_q_heads,
            "num_kv_heads": self.num_kv_heads,
            "embedding_dim": self.embedding_dim,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "use_gated_mlp": self.use_gated_mlp,
            "vocab_size": self.vocab_size,
            "num_pipeline_stages": self.num_pipeline_stages,
            "num_tensor_parallel_workers": self.num_tensor_parallel_workers,
        }


    def add_to_pool(self, task):
        """
        Add a Task to the request pool.
        Request pool is ordered by request arrival time.
        """
        
        # bisect.insort(): Uses binary search algorithm to insert element into sorted list, maintaining list's sorted state
        # self.pending_requests: Target list storing all pending requests
        # task.request: Request object to be inserted
        # key=lambda x: x.arrival_timestamp: Sort key function, sorting by request arrival timestamp
        # lambda x: x.arrival_timestamp is an anonymous function that accepts a parameter x (request object) and returns its arrival_timestamp attribute
        # This ensures the pending_requests list is always sorted by request arrival time
        if task.request not in self.pending_requests:  # If request is not in current pool
            # bisect.insort(self.pending_requests, task.request,  # # Insert sort, insert by arrival time
            #               key=lambda x: x.arrival_timestamp)
            # arrived_at
            bisect.insort(self.pending_requests, task.request,  # # Insert sort, insert by arrival time
                          key=lambda x: x.arrived_at)
            self.request_tasks[task.request] = [task]  # Create task list for new request
        else:
            self.request_tasks[task.request].append(task)  # Otherwise append task