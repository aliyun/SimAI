from math import ceil
from typing import Tuple

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


# Replica represents a model entity, which is a Data Parallelism (DP) unit
# Replica是一个模型实体，即一个DP单位
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
        
        # TODO(tianhao909): decouple pending_requests from replica
        # TODO(tianhao909): 将 pending_requests 从 replica 中解耦
        # self._pending_requests = []
        self.pending_requests = []
        self._pending_tasks = []
        # Scheduler metadata / 调度器元数据
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

        # New variables: track KV cache memory usage
        # 新增变量：跟踪kvcache显存使用情况
        self._allocated_kv_cache_memory = 0  # Allocated KV cache memory (bytes) / 已分配的kvcache显存
        self._max_kv_cache_memory = None  # Max KV cache capacity (bytes) / 最大kvcache显存容量
        self._kv_cache_allocation_map = {}  # Track per-request KV cache allocation / 跟踪每个请求分配的kvcache大小
        
        
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

    def get_kv_cache_per_token(self) -> int:
        """
        Calculate per-token KV Cache size (unit: Bytes).
        计算每个token的KV Cache大小 (单位: Bytes)
        
        Formula / 公式: 2 * num_kv_heads * head_dim * num_layers * bytes_per_element
        
        Returns:
            int: Per-token KV Cache size (Bytes)
        """
        # Determine bytes per element / 确定每个元素的字节数
        dtype_to_bytes = {
            'float16': 2, 'bfloat16': 2,
            'float32': 4, 'float64': 8,
            'fp8': 1, 'int8': 1,
            'int16': 2, 'int32': 4, 'int64': 8
        }
        bytes_per_element = dtype_to_bytes.get(self.pd_p2p_comm_dtype, 2)
        
        # KV Cache size per token / KV Cache每 token的大小
        kv_cache_per_token = (
            2                        # K和V两个缓存
            * self.num_kv_heads      # KV heads数量
            * self.attention_head_dim  # 每个head的维度
            * self.num_layers        # 层数
            * bytes_per_element      # 每个元素的字节数
        )
        return kv_cache_per_token

    def get_remaining_kv_cache_capacity(self, avg_tokens_per_request=None) -> Tuple[int, int]:
        """
        Calculate remaining KV cache memory capacity and how many requests it can serve.
        计算当前副本剩余的kvcache显存容量，以及还能容纳多少个request
        
        Args:
            avg_tokens_per_request: Avg tokens per request (default: max_request_tokens)
                每个请求的平均token数
        
        Returns:
            (remaining_kv_cache_bytes, remaining_request_capacity)
        """
        from vidur.scheduler.utils.memory_planner import MemoryPlanner
        memory_planner = MemoryPlanner(self._replica_config, self)

        # ===== 1. Init max KV cache capacity (computed on first call) =====
        # ===== 1. 初始化最大kvcache容量 (首次调用时计算) =====
        if self._max_kv_cache_memory is None:
            # Get real KV cache available memory (bytes) from memory_planner
            # 直接从 memory_planner 获取真实的 KV cache 可用内存 (bytes)
            # Correct calculation: available memory - model parameter memory
            # 这是正确的计算: 可用内存 - 模型参数内存
            self._max_kv_cache_memory = memory_planner.get_kv_cache_available_memory()
            
            # Compute per-request KV cache for display
            # 计算每请求 KV cache 用于显示
            kv_cache_per_token = self.get_kv_cache_per_token()
            tokens_per_req = avg_tokens_per_request or self.max_request_tokens
            kv_cache_per_request = kv_cache_per_token * tokens_per_req
            max_requests = int(self._max_kv_cache_memory / kv_cache_per_request) if kv_cache_per_request > 0 else 0
            
            logger.info(f"[Replica] KV Cache Capacity Init (KV Cache容量初始化):")
            logger.info(f"  Total GPU mem (GPU总内存): {self.total_memory_gb:.2f} GB")
            logger.info(f"  Mem margin (内存保留比例): {self.memory_margin_fraction*100:.1f}%")
            logger.info(f"  Max KV cache capacity (最大KV cache容量): {self._max_kv_cache_memory/(1024**3):.2f} GB")
            logger.info(f"  KV cache per token (每token KV cache): {kv_cache_per_token} bytes = {kv_cache_per_token/1024:.2f} KB")
            logger.info(f"  Avg tokens per req (每请求平均token数): {tokens_per_req}")
            logger.info(f"  KV cache per req (每请求KV cache): {kv_cache_per_request/(1024**3):.4f} GB")
            logger.info(f"  Max servable reqs (最大可服务请求数): {max_requests}")

        # ===== 2. Compute remaining KV cache memory =====
        # ===== 2. 计算剩余kvcache显存 =====
        remaining_kv_cache = self._max_kv_cache_memory - self._allocated_kv_cache_memory

        # ===== 3. Compute remaining request capacity =====
        # ===== 3. 计算剩余容量可服务的请求数 =====
        # Unified calculation: per-token KV cache * avg tokens per request
        # 使用统一的计算方式
        kv_cache_per_token = self.get_kv_cache_per_token()
        tokens_per_req = avg_tokens_per_request or self.max_request_tokens
        kv_cache_per_request = kv_cache_per_token * tokens_per_req
        
        if kv_cache_per_request > 0:
            remaining_request_capacity = int(remaining_kv_cache / kv_cache_per_request)
        else:
            remaining_request_capacity = 0

        # ===== 4. Print debug info =====
        # ===== 4. 打印调试信息 =====
        logger.debug(f"Remaining KV cache: {remaining_kv_cache / (1024**3):.2f} GB ({remaining_kv_cache / (1024**2):.2f} MB)")
        logger.debug(f"Per-request KV cache: {kv_cache_per_request/(1024**3):.4f} GB ({tokens_per_req} tokens)")
        logger.debug(f"Remaining request capacity: {remaining_request_capacity}")

        return remaining_kv_cache, remaining_request_capacity

    def release_request_kv_cache_memory(self, request) -> None:
        """
        Release KV cache memory occupied by the specified request.
        释放指定request占用的kvcache显存
        """
        # Get KV cache size occupied by this request from allocation map
        # 从分配映射中获取这个request占用的kvcache大小
        if request.id in self._kv_cache_allocation_map:
            kv_cache_size = self._kv_cache_allocation_map[request.id]
            assert kv_cache_size > 0, f"fth debug: request {request.id} kv cache size should be positive"
                   
            # Subtract this request's KV cache from allocated total
            # 从已分配的kvcache中减去这个request的占用
            self._allocated_kv_cache_memory = max(0, self._allocated_kv_cache_memory - kv_cache_size)
            
            # Remove this request from allocation map
            # 从分配映射中移除这个请求
            del self._kv_cache_allocation_map[request.id]

            logger.debug(f"Released KV cache for request {request.id}: {kv_cache_size / (1024**3):.2f} GB ({kv_cache_size / (1024**2):.2f} MB)")
        else:
            logger.warning(f"Request {request.id} not found in KV cache allocation map")

    # def allocate_request_kv_cache_memory(self, request, num_blocks):
    #     """
    #     为指定request分配kvcache显存，使用类似_allocation_map的方式跟踪每个请求的分配情况
    #     根据分配的块数来计算kvcache大小
    #     """
    #     # 根据分配的块数计算这个request占用的kvcache大小
    #     kv_cache_size = request.estimate_kv_cache_size(num_blocks, self)
    
    def allocate_request_kv_cache_memory(self, request, num_blocks, block_size) -> None:
        """
        Allocate KV cache memory for a request, tracking per-request allocation.
        为指定request分配kvcache显存，跟踪每个请求的分配情况
        
        Previously num_blocks was passed directly as num_tokens,
        causing KV cache tracking to be underestimated by block_size times.
        Now correctly converts num_blocks * block_size to num_tokens.
        之前 num_blocks 直接作为 num_tokens 传入，导致跟踪量被低估 block_size 倍。
        
        Args:
            request: Request object / 请求对象
            num_blocks: Number of allocated memory blocks / 分配的内存块数
            block_size: Tokens per block / 每个块包含的token数
        """
        # Correct conversion: num_tokens = num_blocks * block_size
        # 正确转换
        num_tokens = num_blocks * block_size
        kv_cache_size = request.estimate_kv_cache_size(num_tokens, self)
        logger.debug(f"allocate_request_kv_cache_memory: "
                    f"req={request.id}, num_blocks={num_blocks}, block_size={block_size}, "
                    f"num_tokens={num_tokens}, kv_cache_size={kv_cache_size/(1024**2):.2f} MB")

        # Update allocation map / 更新分配映射
        if request.id not in self._kv_cache_allocation_map:
            self._kv_cache_allocation_map[request.id] = kv_cache_size
        else:
            # If already allocated, accumulate (for incremental allocation)
            # 如果已有分配，则累加
            self._kv_cache_allocation_map[request.id] += kv_cache_size

        # Increase allocated KV cache / 增加已分配的kvcache
        self._allocated_kv_cache_memory += kv_cache_size

        logger.debug(f"Allocated KV cache for request {request.id}: {kv_cache_size / (1024**3):.2f} GB, "
                    f"total allocated: {self._allocated_kv_cache_memory / (1024**3):.2f} GB")

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


    def add_to_pool(self, task) -> None:
        """
        Add a Task to the request pool.
        Request pool is ordered by request arrival time.
        """
        
        # bisect.insort(): Uses binary search algorithm to insert element into sorted list, maintaining list's sorted state
        # self.pending_requests: Target list storing all pending requests
        # task.request: Request object to be inserted
        # key=lambda x: x.arrival_timestamp: Sort key function, sorting by request arrival time
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