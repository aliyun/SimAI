from typing import Tuple

from vidur.entities.base_entity import BaseEntity
from vidur.logger import init_logger

from vidur.entities.task import Task
import networkx as nx
from vidur.entities.flow import Flow
from enum import IntEnum  # Import IntEnum from enum for defining enumeration types
logger = init_logger(__name__)


# A decorator which checks if the request has been scheduled
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:
            raise ValueError("Request has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


def check_completed(func):
    def wrapper(self, *args, **kwargs):
        if not self._completed:
            raise ValueError("Request has not been completed yet")
        return func(self, *args, **kwargs)

    return wrapper

class RequestType(IntEnum):  # Define task type enumeration class, inheriting from IntEnum
    MIXED = 0 # Mixed, no distinction
    PREFILL = 1  # Prompt task (prefill stage)
    DECODE = 2  # Token task (generation stage)


class Request(BaseEntity):
    def __init__(
        self,
        arrived_at: float,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        num_processed_tokens: int = 0,
    ):
        self._id = Request.generate_id()
        self._arrived_at = arrived_at
        self._num_prefill_tokens = num_prefill_tokens
        self._num_decode_tokens = num_decode_tokens
        self._num_processed_tokens = num_processed_tokens

        self._scheduled_at = 0
        self._execution_time = 0
        self._model_execution_time = 0
        self._scheduling_delay = 0
        self._preempted_time = 0
        self._completed_at = 0
        self._prefill_completed_at = 0
        self._latest_stage_scheduled_at = 0
        self._latest_stage_completed_at = 0
        self._latest_iteration_scheduled_at = 0
        self._latest_iteration_completed_at = 0
        self._latest_iteration_scheduling_delay = 0

        self._scheduled = False
        self._preempted = False
        self._completed = False
        self._is_prefill_complete = False

        self._num_restarts = 0
        
        # DAG property for PD separation
        self.dag = nx.DiGraph()
        self.node_id = 0
        self.nodes = {}
        self.root_node = None
        # self.request_type = RequestType.MIXED
        self.request_type = RequestType.PREFILL
        self.prefill_arrived_at = arrived_at
        self.decode_arrived_at = float('inf')
        self.decode_time = float('inf')
        
        self.prefill_replica_id = None
        self.decode_replica_id = None
        
        # Point-to-point communication size between prefill and decode stages
        self.pd_p2p_comm_size = float('inf')
        self.pd_p2p_comm_time = float('inf')
        self.pd_p2p_comm_bandwidth = 0
        self.pd_p2p_bytes_per_token = None
        self.pd_p2p_comm_dtype = None
        
        # Reference to global_scheduler for obtaining decode replica
        self.global_scheduler = None
        


    @property
    def size(self) -> Tuple[int, int]:
        return (self._num_prefill_tokens, self._num_decode_tokens)

    @property
    @check_scheduled
    def scheduled_at(self) -> float:
        return self._scheduled_at

    @property
    @check_scheduled
    def latest_stage_scheduled_at(self) -> float:
        return self._latest_stage_scheduled_at

    @property
    @check_scheduled
    def latest_stage_completed_at(self) -> float:
        return self._latest_stage_completed_at

    @property
    @check_scheduled
    def latest_iteration_scheduled_at(self) -> float:
        return self._latest_iteration_scheduled_at

    @property
    @check_scheduled
    def latest_iteration_completed_at(self) -> float:
        return self._latest_iteration_completed_at

    @property
    @check_scheduled
    def latest_iteration_scheduling_delay(self) -> float:
        return self._latest_iteration_scheduling_delay

    @property
    @check_scheduled
    def prefill_completed_at(self) -> float:
        return self._prefill_completed_at

    @property
    @check_scheduled
    def scheduling_delay(self) -> float:
        return self._scheduling_delay

    @property
    @check_scheduled
    def preempted_time(self) -> float:
        return self._preempted_time

    @property
    @check_completed
    def completed_at(self) -> float:
        return self._completed_at

    @property
    @check_scheduled
    def e2e_time(self) -> float:
        return self._completed_at - self._arrived_at

    @property
    @check_scheduled
    def e2e_time_normalized(self) -> float:
        return self.e2e_time / self.num_decode_tokens

    @property
    @check_scheduled
    def execution_time(self) -> float:
        return self._execution_time

    @property
    @check_scheduled
    def execution_time_normalized(self) -> float:
        return self._execution_time / self.num_decode_tokens

    @property
    @check_scheduled
    def model_execution_time(self) -> float:
        return self._model_execution_time

    @property
    @check_scheduled
    def model_execution_time_normalized(self) -> float:
        return self._model_execution_time / self.num_decode_tokens

    @property
    def arrived_at(self) -> float:
        return self._arrived_at

    @property
    def num_prefill_tokens(self) -> int:
        return self._num_prefill_tokens

    @property
    def num_decode_tokens(self) -> int:
        return self._num_decode_tokens

    @property
    def pd_ratio(self) -> float:
        return self._num_prefill_tokens / self._num_decode_tokens

    @property
    def num_processed_tokens(self) -> int:
        return self._num_processed_tokens

    @property
    def total_tokens(self) -> int:
        return self._num_prefill_tokens + self._num_decode_tokens

    @property
    def num_processed_prefill_tokens(self) -> int:
        return min(self._num_processed_tokens, self._num_prefill_tokens)

    @property
    def num_processed_decode_tokens(self) -> int:
        return max(self._num_processed_tokens - self._num_prefill_tokens, 0)

    @property
    def scheduled(self) -> bool:
        return self._scheduled

    @property
    def preempted(self) -> bool:
        return self._preempted and not self._completed

    @property
    def completed(self) -> bool:
        return self._completed

    @property
    def num_restarts(self) -> int:
        return self._num_restarts

    @property
    def is_prefill_complete(self) -> bool:
        return self._is_prefill_complete

    @property
    def has_started_decode(self) -> bool:
        return self._num_processed_tokens > self._num_prefill_tokens + 1

    def on_batch_schedule(
        self,
        time: float,
    ) -> None:
        self._latest_iteration_scheduled_at = time
        self._latest_iteration_scheduling_delay = (
            time - self._latest_iteration_completed_at
        )

        if self._scheduled:
            return

        if self._num_restarts > 0:
            self._scheduled = True
            return

        self._scheduled_at = time
        self._scheduling_delay = time - self._arrived_at
        self._scheduled = True

    def on_batch_end(
        self,
        time: float,
        num_tokens_processed: int,
    ) -> None:
        self._num_processed_tokens += num_tokens_processed
        # Absolute time
        self._latest_iteration_completed_at = time

        assert self._num_processed_tokens <= self.total_tokens


        # _num_processed_tokens = 0+2048
        # 2048+1
        if self._num_processed_tokens == self._num_prefill_tokens:
            self._is_prefill_complete = True
            
            self.request_type = RequestType.DECODE
            
            # we get one decode token when the prefill processing completes
            self._num_processed_tokens += 1


            # we must record the prefill completion time only in the first time
            # in the subsequent restarts, we keep adding the previously decoded
            # tokens to the prefill tokens - that is irrelevant to the original prefill
            if self._prefill_completed_at == 0:
                # Record absolute time of prefill completion
                self._prefill_completed_at = time
        
        # Here; decode batching
        # elif self._num_processed_tokens == self._num_prefill_tokens:
        elif self._num_processed_tokens > self._num_prefill_tokens :
            
            assert self._is_prefill_complete == True, "prefill must be complete at this point"
            assert self.request_type == RequestType.DECODE, "request type must be DECODE at this point"


        elif self._num_processed_tokens < self._num_prefill_tokens:
            pass
        
        # check if request is completed
        if self._num_processed_tokens == self.total_tokens:
            self._completed_at = time
            self._completed = True
            self.decode_time = self._completed_at - self.prefill_completed_at
            assert self.decode_time > 0 and self.decode_time < float("inf"), "decode_time must be positive and finite"
            
            
            logger.debug(f"Request {self._id} completed at {self._completed_at}")
            
            
        if self._num_processed_tokens >= self._num_prefill_tokens:
            assert self.request_type == RequestType.DECODE and self.prefill_completed_at > 0 and self._is_prefill_complete == True, \
                "post-prefill request must be DECODE with valid prefill_completed_at"

        

    def on_batch_stage_schedule(
        self,
        time: float,
    ) -> None:
        self._latest_stage_scheduled_at = time
        if self._latest_stage_completed_at == 0:
            self._preempted_time = 0
        else:
            # TODO: verify preempted_time calculation each iteration
            self._preempted_time += time - self._latest_stage_completed_at
        self._preempted = False

    def on_batch_stage_end(
        self,
        time: float,
        execution_time: float,
        model_execution_time: float,
    ) -> None:
        self._execution_time += execution_time
        self._model_execution_time += model_execution_time
        self._latest_stage_completed_at = time
        self._preempted = True

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "arrived_at": self._arrived_at,
            "execution_time": self._execution_time,
            "model_execution_time": self._model_execution_time,
            "scheduled_at": self._scheduled_at,
            "scheduling_delay": self._scheduling_delay,
            "preempted_time": self._preempted_time,
            "completed_at": self._completed_at,
            "num_prefill_tokens": self._num_prefill_tokens,
            "num_decode_tokens": self._num_decode_tokens,
            "num_processed_tokens": self._num_processed_tokens,
            "scheduled": self._scheduled,
            "preempted": self._preempted,
            "completed": self._completed,
            "latest_stage_scheduled_at": self._latest_stage_scheduled_at,
            "latest_stage_completed_at": self._latest_stage_completed_at,
            "latest_iteration_scheduled_at": self._latest_iteration_scheduled_at,
            "latest_iteration_completed_at": self._latest_iteration_completed_at,
            "num_restarts": self._num_restarts,
        }

    def restart(self):
        logger.debug(f"Restarting request {self._id}")

        # when we restart the request, we can process all the previously
        # decoded tokens in parallel (i.e., we can prefill all the tokens)
        total_tokens = self._num_prefill_tokens + self._num_decode_tokens
        self._num_prefill_tokens = self._num_processed_tokens
        self._num_decode_tokens = total_tokens - self._num_prefill_tokens

        self._num_processed_tokens = 0
        self._scheduled = False
        self._preempted = False
        self._completed = False
        self._is_prefill_complete = False

        self._num_restarts += 1
    
    def create_task(self, task_type, **kwargs):
        """
        Creates a Task and adds it to the DAG.
        """
        
        task = Task.from_type(task_type=task_type,
                              node_id=self.node_id,
                              request=self,
                              **kwargs)
        self.node_id += 1
        self.dag.add_node(task)
        self.nodes[task.node_id] = task
        return task
    
    def create_flow(self, flow_type, **kwargs):
        """
        Create a flow and add it to the DAG.
        """
        flow = Flow.from_type(flow_type=flow_type,
                              node_id=self.node_id,  # Generate unique node ID
                              request=self,
                              **kwargs)  # Create flow based on flow type
        self.node_id += 1
        self.dag.add_node(flow)  # Add flow to DAG
        self.nodes[flow.node_id] = flow  # Add flow to node dictionary
        return flow  # Return created flow
    
    def successors(self, node):
        """
        Returns the next Task or Flow to be executed after node.
        """
        return self.dag.successors(node)
    
    def estimate_kv_cache_size(self, num_tokens=None, replica=None):
        """
        Calculate KV Cache size for the given number of tokens (unit: Bytes).
        计算指定token数量的KV Cache大小 (单位: Bytes)
        
        KV Cache formula / 公式:
        kv_cache_size = 2 (K+V) * num_tokens * num_kv_heads * head_dim * num_layers * bytes_per_element
        
        Args:
            num_tokens: Token count (prefill_tokens + decode_tokens)
            replica: Replica instance with model config
            
        Returns:
            int: KV Cache size (Bytes)
        """
        # ===== 1. Determine bytes per element (by data type) =====
        # ===== 1. 确定每个元素的字节数 (根据数据类型) =====
        dtype_to_bytes = {
            'float16': 2, 'bfloat16': 2,
            'float32': 4, 'float64': 8,
            'fp8': 1, 'int8': 1,
            'int16': 2, 'int32': 4, 'int64': 8
        }
        bytes_per_element = dtype_to_bytes.get(replica.pd_p2p_comm_dtype, 2)  # Default 2 bytes / 默认2字节
        
        # Save to instance for reuse elsewhere
        # 保存到实例属性供其他地方使用
        self.pd_p2p_bytes_per_token = bytes_per_element
        self.pd_p2p_comm_dtype = replica.pd_p2p_comm_dtype
        
        # ===== 2. Get KV Cache related dimensions =====
        # ===== 2. 获取KV Cache相关维度 =====
        # Use correct KV cache dims: num_kv_heads * attention_head_dim
        # 使用正确的KV cache维度: num_kv_heads * attention_head_dim
        # (NOT mlp_hidden_dim, which is MLP's dimension)
        # (而不是mlp_hidden_dim, 那是MLP的维度)
        num_kv_heads = replica.num_kv_heads
        head_dim = replica.attention_head_dim  # embedding_dim // num_q_heads
        num_layers = replica.num_layers
        
        # ===== 3. Calculate KV Cache size =====
        # ===== 3. 计算KV Cache大小 =====
        # Formula: 2(K+V) * num_tokens * num_kv_heads * head_dim * num_layers * bytes_per_element
        # 公式同上
        kv_cache_size = (
            2                    # K和V两个缓存
            * num_tokens         # token数量
            * num_kv_heads       # KV heads数量
            * head_dim           # 每个head的维度
            * num_layers         # 层数
            * bytes_per_element  # 每个元素的字节数
        )
        
        # ===== 4. Print debug info (first call only) =====
        # ===== 4. 打印调试信息 (首次调用时) =====
        if not hasattr(self, '_kv_cache_debug_printed'):
            logger.debug(f"[KV Cache] params: num_tokens={num_tokens}, num_kv_heads={num_kv_heads}, "
                        f"head_dim={head_dim}, num_layers={num_layers}, bytes={bytes_per_element}")
            logger.debug(f"[KV Cache] result: {kv_cache_size} bytes = {kv_cache_size/(1024**3):.4f} GB")
            self._kv_cache_debug_printed = True
        
        return kv_cache_size


    

# class GenerativeLLMRequest(Request):
    