from typing import Tuple

from vidur.entities.base_entity import BaseEntity
from vidur.logger import init_logger

# >
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
        
        # >: Add DAG property
        # self.dag: nx.DiGraph = field(default_factory=nx.DiGraph)
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
        
        # > add: Convenient for obtaining the replica corresponding to decode_replica_id through global_scheduler
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

        # if self._num_processed_tokens == self.total_tokens:
        #     print(f"> Debug: ")
        
        # print(f"> Debug: req on_batch_end Request {self._id} processed {num_tokens_processed} tokens, \
            # total processed {self._num_processed_tokens} tokens.")
        # print(f"> Debug: req on_batch_end num_processed_tokens={self._num_processed_tokens}, \
            # total_tokens={self.total_tokens} request_type={self.request_type}")
        # print(f"> Debug: req on_batch_end At time={time}, \
            # this request's self._completed_at={self._completed_at} self._completed={self._completed}")
        assert self._num_processed_tokens <= self.total_tokens


        # _num_processed_tokens = 0+2048
        # 2048+1
        if self._num_processed_tokens == self._num_prefill_tokens:
            self._is_prefill_complete = True
            
            # >
            self.request_type = RequestType.DECODE
            
            # we get one decode token when the prefill processing completes
            self._num_processed_tokens += 1
            # print(f"> Debug: self._num_processed_tokens += 1 \
                # Request {self._id} processed {num_tokens_processed} tokens, \
                # total processed {self._num_processed_tokens} tokens")


            # we must record the prefill completion time only in the first time
            # in the subsequent restarts, we keep adding the previously decoded
            # tokens to the prefill tokens - that is irrelevant to the original prefill
            if self._prefill_completed_at == 0:
                # > At this point it is absolute time,
                self._prefill_completed_at = time
        
        # Here; decode batching
        # elif self._num_processed_tokens == self._num_prefill_tokens:
        elif self._num_processed_tokens > self._num_prefill_tokens :
            
            # >
            assert self._is_prefill_complete == True, "> debug"
            assert self.request_type == RequestType.DECODE, "> debug"
            
            # we get one decode token when the prefill processing completes
            # self._num_processed_tokens += 1
            # print(f"> Debug: Request {self._id} at this point _num_processed_tokens > _num_prefill_tokens, \
                # total processed {self._num_processed_tokens} tokens")


        elif self._num_processed_tokens < self._num_prefill_tokens:
            # print(f"> Debug: Request {self._id} at this point _num_processed_tokens < _num_prefill_tokens, \
                # total processed {self._num_processed_tokens} tokens")
            pass
        
        # check if request is completed
        if self._num_processed_tokens == self.total_tokens:
            self._completed_at = time
            self._completed = True
            self.decode_time = self._completed_at - self.prefill_completed_at
            assert self.decode_time > 0 and self.decode_time < float("inf") , "> Debug: decode time error"
            # print(f"> Debug: At this point the request should end!!, \
                # Request {self._id} completed at {self._completed_at} ")
            
            
            logger.debug(f"Request {self._id} completed at {self._completed_at}")
            
            
        if self._num_processed_tokens >= self._num_prefill_tokens:
            # print(f"> Debug: request ID={self._id} self.decode_arrived_at={self.decode_arrived_at} self.request_type={self.request_type} self.prefill_completed_at={self.prefill_completed_at} self._is_prefill_complete={self._is_prefill_complete}")
            # assert self.decode_arrived_at < float("inf")  and self.request_type == RequestType.DECODE and self.prefill_completed_at > 0 and self._is_prefill_complete == True, "> debug"
            assert self.request_type == RequestType.DECODE and self.prefill_completed_at > 0 and self._is_prefill_complete == True, "> debug"

        

    def on_batch_stage_schedule(
        self,
        time: float,
    ) -> None:
        self._latest_stage_scheduled_at = time
        if self._latest_stage_completed_at == 0:
            self._preempted_time = 0
        else:
            # TODO > fy test each time
            # print(f"> Debug: request_id={self._id} time={time} self._latest_stage_completed_a={self._latest_stage_completed_at}")
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
    
    # >
    def create_task(self, task_type, **kwargs):
        """
        Creates a Task and adds it to the DAG.
        """
        
        # task = Task.from_type(task_type=task_type,
        #                       node_id=next(self.node_id),
        #                       request=self,
        #                       **kwargs)
        task = Task.from_type(task_type=task_type,
                              node_id=self.node_id,
                              request=self,
                              **kwargs)
        self.node_id += 1
        self.dag.add_node(task)
        self.nodes[task.node_id] = task
        # print(f"> self.dag={self.dag} self.nodes={self.nodes}")
        # print(f"> self.dag={self.dag} ")
        # import pdb; pdb.set_trace() # >
        return task
    
    def create_flow(self, flow_type, **kwargs):
        """
        Create a flow and add it to the DAG.
        """
        # flow = Flow.from_type(flow_type=flow_type,
        #                       node_id=next(self.node_id),  # Generate unique node ID
        #                       request=self,
        #                       **kwargs)  # Create flow based on flow type
        flow = Flow.from_type(flow_type=flow_type,
                              node_id=self.node_id,  # Generate unique node ID
                              request=self,
                              **kwargs)  # Create flow based on flow type
        self.node_id += 1
        self.dag.add_node(flow)  # Add flow to DAG
        self.nodes[flow.node_id] = flow  # Add flow to node dictionary
        return flow  # Return created flow
    
    # >
    def successors(self, node):
        """
        Returns the next Task or Flow to be executed after node.
        """
        return self.dag.successors(node)
    
    # estimate_kv_cache_size
    # def estimate_kv_cache_size(self, num_tokens=None, model=None):
    def estimate_kv_cache_size(self, num_tokens=None, replica=None):
        """
        返回生成num_tokens后的KV缓存大小。
        需要请求的根节点分配到某个实例上。
        Returns the KV-cache size after generating num_tokens
        Requires the Request root node to be allocated on an Instance.
        """
        # if num_tokens is None:  # If num_tokens is not specified
        #     num_tokens = self.generated_tokens  # Use the number of generated tokens
        # if model is None:  # If model is not specified
        #     # model = self.root_node.instance.model  # Use root node's model
        #     model = self.root_node.replica.model  # Use root node's model
                    
        # return 2 * self.batch_size * num_tokens * model.architecture.hidden_size \
        #         * model.architecture.num_layers * model.size.dtype_size  # Calculate KV cache size
        # return 2 * self.batch_size * num_tokens * replica.mlp_hidden_dim \
        #         * replica.num_layers * replica.size.dtype_size  # Calculate KV cache size
        # TODO  :p2p   > self.batch_size and replica.size.dtype_size from vidur
        # Point-to-point communication padding; Global parameters; Comm size/bandwidth; 
        # TODO Another version of ns3; Support writing a stream in config; For later
        
        if replica.pd_p2p_comm_dtype == 'float16':
            pd_p2p_bytes_per_token = 2
        elif replica.pd_p2p_comm_dtype == 'float32':
            pd_p2p_bytes_per_token = 4
        elif replica.pd_p2p_comm_dtype == 'float64':
            pd_p2p_bytes_per_token = 8
        elif replica.pd_p2p_comm_dtype == 'bfloat16':
            pd_p2p_bytes_per_token = 2
        elif replica.pd_p2p_comm_dtype == 'int8':
            pd_p2p_bytes_per_token = 1
        elif replica.pd_p2p_comm_dtype == 'int16':
            pd_p2p_bytes_per_token = 2
        elif replica.pd_p2p_comm_dtype == 'int32':
            pd_p2p_bytes_per_token = 4
        elif replica.pd_p2p_comm_dtype == 'int64':
            pd_p2p_bytes_per_token = 8

        self.pd_p2p_bytes_per_token = pd_p2p_bytes_per_token
        self.pd_p2p_comm_dtype = replica.pd_p2p_comm_dtype
        
        assert self.pd_p2p_bytes_per_token is not None and self.pd_p2p_comm_dtype is not None, "> Debug: PD P2P dtype is not set"
        
        # TODO : >: double check this
        return 2 * num_tokens * replica.mlp_hidden_dim \
                * replica.num_layers * pd_p2p_bytes_per_token  # Calculate KV cache size

        
        # return 2 * num_tokens * replica.mlp_hidden_dim \
        #         * replica.num_layers  # Calculate KV cache size


    

# class GenerativeLLMRequest(Request):
    