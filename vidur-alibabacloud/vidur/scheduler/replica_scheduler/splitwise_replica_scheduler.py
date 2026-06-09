from abc import ABC, abstractmethod  # Import ABC and abstractmethod decorator to define abstract methods that must be implemented by subclasses
from typing import List  # Import List type hint for specifying list-type variables or return values


# from vidur.config import Config  # Import configuration class for getting system configuration parameters

from vidur.config import (  
    BaseReplicaSchedulerConfig,  
    BaseRequestGeneratorConfig,  
    ReplicaConfig,  
)

from vidur.config import SimulationConfig  # Import simulation configuration class

from vidur.entities import Batch, Replica, Request  
from vidur.execution_time_predictor import BaseExecutionTimePredictor 
from vidur.logger import init_logger  
from vidur.scheduler.replica_stage_scheduler import ReplicaStageScheduler  
from vidur.scheduler.utils.memory_planner import MemoryPlanner 
from vidur.scheduler.replica_scheduler.base_replica_scheduler import BaseReplicaScheduler  

from collections import defaultdict
import sys
from vidur.entities.node import NodeState, Node
from vidur.entities.task import TaskType, Task
from vidur.entities.replica import Replica, ReplicaType
from vidur.entities.request import Request, RequestType
from math import ceil  # Import ceil function from math module for rounding up

logger = init_logger(__name__)  # Initialize logger for current module to output log information

class SplitwiseReplicaScheduler(BaseReplicaScheduler):  # Define SplitwiseReplicaScheduler class inheriting from BaseReplicaScheduler
    

    def __init__(
        self,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        request_generator_config: BaseRequestGeneratorConfig,
        replica: Replica,
        num_stages: int,
        execution_time_predictor: BaseExecutionTimePredictor,
    ) -> None:  # Initialization method
        
        
        # Call parent class initialization method
        super().__init__(
            replica_config,
            replica_scheduler_config,
            request_generator_config,
            replica,
            num_stages,
            execution_time_predictor
        )
        
        # >
        self.replica = replica
        self.scheduled_batches = []
        self.batch = None
        
        # self._config = config  # Save configuration object as private instance attribute
        self._replica_id = replica.id  # Save replica ID as private instance attribute

        # self.model = model  # Model object being used
        self.model = replica_config.model_config
        # self.processors = processors  # Processor list assigned to this instance
        self.processors = replica_config.device_config
        # self.overheads = overheads  # Overhead information during instance runtime
        # self.debug = debug  # Whether debugging mode is enabled

        ## Other instance metadata
        # self.metrics = InstanceMetrics()  # Instance performance metrics recorder
        self.servers = set() # Store set of servers used by this instance
        processors = self.processors    
        self.completion_events = {}  # Dictionary storing completion events

        ## Task queues
        self.pending_queue = []  # Pending task queue
        self.completed_queue = []  # Completed task queue
        self.blocked_queue = []  # Blocked task queue
        self.batch = []  # Current executing task batch




        # TODO(tianhao909): delete unused prompt/token task tracking code
        # TODO(tianhao909): 删除未使用的 prompt/token task 跟踪代码
        self.prompt_tasks_in_batch = []
        self.token_tasks_in_batch = []

        ## token 级别的跟踪元数据
        ## Token-level tracking metadata
        self.pending_tokens = 0  # Number of pending tokens
        self.batch_tokens = 0  # Total tokens in current batch
        
        # ORCAInstance 没有最大 batch tokens 限制
        # ORCAInstance has no max_batch_tokens limit
        self.max_batch_tokens = sys.maxsize

        # 连续迭代相关元数据 
        # contiguous iterations metadata
        self.iteration_duration = 0.  # Duration of single iteration
        self.num_contiguous_iterations = 0  # Number of contiguous iterations
        self.pause_next_iteration = False  # Whether to pause next iteration

        # 队列管理
        # Queue management
        
        # 按到达时间排序的待处理请求列表 
        # pending requests (not tasks) ordered by arrival time
        
        # TODO(tianhao909): evaluate if pending_requests can be removed
        # TODO(tianhao909): 看看需不需要删除 pending_requests
        self.pending_requests = []
        
        # 专门用于提示任务的待处理队列（优先处理提示） 
        # separate pending queue for prompt tasks (to prioritize prompts)
        
        # TODO(tianhao909): remove redundant prompt queue
        # TODO(tianhao909): 删除冗余的 prompt 队列
        self.pending_prompt_queue = []
        
        # 请求到任务的映射关系 
        # map requests->tasks on this instance
        self.request_tasks = {}

        
        # > add vllm sche
        self._preempted_requests: List[Request] = []  # Store list of preempted requests
        self._num_running_batches = 0  # Current number of running batches
        
        # 对于 vLLM 及其衍生版本，我们只需要设置一个宽松的最大批大小
        # 内存需求由调度器显式管理
        # For vLLM and its derivatives, we only need to set a loose max batch size
        # Memory requirements are explicitly managed by scheduler
        
        self._max_micro_batch_size = self._config.batch_size_cap // self._num_stages  # Calculate maximum micro-batch size
        self._watermark_blocks = int(  # Calculate memory watermark block count
            self._config.watermark_blocks_fraction * self._config.num_blocks  # Multiply total blocks by coefficient
        )
        
    # def get_replica_scheduler(self, config: Config, replica: Replica, num_stages: int, execution_time_predictor: BaseExecutionTimePredictor):  # Used to get specific replica scheduler
    def get_replica_scheduler(self, config: SimulationConfig, replica: Replica, num_stages: int, execution_time_predictor: BaseExecutionTimePredictor):
        from vidur.scheduler.replica_scheduler.replica_scheduler_registry import ReplicaSchedulerRegistry  # Import replica scheduler registry
        # Get corresponding replica scheduler instance from registry based on configured scheduler provider name and return
        return ReplicaSchedulerRegistry.get_from_str(config.replica_scheduler_provider, config, replica, num_stages, execution_time_predictor)

    def _can_allocate_request(self, request: Request) -> bool:  # Check if resources can be allocated for request
        if request.id not in self._allocation_map:  # If this is a new request (ID not in allocation map)
            # New request
            num_required_blocks = ceil(  # Calculate required blocks for this request, round up
                (request.num_prefill_tokens) / self._config.block_size
            )
            return (  # Check if remaining blocks meet watermark requirement
                self._config.num_blocks
                - self._num_allocated_blocks
                - num_required_blocks
                >= self._watermark_blocks
            )

        # vllm 至少需要一个可用块才能继续执行
        # vllm needs at least one available block to continue execution
        return self._config.num_blocks - self._num_allocated_blocks >= 1  # For existing requests, remaining blocks must be ≥ 1

    # 分配内存块block给req
    # Allocate memory blocks to request
    def _allocate_request(self, request: Request) -> None:  
        if request.id not in self._allocation_map:  # If this is a new request
            # New request
            num_required_blocks = ceil(  # Calculate required blocks
                (request.num_prefill_tokens) / self._config.block_size
            )
            self.allocate(request.id, num_required_blocks)  # Call allocate method to assign blocks
            return  
        
        # 出现2的情况 num_processed_tokens = 2048 +1 （p 结束的时候+1） + 1 （组batch+1）
        # Case where 2 appears: num_processed_tokens = 2048 +1 (added when p completes) + 1 (added when batching)
        num_tokens_reserved = self._allocation_map[request.id] * self._config.block_size  # Allocated token capacity
        num_tokens_required = max(0, request.num_processed_tokens - num_tokens_reserved)  # Remaining tokens needed
        # print(f"> Debug: in _allocate_request: req id={request._id} request._num_processed_tokens={request._num_processed_tokens} num_tokens_required={num_tokens_required} num_tokens_reserved={num_tokens_reserved} num_tokens_required={num_tokens_required}")
        # print(f"> Debug: self._allocation_map[request.id]={self._allocation_map[request.id]} self._config.block_size={self._config.block_size}")
        
        # 要么不需要额外 token，要么只需要 1 个
        # Either no extra tokens needed, or only 1 needed
        assert (  
            num_tokens_required == 0 or num_tokens_required == 1
        ), f"num_tokens_required: {num_tokens_required}"
     

        if num_tokens_required == 0:  # If no additional allocation needed
            return

        # =1 没有向上取整； 这个block 满了；
        # =1 without rounding up; this block is full;
        
        # 16； 32； 64；  
        # 2048 / 16 = 128 
        self.allocate(request.id, 1)  # Allocate one additional memory block

    
    # fth 260122 这边释放资源； 进入decode的逻辑 写在handle_event里面，用于生成新的event， 这边不生成event； 
    def on_batch_end(self, batch: Batch) -> None:  # Called when a batch finishes execution
        self._num_running_batches -= 1  # Decrement running batch count
        
        # 判断是否是pd 分离
        # Check if PD separation is enabled
        if self.replica.replica_type == ReplicaType.MIXED:
            assert False, "PD separation doesn't support mixed mode yet, must be separated"
            pass
        elif self.replica.replica_type == ReplicaType.PREFILL:
            for request in batch.requests:
                if request.completed:
                    self.free(request.id)
                    

                    # 在移除请求之前，先计算当前的kvcache使用情况
                    logger.debug(f"Before removing request {request.id}:")
                    self.replica.get_remaining_kv_cache_capacity()
                    # 移除请求 留给batch_end_event.py去做
                    # self.replica.pending_requests.remove(request)
                    # 移除请求后释放相应的显存
                    self.replica.release_request_kv_cache_memory(request)
                    logger.debug(f"Request {request.id} removed from replica and GPU memory released")
                    self.replica.get_remaining_kv_cache_capacity()
                    
                elif request.is_prefill_complete == True:
                    # 通过 request 找到对应 decode replica；
                    # Find corresponding decode replica through request;
                    self.free(request.id)
                    
                    # 在移除请求之前，先计算当前的kvcache使用情况
                    logger.debug(f"Before removing request {request.id}:")
                    self.replica.get_remaining_kv_cache_capacity()
                    # 移除请求 留给batch_end_event.py去做
                    # self.replica.pending_requests.remove(request)
                    # 移除请求后释放相应的显存
                    self.replica.release_request_kv_cache_memory(request)
                    logger.debug(f"Request {request.id} removed from replica and GPU memory released")
                    self.replica.get_remaining_kv_cache_capacity()
                    
                    d_replica_scheduler = request.global_scheduler.get_replica_scheduler(request.decode_replica_id)
                    # d_replica_scheduler._preempted_requests.append(request)
                    d_replica_scheduler._request_queue.append(request)
                elif request.is_prefill_complete == False:
                    self._preempted_requests.append(request)  
        elif self.replica.replica_type == ReplicaType.DECODE:
            for request in batch.requests:
                if request.completed:
                    # vllm 和 sarathi的 free方法 和 orca的 free 方法不同
                    # vllm and sarathi free methods differ from orca's free method
                    self.free(request.id)
                    
                    # 在移除请求之前，先计算当前的kvcache使用情况
                    logger.debug(f"Before removing request {request.id}:")
                    self.replica.get_remaining_kv_cache_capacity()
                    # 移除请求 留给batch_end_event.py去做
                    # self.replica.pending_requests.remove(request)
                    # 移除请求后释放相应的显存
                    self.replica.release_request_kv_cache_memory(request)
                    logger.debug(f"Request {request.id} removed from replica and GPU memory released")
                    self.replica.get_remaining_kv_cache_capacity()
                    
                    
                    
                elif request.is_prefill_complete == True:
                    self._preempted_requests.append(request)
                elif request.is_prefill_complete == False:
                    assert request.decode_arrived_at == float("inf"), "decode_arrived_at must be infinity for incomplete prefill"
                    
        


    # Implement get next batch using orca approach.
    def _get_next_batch(self) -> Batch:
        
        """
        选择要运行的任务批次。
        保留现有任务，并从请求池中添加新任务到批次中。
        返回：被抢占的任务列表、新增任务列表
        
        Select a batch of tasks to run.
        Keep existing tasks and add new tasks from the request pool to the batch.
        Return: List of preempted tasks, List of new tasks
        """
        
        requests = []  # Store requests to be processed in this batch
        num_tokens = []  # Store token counts for corresponding requests
        num_batch_tokens = 0  # Total tokens in current batch
        
        if self.replica.replica_type == ReplicaType.MIXED:
            pass
        elif self.replica.replica_type == ReplicaType.PREFILL:
            # req弹出； 原版 req 弹出 并没有塞回来，
            # Request popping; original request popping didn't put them back, 
            # batch； 
            tmp_requests_to_remove = list() # Record requests to be removed from queue
            
            # 对于batch end 加回来的请求（默认之前的放得下）# （没完成） # 因此 batch end 不能把完成p的request 放回p replica； 但可以放到 d replica中； 不过目前逻辑不需要放入d replica中
            # For requests added back by batch end (assuming previous ones fit) # (not completed) # Therefore batch end cannot put completed p requests back to p replica; but can put them in d replica; however current logic doesn't require putting them in d replica
            while self._preempted_requests:  # Iterate through preempted requests
                if len(requests) == self._max_batch_size:  # If reached maximum batch size, break loop
                    break

                request = self._preempted_requests.pop(0)  # Take first preempted request
                next_num_tokens = self._get_request_next_num_tokens(request)  # Get next token count needed by this request
                requests.append(request)  # Add to request list
                num_tokens.append(next_num_tokens)  # Record token count
            
            # TODO(tianhao909): implement GPU memory pool with space validation for KV cache transfer
            # TODO(tianhao909): 实现显存池，判断空间是否足够再传递 KV cache
            # TODO(tianhao909): handle extreme cases where kvcache queued at decode side before p2p transfer
            
            
            # For unprocessed requests;
            for request in self._request_queue:
                if request.request_type == RequestType.PREFILL and request.is_prefill_complete == False:
                    
                    # 组batch + 判断到达时间
                    # Form batch + Check arrival time

                    next_num_tokens = self._get_request_next_num_tokens(request)  # Get next token count needed by this request
                    assert next_num_tokens == request.num_prefill_tokens
                    if num_batch_tokens + next_num_tokens > self._config.max_tokens_in_batch:  # If total batch tokens plus current request tokens exceed limit
                        break

                    if len(self._allocation_map) == self._config.batch_size_cap: # If allocation map size reaches batch capacity limit
                        break

                    if len(requests) == self._max_micro_batch_size:  # If request list size reaches maximum micro-batch size
                        break
                    
                    # vllm sarathi method
                    # if not self._can_allocate_request(request):  # If request cannot be allocated
                    #     break
                    
                    # orca method
                    if not self.can_allocate(self._max_blocks_per_sequence):
                        break
                    
                    # pop(0) cannot be written inside queue iteration loop, write after iteration completes
    
                    # request = self._request_queue.pop(0)  # Remove and get first request from request queue

                    # vllm and sarathi allocation approach in vidur:
                    # self._allocate_request(request)  # Allocate request resources
                    
                    # orca allocation approach in vidur: allocate maximum blocks for request
                    self.allocate(request.id, self._max_blocks_per_sequence)
                    # self.replica.allocate_request_kv_cache_memory(request, self._max_blocks_per_sequence)
                    # Pass block_size to correctly convert blocks to tokens
                    # 传入 block_size，正确将 blocks 转为 tokens
                    self.replica.allocate_request_kv_cache_memory(
                        request, self._max_blocks_per_sequence, self._config.block_size)
                    
                    
                    requests.append(request)  # Add request to request list
                    tmp_requests_to_remove.append(request)
                    # Determine token count; prefill tokens
                    num_tokens.append(next_num_tokens)  # Add token count to token count list
                    num_batch_tokens += next_num_tokens  # Update total batch tokens
                     
                elif request.request_type == RequestType.DECODE:
                    continue 
            if not requests:
                return
            else:
                
                # 遍历完成后，从_request_queue中移除已处理的请求
                # After iteration completes, remove processed requests from _request_queue
                for request in tmp_requests_to_remove:
                    self._request_queue.remove(request)
                    
                
                return Batch(self._replica_id, requests, num_tokens)  # Create and return Batch object
        elif self.replica.replica_type == ReplicaType.DECODE:
            tmp_requests_to_remove = list()
            
            # 对于batch end 加回来的请求（默认之前的放得下）# （没完成） # 因此 batch end 不能把完成p的request 放回p replica； 但可以放到 d replica中； 不过目前逻辑不需要放入d replica中
            # For requests added back by batch end (assuming previous ones fit) # (not completed) # Therefore batch end cannot put completed p requests back to p replica; but can put them in d replica; however current logic doesn't require putting them in d replica
            while self._preempted_requests:  # Iterate through preempted requests
                if len(requests) == self._max_batch_size:  # If reached maximum batch size, break loop
                    break

                request = self._preempted_requests.pop(0)  # Take first preempted request
                next_num_tokens = self._get_request_next_num_tokens(request)  # Get next token count needed by this request
                requests.append(request)  # Add to request list
                num_tokens.append(next_num_tokens)  # Record token count
            
            for request in self._request_queue:
                
                if request.request_type == RequestType.PREFILL:
                    continue
                elif request.request_type == RequestType.DECODE:
                    # 判断time ； 和 是否已完成prefill； 如果已完成 才能组batch； 不然直接返回
                    # Check time and whether prefill is complete; can only form batch if complete; otherwise return directly
                    
                    if request.is_prefill_complete == True:
                        # 组batch + 判断到达时间
                        # Form batch + Check arrival time
                        assert request.decode_arrived_at != float('inf'), "decode_arrived_at must be set before decode batching"
                        
                        # if request.decode_arrived_at == float('inf'):
                        #     continue
                        
                        # vllm sarathi orca all use this to get next token count needed by request
                        next_num_tokens = self._get_request_next_num_tokens(request)  
                        
                        # decode next_num_tokens can only be 1
                        assert next_num_tokens == 1, "decode next_num_tokens must be 1"
                        
                        # 如果批处理token总数加上当前请求token数超过最大限制
                        # If total batch tokens plus current request tokens exceed limit
                        if num_batch_tokens + next_num_tokens > self._config.max_tokens_in_batch: 
                            logger.debug("break: num_batch_tokens + next_num_tokens > max_tokens_in_batch") 
                            break
                        
                        # sarathi、vllm的方法：如果分配映射大小达到批处理容量上限
                        # sarathi, vllm method: if allocation map size reaches batch capacity limit
                        # if len(self._allocation_map) == self._config.batch_size_cap:  
                        #     break
                        
                        # 如果请求列表大小达到最大微批处理大小
                        # If request list size reaches maximum micro-batch size
                        if len(requests) == self._max_micro_batch_size:  
                            break
                        
                        # vllm, sarathi method: if request cannot be allocated
                        # if not self._can_allocate_request(request):  
                        #     break
                        
                        # orca method
                        if not self.can_allocate(self._max_blocks_per_sequence):
                            break
                        
                        # sarathi and vllm method
                        # self._allocate_request(request)  # Allocate request resources
                        
                        # orca的方法： 为请求分配最大块数的资源
                        # orca method: allocate maximum blocks for request
                        self.allocate(request.id, self._max_blocks_per_sequence)
                        # Pass block_size to correctly convert blocks to tokens
                        # 传入 block_size，正确将 blocks 转为 tokens
                        self.replica.allocate_request_kv_cache_memory(
                            request, self._max_blocks_per_sequence, self._config.block_size)
                        
                        # self.replica.allocate_request_kv_cache_memory(request, self._max_blocks_per_sequence)
                        
                        requests.append(request)  # Add request to request list
                        tmp_requests_to_remove.append(request)
                        # Determine token count; prefill tokens
                        num_tokens.append(next_num_tokens)  # Add token count to token count list
                        num_batch_tokens += next_num_tokens  # Update total batch tokens
                        
                    else:
                        continue
            
            if requests:
                # After iteration completes, remove processed requests from _request_queue
                # 遍历完成后，从_request_queue中移除已处理的请求
                for request in tmp_requests_to_remove:
                    self._request_queue.remove(request)
                return Batch(self._replica_id, requests, num_tokens)  # Create and return Batch object
            
            if not requests:
                return