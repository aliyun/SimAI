from typing import List

from vidur.entities import Batch
from vidur.events import BaseEvent
from vidur.logger import init_logger
from vidur.metrics import MetricsStore
from vidur.scheduler import BaseGlobalScheduler
from vidur.types import EventType

# >
from vidur.entities.request import RequestType
from vidur.entities.replica import ReplicaType


logger = init_logger(__name__)


# 一个micro-batch在pipeline上执行结束
# A micro-batch execution ends in the pipeline
class BatchEndEvent(BaseEvent):
    def __init__(self, time: float, replica_id: int, batch: Batch):
        super().__init__(time, EventType.BATCH_END)

        self._replica_id = replica_id
        self._batch = batch

    def handle_event(
        self, scheduler: BaseGlobalScheduler, metrics_store: MetricsStore
    ) -> List[BaseEvent]:
        from vidur.events.replica_schedule_event import ReplicaScheduleEvent

        # > batch结束会触发下一个
        # > batch completion triggers the next one
        self._batch.on_batch_end(self.time)
        replica_scheduler = scheduler.get_replica_scheduler(self._replica_id)
        replica_scheduler.on_batch_end(self._batch)

        memory_usage_percent = replica_scheduler.memory_usage_percent
        metrics_store.on_batch_end(
            self.time, self._batch, self._replica_id, memory_usage_percent
        )

        
        
        logger.debug(f"time={self._time} Generates ReplicaScheduleEvent from event {self._id} {self._event_type}, "
            f"replica_id={self._replica_id}")
            
        # replica继续将下一个micro-batch加入pipeline
        # replica continues to add the next micro-batch to the pipeline
        
        # 获取全局调度器类型，判断是否为Splitwise调度策略
        # Get global scheduler type to determine if it's Splitwise scheduling policy
        events = [ReplicaScheduleEvent(self.time, self._replica_id)]
        
        # >: 之前是vidur原生的代码； 后面的pd分离是增加的处理； 没有pd分离则不会进入以下的路径
        # >: Previous code was native vidur; PD separation is added processing; Without PD separation, it won't enter the following path
       
        # Check if Splitwise scheduling policy is used
        # TODO(tianhao909): test if non-PD separation works normally
        # TODO(tianhao909): 测试非 PD 分离模式是否正常工作
        if hasattr(scheduler, '__class__') and scheduler.__class__.__name__ == 'SplitwiseGlobalScheduler':
            # 对于批次中的每个请求，检查是否需要转移到D副本
            # For each request in the batch, check if it needs to be transferred to D replica
            for request in self._batch.requests:
                #  fy： batch 类型： p batch； d batch； 在外面判断batch 里面所有的request type； 
                # fy: batch types: p batch; d batch; Determine all request types inside the batch from outside;
               
                # 判断是否纯p batch； 纯 d batch； 还是 pd req 混batch
                # Determine if it's pure p batch; pure d batch; or mixed pd req batch
              
                # 如果请求已完成prefill阶段
                # If the request has completed prefill stage
                if request.is_prefill_complete and request.request_type == RequestType.DECODE \
                    and replica_scheduler.replica.replica_type == ReplicaType.PREFILL:
                    # 修改请求类型为DECODE
                    # Modify request type to DECODE
                    # request.request_type = RequestType.DECODE

                    
                    # TODO(tianhao909): add P2P transmission bandwidth delay overhead here
                    # TODO(tianhao909): 在这里添加 P2P 传输带宽时延开销
                    # transfer_delay = calculate_p2p_transfer_delay(request)
                    # request.decode_arrived_at += transfer_delay
                    # transfer_delay = 1 # > assumption
                    # transfer_delay = 10 # > assumption
                    
                    # request.pd_p2p_comm_size = request.estimate_kv_cache_size()
                    assert request.num_processed_tokens == request.num_prefill_tokens + 1, \
                        "processed tokens must equal prefill tokens + 1 at this point"
                    request.pd_p2p_comm_size = request.estimate_kv_cache_size( request.num_processed_tokens, replica_scheduler.replica)

                    # replica_scheduler.replica
                    # replica_scheduler.replica.
                    # transfer_delay = request.pd_p2p_comm_size / (request.bandwidth - request.bandwidth_used)
                    # transfer_delay = request.pd_p2p_comm_size / request.bandwidth
                    
                    # TODO(tianhao909): determine bandwidth from topology with contention modeling
                    # TODO(tianhao909): bandwidth 应该从 topo 获取，并考虑竞争
                   
                    # request.pd_p2p_comm_bandwidth = 400*1024*1024*1024
                    request.pd_p2p_comm_bandwidth = replica_scheduler.replica.pd_p2p_comm_bandwidth*1024*1024*1024/8
                    assert request.pd_p2p_comm_size < float('inf') and request.pd_p2p_comm_size > 0 and request.pd_p2p_comm_bandwidth > 0, \
                        "P2P communication size and bandwidth must be valid"
                    request.pd_p2p_comm_time = request.pd_p2p_comm_size / request.pd_p2p_comm_bandwidth
                    
                    
                    # 设置decode阶段的到达时间为prefill完成时间
                    # Set decode stage arrival time to prefill completion time
                    request.decode_arrived_at = request.prefill_completed_at + request.pd_p2p_comm_time
                    
                    # 从P副本中删除请求
                    # Remove request from P replica
                    
                    # TODO(tianhao909): write small-token test cases for memory logic validation
                    # TODO(tianhao909): 写两个 req p 和 d 的 token 数目都很少；测试内存判断的逻辑对不对
                  
                    # > 隐患 replica 清除 req时候， 对应的内存块也要清除
                    # > risk: When replica clears requests, corresponding memory blocks should also be cleared
                    p_replica_scheduler = replica_scheduler
                    if request in p_replica_scheduler.replica.pending_requests:
                        # 在移除请求之前，先计算当前的kvcache使用情况
                        # print(f"> 在移除请求 {request.id} 之前:")
                        # p_replica_scheduler.replica.get_remaining_kv_cache_capacity()
                        
                        # 移除请求
                        p_replica_scheduler.replica.pending_requests.remove(request)
                        
                        # 移除请求后释放相应的显存
                        # p_replica_scheduler.replica.release_request_kv_cache_memory(request)
                        # print(f"> 请求 {request.id} 已从Prefill副本移除并释放显存")
                        # p_replica_scheduler.replica.get_remaining_kv_cache_capacity()
                        
                    # TODO(tianhao909): ensure corresponding storage is also cleared
                    # TODO(tianhao909): 确保对应的存储也清空了
                    
                    # 将请求添加到D副本，获取对应的D副本并添加请求
                    # Add request to D replica, get corresponding D replica and add request
                  
                    d_replica_scheduler = scheduler.get_replica_scheduler(request.decode_replica_id)
                    # d_replica.pending_requests.append(request)
                    
                    # 生成D副本的调度事件
                    # Generate D replica scheduling event
                    events.append(ReplicaScheduleEvent(request.decode_arrived_at, request.decode_replica_id))
                    
                    logger.debug(f"pd d-path time={self._time} Generates ReplicaScheduleEvent from event {self._id} {self._event_type}, "
                        f"decode_replica_id={request.decode_replica_id} len(events)={len(events)}")
        
                
                if request._num_processed_tokens >= request._num_prefill_tokens:            
                    # print(f"> self.decode_arrived_at={self.decode_arrived_at} self.request_type={self.request_type} self.prefill_completed_at={self.prefill_completed_at} self._is_prefill_complete={self._is_prefill_complete}")
                    assert request.decode_arrived_at < float("inf") and request.request_type == RequestType.DECODE and request.prefill_completed_at > 0 and request._is_prefill_complete == True, \
                        "post-prefill request must have valid decode_arrived_at and be in DECODE state"

        # Call memory info logging function (disabled)
        # 调用显存信息日志函数（已禁用）
        # self._log_memory_info(scheduler)
                        
        return events

    def _log_memory_info(self, scheduler: BaseGlobalScheduler) -> None:
        """
        Get and print memory capacity info for prefill and decode replicas.
        获取并打印prefill和decode副本的各种显存容量信息
        """
        # Get all replicas from scheduler
        # 获取scheduler中所有的replica
        # Use scheduler._replica_schedulers to get all replica IDs
        # 使用scheduler的_replica_schedulers属性获取所有副本ID
        replica_ids = list(scheduler._replica_schedulers.keys())
        
        # Separate prefill and decode replica info
        # 分别记录prefill和decode副本的信息
        prefill_replica_info = {}
        decode_replica_info = {}
        
        for replica_id in replica_ids:
            replica_scheduler = scheduler.get_replica_scheduler(replica_id)
            replica = replica_scheduler.replica
            
            # Get TP and PP parameters / 获取TP和PP参数
            tensor_parallel_size = replica._replica_config.tensor_parallel_size
            pipeline_parallel_size = replica._replica_config.num_pipeline_stages
            
            # Create param_counter from replica_config
            # 从replica_config创建param_counter
            param_counter = replica._replica_config._param_counter if hasattr(replica._replica_config, '_param_counter') else None
            if param_counter is None:
                # If replica has no _param_counter, create from replica_config
                # 如果replica本身没有_param_counter，尝试从replica_config创建
                from vidur.utils.param_counter import ParamCounter
                param_counter = ParamCounter(replica._replica_config)
            
            # Get model params memory usage / 获取模型参数占用的显存
            total_params = param_counter.get_num_parameters_per_device()
            # Convert bytes to GB / 将bytes转换为GB
            total_params_gb = total_params / (1024**3)
            
            # Create memory_planner from replica_config and replica
            # 从replica_config和replica创建memory_planner
            from vidur.scheduler.utils.memory_planner import MemoryPlanner
            memory_planner = MemoryPlanner(replica._replica_config, replica)
            
            # Get reserved KV cache memory / 获取kvcache预留的显存
            max_batch_size = memory_planner.get_max_batch_size()
            kv_cache_per_request = memory_planner._get_kv_cache_memory_per_device_per_request()
            memory_for_kv_cache = kv_cache_per_request * max_batch_size
            
            # Convert bytes to GB / 将bytes转换为GB
            memory_for_kv_cache_gb = memory_for_kv_cache / (1024**3)
            
            # Get actual running requests' KV cache memory
            # Note: Replica has no running_requests attr, so we only count pending_requests
            # 获取实际运行的request的kvcache显存容量
            # 注意：Replica对象没有running_requests属性，只统计pending_requests
            pending_requests_count = len(replica.pending_requests)
            actual_kv_cache_memory = kv_cache_per_request * pending_requests_count
            actual_kv_cache_memory_gb = actual_kv_cache_memory / (1024**3)
            
            # Compute whole-replica values (per-GPU * TP * PP)
            # 计算整个replica的值（单GPU值 × TP × PP）
            total_memory_replica_gb = replica.total_memory_gb * tensor_parallel_size * pipeline_parallel_size
            params_memory_replica_gb = total_params_gb * tensor_parallel_size * pipeline_parallel_size
            reserved_kv_cache_memory_replica_gb = memory_for_kv_cache_gb * tensor_parallel_size * pipeline_parallel_size
            actual_running_kv_cache_memory_replica_gb = actual_kv_cache_memory_gb * tensor_parallel_size * pipeline_parallel_size
            
            # Store info / 存储信息
            replica_info = {
                'total_memory_gb': replica.total_memory_gb,
                'total_memory_replica_gb': total_memory_replica_gb,
                'params_memory_gb': total_params_gb,
                'params_memory_replica_gb': params_memory_replica_gb,
                'reserved_kv_cache_memory_gb': memory_for_kv_cache_gb,
                'reserved_kv_cache_memory_replica_gb': reserved_kv_cache_memory_replica_gb,
                'actual_running_kv_cache_memory_gb': actual_kv_cache_memory_gb,
                'actual_running_kv_cache_memory_replica_gb': actual_running_kv_cache_memory_replica_gb,
                'active_requests_count': pending_requests_count,
                'max_batch_size': max_batch_size,
                'tp': tensor_parallel_size,
                'pp': pipeline_parallel_size
            }
            
            if replica.replica_type == ReplicaType.PREFILL:
                prefill_replica_info[replica_id] = replica_info
            elif replica.replica_type == ReplicaType.DECODE:
                decode_replica_info[replica_id] = replica_info
        
        # 打印信息 | Print memory info
        logger.info("=" * 100)
        logger.info("Memory Usage Statistics (GB) (显存使用情况统计):")
        logger.info("-" * 100)
        
        if prefill_replica_info:
            logger.info("Prefill Replica Memory Info (Prefill副本显存信息):")
            for pid, info in prefill_replica_info.items():
                logger.info(f"  Replica ID {pid} (TP={info['tp']}, PP={info['pp']}):")
                logger.info(f"    Per-GPU total mem (单GPU总显存容量): {info['total_memory_gb']:.2f} GB ({info['total_memory_gb']*1024:.2f} MB)")
                logger.info(f"    Replica total mem (整个Replica总显存容量): {info['total_memory_replica_gb']:.2f} GB ({info['total_memory_replica_gb']*1024:.2f} MB)")
                logger.info(f"    Per-GPU model params (单GPU模型参数占用显存): {info['params_memory_gb']:.2f} GB ({info['params_memory_gb']*1024:.2f} MB)")
                logger.info(f"    Replica model params (整个Replica模型参数占用显存): {info['params_memory_replica_gb']:.2f} GB ({info['params_memory_replica_gb']*1024:.2f} MB)")
                logger.info(f"    Per-GPU reserved KV cache (单GPU预留kvcache显存): {info['reserved_kv_cache_memory_gb']:.2f} GB ({info['reserved_kv_cache_memory_gb']*1024:.2f} MB)")
                logger.info(f"    Replica reserved KV cache (整个Replica预留kvcache显存): {info['reserved_kv_cache_memory_replica_gb']:.2f} GB ({info['reserved_kv_cache_memory_replica_gb']*1024:.2f} MB)")
                logger.info(f"    Per-GPU actual KV cache (单GPU实际kvcache显存): {info['actual_running_kv_cache_memory_gb']:.2f} GB ({info['actual_running_kv_cache_memory_gb']*1024:.2f} MB)")
                logger.info(f"    Replica actual KV cache (整个Replica实际kvcache显存): {info['actual_running_kv_cache_memory_replica_gb']:.2f} GB ({info['actual_running_kv_cache_memory_replica_gb']*1024:.2f} MB)")
                logger.info(f"    Active requests (当前活跃请求数): {info['active_requests_count']}")
                logger.info(f"    Max batch size (最大批处理大小): {info['max_batch_size']}")
            logger.info("-" * 100)
        
        if decode_replica_info:
            logger.info("Decode Replica Memory Info (Decode副本显存信息):")
            for did, info in decode_replica_info.items():
                logger.info(f"  Replica ID {did} (TP={info['tp']}, PP={info['pp']}):")
                logger.info(f"    Per-GPU total mem (单GPU总显存容量): {info['total_memory_gb']:.2f} GB ({info['total_memory_gb']*1024:.2f} MB)")
                logger.info(f"    Replica total mem (整个Replica总显存容量): {info['total_memory_replica_gb']:.2f} GB ({info['total_memory_replica_gb']*1024:.2f} MB)")
                logger.info(f"    Per-GPU model params (单GPU模型参数占用显存): {info['params_memory_gb']:.2f} GB ({info['params_memory_gb']*1024:.2f} MB)")
                logger.info(f"    Replica model params (整个Replica模型参数占用显存): {info['params_memory_replica_gb']:.2f} GB ({info['params_memory_replica_gb']*1024:.2f} MB)")
                logger.info(f"    Per-GPU reserved KV cache (单GPU预留kvcache显存): {info['reserved_kv_cache_memory_gb']:.2f} GB ({info['reserved_kv_cache_memory_gb']*1024:.2f} MB)")
                logger.info(f"    Replica reserved KV cache (整个Replica预留kvcache显存): {info['reserved_kv_cache_memory_replica_gb']:.2f} GB ({info['reserved_kv_cache_memory_replica_gb']*1024:.2f} MB)")
                logger.info(f"    Per-GPU actual KV cache (单GPU实际kvcache显存): {info['actual_running_kv_cache_memory_gb']:.2f} GB ({info['actual_running_kv_cache_memory_gb']*1024:.2f} MB)")
                logger.info(f"    Replica actual KV cache (整个Replica实际kvcache显存): {info['actual_running_kv_cache_memory_replica_gb']:.2f} GB ({info['actual_running_kv_cache_memory_replica_gb']*1024:.2f} MB)")
                logger.info(f"    Active requests (当前活跃请求数): {info['active_requests_count']}")
                logger.info(f"    Max batch size (最大批处理大小): {info['max_batch_size']}")
            logger.info("-" * 100)
        logger.info("=" * 100)


    def to_dict(self) -> dict:
        return {
            "time": self.time,
            "event_type": self.event_type,
            "batch_id": self._batch.id,
        }