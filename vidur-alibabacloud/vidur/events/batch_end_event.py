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

        
        
        print(f"> Debug: time={self._time} Generates ReplicaScheduleEvent from event {self._id} {self._event_type}, \
            replica_id={self._replica_id}")
            
        # replica继续将下一个micro-batch加入pipeline
        # replica continues to add the next micro-batch to the pipeline
        
        # 获取全局调度器类型，判断是否为Splitwise调度策略
        # Get global scheduler type to determine if it's Splitwise scheduling policy
        events = [ReplicaScheduleEvent(self.time, self._replica_id)]
        
        # >: 之前是vidur原生的代码； 后面的pd分离是增加的处理； 没有pd分离则不会进入以下的路径
        # >: Previous code was native vidur; PD separation is added processing; Without PD separation, it won't enter the following path
       
        # Check if Splitwise scheduling policy is used
        # TODO 250911 > test if non-PD separation works normally
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

                    
                    # TODO: > 在这里添加P2P传输带宽时延开销
                    # TODO: > Add P2P transmission bandwidth delay overhead here
                    # transfer_delay = calculate_p2p_transfer_delay(request)
                    # request.decode_arrived_at += transfer_delay
                    # transfer_delay = 1 # > assumption
                    # transfer_delay = 10 # > assumption
                    
                    # request.pd_p2p_comm_size = request.estimate_kv_cache_size()
                    assert request.num_processed_tokens == request.num_prefill_tokens + 1 , "> debug"
                    request.pd_p2p_comm_size = request.estimate_kv_cache_size( request.num_processed_tokens, replica_scheduler.replica)

                    # replica_scheduler.replica
                    # replica_scheduler.replica.
                    # transfer_delay = request.pd_p2p_comm_size / (request.bandwidth - request.bandwidth_used)
                    # transfer_delay = request.pd_p2p_comm_size / request.bandwidth
                    
                    # TODO >: request.bandwidth 具体怎么赋值， 怎么传，应该是个topo； 或者考虑竞争？
                    # TODO >: How exactly is request.bandwidth assigned and passed, should be a topology; Or consider contention?
                   
                    # request.pd_p2p_comm_bandwidth = 400*1024*1024*1024
                    request.pd_p2p_comm_bandwidth = replica_scheduler.replica.pd_p2p_comm_bandwidth*1024*1024*1024/8
                    assert request.pd_p2p_comm_size < float('inf') and request.pd_p2p_comm_size > 0 and request.pd_p2p_comm_bandwidth > 0 , "> debug"
                    request.pd_p2p_comm_time = request.pd_p2p_comm_size / request.pd_p2p_comm_bandwidth
                    
                    
                    # 设置decode阶段的到达时间为prefill完成时间
                    # Set decode stage arrival time to prefill completion time
                    request.decode_arrived_at = request.prefill_completed_at + request.pd_p2p_comm_time
                    
                    # 从P副本中删除请求
                    # Remove request from P replica
                    
                    # TODO: > 250911 写两个req p 和 d的token数目都很少； 测试内存判断的逻辑对不对；整体等逻辑对不对
                    # TODO: > 250911 Write two requests with few tokens for both p and d; Test if memory judgment logic is correct; Overall logic correctness
                  
                    # > 隐患 replica 清除 req时候， 对应的内存块也要清除
                    # > risk: When replica clears requests, corresponding memory blocks should also be cleared
                    p_replica_scheduler = replica_scheduler
                    if request in p_replica_scheduler.replica.pending_requests:
                        p_replica_scheduler.replica.pending_requests.remove(request)
                        
                    # TODO：> 确保对应的存储也清空了
                    # TODO: > Ensure corresponding storage is also cleared
                    
                    # 将请求添加到D副本，获取对应的D副本并添加请求
                    # Add request to D replica, get corresponding D replica and add request
                  
                    d_replica_scheduler = scheduler.get_replica_scheduler(request.decode_replica_id)
                    # d_replica.pending_requests.append(request)
                    
                    # 生成D副本的调度事件
                    # Generate D replica scheduling event
                    events.append(ReplicaScheduleEvent(request.decode_arrived_at, request.decode_replica_id))
                    
                    print(f"> Debug: pd d-path time={self._time} Generates ReplicaScheduleEvent from event {self._id} {self._event_type}, \
                        decode_replica_id={request.decode_replica_id} len(events)={len(events)}")
        
                
                if request._num_processed_tokens >= request._num_prefill_tokens:            
                    # print(f"> self.decode_arrived_at={self.decode_arrived_at} self.request_type={self.request_type} self.prefill_completed_at={self.prefill_completed_at} self._is_prefill_complete={self._is_prefill_complete}")
                    assert request.decode_arrived_at < float("inf")  and request.request_type == RequestType.DECODE and request.prefill_completed_at > 0 and request._is_prefill_complete == True, "> debug"

                        
        return events


    def to_dict(self):
        return {
            "time": self.time,
            "event_type": self.event_type,
            "batch_id": self._batch.id,
        }
