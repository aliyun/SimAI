from typing import List, Tuple

from vidur.entities import Request
from vidur.scheduler.global_scheduler.base_global_scheduler import BaseGlobalScheduler


class LORGlobalScheduler(BaseGlobalScheduler):
    """
    Least outstanding requests (LOR) global scheduler.
    """

    # Define a method named schedule that takes no parameters except self, returning a list where each element is a tuple containing an integer and a Request object.
    def schedule(self) -> List[Tuple[int, Request]]:
        # 对请求队列进行排序 pd分离也需要 按照_arrived_at排序
        # Sort the request queue - PD separation also needs sorting by _arrived_at
        self.sort_requests()
        
        # 存储请求映射结果的列表 pd分离需要 p集群 和 d 集群 ， 对应request； request也要区分成prefill实例和decode实例
        # Store the request mapping results - PD separation requires P cluster and D cluster, corresponding to requests; requests also need to be distinguished into prefill instances and decode instances
        request_mapping = []
        # keep a map of replica_id -> replica_scheduler
        # this is used to find the replica with the least outstanding requests
        
        # This is a dictionary comprehension that creates a dictionary named pending_requests_map.
        # It iterates through self._replica_schedulers.values() (all replica schedulers), creating a key-value pair for each replica scheduler:

        # Key: replica_scheduler.replica_id - Replica ID
        # Value: replica_scheduler.num_pending_requests - Number of pending requests for this replica
        # This dictionary tracks how many pending requests each replica currently has, to find the least loaded replica.
        
        # Detailed breakdown
        # replica_scheduler.replica_id - Key

        # This is the unique identifier for each replica scheduler
        # replica_scheduler.num_pending_requests - Value

        # This is the current number of pending requests for each replica scheduler
        # for replica_scheduler in self._replica_schedulers.values() - Iteration part

        # self._replica_schedulers is a dictionary storing all replica schedulers
        # .values() method returns all values in the dictionary (i.e., all replica scheduler objects)
        # replica_scheduler is the loop variable representing each replica scheduler object
        pending_requests_map = {
            replica_scheduler.replica_id: replica_scheduler.num_pending_requests
            for replica_scheduler in self._replica_schedulers.values()
        }
        
        # print(f"> Debug: pending_requests_map={pending_requests_map}")

        # This is a while loop that continues as long as self._request_queue (request queue) is not empty. This processes all requests in the queue one by one
        # using a very simple implementation here, to keep wiring simple
        while self._request_queue:
            request = self._request_queue.pop(0)
            
            # Find the replica ID with the fewest pending requests
            # pending_requests_map.items() - Get all key-value pairs from the dictionary ((replica_id, num_pending_requests))
            # min(..., key=lambda x: x[1]) - Find the key-value pair with the smallest value, where:
            # key=lambda x: x[1] is a lambda function used to specify the comparison criteria
            # x is a key-value pair (replica_id, num_pending_requests)
            # x[1] is the value in the key-value pair (i.e., number of pending requests)
            # [0] - Extract the key (replica_id) from the returned key-value pair
            replica_id = min(pending_requests_map.items(), key=lambda x: x[1])[0] 
            pending_requests_map[replica_id] += 1
            request_mapping.append((replica_id, request))
            
            # import pdb; pdb.set_trace() # >
        # print(f"> Debug: {request_mapping}{pending_requests_map}")
        return request_mapping
