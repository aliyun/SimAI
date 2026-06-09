import logging  # Import logging module for recording runtime information

from dataclasses import dataclass, field  # Import dataclass tools for defining classes
from enum import IntEnum  # Import IntEnum for defining integer enumeration types

# from flow import Flow  
# from processor import CPU, GPU  
# from simulator import clock, schedule_event, cancel_event, reschedule_event  
# from server import Server 

# >
from vidur.entities.flow import Flow
from vidur.entities.processor import CPU, GPU 
from vidur.entities.replica import Replica


class LinkType(IntEnum):  # Define LinkType enumeration class for link types
    DEFAULT = 0  
    PCIeLink = 1  
    EthernetLink = 2 
    IBLink = 3  
    NVLink = 4 
    RDMADirectLink = 5  
    DummyLink = 6  # Virtual link type for simulating latency


@dataclass(kw_only=True) # Use dataclass decorator, only allow keyword arguments
class Link():  # Link class represents network link
    """
    Links are unidirectional edges in the cluster interconnect topology graph.
    They are the lowest-level networking equivalent of Processors.
    Instead of Tasks, Links can run (potentially multiple) Flows.
    Links have a maximum bandwidth they can support, after which point they become congested.

    TODO: replace with a higher-fidelity network model (e.g., ns-3).

    Attributes:
        link_type (LinkType): Type of the Link (e.g., NVLink, IB, etc).
        src (object): Source endpoint
        dest (object): Destination endpoint
        bandwidth (float): The maximum bandwidth supported by the Link.
        bandwidth_used (float): The bandwidth used by the Link.
        server (Server): The Server that the Processor belongs to.
        flows (List[Flow]): Flows running on this Link.
        max_flows (int): Maximum number of flows that can run in parallel on the link.
    """
    """
    链接是集群互连拓扑图中的单向边。
    它们是处理器的最低级别网络等价物。
    与任务不同，链接可以运行（可能是多个）流。
    链接有一个最大带宽限制，超过该限制后会变得拥塞。

    TODO: 使用更高保真度的网络模型替换（例如，ns-3）。
    """

    link_type: LinkType = LinkType.DEFAULT  # Link type, default to DEFAULT
    name: str  # Link name
    src: object  # Source endpoint object
    dest: object  # Destination endpoint object
    bandwidth: float  # Link bandwidth limit
    bandwidth_used: float  # Currently used bandwidth
    _bandwidth_used: float = 0  # Actual stored bandwidth usage value
    max_flows: int  # Maximum concurrent flow count
    retry: bool = True  # Retry flag
    retry_delay: float = 1.  # Retry delay time
    overheads: dict = field(default_factory=dict)  # Overhead dictionary, empty by default
    
    # Queue definitions
    pending_queue: list[Flow] = field(default_factory=list)  
    executing_queue: list[Flow] = field(default_factory=list) 
    completed_queue: list[Flow] = field(default_factory=list) 

    @property
    def bandwidth_used(self):  # Get current bandwidth usage
        return self._bandwidth_used

    @bandwidth_used.setter
    def bandwidth_used(self, bandwidth_used):  # Set bandwidth usage with validation
        if type(bandwidth_used) is property:  # If it's a property type, set to 0
            bandwidth_used = 0
        if bandwidth_used < 0:  # Bandwidth usage cannot be negative
            raise ValueError("Bandwidth used cannot be negative")
        elif bandwidth_used > self.bandwidth:  # Cannot exceed maximum bandwidth
            raise ValueError("Cannot exceed link bandwidth")
        self._bandwidth_used = bandwidth_used  # Set actual bandwidth usage value

    @property
    def bandwidth_free(self):  # Calculate remaining available bandwidth
        return self.bandwidth - self.bandwidth_used

    @property
    def peers(self):  # Get peer devices (not implemented)
        pass

    def flow_arrival(self, flow):  # Data flow arrival processing function
        """
        Flow arrives at the Link.
        """
        flow.instance = self  # Set the link instance the flow belongs to
        flow.arrive()  # Call flow's arrive method
        self.pending_queue.append(flow)  # Add flow to pending queue
        if len(self.pending_queue) > 0 and len(self.executing_queue) < self.max_flows:  # If there are free slots and pending flows
            if flow.dest.memory + flow.request.memory <= flow.dest.max_memory:  # Check if destination device has enough memory
                self.run_flow(flow)  # Run the flow
            elif self.retry:  # If retry is needed
                schedule_event(self.retry_delay, lambda link=self,flow=flow: link.retry_flow(flow))  # Schedule retry event
            else:
                # will lead to OOM
                self.run_flow(flow)  # Force run, may cause out of memory


    def flow_completion(self, flow):
        """
        Flow completes on this Link.
        """
        flow.complete()  # Call flow object's complete method to mark flow as completed
        self.executing_queue.remove(flow)  # Remove flow from executing queue
        self.completed_queue.append(flow)  # Add to completed queue
        flow.executor.finish_flow(flow, self)  # Notify flow executor that flow is completed
        if flow.notify:  # If need to notify source that flow is completed
            flow.src.notify_flow_completion(flow)   # Source performs callback processing
        self.bandwidth_used -= (self.bandwidth - self.bandwidth_used)  # Update bandwidth usage (release bandwidth occupied by current flow)
        if len(self.pending_queue) > 0 and len(self.executing_queue) < self.max_flows: # If there are pending flows and free concurrent slots
            next_flow = self.pending_queue[0]  # Get first pending flow
            if next_flow.dest.memory + next_flow.request.memory <= next_flow.dest.max_memory:  # Check if there's enough memory to run next flow
                self.run_flow(next_flow)  # Run next flow
            elif self.retry:  # Otherwise try retry mechanism
                schedule_event(self.retry_delay, lambda link=self,flow=flow: link.retry_flow(flow))  # Schedule retry event
            else:
                # will lead to OOM
                self.run_flow(next_flow) # Force run, may cause out of memory

    def retry_flow(self, flow):
        """
        Flow is retried on this Link.
        """
        if flow not in self.pending_queue:  # If flow is not in pending queue, return directly
            return
        if (len(self.executing_queue) < self.max_flows) and (flow.dest.memory + flow.request.memory <= flow.dest.max_memory): # If there are resources and destination device has enough memory
            self.run_flow(flow)  # Try to run the flow again
        elif self.retry:  # Otherwise continue to schedule retry
            schedule_event(self.retry_delay, lambda link=self,flow=flow: link.retry_flow(flow))  # Try again after delay
        else:
            # will lead to OOM
            self.run_flow(flow)  # Force run, may cause out of memory

    def get_duration(self, flow):
        """
        FIXME: this can be shorter than prompt duration
        """
        return flow.size / (self.bandwidth - self.bandwidth_used)  # Calculate transfer time = data size / available bandwidth

    def run_flow(self, flow):
        """
        Run a Flow on this Link.
        """
        flow.run()  # Start running the data flow
        self.pending_queue.remove(flow)  # Remove from pending queue
        self.executing_queue.append(flow)  # Add to executing queue
        flow.duration = self.get_duration(flow)  # Set the flow's duration
        # TODO: policy on how to allocate bandwidth to multiple flows
        self.bandwidth_used += (self.bandwidth - self.bandwidth_used)  # Occupy all available bandwidth
        schedule_event(flow.duration,  # Schedule an event to call flow_completion after duration time
                       lambda link=self,flow=flow: link.flow_completion(flow))

    def preempt_flow(self, flow):
        """
        Preempt a flow on this Link.
        """
        flow.preempt()  # Perform preemption operation
        raise NotImplementedError  # Current preemption logic not implemented, throw exception


@dataclass(kw_only=True)
class PCIeLink(Link):  # PCIe link class, inherits from Link
    """
    PCIeLink is a specific type of Link between CPUs and GPUs.
    """
    link_type: LinkType = LinkType.PCIeLink  # Link type is PCIe
    src: CPU  # Source device is CPU
    dest: GPU  # Destination device is GPU


# @dataclass(kw_only=True)
# class EthernetLink(Link):  # Ethernet link class
#     """
#     EthernetLink is standard Ethernet between Servers.
#     """
#     link_type: LinkType = LinkType.EthernetLink  # Link type is Ethernet
#     src: Server  # Source device is server
#     dest: Server  # Destination device is also server


# @dataclass(kw_only=True)
# class IBLink(Link):  
#     """
#     IBLink is the Infiniband Link between Servers.
#     """
#     link_type: LinkType = LinkType.IBLink 
#     src: Server  # Source device is server
#     dest: Server  # Destination device is also server


@dataclass(kw_only=True)
class NVLink(Link):  # NVLink class for GPU communication
    """
    NVLink is a specific type of Link between GPUs.
    """
    link_type: LinkType = LinkType.NVLink  # Link type is NVLink
    src: GPU  # Source device is GPU
    dest: GPU  # Destination device is also GPU

@dataclass(kw_only=True)
class RDMADirectLink(Link):  
    """
    RDMADirect is the Infiniband link between GPUs across/within Servers.
    """
    link_type: LinkType = LinkType.RDMADirectLink
    src: GPU  # Source device is GPU
    dest: GPU # Destination device is also GPU


@dataclass(kw_only=True)
class DummyLink(Link): # Virtual link class, simulates latency without consuming bandwidth
    """
    A Link whose bandwidth is never actually used and can hold infinite flows.
    Used to simulate delay.
    """
    link_type: LinkType = LinkType.DummyLink  
    src: object = None  # Source device is empty
    dest: object = None  # Destination device is also empty
    max_flows: float = float("inf")  # Maximum concurrent flow count is infinite

    @property
    def bandwidth_used(self): # Bandwidth usage always returns internal stored value
        return self._bandwidth_used

    @bandwidth_used.setter
    def bandwidth_used(self, bandwidth_used):  # Setting bandwidth usage does nothing
        return