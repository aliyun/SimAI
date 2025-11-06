import logging

from dataclasses import dataclass, field  
from enum import IntEnum  

# from instance import Instance  
# from metrics import FlowMetrics, FlowSLO  
# from model import Model, ModelArchitecture  
# from node import Node  

# from simulator import clock, schedule_event, cancel_event, reschedule_event  

# >
from vidur.entities.node import Node
from vidur.entities.replica import Replica

class FlowType(IntEnum):  # Define FlowType as a subclass of IntEnum to represent flow types
    DEFAULT = 0  # Default flow type
    KVCacheTransfer = 1  # KV cache transfer flow type


@dataclass(kw_only=True)  # Using dataclass decorator, parameters must be passed via keyword
class Flow(Node):  # Flow class inherits from Node class
    """
    Flows are communication nodes in the Request DAG that execute on Links.
    Flows are the networking counterparts of Tasks.
    """
    flow_type: FlowType  # Type of flow
    # src: Instance  # Source instance of flow
    # dest: Instance  # Destination instance of flow
    src: Replica
    dest: Replica
    batch_size: int = 1  # Batch size, default value is 1
    size: float = 0.  # Size of flow, default value is 0
    duration: float = 0.  # Duration, default value is 0
    notify: bool = False  # Whether to notify, default value is False
    # metrics: FlowMetrics = field(default_factory=FlowMetrics)  # Flow metrics, default initialized to FlowMetrics object
    # slo: FlowSLO = field(default_factory=FlowSLO)  # SLO, default initialized to FlowSLO object
    # executor: 'Executor' = None  # Executor, default value is None
    links = []  # Link list
    _link = None  # Current link, default value is None
    def __hash__(self):  # Implement __hash__ method
        return hash(self.node_id)  # Return hash value of node_id

    @property  # Define property decorator
    def link(self):  # Get current link
        return self._link  # Return _link attribute

    @link.setter  # Method to set link property
    def link(self, link):  # Set current link
        if link is self._link:  # If the incoming link is the same as the current link
            return  # Return directly
        self._link = link # Update _link attribute
        if link is not None:  
            self.links.append(link)  # Add link to links list

    @property  # Define property decorator
    def duration(self):  
        return self._duration  

    @duration.setter  # Method to set duration property
    def duration(self, duration): 
        self._duration = duration  

    @property  # Define property decorator
    def memory(self):  
        return 0  # Return 0, indicating no memory usage

    def run(self):  
        super().run()  # Call parent class's run method

        # manage memory 
        self.dest.alloc_memory(self.request, self.request.memory)  # Allocate memory on destination instance

    def complete(self):  
        super().complete() # Call parent class's complete method

        # manage memory 
        self.src.free_memory(self.request, self.request.memory)  # Free memory on source instance

    @classmethod  # Class method decorator
    def from_type(cls, flow_type, **kwargs): # Create corresponding flow object based on flow type
        if flow_type == FlowType.DEFAULT: 
            return Flow(**kwargs)  # Create and return Flow object
        elif flow_type == FlowType.KVCacheTransfer:  
            return KVCacheTransferFlow(**kwargs)  # Create and return KVCacheTransferFlow object
        else:  # If flow type is invalid
            raise ValueError(f"Invalid FlowType {flow_type}")   # Raise exception


@dataclass(kw_only=True)  # Using dataclass decorator, parameters must be passed via keyword
class KVCacheTransferFlow(Flow):  # KVCacheTransferFlow class inherits from Flow class
    """
    Flow for transferring KV cache between instances.
    """
    flow_type: FlowType = FlowType.KVCacheTransfer  # Flow type fixed to KVCacheTransfe

    def __hash__(self):  # Implement __hash__ method
        return hash(self.node_id)  # Return hash value of node_id