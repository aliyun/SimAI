import logging
from dataclasses import dataclass, field 
from enum import IntEnum 

# from metrics import NodeMetrics  
# from simulator import clock, schedule_event, cancel_event, reschedule_event  

class NodeState(IntEnum):  # Define integer enumeration representing various node states
    NONE = 0  # Uninitialized state
    QUEUED = 1  # Queued waiting for scheduling
    RUNNING = 2  
    BLOCKED = 3  # Blocked (e.g., preempted)
    COMPLETED = 4 
    ABORTED = 5  # Aborted/terminated
@dataclass(kw_only=True)  # Define dataclass, all parameters must be passed using keywords
class Node():  # Node class, basic unit in tasks and requests
    """
    Base class for Tasks and Nodes in a Request
    Simplest element of the Request DAG
    """
    node_id: int  # Unique node ID
    num_preemptions: int = 0  # Number of preemptions, default is 0
    request: 'Request' = None  # Associated request object (type is Request, defined later)
    state: NodeState = NodeState.NONE  # Current node state, default uninitialized
    # Object recording timestamps/statistical factors
    # metrics: NodeMetrics = field(default_factory=NodeMetrics)
    # chain of nodes that must be executed back-to-back
    # only stored in the first node of the chain
    # Chain of nodes to be executed consecutively, only stored in the first node of the chain
    chain: list = field(default_factory=list)

    def __hash__(self):  # Define hash function for use in sets, etc.
        """
        NOTE: hash functions get overridden to None in child classes
        """
        return hash(self.node_id)  # Use node ID as hash value

    def __eq__(self, other):  # Node equality judgment logic
        # Nodes are considered equal if node IDs are the same
        return self.node_id == other.node_id

    def arrive(self):  # Node arrives at queue (enqueued)
        assert self.state == NodeState.NONE  # Ensure node was previously in uninitialized state
        self.metrics.arrival_timestamp = clock() # Record arrival time
        self.state = NodeState.QUEUED  # Set state to queued

    def run(self):  # Node starts running
        assert self.state == NodeState.QUEUED # Must be in queue
        self.metrics.run_timestamp = clock()  # Record run timestamp
        self.metrics.start_timestamp = clock()  # Also record start timestamp
        # Calculate accumulated queue time
        self.metrics.queue_time += clock() - self.metrics.arrival_timestamp
        # If current node is root node
        if self.request.root_node is self:
            # Record prompt start time for request metrics
            self.request.metrics.prompt_start_timestamp = clock()
            self.request.metrics.queue_time = clock() - \
                            self.request.metrics.router_arrival_timestamp # Record total wait time for request
        self.state = NodeState.RUNNING  # Change state to running

    def run_after_preempt(self):  # Node runs after being preempted
        assert self.state == NodeState.BLOCKED  # Can only re-run in blocked stat
        self.metrics.run_timestamp = clock()   # Re-record run time
        self.metrics.blocked_time += clock() - self.metrics.preempt_timestamp  # Add blocked time
        self.state = NodeState.RUNNING  # State becomes running

    def complete(self): # Node completes
        assert self.state == NodeState.RUNNING  # Must be in running state to complete
        self.metrics.completion_timestamp = clock()  # Mark completion time
        self.metrics.service_time += clock() - self.metrics.run_timestamp  # Add this service time
        self.metrics.response_time = clock() - self.metrics.arrival_timestamp  # Calculate response time
        self.state = NodeState.COMPLETED  # Update state to completed


    def preempt(self):  # Node is preempted (paused)
        assert self.state == NodeState.RUNNING  # Can only preempt running nodes
        self.metrics.preempt_timestamp = clock()  # Record preemption timestamp
        self.metrics.service_time += clock() - self.metrics.run_timestamp # Add served time in this round
        self.state = NodeState.BLOCKED  # Change state to blocked

    def abort(self):  # Node terminates/aborts (may come from different states)
        if self.state == NodeState.QUEUED:  # If in queue (not yet scheduled)
            self.metrics.queue_time += clock() - self.metrics.arrival_timestamp  # Accumulate queue time to now
            if self.request.root_node is self:  # If root node
                self.request.metrics.queue_time = clock() - \
                                self.request.metrics.router_arrival_timestamp  # Record queue time for entire request
        elif self.state == NodeState.RUNNING:  # If running
            self.metrics.service_time += clock() - self.metrics.run_timestamp  # Add served time
        elif self.state == NodeState.BLOCKED:  # If blocked
            self.metrics.blocked_time += clock() - self.metrics.preempt_timestamp  # Add blocked time
        self.state = NodeState.ABORTED  # Change state to blocked
