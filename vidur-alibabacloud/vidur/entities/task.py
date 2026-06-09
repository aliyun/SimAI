import logging  

from dataclasses import dataclass, field  
from enum import IntEnum  


# from simulator import clock, schedule_event, cancel_event, reschedule_event 

# from metrics import TaskMetrics, TaskSLO  # Import TaskMetrics (task metrics) and TaskSLO (task SLO) from metrics module
# from node import Node  # Import Node class from node module, Task class will inherit this class
from vidur.entities.node import Node

class TaskType(IntEnum):  # Define task type enumeration class inheriting from IntEnum
    COMPUTE = 0  # Compute task
    PROMPT = 1  # Prompt task (prefill stage)
    TOKEN = 2  # Token task (generation stage)


@dataclass(kw_only=True)  # Use dataclass to define Task class, kw_only=True means only keyword arguments can be used for initialization
class Task(Node):  # Task class inherits from Node, representing a compute node
    """
    Task 是请求 DAG（有向无环图）中的计算节点
    任务在实例（Instance）上执行
    任务是 Flow 的计算对应部分
    
    Tasks are computation nodes in the Request DAG.
    Tasks execute on Instances.

    Tasks are the computational counterparts of Flows.
    """
    task_type: TaskType  # Task type (compute, prompt, token)
    batch_size: int = 1  
    duration: float = 0.  # Total task duration
    remaining_duration: float = 0.  # Remaining execution time
    cleanup_memory: bool = True  # Whether to clean up memory after completion
    # metrics: TaskMetrics = field(default_factory=TaskMetrics)  # Performance metrics of the task
    # slo: TaskSLO = field(default_factory=TaskSLO)  # Service level objective of the task
    executor: 'Executor' = None  # Executor that executes this task
    instances = []  # List of instances executing this task (class variable)
    _instance = None  # Currently bound instance

    def __hash__(self):  
        return hash(self.node_id)  

    @property
    def instance(self):  # Get currently bound instance
        return self._instance  # Return instance object

    @instance.setter
    def instance(self, instance):  
        if instance is self._instance:  
            return
        self._instance = instance  
        if instance is not None: 
            self.instances.append(instance)  

    @property
    def memory(self):  # Get memory required by task (base class defaults to 0)
        return 0

    @classmethod
    def from_type(cls, task_type, **kwargs):  # Build different task objects based on task type
        if task_type == TaskType.COMPUTE:  # If it's a compute task
            return ComputeTask(**kwargs)
        elif task_type == TaskType.PROMPT:  # If it's a prompt task
            return PromptTask(**kwargs)
        elif task_type == TaskType.TOKEN:  # If it's a token task
            return TokenTask(**kwargs)
        else:
            raise ValueError(f"Invalid TaskType {task_type}")  # Raise exception for invalid type


@dataclass(kw_only=True)
class ComputeTask(Task):  # Compute task class
    """
    计算任务表示任意计算过程
    """
    task_type: TaskType = TaskType.COMPUTE  # Default is compute task type

    def __hash__(self):
        return hash(self.node_id)  

    @property
    def memory(self):  # Compute task does not require extra memory
        return 0


@dataclass(kw_only=True)
class PromptTask(Task):  # Prompt task class
    """
    Prompt task represents the prompt phase (prefill computation) in generative LLMs
    They are typically root tasks of generative LLM requests
    """
    prompt_size: int  # Number of prompt tokens
    tokens_per_iteration: int = 0 # Tokens per iteration
    processing_tokens: int = 0  # Current processing token count
    processed_tokens: int = 0   # Total processed tokens
    generating_tokens: int = 0  # Current generating token count
    generated_tokens: int = 0  # Total generated tokens
    task_type: TaskType = TaskType.PROMPT  # Default type is prompt task
    cleanup_memory: bool = False # Prompt task does not clean up memory by default (subsequent tokens still needed)
    is_prefill_complete = False  # Prefill is not completed by default

    def __post_init__(self):  # Method called after dataclass initialization
        self.tokens_per_iteration = self.prompt_size  # Prompt task processes entire prompt in one iteration

    def __hash__(self):
        return hash(self.node_id)

    @property
    def memory(self):  # Calculate KV Cache memory required for prompt phase
        num_tokens = self.prompt_size + 1  # +1 usually for the first generated token
        # return self.request.estimate_kv_cache_size(num_tokens=num_tokens,
        #                                            model=self.instance.model)
        return self.request.estimate_kv_cache_size(num_tokens=num_tokens,
                                                   replica=self.instance)

    def max_memory(self, instance):  # Calculate maximum memory required under specified instance
        num_tokens = self.prompt_size + 1
        # return self.request.estimate_kv_cache_size(num_tokens=num_tokens,
        #                                            model=instance.model)
        return self.request.estimate_kv_cache_size(num_tokens=num_tokens,
                                                   replica=instance)

    def run(self):  # Execute task
        super().run()  # Call parent class run logic

        # 管理内存分配
        self.instance.alloc_memory(self.request, self.memory)  # Allocate memory on instance
        self.request.memory += self.memory  # Request object records memory usage

    def complete_iteration(self):  # Statistics and state update after completing one iteration
        self.processed_tokens += self.processing_tokens  # Accumulate processed tokens
        self.request.processed_tokens += self.processing_tokens  # Request-level update
        self.generated_tokens += self.generating_tokens  # Accumulate generated tokens
        self.request.generated_tokens += self.generating_tokens  # Request-level update
        self.processing_tokens = 0  # Reset current processing token count
        self.generating_tokens = 0  # Reset current generating token count
        
    def is_complete(self):  # Check if task is complete (generated 1 token means complete)
        return self.generated_tokens == 1

    def complete(self):  # Post-completion processing
        super().complete()  # Call parent class completion logic

        # Update scheduler pending token count (subtract entire prompt)
        self.instance.sched_pending_tokens -= self.prompt_size

        # Record TTFT (Time To First Token)
        self.request.metrics.prompt_end_timestamp = clock()
        self.request.metrics.TTFT = clock() - \
                                self.request.metrics.router_arrival_timestamp

        # Ensure processed and generated token counts meet expectations
        assert self.processed_tokens == self.prompt_size
        assert self.request.processed_tokens == self.request.prompt_size
        assert self.generated_tokens == 1

        ## Free memory if needed
        if self.cleanup_memory:
            self.instance.free_memory(self.request, self.request.memory)
            self.request.memory = 0


@dataclass(kw_only=True)
class TokenTask(Task):  # Token task class
    """
    Token task represents the decoding phase of generative LLMs
    """
    token_size: int  # Number of tokens to generate
    tokens_per_iteration: int = 1  # Process 1 token per iteration
    processing_tokens: int = 0 
    processed_tokens: int = 0 
    generating_tokens: int = 0  
    generated_tokens: int = 0  
    task_type: TaskType = TaskType.TOKEN  
    is_prefill_complete = True  # Prefill is completed by default
    

    def __hash__(self):
        return hash(self.node_id)

    @property
    def memory(self):  # Calculate KV Cache memory required for token phase
        num_tokens = self.token_size
        # return self.request.estimate_kv_cache_size(num_tokens=num_tokens,
        #                                            model=self.instance.model)
        return self.request.estimate_kv_cache_size(num_tokens=num_tokens,
                                                   replica=self.instance)

    def max_memory(self, instance):  # Calculate maximum memory requirement under specified instance
        num_tokens = self.token_size
        # return self.request.estimate_kv_cache_size(num_tokens=num_tokens,
        #                                            model=instance.model)
        return self.request.estimate_kv_cache_size(num_tokens=num_tokens,
                                                   replica=instance)

    def run(self):  # Execute token phase
        super().run()
        # Manage memory allocation
        self.instance.alloc_memory(self.request, self.memory)
        self.request.memory += self.memory

    def complete_iteration(self):  # Update after each iteration completion
        self.processed_tokens += self.processing_tokens
        self.request.processed_tokens += self.processing_tokens
        self.generated_tokens += self.generating_tokens
        self.request.generated_tokens += self.generating_tokens
        self.processing_tokens = 0
        self.generating_tokens = 0

    def is_complete(self):  # Check if complete
        return self.generated_tokens == self.token_size

    def complete(self): # Complete task
        super().complete()
        # Update scheduler pending token count
        self.instance.sched_pending_tokens -= 1

        # Check if token count meets expectations
        assert self.processed_tokens == self.token_size
        assert self.generated_tokens == self.token_size
        assert self.request.generated_tokens == self.request.token_size
        assert self.request.processed_tokens == self.request.prompt_size + \
                                                self.request.token_size - 1

        # Free memory if needed
        if self.cleanup_memory:
            self.instance.free_memory(self.request, self.request.memory)
            self.request.memory = 0


if __name__ == "__main__":  # If running current file directly
    pass  # Do nothing
