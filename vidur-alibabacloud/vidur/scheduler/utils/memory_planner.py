from vidur.config import ReplicaConfig
from vidur.entities.replica import Replica
from vidur.utils.param_counter import ParamCounter


class MemoryPlanner:
    def __init__(self, replica_config: ReplicaConfig, replica: Replica) -> None:
        self._param_counter = ParamCounter(replica_config)
        self._replica = replica

    def _get_kv_cache_memory_per_layer_per_request(self) -> int:
        return (
            2  # 2 bytes per float
            * 2  # one for key, one for value
            * self._replica.attention_head_dim
            * self._replica.kv_heads_per_tensor_parallel_worker
            * self._replica.max_request_tokens
        )

    def _get_parameter_memory_per_device(self) -> int:
        return 2 * self._param_counter.get_num_parameters_per_device()

    def _get_kv_cache_memory_per_device_per_request(self) -> int:
        return (
            self._get_kv_cache_memory_per_layer_per_request() * self._replica.num_layers
        )

    def get_max_batch_size(self) -> int:
        available_memory = (
            self._replica.total_memory_gb
            * 1024**3
            * (1 - self._replica.memory_margin_fraction)
        )
        parameter_memory_per_device = self._get_parameter_memory_per_device()
        kv_cache_memory_per_device_per_request = (
            self._get_kv_cache_memory_per_device_per_request()
        )

        memory_for_kv_cache = available_memory - parameter_memory_per_device
        number_of_requests = (
            memory_for_kv_cache // kv_cache_memory_per_device_per_request
        )

        assert (
            number_of_requests > 0
        ), "Not enough memory to store even a single request"

        return number_of_requests

    def get_max_request_slots(self) -> int:
        return self.get_max_batch_size() * self._replica.num_pipeline_stages
