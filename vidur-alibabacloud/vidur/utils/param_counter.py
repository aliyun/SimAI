from math import ceil

from vidur.config import ReplicaConfig


class ParamCounter:
    def __init__(self, replica_config: ReplicaConfig) -> None:
        self._replica_config = replica_config
        self._model_config = self._replica_config.model_config

        assert (
            self._model_config.num_q_heads % self._replica_config.tensor_parallel_size
            == 0
        )
        assert (
            self._model_config.num_layers % self._replica_config.num_pipeline_stages
            == 0
        )
        assert (
            self._model_config.embedding_dim % self._replica_config.tensor_parallel_size
            == 0
        )
        assert self._model_config.embedding_dim % self._model_config.num_q_heads == 0

        self._num_layers_per_pipeline_stage = (
            self._model_config.num_layers // self._replica_config.num_pipeline_stages
        )
        self._attention_head_dim = (
            self._model_config.embedding_dim // self._model_config.num_q_heads
        )
        self._q_heads_per_tensor_parallel_worker = (
            self._model_config.num_q_heads // self._replica_config.tensor_parallel_size
        )
        self._kv_heads_per_tensor_parallel_worker = ceil(
            self._model_config.num_kv_heads / self._replica_config.tensor_parallel_size
        )

    def get_num_parameters_per_layer(self) -> int:
        num_parameters = 0
        # weights for attention metrics Wq, Wk, Wv
        num_parameters += (
            self._model_config.embedding_dim
            * self._attention_head_dim
            * (
                self._q_heads_per_tensor_parallel_worker
                + 2 * self._kv_heads_per_tensor_parallel_worker
            )
        )
        # weights for attention metrics Wo
        num_parameters += (
            self._model_config.embedding_dim
            * self._attention_head_dim
            * self._q_heads_per_tensor_parallel_worker
        )
        # fc layer weights
        if self._model_config.use_gated_mlp:
            num_parameters += (
                3
                * self._model_config.embedding_dim
                * self._model_config.mlp_hidden_dim
                // self._replica_config.tensor_parallel_size
            )
        else:
            num_parameters += (
                2
                * self._model_config.embedding_dim
                * self._model_config.mlp_hidden_dim
                // self._replica_config.tensor_parallel_size
            )

        return num_parameters

    def get_num_parameters_per_device(self) -> int:
        num_parameters_per_layer = self.get_num_parameters_per_layer()
        return num_parameters_per_layer * self._num_layers_per_pipeline_stage
