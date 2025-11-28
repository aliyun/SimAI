from abc import ABC, abstractmethod

from vidur.execution_time_predictor.communication_time_predictor import TPTimePredictor

from vidur.config import (
    BaseExecutionTimePredictorConfig,
    BaseReplicaSchedulerConfig,
    MetricsConfig,
    ReplicaConfig,
    SimulationConfig,
)
from vidur.entities import Batch, ExecutionTime


# 返回单个micro-batch在单个TP shard，单个PP stage上的执行时间
# Returns execution time for a single micro-batch on a single TP shard and a single PP stage
class BaseExecutionTimePredictor(ABC):
    def __init__(
        self,
        predictor_config: BaseExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
        simulation_config: SimulationConfig,
    ) -> None:
        self._config = predictor_config
        self._replica_config = replica_config
        self._model_config = replica_config.model_config

        # get configs
        self._replica_scheduler_provider = str(replica_scheduler_config.get_type())
        self._block_size = replica_scheduler_config.block_size
        self._cache_dir = metrics_config.cache_dir
        self._num_layers_per_pipeline_stage = (
            self._model_config.num_layers // self._replica_config.num_pipeline_stages
        )
        self._tp_time_predictor = TPTimePredictor(
            self._model_config,
            self._replica_config,
            self._config
        )
        # > add
        self.replica_scheduler_config = replica_scheduler_config
        self.simulation_config = simulation_config

    def get_execution_time(self, batch: Batch, pipeline_stage: int) -> ExecutionTime:
        if pipeline_stage == self._replica_config.num_pipeline_stages - 1:
            pipeline_parallel_communication_time = 0
        else:
            pipeline_parallel_communication_time = (
                # 这里PP没有考虑async io
                # PP does not consider async IO here
                self._get_pipeline_parallel_communication_time(batch)
            )

        if self._replica_config.tensor_parallel_size == 1:
            tensor_parallel_communication_time = 0
        else:
            # if self._config.simai_enable:
            # if self._config.simai_simulation_enable:
            if self._config.backend == "simai_simulation":
                tensor_parallel_communication_time = self._tp_time_predictor.get_execution_time(batch)
                
                # TODO: chentong fix it
                # fy：有可能跑出来结果是-1
                # fy: Result may be -1
                assert tensor_parallel_communication_time >= 0, "> Debug: tensor_parallel_communication_time must be greater than 0"
                
                # >: 如果simai 后端返回-1，则调用vidur的查表方法
                # >: If simai backend returns -1, call vidur's lookup table method 
                if tensor_parallel_communication_time == -1:
                    tensor_parallel_communication_time = self._get_tensor_parallel_communication_time(batch)
                    
            # elif self._config.simai_analytical_enable:
            elif self._config.backend == "simai_analytical":
                tensor_parallel_communication_time = self._tp_time_predictor.get_execution_time_by_simai_analytical(batch)
                assert tensor_parallel_communication_time >= 0, "> Debug: tensor_parallel_communication_time must be greater than 0"
                
                # >：如果simai 后端返回-1，则调用vidur的查表方法
                # >: If simai backend returns -1, call vidur's lookup table method 
                if tensor_parallel_communication_time == -1:
                    tensor_parallel_communication_time = self._get_tensor_parallel_communication_time(batch)
            
            elif self._config.backend == "aicb":
                # TODO currently not supported TP communication when using aicb
                tensor_parallel_communication_time = 0
            else:
                assert self._config.backend == "vidur", "> Debug: self._config.backend can only be simai_simulation, simai_analytical, vidur"
                tensor_parallel_communication_time = self._get_tensor_parallel_communication_time(batch)

        if self._config.backend == "aicb":
            # > add self
            # extract AICB params
            import copy

            replica_config = copy.deepcopy(self._replica_config)
            
            # TODO is this correct?
            batch_prefill_replica_id = batch.requests[0].prefill_replica_id
            batch_replica_id = batch.replica_id
            
            # > add
            # print(f"> debug self.replica_type ")
            
            
            if batch_prefill_replica_id == batch_replica_id:
                replica_config.phase = "prefill"
            else:
                replica_config.phase = "decode"

            tp = self._replica_config.tensor_parallel_size
            pp = self._replica_config.num_pipeline_stages
            # dp = 1 # TODO get world_size from dp or somehow get replica size
            dp = self.simulation_config.cluster_config.num_replicas
            ws = tp * pp * dp
            replica_config.world_size = ws

            if replica_config.phase == "prefill":
                bs = 1
                seq = 0
                for request, num_tokens_to_process in zip(batch.requests, batch.num_tokens):
                    if request._is_prefill_complete:
                        continue
                    seq += num_tokens_to_process
            elif replica_config.phase == "decode":
                bs = 0
                seq = 0
                for request, num_tokens_to_process in zip(batch.requests, batch.num_tokens):
                    if request._is_prefill_complete:
                        bs += 1
                        seq += request.num_processed_prefill_tokens + request.num_processed_decode_tokens - 1
            
            replica_config.batch_size = bs
            replica_config.seq_len = seq
            

            return ExecutionTime(
                self._num_layers_per_pipeline_stage,
                self._get_attention_rope_execution_time(batch),
                self._get_attention_kv_cache_save_execution_time(batch),
                self._get_attention_decode_execution_time(batch),
                self._get_attention_prefill_execution_time(batch),
                self._get_attention_layer_pre_proj_execution_time(batch),
                self._get_attention_layer_post_proj_execution_time(batch),
                self._get_mlp_layer_up_proj_execution_time(batch),
                self._get_mlp_layer_down_proj_execution_time(batch),
                self._get_mlp_layer_act_execution_time(batch),
                self._get_attn_norm_layer_act_execution_time(batch),
                self._get_mlp_norm_layer_act_execution_time(batch),
                self._get_add_layer_act_execution_time(batch),
                tensor_parallel_communication_time,
                pipeline_parallel_communication_time,
                self._get_schedule_time(batch),
                self._get_sampler_e2e_time(batch),
                self._get_prepare_inputs_e2e_time(batch),
                self._get_process_model_outputs_time(batch),
                self._get_ray_comm_time(batch),
                self._config,
                replica_config,
                self.replica_scheduler_config
                # self._model_config
            )
        else:
            return ExecutionTime(
                self._num_layers_per_pipeline_stage,
                self._get_attention_rope_execution_time(batch),
                self._get_attention_kv_cache_save_execution_time(batch),
                self._get_attention_decode_execution_time(batch),
                self._get_attention_prefill_execution_time(batch),
                self._get_attention_layer_pre_proj_execution_time(batch),
                self._get_attention_layer_post_proj_execution_time(batch),
                self._get_mlp_layer_up_proj_execution_time(batch),
                self._get_mlp_layer_down_proj_execution_time(batch),
                self._get_mlp_layer_act_execution_time(batch),
                self._get_attn_norm_layer_act_execution_time(batch),
                self._get_mlp_norm_layer_act_execution_time(batch),
                self._get_add_layer_act_execution_time(batch),
                tensor_parallel_communication_time,
                pipeline_parallel_communication_time,
                self._get_schedule_time(batch),
                self._get_sampler_e2e_time(batch),
                self._get_prepare_inputs_e2e_time(batch),
                self._get_process_model_outputs_time(batch),
                self._get_ray_comm_time(batch),
                self._config,
                self._replica_config,
                self.replica_scheduler_config
                
                # self._model_config
            )

    @abstractmethod
    def _get_attention_layer_pre_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_layer_post_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_rope_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_kv_cache_save_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_decode_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_prefill_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_mlp_layer_up_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_mlp_layer_down_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_mlp_layer_act_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_tensor_parallel_communication_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_pipeline_parallel_communication_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_schedule_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_sampler_e2e_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_prepare_inputs_e2e_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_process_model_outputs_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_ray_comm_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_mlp_norm_layer_act_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attn_norm_layer_act_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_add_layer_act_execution_time(self, batch: Batch) -> float:
        pass
