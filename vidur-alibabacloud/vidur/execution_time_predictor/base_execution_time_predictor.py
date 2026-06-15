from abc import ABC, abstractmethod

from vidur.execution_time_predictor.communication_time_predictor import TPTimePredictor
from vidur.logger import init_logger

from vidur.config import (
    BaseExecutionTimePredictorConfig,
    BaseReplicaSchedulerConfig,
    MetricsConfig,
    ReplicaConfig,
    SimulationConfig,
)
from vidur.entities import Batch, ExecutionTime

logger = init_logger(__name__)


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
                
                # TODO(chentong): fix potential -1 return value
                # Result may be -1 in some cases
                # 有可能跑出来结果是 -1
                assert tensor_parallel_communication_time >= 0, "tensor_parallel_communication_time must be non-negative"
                
                # If simai backend returns -1, fall back to vidur's lookup table method
                # 如果 simai 后端返回 -1，则调用 vidur 的查表方法 
                if tensor_parallel_communication_time == -1:
                    tensor_parallel_communication_time = self._get_tensor_parallel_communication_time(batch)
                    
            # elif self._config.simai_analytical_enable:
            elif self._config.backend == "simai_analytical":
                tensor_parallel_communication_time = self._tp_time_predictor.get_execution_time_by_simai_analytical(batch)
                assert tensor_parallel_communication_time >= 0, "tensor_parallel_communication_time must be non-negative"
                
                # If simai backend returns -1, fall back to vidur's lookup table method
                # 如果 simai 后端返回 -1，则调用 vidur 的查表方法 
                if tensor_parallel_communication_time == -1:
                    tensor_parallel_communication_time = self._get_tensor_parallel_communication_time(batch)
            
            elif self._config.backend == "aicb":
                # TODO(tianhao909): add TP communication support for AICB backend
                # TODO(tianhao909): AICB 后端暂不支持 TP 通信
                tensor_parallel_communication_time = 0
            else:
                assert self._config.backend == "vidur", "backend must be one of: simai_simulation, simai_analytical, aicb, vidur"
                tensor_parallel_communication_time = self._get_tensor_parallel_communication_time(batch)

        if self._config.backend == "aicb":
            # ============================================================
            # [AICB Backend] Build per-batch replica_config copy
            # Need to set correct params based on current batch phase (prefill/decode)
            # [AICB Backend] 构建 per-batch 的 replica_config 副本
            # 需要根据当前 batch 的 phase (prefill/decode) 设置正确参数
            # ============================================================
            import copy

            replica_config = copy.deepcopy(self._replica_config)
            
            # Determine current batch phase: prefill or decode
            # 判断当前 batch 的 phase: prefill or decode
            batch_prefill_replica_id = batch.requests[0].prefill_replica_id
            batch_replica_id = batch.replica_id
            
            if batch_prefill_replica_id == batch_replica_id:
                replica_config.phase = "prefill"
            else:
                replica_config.phase = "decode"
            
            # ============================================================
            # [PD-Aware] Set correct TP/PP/WS/EP per phase
            # PD separation: prefill/decode have independent world_size and EP
            #   - prefill: ws = p_tp * p_pp * num_p, ep = ws
            #   - decode:  ws = d_tp * d_pp * num_d, ep = ws
            # Non-PD: ws = tp * pp * total_dp, ep = ws
            #
            # [PD-Aware] 按 phase 设置正确的 TP/PP/WS/EP
            # PD 分离时: prefill/decode 有独立的 world_size 和 EP
            # 非 PD 场景: ws = tp * pp * total_dp, ep = ws
            # ============================================================
            orig_tp = self._replica_config.tensor_parallel_size
            orig_pp = self._replica_config.num_pipeline_stages
            total_dp = self.simulation_config.cluster_config.num_replicas
            
            if replica_config.phase == "prefill" and hasattr(self._replica_config, 'prefill_world_size'):
                # PD separation: use prefill cluster params / PD 分离: 使用 prefill 集群的参数
                tp = getattr(self._replica_config, '_prefill_tp', orig_tp)
                pp = getattr(self._replica_config, '_prefill_pp', orig_pp)
                ws = self._replica_config.prefill_world_size
                ep = getattr(self._replica_config, 'prefill_ep', ws)
            elif replica_config.phase == "decode" and hasattr(self._replica_config, 'decode_world_size'):
                # PD separation: use decode cluster params / PD 分离: 使用 decode 集群的参数
                tp = getattr(self._replica_config, '_decode_tp', orig_tp)
                pp = getattr(self._replica_config, '_decode_pp', orig_pp)
                ws = self._replica_config.decode_world_size
                ep = getattr(self._replica_config, 'decode_ep', ws)
            else:
                # Non-PD: EP = ws = tp * pp * dp / 非 PD 场景
                tp = orig_tp
                pp = orig_pp
                ws = tp * pp * total_dp
                ep = ws
            
            # Write per-phase params to copied replica_config
            # 将 per-phase 参数写入 copy 后的 replica_config
            replica_config.world_size = ws
            replica_config.expert_model_parallel_size = ep
            replica_config.tensor_parallel_size = tp
            replica_config.num_pipeline_stages = pp
            
            # Print current batch AICB params for debugging
            # Note: non-PD mode also has prefill_world_size (unified interface), use pd_node_ratio to determine
            # 打印当前 batch 的 AICB 参数, 方便调试确认
            # 注意: 非PD模式也有 prefill_world_size (统一接口), 用 pd_node_ratio 判断
            pd_mode = "PD-separated" if self._replica_config.pd_node_ratio < 1 else "MIXED(non-PD)"
            logger.debug(f"[AICB Params] phase={replica_config.phase}, tp={tp}, pp={pp}, "
                  f"ws={ws}, ep={ep}, total_dp={total_dp}, mode={pd_mode}")
            if replica_config.phase == "prefill":
                bs = 1
                seq = 0
                for request, num_tokens_to_process in zip(batch.requests, batch.num_tokens):
                    if request._is_prefill_complete:
                        continue
                    seq += num_tokens_to_process
                # Prefill phase does not need first-last interpolation
                # prefill阶段不需要首尾插值
                replica_config.decode_last_seq = None
            elif replica_config.phase == "decode":
                bs = 0
                seq = 0
                decode_last_seq = 0  # Last decode iteration's seq for first-last interpolation / 最后一轮decode的seq值，用于首尾插值预加载
                for request, num_tokens_to_process in zip(batch.requests, batch.num_tokens):
                    if request._is_prefill_complete:
                        bs += 1
                        # Current iteration seq = prefill_tokens + processed_decode_tokens - 1
                        # 当前迭代的seq = prefill_tokens + processed_decode_tokens - 1
                        seq += request.num_processed_prefill_tokens + request.num_processed_decode_tokens - 1
                        # Last iteration seq = prefill_tokens + (decode_tokens - 1) - 1
                        # Because at last iteration processed_decode_tokens = decode_tokens - 1
                        # 最后一轮的seq = prefill_tokens + (decode_tokens - 1) - 1
                        # 因为最后一轮时 processed_decode_tokens = decode_tokens - 1
                        decode_last_seq += request.num_processed_prefill_tokens + (request.num_decode_tokens - 1) - 1
                
                # [First-Last Interpolation] Save last decode iteration seq
                # [首尾插值] 保存最后一轮decode的seq值
                replica_config.decode_last_seq = decode_last_seq
                logger.debug(f"[AICB first-last interpolation] decode current seq={seq}, last iter seq={decode_last_seq}")
            
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
