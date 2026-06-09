import json

from vidur.config import BaseRequestGeneratorConfig, ClusterConfig, MetricsConfig
from vidur.entities.base_entity import BaseEntity
from vidur.entities.replica import Replica
from vidur.logger import init_logger

logger = init_logger(__name__)


# Cluster contains multiple Replicas
class Cluster(BaseEntity):

    def __init__(
        self,
        cluster_config: ClusterConfig,
        metrics_config: MetricsConfig,
        generator_config: BaseRequestGeneratorConfig,
    ) -> None:
        """
        Initialize cluster with replicas based on PD disaggregation config.
        根据 PD 分离配置初始化集群及其 replica

        - pd_node_ratio == 1: MIXED mode, all replicas handle both prefill & decode
          pd_node_ratio == 1: MIXED 模式，所有 replica 同时处理 prefill 和 decode
        - 0 < pd_node_ratio < 1: PD separation, independent prefill/decode clusters
          0 < pd_node_ratio < 1: PD 分离，独立的 prefill/decode 集群
        """
        
        # >: test when cluster is registered
        self._id = Cluster.generate_id()
        self._config = cluster_config

        # get metrics config
        self._output_dir = metrics_config.output_dir

        # Init replica object handles
        self._replicas = {}
        
        rc = self._config.replica_config  # shorthand
        num_replicas = self._config.num_replicas

        # ============================================================
        # PD disaggregation OFF (pd_node_ratio == 1): all replicas are MIXED type
        # PD 分离关闭: 所有 replica 都是 MIXED 类型
        # Each replica handles both prefill and decode
        # 每个 replica 同时处理 prefill 和 decode
        # EP = ws = tp * pp * dp (full cluster world_size)
        # ============================================================
        if rc.pd_node_ratio == 1:
            dp = num_replicas
            full_ws = rc.tensor_parallel_size * rc.num_pipeline_stages * dp
            # [EP Auto] Final EP = full cluster world_size
            # [EP Auto] 最终 EP = 全集群 world_size
            rc.expert_model_parallel_size = full_ws
            
            # [Key] Set per-phase attributes in non-PD mode for uniform interface
            # [关键] 非PD模式也设置 per-phase 属性, 与PD模式保持统一接口
            # In non-PD mode all replicas handle prefill/decode, sharing the same world_size
            # 非PD时所有 replica 同时处理 prefill/decode, 共享同一个 world_size
            rc.prefill_world_size = full_ws
            rc.decode_world_size  = full_ws
            rc.prefill_ep = full_ws
            rc.decode_ep  = full_ws
            rc._num_prefill_replicas = num_replicas  # All replicas do prefill / 所有 replica 都做 prefill
            rc._num_decode_replicas  = num_replicas  # All replicas do decode / 所有 replica 都做 decode
            rc._prefill_tp = rc.tensor_parallel_size
            rc._prefill_pp = rc.num_pipeline_stages
            rc._decode_tp  = rc.tensor_parallel_size
            rc._decode_pp  = rc.num_pipeline_stages
            
            logger.debug(f"{'='*70}")
            logger.debug(f"[Cluster] PD off, MIXED mode (pd_node_ratio=1)")
            logger.debug(f"[Cluster] tp={rc.tensor_parallel_size}, pp={rc.num_pipeline_stages}, "
                         f"dp={dp}, ws={full_ws}, ep={full_ws}")
            logger.debug(f"[Cluster] prefill_ws={rc.prefill_world_size}, decode_ws={rc.decode_world_size} (same)")
            logger.debug(f"{'='*70}")
            
            for _ in range(num_replicas):
                replica = Replica(rc, generator_config)
                self._replicas[replica.id] = replica
        
        # ============================================================
        # PD disaggregation ON (0 < pd_node_ratio < 1)
        # PD 分离开启
        # Prefill/Decode are independent clusters, may have different TP/PP/EP
        # Prefill/Decode 是独立集群, 可有不同 TP/PP/EP
        #
        # Replica count priority / replica 数量确定优先级:
        #   1. num_prefill_replicas (user specified, most flexible)
        #      num_prefill_replicas (用户直接指定, 最灵活)
        #   2. pd_node_ratio (calculated by ratio)
        #      pd_node_ratio (按比例计算)
        # ============================================================
        elif rc.pd_node_ratio > 0 and rc.pd_node_ratio < 1:
            # --- Replica count allocation ---
            # --- replica 数量分配 ---
            if rc.num_prefill_replicas is not None:
                # User specified prefill replica count
                # 用户直接指定 prefill replica 数量
                num_p = rc.num_prefill_replicas
                num_d = num_replicas - num_p
                replica_source = f"num_prefill_replicas={rc.num_prefill_replicas} (user specified, 用户指定)"
            else:
                # Calculate from pd_node_ratio / 通过 pd_node_ratio 计算
                num_p = int(num_replicas * rc.pd_node_ratio)
                num_d = num_replicas - num_p
                replica_source = f"pd_node_ratio={rc.pd_node_ratio} (by ratio, 按比例)"
            
            rc._num_prefill_replicas = num_p
            rc._num_decode_replicas = num_d
            if num_p <= 0 or num_d <= 0:
                raise ValueError(
                    f"[Cluster] _num_prefill_replicas={num_p} and "
                    f"_num_decode_replicas={num_d} must both be > 0, "
                    f"source: {replica_source}")
            
            # --- per-phase TP/PP (fallback to shared values) ---
            # --- per-phase TP/PP (回退到共享值) ---
            p_tp = rc.prefill_tensor_parallel_size or rc.tensor_parallel_size
            p_pp = rc.prefill_num_pipeline_stages or rc.num_pipeline_stages
            d_tp = rc.decode_tensor_parallel_size or rc.tensor_parallel_size
            d_pp = rc.decode_num_pipeline_stages or rc.num_pipeline_stages
            
            # --- per-phase world_size and EP ---
            # --- per-phase world_size 和 EP ---
            # EP = world_size = tp * pp * dp (ref vLLM: EP_SIZE = TP_SIZE x DP_SIZE)
            # EP = world_size = tp * pp * dp (参考 vLLM)
            rc.prefill_world_size = p_tp * p_pp * num_p
            rc.decode_world_size  = d_tp * d_pp * num_d
            rc.prefill_ep = rc.prefill_world_size
            rc.decode_ep  = rc.decode_world_size
            
            # Save per-phase actual TP/PP for later use
            # 保存 per-phase 的实际 TP/PP, 方便后续使用
            rc._prefill_tp = p_tp
            rc._prefill_pp = p_pp
            rc._decode_tp = d_tp
            rc._decode_pp = d_pp
            
            if rc.prefill_world_size <= 0 or rc.decode_world_size <= 0:
                raise ValueError(
                    f"[Cluster] prefill_ws={rc.prefill_world_size} and "
                    f"decode_ws={rc.decode_world_size} must both be > 0")
            
            # --- Verbose PD config printout ---
            # --- 详尽打印 PD 配置 ---
            logger.debug(f"{'='*70}")
            logger.debug(f"[PD Config] PD enabled ({replica_source})")
            logger.debug(f"[PD Config] Total replicas: {num_replicas} "
                         f"(prefill={num_p}, decode={num_d})")
            logger.debug(f"[PD Config] Prefill: tp={p_tp}, pp={p_pp}, dp={num_p}, "
                         f"ws={rc.prefill_world_size}, ep={rc.prefill_ep}")
            logger.debug(f"[PD Config] Decode:  tp={d_tp}, pp={d_pp}, dp={num_d}, "
                         f"ws={rc.decode_world_size}, ep={rc.decode_ep}")
            logger.debug(f"{'='*70}")
            
            for _ in range(num_replicas):
                replica = Replica(rc, generator_config)
                self._replicas[replica.id] = replica

        else:
            raise ValueError(
                f"[Cluster] Invalid pd_node_ratio={rc.pd_node_ratio}. "
                f"Must be in range (0, 1]. "
                f"Use 1 for MIXED mode, or (0, 1) for PD separation."
            )

        if metrics_config.write_json_trace:
            self._write_cluster_info_to_file()
    

    @property
    def replicas(self):
        return self._replicas

    def to_dict(self) -> dict:
        return {
            "id": self._id,
            "num_replicas": len(self._replicas),
        }

    def _write_cluster_info_to_file(self) -> None:
        replica_dicts = [replica.to_dict() for replica in self._replicas.values()]
        cluster_info = {"replicas": replica_dicts}

        cluster_file = f"{self._output_dir}/cluster.json"
        with open(cluster_file, "w") as f:
            json.dump(cluster_info, f)
