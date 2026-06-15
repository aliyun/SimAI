from vidur.entities.base_entity import BaseEntity
from vidur.config import (
    BaseExecutionTimePredictorConfig,
    BaseReplicaSchedulerConfig,
    MetricsConfig,
    ReplicaConfig,
)
from vidur.logger import init_logger

import os
import sys
import subprocess
import json
import time as time_module
from pathlib import Path
import csv
from typing import Dict, Optional, Tuple

logger = init_logger(__name__)

# 获取当前文件目录，用于计算 aicb 的绝对路径
# Get current file directory for calculating absolute path to aicb
_CURRENT_FILE_DIR = Path(__file__).resolve().parent
# execution_time.py is under vidur-alibabacloud/vidur/entities/
# aicb is under workspace_root/aicb/
# Path: entities/ -> vidur/ -> vidur-alibabacloud/ -> workspace_root/ -> aicb/
_AICB_ROOT = _CURRENT_FILE_DIR.parent.parent.parent / "aicb"


# ============================================================
# [AICB Optimization B+C] Global Cache + Linear Interpolation
# [AICB优化 B+C方案] 全局缓存 + 首尾插值
#
# Plan C: Global lookup - avoid repeated AICB CSV reads/runs
# Plan B: Head-tail token strategy - linear interpolation for intermediate seq values
# 方案C: 全局查表 - 避免重复读取/运行AICB
# 方案B: 首尾token策略 - 对中间seq值线性插值
#
# Cache key: (model_name, ws, tp, pp, ep, bs, seq, phase)
# Cache value: {layer_id: {layer_name: {comp_time, comm_size}}}
# ============================================================

# AICB cache data directory
# AICB缓存数据存放目录
_AICB_WORKLOAD_DIR = _CURRENT_FILE_DIR.parent.parent / "data" / "aicb_workload"
_AICB_CACHE_DIR = _AICB_WORKLOAD_DIR / "cache"
_AICB_LOG_DIR = _AICB_WORKLOAD_DIR / "logs"


class AICBGlobalCache:
    """
    [AICB Optimization B+C] Global AICB Data Cache
    [AICB优化 B+C方案] 全局AICB数据缓存
    
    Features / 功能:
    1. Plan C (Lookup): Cache loaded AICB data to avoid repeated CSV reads and subprocess calls
       方案C (查表): 缓存已加载的AICB数据，避免重复CSV读取和subprocess调用
    2. Plan B (Interpolation): Use head-tail token linear interpolation for unmatched seq values
       方案B (插值): 对于没有精确匹配的seq值，使用首尾token线性插值
    3. Persistence: Save cache index and data to disk for cross-run reuse
       持久化: 将缓存索引和数据保存到磁盘，跨运行复用
    4. Logging: Record all AICB calls and cache hit stats
       日志: 记录所有AICB调用和缓存命中情况
    """
    
    def __init__(self):
        # Core cache: (model, ws, tp, pp, ep, bs, seq, phase) -> parsed data
        # 核心缓存
        self._cache: Dict[Tuple, Dict] = {}
        
        # Statistics counters / 统计计数器
        self._stats = {
            'cache_hits': 0,        # Exact cache hits / 精确命中查表
            'interpolations': 0,    # Interpolation hits / 插值命中
            'aicb_calls': 0,        # Actual AICB subprocess calls / 实际AICB subprocess调用
            'csv_loads': 0,         # CSV file load count / CSV文件加载次数
        }
        
        # Ensure directories exist / 确保目录存在
        _AICB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _AICB_LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Log file / 日志文件
        self._log_file = _AICB_LOG_DIR / "aicb_cache_log.txt"
        
        # Load persistent index / 加载持久化索引
        self._index_file = _AICB_WORKLOAD_DIR / "cache_index.json"
        self._load_index()
        
        self._log(f"AICBGlobalCache 初始化完成, 缓存目录: {_AICB_CACHE_DIR}")
    
    def _log(self, msg: str) -> None:
        """Write to log file and print / 写入日志文件并打印"""
        timestamp = time_module.strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {msg}"
        try:
            with open(self._log_file, 'a') as f:
                f.write(log_line + '\n')
        except:
            pass
        logger.debug(f"[AICB Cache] {msg}")
    
    def _make_key(self, model_name, ws, tp, pp, ep, bs, seq, phase) -> Tuple:
        """Generate cache key / 生成缓存key"""
        return (model_name, ws, tp, pp, ep, bs, seq, phase)
    
    def _make_group_key(self, model_name, ws, tp, pp, ep, bs, phase) -> Tuple:
        """Generate group key without seq, for finding seq values to interpolate
        生成不含seq的分组key，用于查找同组的seq值进行插值"""
        return (model_name, ws, tp, pp, ep, bs, phase)
    
    def get(self, model_name, ws, tp, pp, ep, bs, seq, phase) -> Optional[Dict]:
        """
        Retrieve AICB data from cache.
        从缓存获取AICB数据
        
        Lookup strategy (Plan B+C):
        查找策略 (B+C方案):
        1. Exact match (Plan C): return cached data directly
           精确匹配 (方案C): 直接返回缓存数据
        2. Linear interpolation (Plan B): interpolate between two nearest seq values
           线性插值 (方案B): 找同组中seq最近的两个值，线性插值
        3. Nearest neighbor: return the only neighbor if just one exists
           最近邻: 如果只有一个邻居，直接返回
        4. Miss: return None
           未命中: 返回None
        
        Returns:
            Dict or None: AICB data, or None if cache miss
        """
        key = self._make_key(model_name, ws, tp, pp, ep, bs, seq, phase)
        
        # === 1. Exact match (Plan C: lookup) ===
        # === 1. 精确匹配 (方案C: 查表) ===
        if key in self._cache:
            self._stats['cache_hits'] += 1
            self._log(f"[命中] 精确匹配: key={self._format_key(key)}, "
                      f"总命中={self._stats['cache_hits']}")
            return self._cache[key]
        
        # === 2. Try interpolation (Plan B: head-tail token strategy) ===
        # === 2. 尝试插值 (方案B: 首尾token策略) ===
        group_key = self._make_group_key(model_name, ws, tp, pp, ep, bs, phase)
        neighbors = self._find_neighbors(group_key, seq)
        
        if neighbors is not None:
            interpolated = neighbors
            self._stats['interpolations'] += 1
            # Cache interpolated result to avoid repeated computation
            # 缓存插值结果，避免重复计算
            self._cache[key] = interpolated
            self._log(f"[插值] seq={seq}, 使用邻居插值, "
                      f"总插值={self._stats['interpolations']}")
            return interpolated
        
        # === 3. Cache miss ===
        # === 3. 未命中 ===
        self._log(f"[未命中] key={self._format_key(key)}")
        return None
    
    def put(self, model_name, ws, tp, pp, ep, bs, seq, phase, data: Dict) -> None:
        """Store AICB data into cache / 将AICB数据存入缓存"""
        key = self._make_key(model_name, ws, tp, pp, ep, bs, seq, phase)
        self._cache[key] = data
        self._log(f"[缓存] key={self._format_key(key)}, layers={len(data)}")
        
        # Persist to disk cache / 保存到磁盘缓存
        self._save_cache_entry(key, data)
    
    def record_aicb_call(self) -> None:
        """Record an actual AICB subprocess call / 记录一次实际的AICB subprocess调用"""
        self._stats['aicb_calls'] += 1
        self._log(f"[AICB调用] 第{self._stats['aicb_calls']}次subprocess调用")
    
    def record_csv_load(self) -> None:
        """Record a CSV file load / 记录一次CSV文件加载"""
        self._stats['csv_loads'] += 1
    
    def _find_neighbors(self, group_key, target_seq) -> Optional[Dict]:
        """
        [Plan B core] Find neighbors of target_seq in the same group for interpolation.
        [方案B核心] 在同组中查找target_seq的邻居，进行线性插值
        
        Strategy / 策略:
        - Collect all cached seq values in the same group (model, ws, tp, pp, ep, bs, phase)
          找到同组中所有已缓存的seq值
        - Find nearest seq values on both sides of target_seq
          找到target_seq两侧最近的seq值
        - Two neighbors: linear interpolation / 两个邻居: 线性插值
        - One neighbor: nearest neighbor / 一个邻居: 使用最近邻
        - No neighbors: return None / 没有邻居: 返回None
        """
        # Collect all cached seq values in the same group
        # 收集同组的所有已缓存seq值
        cached_seqs = {}
        for key, data in self._cache.items():
            # key = (model, ws, tp, pp, ep, bs, seq, phase)
            key_group = (key[0], key[1], key[2], key[3], key[4], key[5], key[7])
            if key_group == group_key:
                cached_seqs[key[6]] = data  # key[6] = seq
        
        if not cached_seqs:
            return None
        
        seq_values = sorted(cached_seqs.keys())
        
        # Find neighbors on both sides of target_seq
        # 找到target_seq两侧的邻居
        lower_seq = None
        upper_seq = None
        for s in seq_values:
            if s <= target_seq:
                lower_seq = s
            if s >= target_seq and upper_seq is None:
                upper_seq = s
        
        # Exact match (should not reach here, but just in case)
        # 精确匹配（不应该到这里，但安全起见）
        if target_seq in cached_seqs:
            return cached_seqs[target_seq]
        
        # Two neighbors: linear interpolation
        # 两个邻居: 线性插值
        if lower_seq is not None and upper_seq is not None and lower_seq != upper_seq:
            alpha = (target_seq - lower_seq) / (upper_seq - lower_seq)
            interpolated = self._interpolate(cached_seqs[lower_seq], cached_seqs[upper_seq], alpha)
            self._log(f"  插值: seq={target_seq}, lower={lower_seq}, upper={upper_seq}, "
                      f"alpha={alpha:.4f}")
            return interpolated
        
        # Only one neighbor: nearest neighbor
        # 只有一个邻居: 最近邻
        nearest = lower_seq if lower_seq is not None else upper_seq
        if nearest is not None:
            self._log(f"  最近邻: seq={target_seq}, nearest={nearest}")
            return cached_seqs[nearest]
        
        return None
    
    def _interpolate(self, data_low: Dict, data_high: Dict, alpha: float) -> Dict:
        """
        [Plan B core] Linear interpolation between two AICB datasets.
        [方案B核心] 对两组AICB数据进行线性插值
        
        For each layer's metric (comp_time, comm_size):
        对每层的每个指标:
        result = data_low * (1-alpha) + data_high * alpha
        
        Args:
            data_low: AICB data at lower seq / seq较小时的AICB数据
            data_high: AICB data at higher seq / seq较大时的AICB数据
            alpha: interpolation coefficient [0, 1] / 插值系数
        """
        result = {}
        # Merge layer_ids from both datasets
        # 取两个数据共有的layer_id
        all_layers = set(data_low.keys()) | set(data_high.keys())
        
        for layer_id in all_layers:
            result[layer_id] = {}
            low_layer = data_low.get(layer_id, {})
            high_layer = data_high.get(layer_id, {})
            
            # Merge sub-components (attention, mlp, moe, etc.)
            # 取两层共有的子组件
            all_components = set(low_layer.keys()) | set(high_layer.keys())
            
            for comp_name in all_components:
                low_comp = low_layer.get(comp_name, {'comp_time': 0.0, 'comm_size': 0.0})
                high_comp = high_layer.get(comp_name, {'comp_time': 0.0, 'comm_size': 0.0})
                
                result[layer_id][comp_name] = {
                    'comp_time': low_comp['comp_time'] * (1 - alpha) + high_comp['comp_time'] * alpha,
                    'comm_size': low_comp['comm_size'] * (1 - alpha) + high_comp['comm_size'] * alpha,
                }
        
        return result
    
    def _format_key(self, key: Tuple) -> str:
        """Format key as a human-readable string / 格式化key为可读字符串"""
        return (f"model={key[0]}, ws={key[1]}, tp={key[2]}, pp={key[3]}, "
                f"ep={key[4]}, bs={key[5]}, seq={key[6]}, phase={key[7]}")
    
    def _save_cache_entry(self, key: Tuple, data: Dict) -> None:
        """Save a cache entry to disk / 将缓存条目保存到磁盘"""
        try:
            # Encode key info into filename
            # 文件名编码key信息
            filename = (f"aicb-{key[0]}-ws{key[1]}-tp{key[2]}-pp{key[3]}"
                       f"-ep{key[4]}-bs{key[5]}-seq{key[6]}-{key[7]}.json")
            filepath = _AICB_CACHE_DIR / filename
            
            # Convert int keys to str keys (JSON requirement)
            # 将int key转为str key (JSON要求)
            serializable = {}
            for lid, ldata in data.items():
                serializable[str(lid)] = ldata
            
            with open(filepath, 'w') as f:
                json.dump(serializable, f, indent=2)
        except Exception as e:
            self._log(f"[WARNING] 保存缓存条目失败: {e}")
    
    def _load_index(self) -> None:
        """Load cache index and existing data from disk / 从磁盘加载缓存索引和已有数据"""
        try:
            if _AICB_CACHE_DIR.exists():
                json_files = list(_AICB_CACHE_DIR.glob("aicb-*.json"))
                loaded = 0
                for jf in json_files:
                    try:
                        # Parse key from filename
                        # 从文件名解析key
                        key = self._parse_filename(jf.name)
                        if key is None:
                            continue
                        
                        with open(jf, 'r') as f:
                            raw_data = json.load(f)
                        
                        # Restore int keys
                        # 恢复int key
                        data = {}
                        for lid_str, ldata in raw_data.items():
                            data[int(lid_str)] = ldata
                        
                        self._cache[key] = data
                        loaded += 1
                    except:
                        continue
                
                if loaded > 0:
                    self._log(f"从磁盘加载了 {loaded} 条缓存记录")
        except:
            pass
    
    def _parse_filename(self, filename: str) -> Optional[Tuple]:
        """Parse key from cache filename / 从缓存文件名解析key"""
        try:
            # aicb-ModelName-ws32-tp4-pp1-ep4-bs1-seq100-prefill.json
            if not filename.startswith("aicb-") or not filename.endswith(".json"):
                return None
            
            name = filename[5:-5]  # 去掉 "aicb-" 和 ".json"
            parts = name.rsplit('-', 7)  # 从右边分割，取最后7个部分
            if len(parts) < 8:
                return None
            
            # Last 7 parts: ws{}, tp{}, pp{}, ep{}, bs{}, seq{}, phase
            # 最后7个部分
            model_name = parts[0]
            phase = parts[-1]
            seq = int(parts[-2].replace('seq', ''))
            bs = int(parts[-3].replace('bs', ''))
            ep = int(parts[-4].replace('ep', ''))
            pp = int(parts[-5].replace('pp', ''))
            tp = int(parts[-6].replace('tp', ''))
            ws = int(parts[-7].replace('ws', ''))
            
            return (model_name, ws, tp, pp, ep, bs, seq, phase)
        except:
            return None
    
    def print_stats(self) -> None:
        """Print cache statistics / 打印缓存统计信息"""
        total_queries = (self._stats['cache_hits'] + self._stats['interpolations'] 
                        + self._stats['aicb_calls'])
        logger.debug(f"\n{'='*70}")
        logger.debug(f"[AICB Cache Stats Report]")
        logger.debug(f"{'='*70}")
        logger.debug(f"  Cache entries:     {len(self._cache)}")
        logger.debug(f"  Exact hits: {self._stats['cache_hits']}")
        logger.debug(f"  Interpolated hits:       {self._stats['interpolations']}")
        logger.debug(f"  AICB real calls:   {self._stats['aicb_calls']}")
        logger.debug(f"  CSV file loads:    {self._stats['csv_loads']}")
        if total_queries > 0:
            hit_rate = (self._stats['cache_hits'] + self._stats['interpolations']) / total_queries * 100
            logger.debug(f"  Cache hit rate:     {hit_rate:.1f}%")
        logger.debug(f"  Cache dir:       {_AICB_CACHE_DIR}")
        logger.debug(f"  Log file:       {self._log_file}")
        logger.debug(f"{'='*70}\n")
        
        # Also write to log file / 也写入日志
        self._log(f"统计: hits={self._stats['cache_hits']}, "
                  f"interp={self._stats['interpolations']}, "
                  f"calls={self._stats['aicb_calls']}, "
                  f"entries={len(self._cache)}")
    
    def save_lookup_table(self) -> None:
        """Save complete lookup index to JSON for inspection / 保存完整的查表索引到JSON，方便查看"""
        try:
            table = {}
            for key, data in self._cache.items():
                key_str = self._format_key(key)
                table[key_str] = {
                    'num_layers': len(data),
                    'layer_ids': sorted([int(k) for k in data.keys()]),
                }
            
            table_file = _AICB_WORKLOAD_DIR / "lookup_table.json"
            with open(table_file, 'w') as f:
                json.dump(table, f, indent=2, ensure_ascii=False)
            self._log(f"查表索引已保存到 {table_file}")
        except Exception as e:
            self._log(f"[WARNING] 保存查表索引失败: {e}")


# ============================================================
# Global singleton: all ExecutionTime objects share the same cache
# 全局单例: 所有 ExecutionTime 对象共享同一个缓存
# ============================================================
_GLOBAL_AICB_CACHE = AICBGlobalCache()

# [首尾插值] 记录预加载失败的key，避免重复尝试
# Record failed preload keys to avoid repeated attempts
_FAILED_PRELOAD_KEYS = set()

class ExecutionTime(BaseEntity):
    def __init__(
        self,
        num_layers_per_pipeline_stage: int,
        attention_rope_execution_time: float,
        attention_kv_cache_save_execution_time: float,
        attention_decode_execution_time: float,
        attention_prefill_execution_time: float,
        attention_layer_pre_proj_execution_time: float,
        attention_layer_post_proj_execution_time: float,
        mlp_layer_up_proj_execution_time: float,
        mlp_layer_down_proj_execution_time: float,
        mlp_layer_act_execution_time: float,
        attn_norm_time: float,
        mlp_norm_time: float,
        add_time: float,
        tensor_parallel_communication_time: float,
        pipeline_parallel_communication_time: float,
        schedule_time: float,
        sampler_e2e_time: float,
        prepare_inputs_e2e_time: float,
        process_model_outputs_time: float,
        ray_comm_time: float,
        predictor_config: BaseExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
    ) -> None:
        self._id = ExecutionTime.generate_id()

        self._num_layers_per_pipeline_stage = num_layers_per_pipeline_stage
        self._attention_rope_execution_time = attention_rope_execution_time
        self._attention_kv_cache_save_execution_time = (
            attention_kv_cache_save_execution_time
        )
        self._attention_decode_execution_time = attention_decode_execution_time
        self._attention_prefill_execution_time = attention_prefill_execution_time
        self._attention_layer_pre_proj_execution_time = (
            attention_layer_pre_proj_execution_time
        )
        self._attention_layer_post_proj_execution_time = (
            attention_layer_post_proj_execution_time
        )
        self._mlp_layer_up_proj_execution_time = mlp_layer_up_proj_execution_time
        self._mlp_layer_down_proj_execution_time = mlp_layer_down_proj_execution_time
        self._mlp_layer_act_execution_time = mlp_layer_act_execution_time
        self._mlp_norm_time = mlp_norm_time
        self._attn_norm_time = attn_norm_time
        self._add_time = add_time
        self._tensor_parallel_communication_time = tensor_parallel_communication_time
        self._pipeline_parallel_communication_time = (
            pipeline_parallel_communication_time
        )
        self._schedule_time = schedule_time
        self._sampler_e2e_time = sampler_e2e_time
        self._prepare_inputs_e2e_time = prepare_inputs_e2e_time
        self._process_model_outputs_time = process_model_outputs_time
        self._ray_comm_time = ray_comm_time
        
        self._config = predictor_config
        self._replica_config = replica_config
        self._model_config = replica_config.model_config
        self.replica_scheduler_config = replica_scheduler_config
        
        # Cache AICB data to avoid repeated loading
        # 缓存 AICB 数据，避免重复加载
        # Optional[Dict[str, float]]: can be None or a dict mapping str to float
        # Optional[Dict[str, float]] 表示这个变量可以是 None 或者是一个键为字符串、值为浮点数的字典。
        self._aicb_data: Optional[Dict[str, float]] = None
        
    # Two allreduces in mlp and attention layers are implemented here
    # mlp和attention中的两次allreduce在这里实现
    def _get_mlp_layer_execution_time(self) -> float:
        assert self._mlp_layer_up_proj_execution_time \
            + self._mlp_layer_down_proj_execution_time \
            + self._mlp_layer_act_execution_time \
            + self._tensor_parallel_communication_time \
            + self._mlp_norm_time > 0, "MLP layer execution time must be positive"
        return (
            self._mlp_layer_up_proj_execution_time
            + self._mlp_layer_down_proj_execution_time
            + self._mlp_layer_act_execution_time
            + self._tensor_parallel_communication_time
            + self._mlp_norm_time
        )

    def _get_attention_layer_execution_time(self) -> float:
        assert             self._attention_layer_pre_proj_execution_time \
            + self._attention_layer_post_proj_execution_time \
            + self._attention_rope_execution_time \
            + self._attention_kv_cache_save_execution_time \
            + self._attention_decode_execution_time \
            + self._attention_prefill_execution_time \
            + self._tensor_parallel_communication_time \
            + self._attn_norm_time > 0, "Attention layer execution time must be positive"
        return (
            self._attention_layer_pre_proj_execution_time
            + self._attention_layer_post_proj_execution_time
            + self._attention_rope_execution_time
            + self._attention_kv_cache_save_execution_time
            + self._attention_decode_execution_time
            + self._attention_prefill_execution_time
            + self._tensor_parallel_communication_time
            + self._attn_norm_time
        )
    
    def _get_attention_layer_execution_time_from_aicb(self,layer_id) -> float:
    
        if self._aicb_data is None:
            self._aicb_data = self._load_aicb_data()
        
        # If AICB data is empty, return a small default to avoid division by zero
        # 如果 AICB 数据为空，返回一个小的默认值避免除零
        if not self._aicb_data:
            logger.warning("AICB data is empty, using default attention execution time")
            return 1e-6  # 1 microsecond as default
                    
        layer_data = self._aicb_data.get(layer_id, {}).get("attention", {})
                
        # Convert unit from ns to s / 单侍从ns转换为s
        attention_comp_time = layer_data.get('comp_time', 0.0) * 1e-9 
        
        # Unit: Byte / 单位Byte
        attention_comm_size = layer_data.get('comm_size', 0.0) 
        
        attention_time = attention_comp_time
        return attention_time if attention_time > 0 else 1e-6
    
    # def _get_mlp_layer_execution_time_from_dpsk_and_aiob(self) -> float:
    # def _get_mlp_layer_execution_time_from_aicb(self) -> float:
    def _get_mlp_layer_execution_time_from_aicb(self, layer_id) -> float:
        if self._aicb_data is None:
            self._aicb_data = self._load_aicb_data()
        
        # If AICB data is empty, return a small default to avoid division by zero
        # 如果 AICB 数据为空，返回一个小的默认值避免除零
        if not self._aicb_data:
            logger.warning("AICB data is empty, using default MLP execution time")
            return 1e-6  # 1 microsecond as default
                    
        layer_data = self._aicb_data.get(layer_id, {}).get("mlp", {})
                
        # Convert unit from ns to s / 单侍从ns转换为s
        mlp_comp_time = layer_data.get('comp_time', 0.0) * 1e-9
        
        # Unit: Byte / 单位Byte
        mlp_comm_size = layer_data.get('comm_size', 0.0)
    
        mlp_time = mlp_comp_time
        return mlp_time if mlp_time > 0 else 1e-6
    
    def _get_moe_layer_execution_time_from_aicb(self, layer_id) -> float:
        if self._aicb_data is None:
            self._aicb_data = self._load_aicb_data()
        # return self._aicb_data.get("moe")
        
        # If AICB data is empty, return a small default to avoid division by zero
        # 如果 AICB 数据为空，返回一个小的默认值避免除零
        if not self._aicb_data:
            logger.warning("AICB data is empty, using default MoE execution time")
            return 1e-6  # 1 microsecond as default
        
        # Get corresponding values from the data structure
        # 从数据结构中获取对应的值
        layer_data = self._aicb_data.get(layer_id, {}).get("moe", {})
        # +comm
        # return layer_data.get('comp_time', 0.0)

        replica_stage = "prefill"  # NOTE: hardcoded prefill; decode uses different bandwidth
    
        # Convert unit from ns to s / 单侍从ns转换为s
        moe_comp_time = layer_data.get('comp_time', 0.0) * 1e-9 
        
        # Unit: Byte / 单位Byte
        moe_comm_size = layer_data.get('comm_size', 0.0) 
    
        if replica_stage == "prefill": # normal kernel
            # Convert Gbps to Byte/s / Gbps换算成 Byte/s
            cur_bw = self._replica_config.rdma_bandwidth * 1024 * 1024 * 1024 / 8 
        elif replica_stage == "decode": # low_latency kernel
            # Convert Gbps to Byte/s / Gbps换算成 Byte/s
            cur_bw = self._replica_config.nvlink_bandwidth * 1024 * 1024 * 1024 / 8 
        moe_comm_time = moe_comm_size / cur_bw # 秒
        moe_time = moe_comp_time + moe_comm_time
        return moe_time if moe_time > 0 else 1e-6
    
    def _get_aicb_params(self):
        """
        Get AICB invocation parameters.
        获取 AICB 调用参数
        
        Automatically reads per-phase TP/PP/WS/EP from replica_config.
        These values are set correctly per phase in base_execution_time_predictor.py.
        自动从 replica_config 读取 per-phase 的 TP/PP/WS/EP
        这些值已在 base_execution_time_predictor.py 中按 phase 正确设置
        
        Returns:
            (model_name, model_json_file, tp, pp, ws, ep, bs, seq, phase)
        """
        if self._replica_config.model_name == 'deepseek-671B':
            model_name = "DeepSeek-671B"
            model_json_file = "./scripts/inference_configs/deepseek_default.json"
        elif self._replica_config.model_name == 'qwen3-moe-235B':
            model_name = "Qwen3-Moe-235B"
            model_json_file = "./scripts/inference_configs/qwen3_moe_default.json"
        elif self._replica_config.model_name == 'qwen3-next-80B':
            model_name = "Qwen3-Next-80B"
            model_json_file = "./scripts/inference_configs/qwen3_next_default.json"
            
        # [PD-Aware] These values are set correctly per phase in base_execution_time_predictor.py
        # [PD-Aware] 这些值已在 base_execution_time_predictor.py 中按 phase 正确设置
        tp = self._replica_config.tensor_parallel_size
        pp = self._replica_config.num_pipeline_stages
        ws = self._replica_config.world_size
        # ep = self._replica_config.expert_model_parallel_size
        ep = self._replica_config.expert_model_parallel_size  # [EP Auto] = per-phase world_size
        bs = self._replica_config.batch_size
        seq = self._replica_config.seq_len
        phase = self._replica_config.phase

        return model_name, model_json_file, tp, pp, ws, ep, bs, seq, phase
    
    def _get_aicb_csv_path(self) -> str:
        """Generate expected AICB CSV path based on current configuration
        根据当前配置生成 AICB CSV 的预期路径"""
        model_name, _, tp, pp, ws, ep, bs, seq, phase = self._get_aicb_params()
        logger.debug(f'get aicb csv path: {model_name} world_size{ws}-tp{tp}-pp{pp}-ep{ep}-bs{bs}-seq{seq}-{phase}')

        filename = (
            f"vidur-{model_name}-world_size{ws}-tp{tp}-pp{pp}-ep{ep}"
            f"-bs{bs}-seq{seq}-{phase}.csv"
        )
        # Use absolute path based on code file location
        # 使用基于代码文件位置的绝对路径
        return str(_AICB_ROOT / "results" / "workload" / filename)
    
    def _generate_aicb_csv(self):
        """Generate AICB CSV file / 生成AICB CSV文件"""
        model_name, model_json_file, tp, pp, ws, ep, bs, seq, phase = self._get_aicb_params()
        logger.debug(f"_generate_aicb_csv: model_name={model_name} model_json_file={model_json_file} tp={tp} pp={pp} ws={ws} ep={ep} bs={bs} seq={seq} phase={phase}")
        # Use absolute path based on code file location
        # 使用基于代码文件位置的绝对路径
        cwd = str(_AICB_ROOT)
        cwd_path = Path(cwd)
        
        logger.debug(f'\n{"="*80}')
        logger.debug(f'===== AICB CSV Generation Debug Info =====')
        logger.debug(f'{"="*80}')
        
        # Check if AICB directory exists / 检查 AICB 目录是否存在
        if not cwd_path.exists():
            logger.error(f'AICB directory does not exist: {cwd}')
            logger.error(f'Please ensure AICB is properly installed')
            return False
        
        logger.debug(f'AICB Root Directory: {cwd_path}')
        logger.debug(f'AICB Root exists: {cwd_path.exists()}')
        
        # Check results/workload directory / 检查 results/workload 目录
        results_dir = cwd_path / "results" / "workload"
        logger.debug(f'Results directory: {results_dir}')
        logger.debug(f'Results directory exists: {results_dir.exists()}')
        
        # Create directory if not exists / 如果不存在，创建目录
        if not results_dir.exists():
            logger.debug(f'Creating results directory: {results_dir}')
            results_dir.mkdir(parents=True, exist_ok=True)
        
        # List existing files in results/workload directory
        # 列出 results/workload 目录下已有的文件
        if results_dir.exists():
            existing_files = list(results_dir.glob('*.csv'))
            logger.debug(f'Existing CSV files in results/workload ({len(existing_files)} files):')
            for f in existing_files[:10]:
                logger.debug(f'  - {f.name}')
            if len(existing_files) > 10:
                logger.debug(f'  ... and {len(existing_files) - 10} more files')
        
        # Build AICB command / 构建AICB命令
        cmd = [
            sys.executable,
            "-m", "workload_generator.Vidur_workload_generator",
            model_name,
            model_json_file,
            "--seq_length", str(seq),
            "--micro_batch", str(bs),
            "--world_size", str(ws),
            "--tensor_model_parallel_size", str(tp),
            "--expert_model_parallel_size", str(ep),
            "--aiob_enable",
            "--phase", phase,
        ]
        
        if pp > 1:
            cmd.extend(["--pipeline_model_parallel", str(pp)])

        # Print command for manual execution / 打印可以手动执行的命令
        cmd_str = " ".join(cmd)
        logger.debug(f'\n===== Command Details =====')
        logger.debug(f'Working directory: cd {cwd_path}')
        logger.debug(f'Full command: {cmd_str}')
        logger.debug(f'One-liner: cd {cwd_path} && {cmd_str}')
        
        # Expected output CSV file path / 预期生成的文件路径
        expected_csv = self._get_aicb_csv_path()
        logger.debug(f'Expected output CSV file: {expected_csv}')
        
        try:
            logger.debug(f'\n===== Executing Command =====')
            result = subprocess.run(cmd, cwd=cwd_path, capture_output=True, text=True, timeout=300)
            
            logger.debug(f'Return code: {result.returncode}')
            if result.stdout.strip():
                logger.debug(f'STDOUT: {result.stdout.strip()}')
            if result.stderr.strip():
                logger.debug(f'STDERR: {result.stderr.strip()}')
            
            # Check results/workload directory after command execution
            # 检查命令执行后 results/workload 目录的变化
            logger.debug(f'===== Post-execution Check =====')
            if results_dir.exists():
                new_files = list(results_dir.glob('*.csv'))
                logger.debug(f'CSV files after execution ({len(new_files)} files)')
                    
                if os.path.exists(expected_csv):
                    logger.debug(f'SUCCESS: Expected CSV file was created!')
                else:
                    logger.warning(f'Expected CSV file was NOT created!')
                    similar_files = list(results_dir.glob(f'*{model_name}*{phase}*.csv'))
                    if similar_files:
                        logger.debug(f'Similar files found: {[f.name for f in similar_files]}')
                    else:
                        logger.debug(f'No similar files found matching pattern: *{model_name}*{phase}*.csv')
            
            if result.returncode != 0:
                logger.error(f'AICB command failed with return code {result.returncode}')
                return False
            else:
                logger.debug(f'AICB command succeeded (returncode=0)')
                return True
                
        except subprocess.TimeoutExpired:
            logger.error('AICB command timed out after 300 seconds')
            return False
        except Exception as e:
            logger.error(f'Failed to run AICB command: {e}', exc_info=True)
            return False


    def _generate_or_find_bs1_csv(
        self, model_name, ws, tp, pp, ep, bs, seq, phase, original_csv_path
    ) -> str:
        """
        [AICB Safe Mode] Always use bs=1 to generate or find CSV.
        [AICB Safe Mode] 始终使用 bs=1 生成或查找 CSV
        
        Reason: DeepSeek-V3-671B prefill AICB profiling fails on H20 (SM90)
        when tp>=4, due to FlashMLA `flash_mla_sparse_fwd` kernel's h_q
        alignment requirement (B_H=64). This does not affect decode or
        other models.
        原因: DeepSeek-V3-671B prefill 在 H20 (SM90) 上当 tp>=4 时 AICB
        profiling 失败，因 FlashMLA flash_mla_sparse_fwd kernel 的 h_q
        对齐要求 (B_H=64)。不影响 decode 或其他模型。
        
        Strategy / 策略:
          1. If requested bs=1, generate directly / 如果请求的就是 bs=1, 直接生成
          2. If bs>1, look for existing bs=1 CSV / 如果 bs>1, 查找已有的 bs=1 CSV
          3. If bs=1 CSV missing, generate it / 如果 bs=1 CSV 不存在, 生成它
        
        Args:
            model_name: Model name / 模型名
            ws, tp, pp, ep: Parallelism config / 并行配置
            bs: Original requested batch size (may be > 1)
            seq: Sequence length / 序列长度
            phase: Stage (prefill/decode) / 阶段
            original_csv_path: Original CSV path (may be bs>1)
        
        Returns:
            Path to found or generated CSV (may be bs=1 CSV)
        """
        logger.debug(f'[AICB Safe Mode] CSV不存在: {original_csv_path}')
        logger.debug(f'[AICB Safe Mode] 请求参数: model={model_name}, bs={bs}, seq={seq}, phase={phase}')
        
        # ---- Case 1: bs=1 already, generate directly ----
        # ---- 情况1: 本身就是 bs=1, 直接生成 ----
        if bs == 1:
            logger.debug(f'[AICB Safe Mode] bs=1, 直接生成...')
            _GLOBAL_AICB_CACHE.record_aicb_call()
            if self._generate_aicb_csv():
                if os.path.exists(original_csv_path):
                    logger.debug(f'[AICB Safe Mode] Successfully generated bs=1 CSV: {original_csv_path}')
                    return original_csv_path
                else:
                    logger.warning('[AICB Safe Mode] 生成后未找到 CSV (文件名可能不匹配)')
            else:
                logger.warning('[AICB Safe Mode] bs=1 生成失败')
            return original_csv_path
        
        # ---- Case 2: bs > 1, skip original bs, use bs=1 instead ----
        # ---- 情况2: bs > 1, 跳过原始 bs, 直接使用 bs=1 ----
        logger.debug(f'[AICB Safe Mode] bs={bs} > 1, 跳过原始bs (避免CUDA kernel错误), 使用 bs=1')
        
        # Temporarily switch to bs=1 to get the bs=1 CSV path
        # 临时切换到 bs=1 以获取 bs=1 的 CSV 路径
        original_bs = self._replica_config.batch_size
        self._replica_config.batch_size = 1
        bs1_csv_path = self._get_aicb_csv_path()
        self._replica_config.batch_size = original_bs  # Restore immediately / 立即恢复
        
        logger.debug(f'[AICB Safe Mode] bs=1 CSV 路径: {bs1_csv_path}')
        
        # Check if bs=1 CSV already exists
        # 检查 bs=1 CSV 是否已存在
        if os.path.exists(bs1_csv_path):
            logger.debug('[AICB Safe Mode] 找到已有 bs=1 CSV (无需生成)')
            return bs1_csv_path
        
        # bs=1 CSV doesn't exist, generate it
        # bs=1 CSV 不存在, 生成它
        logger.debug('[AICB Safe Mode] bs=1 CSV 不存在, 开始生成...')
        original_bs = self._replica_config.batch_size
        self._replica_config.batch_size = 1
        
        _GLOBAL_AICB_CACHE.record_aicb_call()
        gen_ok = self._generate_aicb_csv()
        
        self._replica_config.batch_size = original_bs  # Restore / 恢复
        
        if gen_ok and os.path.exists(bs1_csv_path):
            logger.debug(f'[AICB Safe Mode] Successfully generated bs=1 CSV: {bs1_csv_path}')
            return bs1_csv_path
        else:
            logger.warning('[AICB Safe Mode] bs=1 生成失败或未找到 CSV')
            return original_csv_path  # 返回原路径, 后续 fallback 逻辑会处理

    def _load_aicb_data(self) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        [AICB Optimization B+C] Load AICB data, preferring global cache and interpolation.
        [AICB优化 B+C方案] 加载AICB数据，优先使用全局缓存和插值
        
        Lookup flow / 查找流程:
        1. Check global cache exact match (Plan C: lookup)
           检查全局缓存精确匹配 (方案C: 查表)
        2. Check global cache interpolation (Plan B: head-tail token strategy)
           检查全局缓存插值 (方案B: 首尾token策略)
        3. If both miss, read/generate CSV and cache
           如果都没有，读取/生成CSV并缓存
        
        Returns:
            {layer_id: {layer_name: {comp_time: value, comm_size: value}}}
        """
        global _GLOBAL_AICB_CACHE
        
        if self._aicb_data is not None:
            return self._aicb_data

        # Get current parameters / 获取当前参数
        model_name, _, tp, pp, ws, ep, bs, seq, phase = self._get_aicb_params()
        
        # === Step 1+2: Try global cache (exact match or interpolation) ===
        # === 步骤1+2: 尝试从全局缓存获取 (精确匹配 或 插值) ===
        cached_data = _GLOBAL_AICB_CACHE.get(model_name, ws, tp, pp, ep, bs, seq, phase)
        if cached_data is not None:
            self._aicb_data = cached_data
            # [Head-tail interp] On cache hit, also ensure last_seq is preloaded
            # [首尾插值] 缓存命中时，也确保last_seq已预加载
            # So even old cache without last_seq can be supplemented
            # 这样即使旧缓存中没有last_seq，也能补充加载
            self._ensure_decode_endpoint_preloaded(phase, seq)
            return cached_data
        
        # === Step 3: Cache miss, need to read/generate CSV ===
        # === 步骤3: 缓存未命中，需要读取/生成CSV ===
        logger.debug(f"[AICB] Cache miss, loading CSV (缓存未命中，需要加载CSV): "
              f"model={model_name}, bs={bs}, seq={seq}, phase={phase}")
        
        csv_path = self._get_aicb_csv_path()
        full_csv_path = csv_path

        if not os.path.exists(full_csv_path):
            if self._config.aicb_force_bs1:
                # [AICB Fallback] User explicitly enabled force-bs1 mode
                # This forces bs=1 CSV generation regardless of actual batch size.
                # Known issue: DeepSeek-V3-671B prefill AICB profiling fails on H20
                # (SM90) when tp>=4, due to FlashMLA flash_mla_sparse_fwd kernel's
                # h_q alignment requirement (B_H=64). Does not affect decode or
                # other models.
                full_csv_path = self._generate_or_find_bs1_csv(
                    model_name, ws, tp, pp, ep, bs, seq, phase, full_csv_path
                )
            else:
                # Normal mode: generate CSV with actual batch size
                _GLOBAL_AICB_CACHE.record_aicb_call()
                self._generate_aicb_csv()
            
            if not os.path.exists(full_csv_path):
                # ============================================================
                # [AICB Fallback] Search for existing CSV of the same model
                #                 in results/workload/ directory
                # [AICB Fallback] 在 results/workload/ 目录搜索同模型的已有CSV
                #
                # Search strategy (by priority) / 搜索策略 (按优先级):
                #   1. Same model + same ws + same phase (different bs/seq)
                #      同模型 + 同ws + 同phase (不同 bs/seq)
                #   2. Same model + same phase (different ws/bs/seq)
                #      同模型 + 同phase (不同 ws/bs/seq)
                #   3. Any model's fallback CSV
                #      任意模型的兖底 CSV
                # ============================================================
                import glob
                search_dir = os.path.dirname(full_csv_path)
                
                # Get correct model name (from _get_aicb_params, already proper case)
                # 获取正确的模型名
                found_fallback = False
                
                # Priority 1: same model + same ws + same phase
                # 优先级1: 同模型 + 同ws + 同phase
                pattern1 = os.path.join(search_dir, 
                    f"vidur-{model_name}-world_size{ws}-tp{tp}-pp{pp}-ep{ep}-bs*-seq*-{phase}.csv")
                matches1 = sorted(glob.glob(pattern1))
                if matches1:
                    full_csv_path = matches1[0]
                    logger.debug(f'[AICB Fallback] Found same model+ws: {full_csv_path}')
                    found_fallback = True
                
                # Priority 2: same model + same phase (any ws/ep)
                # 优先级2: 同模型 + 同phase (任意 ws/ep)
                if not found_fallback:
                    pattern2 = os.path.join(search_dir, f"vidur-{model_name}-*-{phase}.csv")
                    matches2 = sorted(glob.glob(pattern2))
                    if matches2:
                        full_csv_path = matches2[0]
                        logger.debug(f'[AICB Fallback] Found same model: {full_csv_path}')
                        found_fallback = True
                
                # Priority 3: any CSV as fallback
                # 优先级3: 任意 CSV 兖底
                if not found_fallback:
                    all_csvs = sorted(glob.glob(os.path.join(search_dir, "vidur-*.csv")))
                    if all_csvs:
                        full_csv_path = all_csvs[0]
                        logger.debug(f'[AICB Fallback] Using any available CSV: {full_csv_path}')
                        found_fallback = True
                
                if not found_fallback:
                    logger.error('无法找到任何AICB CSV文件')
                    return {}

        # === Parse CSV ===
        # === 解析CSV ===
        _GLOBAL_AICB_CACHE.record_csv_load()
        data = self._parse_aicb_csv(full_csv_path)
        
        if data:
            # Store into global cache / 存入全局缓存
            _GLOBAL_AICB_CACHE.put(model_name, ws, tp, pp, ep, bs, seq, phase, data)
            
            # Also copy CSV to aicb_workload/cache dir for inspection
            # 同时复制CSV到aicb_workload/cache目录，方便查看
            try:
                import shutil
                cache_csv = _AICB_CACHE_DIR / os.path.basename(full_csv_path)
                if not cache_csv.exists():
                    shutil.copy2(full_csv_path, cache_csv)
            except:
                pass
            
            # ============================================================
            # [Head-tail interp optimization] Preload AICB data for decode's last round
            # [首尾插值优化] 预加载decode最后一轮的AICB数据
            # ============================================================
            self._ensure_decode_endpoint_preloaded(phase, seq)
        
        self._aicb_data = data
        return data

    def _ensure_decode_endpoint_preloaded(self, phase: str, current_seq: int):
        """
        [Head-tail interp] Ensure AICB data for decode's last round is preloaded.
        [首尾插值] 确保decode最后一轮的AICB数据已预加载
        
        Called from both cache-hit and cache-miss paths.
        _preload_decode_endpoint internally checks cache, returns immediately if loaded.
        无论是缓存命中还是缓存未命中路径都会调用此方法。
        
        Args:
            phase: Current stage ("prefill" or "decode") / 当前阶段
            current_seq: Current iteration's seq value / 当前迭代的seq值
        """
        if phase == "decode" and hasattr(self._replica_config, 'decode_last_seq'):
            last_seq = self._replica_config.decode_last_seq
            if last_seq is not None and last_seq != current_seq:
                self._preload_decode_endpoint(last_seq)

    
    def _preload_decode_endpoint(self, last_seq: int):
        """
        [Head-tail interp optimization] Preload AICB data for decode's last round.
        [首尾插值优化] 预加载decode最后一轮的AICB数据
        
        Purpose: When first loading decode round 1 CSV, also load the last round's
        seq CSV so intermediate iterations can get more accurate execution times
        via linear interpolation.
        目的: 在首次加载decode第一轮CSV时，同时加载最后一轮的seq对应的CSV
        
        Principle / 原理:
        - KV Cache grows during decode, so computation changes with seq
          Transformer推理中，decode阶段的KV Cache随seq增长
        - Using only round 1 data (nearest neighbor) ignores this growth trend
          只用第一轮数据(最近邻)会忽略这种增长趋势
        - Head-tail interpolation captures the linear growth
          首尾两点线性插值可以捕捉这种线性增长
        
        Args:
            last_seq: Seq_len value of the last decode round / 最后一轮decode的seq_len值
        """
        global _GLOBAL_AICB_CACHE, _FAILED_PRELOAD_KEYS
        
        model_name, _, tp, pp, ws, ep, bs, _, phase = self._get_aicb_params()
        
        # Note: Must use exact match check, not get() (which triggers nearest neighbor/interpolation)
        # 注意: 必须用精确匹配检查，不能用 get() (它会触发最近邻/插值)
        # Otherwise seq=106 would match seq=100 via nearest neighbor, skipping real CSV load
        # 否则 seq=106 会被最近邻匹配到 seq=100 的数据，跳过真正的CSV加载
        exact_key = _GLOBAL_AICB_CACHE._make_key(model_name, ws, tp, pp, ep, bs, last_seq, phase)
        if exact_key in _GLOBAL_AICB_CACHE._cache:
            # Already has real data, no need to preload
            # 已有真实数据，无需预加载
            return
        
        # Avoid repeated attempts on failed preloads
        # 避免重复尝试已失败的预加载
        if exact_key in _FAILED_PRELOAD_KEYS:
            return
        
        logger.debug(f"[AICB] Preloading decode last round: "
                    f"model={model_name}, bs={bs}, seq={last_seq}, phase={phase}")
        
        # Temporarily modify seq_len to generate corresponding CSV path
        # 临时修改seq_len以生成对应的CSV路径
        original_seq = self._replica_config.seq_len
        self._replica_config.seq_len = last_seq
        
        csv_path = self._get_aicb_csv_path()
        
        if not os.path.exists(csv_path):
            if self._config.aicb_force_bs1:
                # [AICB Safe Mode] Use bs=1, avoid CUDA kernel compatibility issues
                logger.debug(f"[AICB] last_seq CSV not found: {csv_path}")
                csv_path = self._generate_or_find_bs1_csv(
                    model_name, ws, tp, pp, ep, bs, last_seq, phase, csv_path
                )
            else:
                # Normal mode: generate with actual batch size
                _GLOBAL_AICB_CACHE.record_aicb_call()
                self._generate_aicb_csv()
        
        if os.path.exists(csv_path):
            _GLOBAL_AICB_CACHE.record_csv_load()
            data = self._parse_aicb_csv(csv_path)
            if data:
                _GLOBAL_AICB_CACHE.put(model_name, ws, tp, pp, ep, bs, last_seq, phase, data)
                logger.debug(f"[AICB] Successfully cached last_seq={last_seq}, layers={len(data)}")
                
                # Copy CSV to cache directory / 复制CSV到缓存目录
                try:
                    import shutil
                    cache_csv = _AICB_CACHE_DIR / os.path.basename(csv_path)
                    if not cache_csv.exists():
                        shutil.copy2(csv_path, cache_csv)
                except:
                    pass
            else:
                logger.warning("[AICB首尾插值] last_seq CSV解析为空")
                _FAILED_PRELOAD_KEYS.add(exact_key)
        else:
            logger.warning(f"[AICB首尾插值] 生成后仍未找到last_seq CSV: {csv_path}")
            _FAILED_PRELOAD_KEYS.add(exact_key)
        
        # Restore original seq_len / 恢复原始seq_len
        self._replica_config.seq_len = original_seq
    
    def _parse_aicb_csv(self, csv_path: str) -> Dict[int, Dict[str, Dict[str, float]]]:
        """
        Parse AICB CSV file and return structured data.
        解析AICB CSV文件，返回结构化数据
        
        Extracted from _load_aicb_data for clean separation.
        从 _load_aicb_data 中提取的CSV解析逻辑。
        
        Returns:
            {layer_id: {layer_name: {comp_time: float, comm_size: float}}}
        """
        data: Dict[int, Dict[str, Dict[str, float]]] = {}
        
        try:
            with open(csv_path, newline='') as f:
                reader = csv.DictReader(f, delimiter='\t')
                logger.debug(f"[AICB优化] CSV列名: {reader.fieldnames}")
                
                if reader.fieldnames and len(reader.fieldnames) == 1:
                    actual_fieldnames = reader.fieldnames[0].split('\t')
                    if 'layer_id' in actual_fieldnames and 'layer_name' in actual_fieldnames:
                        f.seek(0)
                        lines = f.readlines()
                        headers = lines[0].strip().split('\t')
                        
                        for line_num, line in enumerate(lines[1:], 1):
                            values = line.strip().split('\t')
                            if len(values) == len(headers):
                                row = dict(zip(headers, values))
                                layer_id = int(row['layer_id'])
                                layer_name = row['layer_name']
                                comp_time = float(row['comp_time'])
                                comm_size = float(row['comm_size'])
                                
                                if layer_id not in data:
                                    data[layer_id] = {}
                                data[layer_id][layer_name] = {
                                    'comp_time': comp_time,
                                    'comm_size': comm_size
                                }
                    else:
                        return {}
                else:
                    for row_num, row in enumerate(reader, 1):
                        if 'layer_id' not in row or 'layer_name' not in row:
                            continue
                        
                        layer_id = int(row['layer_id'])
                        layer_name = row['layer_name']
                        comp_time = float(row['comp_time'])  
                        comm_size = float(row['comm_size'])
                        
                        if layer_id not in data:
                            data[layer_id] = {}
                        data[layer_id][layer_name] = {
                            'comp_time': comp_time,
                            'comm_size': comm_size
                        }
        except Exception as e:
            logger.error(f"读取CSV文件失败: {e}", exc_info=True)
            return {}

        logger.debug(f"[AICB] Successfully parsed CSV: {len(data)} layers from {csv_path}")
        return data
    
    def _get_block_execution_time(self) -> float:
        return (
            self._get_attention_layer_execution_time()
            + self._get_mlp_layer_execution_time()
            + self._add_time
        )
    def _get_block_execution_time_by_layer_id(self, layer_id: int = 0) -> float:
        
        if self._replica_config.model_name in ['deepseek-671B', 'qwen3-moe-235B', 'qwen3-next-80B'] and self._config.backend == 'aicb':   
            att_time = self._get_attention_layer_execution_time_from_aicb(layer_id)
            # 根据模型类型确定使用的层类型
            # Determine layer type based on model
            
            att_time = self._get_attention_layer_execution_time_from_aicb(layer_id)
            mlp_time = self._get_mlp_layer_execution_time_from_aicb(layer_id)
            moe_time = self._get_moe_layer_execution_time_from_aicb(layer_id)
            assert att_time >= 0 and mlp_time >= 0 and moe_time >= 0, "AICB layer times must be non-negative"
            return att_time + mlp_time + moe_time
        
        else:
            
            return (
                self._get_attention_layer_execution_time()
                + self._get_mlp_layer_execution_time()
                + self._add_time
            )

    def _get_cpu_overhead(self) -> float:
        return (
            self._schedule_time
            + self._sampler_e2e_time
            + self._prepare_inputs_e2e_time
            + self._process_model_outputs_time
            + self._ray_comm_time
        )

    @property
    def num_layers(self) -> int:
        return self._num_layers_per_pipeline_stage

    @property
    def mlp_layer_up_proj_execution_time(self) -> float:
        return self._mlp_layer_up_proj_execution_time

    @property
    def mlp_layer_down_proj_execution_time(self) -> float:
        return self._mlp_layer_down_proj_execution_time

    @property
    def mlp_layer_act_execution_time(self) -> float:
        return self._mlp_layer_act_execution_time

    @property
    def mlp_all_reduce_time(self) -> float:
        return self._tensor_parallel_communication_time

    @property
    def attention_pre_proj_time(self) -> float:
        return self._attention_layer_pre_proj_execution_time

    @property
    def attention_post_proj_time(self) -> float:
        return self._attention_layer_post_proj_execution_time

    @property
    def attention_all_reduce_time(self) -> float:
        return self._tensor_parallel_communication_time

    @property
    def attention_rope_execution_time(self) -> float:
        return self._attention_rope_execution_time

    @property
    def attention_kv_cache_save_execution_time(self) -> float:
        return self._attention_kv_cache_save_execution_time

    @property
    def attention_decode_execution_time(self) -> float:
        return self._attention_decode_execution_time

    @property
    def attention_prefill_execution_time(self) -> float:
        return self._attention_prefill_execution_time

    @property
    def pipeline_parallel_communication_time(self) -> float:
        return self._pipeline_parallel_communication_time

    @property
    def schedule_time(self) -> float:
        return self._schedule_time

    @property
    def sampler_e2e_time(self) -> float:
        return self._sampler_e2e_time

    @property
    def prepare_inputs_e2e_time(self) -> float:
        return self._prepare_inputs_e2e_time

    @property
    def process_model_outputs_time(self) -> float:
        return self._process_model_outputs_time

    @property
    def ray_comm_time(self) -> float:
        return self._ray_comm_time

    @property
    def mlp_norm_time(self) -> float:
        return self._mlp_norm_time

    @property
    def attn_norm_time(self) -> float:
        return self._attn_norm_time

    @property
    def add_time(self) -> float:
        return self._add_time

    @property
    def model_time(self) -> float:
        # 对于特定模型，需要逐层计算执行时间
        # For specific models, the execution time needs to be calculated layer by layer.
        if self._replica_config.model_name in ['deepseek-671B', 'qwen3-moe-235B', 'qwen3-next-80B'] and self._config.backend == 'aicb':
            # 计算当前 pipeline stage 包含的 layer_id 范围
            # Calculate the range of layer_ids included in the current pipeline stage
            
            # NOTE: PP>1 not yet supported, hardcoded stage 0
            if self._replica_config.num_pipeline_stages == 1:
                self._pipeline_stage_id = 0
                start_layer = self._pipeline_stage_id * self._num_layers_per_pipeline_stage
                end_layer = start_layer + self._num_layers_per_pipeline_stage
            total_block_time = 0.0
            
            # 遍历每个 layer_id
            # Iterate through each layer_id
            for layer_id in range(start_layer, end_layer):
                self._current_layer_id = layer_id
                # block_time = self._get_block_execution_time()
                block_time = self._get_block_execution_time_by_layer_id(layer_id)
                total_block_time += block_time
                

            self._current_layer_id = None  # Clean up
            return (total_block_time + self.pipeline_parallel_communication_time) * 1e-3
            
            # total_execution_time = 0.0
            # for layer_id in range(self._num_layers_per_pipeline_stage):
            #     block_execution_time = self._get_block_execution_time(layer_id)
            #     total_execution_time += block_execution_time
                
            # # return in seconds
            # return (
            #     total_execution_time + self.pipeline_parallel_communication_time
            # ) * 1e-3
        
        else:
            # we are not counting the execution time for the embedding layer and last softmax layer
            block_execution_time = self._get_block_execution_time()
            # 单个replica stage中的多个layer在这里实现
            # Multiple layers in a single replica stage are implemented here
            pipeline_stage_execution_time = (
                block_execution_time * self._num_layers_per_pipeline_stage
            )
            # return in seconds
            return (
                pipeline_stage_execution_time + self.pipeline_parallel_communication_time
            ) * 1e-3

    @property
    def model_time_ms(self) -> float:
        return self.model_time * 1e3

    @property
    def total_time(self) -> float:
        # return in seconds
        return self.model_time + self._get_cpu_overhead() * 1e-3
