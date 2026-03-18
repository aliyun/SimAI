from math import ceil

from vidur.config import ReplicaConfig
from vidur.logger import init_logger
import os
import json

logger = init_logger(__name__)


class ParamCounter:
    def __init__(self, replica_config: ReplicaConfig) -> None:
        self._replica_config = replica_config
        self._model_config = self._replica_config.model_config
        self.config = self._model_config

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
        
        # TODO(tianhao909): support FP8 precision quantization
        # TODO(tianhao909): 支持 FP8 精度量化
        if self._replica_config.pd_p2p_comm_dtype == "fp8":
            logger.debug(f"FP8 enabled, dtype={self._replica_config.pd_p2p_comm_dtype}")
            self.use_fp8 = True
        else:
            logger.debug(f"FP8 disabled, dtype={self._replica_config.pd_p2p_comm_dtype}")
            self.use_fp8 = False
        self.tp = self._replica_config.tensor_parallel_size 
        self.ep = self._replica_config.expert_model_parallel_size
        
        # 标记是否已经打印过调试信息 | Flag to track if debug info has been printed
        self._debug_printed = False
        
        if self._replica_config.model_name in ['deepseek-671B', 'qwen3-moe-235B', 'qwen3-next-80B']:
            self.model_config_postprocessing()
            # self._model_config
            
    def model_config_postprocessing(self, ):
        # 初始化配置字典 | Initialize configuration dictionary
        d = dict()
        if self._replica_config.model_name == 'deepseek-671B':
            # 使用相对路径定位配置文件 | Use relative path to locate config file
            config_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "hf_configs", "deepseek_v3_config.json")
        elif self._replica_config.model_name == 'qwen3-moe-235B':
            # TODO(tianhao909): add corresponding JSON config file for Qwen3-MoE
            # TODO(tianhao909): 增加对应的 JSON 配置文件
            config_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "hf_configs", "qwen3_moe_config.json")
        elif self._replica_config.model_name == 'qwen3-next-80B':
            config_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "hf_configs", "qwen3-next-80B-A3B_config.json")
        logger.debug(f"config_path={config_path}")
        # 检查配置文件是否存在 | Check if config file exists
        if not os.path.exists(config_path):
            logger.warning(f"Config file {config_path} not found, using default config")
            return
        # 以只读模式加载JSON配置 | Load JSON config in read-only mode
        with open(config_path, "r") as f:
            d = json.load(f)

        # 模型隐藏层维度 | Model hidden size dimension
        self._model_config.hidden_size = d["hidden_size"]
        # 隐藏层数量 | Number of hidden layers
        self._model_config.num_hidden_layers = d["num_hidden_layers"]

        # 判断是否使用混合注意力机制（全注意力与线性注意力交替）
        # Determine if using hybrid attention (alternating full and linear attention)
        self._model_config.is_hybrid_linear = d.get("full_attention_interval") is not None
        if self._model_config.is_hybrid_linear:
            # 全注意力层数量：每隔N层插入一次 | Full attention layers: inserted every N layers
            self._model_config.num_full_attn_layers = (
                self._model_config.num_hidden_layers // d["full_attention_interval"]
            )
            # 线性注意力层数量：总层数减去全注意力层数 | Linear attention layers: total - full attention
            self._model_config.num_linear_attn_layers = (
                self._model_config.num_hidden_layers - self._model_config.num_full_attn_layers
            )
            # 线性注意力卷积核维度 | Linear attention convolution kernel dimension
            self._model_config.linear_conv_kernel_dim = d["linear_conv_kernel_dim"]
            # 线性注意力键向量头维度 | Linear attention key head dimension
            self._model_config.linear_key_head_dim = d["linear_key_head_dim"]
            # 线性注意力键向量头数 | Linear attention number of key heads
            self._model_config.linear_num_key_heads = d["linear_num_key_heads"]
            # 线性注意力值向量头维度 | Linear attention value head dimension
            self._model_config.linear_value_head_dim = d["linear_value_head_dim"]
            # 线性注意力值向量头数 | Linear attention number of value heads
            self._model_config.linear_num_value_heads = d["linear_num_value_heads"]

        self._model_config.attn_type = "MHA/GQA"  # Default attention: MHA or GQA / 默认注意力类型
        if "kv_lora_rank" in d:  # If kv_lora_rank present, use MLA attention / 如果配置中包含 kv_lora_rank，则使用MLA
            self._model_config.attn_type = "MLA"

        # Attention mechanism parameter setup / 注意力机制相关参数设置
        if self._model_config.attn_type == "MHA/GQA":  # MHA/GQA type
            self._model_config.num_attention_heads = d["num_attention_heads"]  # Number of attention heads / 注意力头数量
            self._model_config.num_key_value_heads = d["num_key_value_heads"]  # KV heads for GQA / 键和值的头数量
            if "head_dim" in d:  # If head_dim specified in config
                self._model_config.head_dim = d["head_dim"]
            else:
                self._model_config.head_dim = self._model_config.hidden_size // self._model_config.num_attention_heads  # Compute from hidden_size / heads
        elif self._model_config.attn_type == "MLA":  # MLA type
            self._model_config.q_lora_rank = d["q_lora_rank"]  # Query LoRA rank / 查询向量LoRA的秩
            self._model_config.qk_nope_head_dim = d["qk_nope_head_dim"]  # QK no-position head dim / 无位置编码的QK头维度
            self._model_config.qk_rope_head_dim = d["qk_rope_head_dim"]  # QK RoPE head dim / 使用RoPE编码的QK头维度
            self._model_config.kv_lora_rank = d["kv_lora_rank"]  # KV LoRA rank / 键值对LoRA的秩
            self._model_config.num_attention_heads = d["num_attention_heads"]  # Total attention heads / 注意力头总数
            self._model_config.v_head_dim = d["v_head_dim"]  # Value head dim / 值向量每个头的维度
            self._model_config.qk_head_dim = self._model_config.qk_nope_head_dim + self._model_config.qk_rope_head_dim  # QK total head dim = nope + rope

        # FFN/MoE (Feed-Forward Network / Mixture of Experts) configuration
        # FFN/MoE（前馈网络/专家混合模型）配置
        self._model_config.is_moe = True  # Default enable MoE / 默认启用MoE
        if "num_routed_experts" in d:  # Routed expert count / 路由专家数量
            self._model_config.num_routed_experts = d["num_routed_experts"]
        elif "num_experts" in d:  # Fallback to num_experts field
            self._model_config.num_routed_experts = d["num_experts"]
        else:
            self._model_config.is_moe = False  # No MoE if no expert fields / 不使用MoE
            self._model_config.num_routed_experts = 1  # Single expert (standard FFN) / 单一专家

        if self._model_config.is_moe:  # If MoE enabled / 如果启用了MoE
            self._model_config.num_experts_per_tok = d["num_experts_per_tok"]  # Experts activated per token / 每个token激活的专家数
            self._model_config.intermediate_size = d["moe_intermediate_size"]  # Per-expert intermediate size / 每个专家的中间层大小
            self._model_config.num_shared_experts = d.get("num_shared_experts", 0)  # Shared expert count / 共享专家数量
        else:  # Standard FFN (no MoE) / 未启用MoE
            self._model_config.num_experts_per_tok = 1  # Single "expert" / 标准FFN
            self._model_config.intermediate_size = d["intermediate_size"]  # Standard FFN intermediate size / 标准FFN中间层大小
            self._model_config.num_shared_experts = 0  # No shared experts / 无共享专家

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
    
    # Layer Dimension
        # First 3 layers are dense (no gate). Based on the above calculation,
        # each of the first 3 layers of DeepSeek V3 has parameter count:
        # Layer 维度
        # 前 3 层是 dense，没有 gate，基于上面的计算，DeepSeek V3 前 3 层每层的参数量是：
            # (单层MLA中Q的LoRA参数量48,760,320 + 单层MLA中KV的LoRA参数量20,906,496 + 单层 MLA中WO的参数量117,440,512 + （pre+post）attention layernorm的参数14336（即7168+7168）） + （每个专家的参数量44,040,192 * 9 （9 因为前 3 层 dense，每层固定激活8个路由专家和一个共享专家））
            # (48,760,320 + 20,906,496 + 117,440,512 + 14336) + (44,040,192 * 9) = 583,483,392
        # Last 58 layers are MoE sparse-activated experts. DeepSeek V3 per-layer params:
        # 后 58 层是 MoE 稀疏激活专家，基于上面的计算，DeepSeek V3 后 58 层每层的参数量是：
            # (48,760,320 + 20,906,496 + 117,440,512 + 14336) + (44,040,192 * 257 + 1,835,264) = 11,507,286,272
            # (单层MLA中Q的LoRA参数量48,760,320 + 单层MLA中KV的LoRA参数量20,906,496 + 单层 MLA中WO的参数量117,440,512 + （pre+post）attention layernorm的参数14336（即7168+7168）） + （每个专家的参数量44,040,192 * 257 （256个路由专家和一个共享专家） + 路由 Gate 的参数量1,835,264）
    def get_num_parameters_per_layer_by_layer_id(self, layer_id: int = 0) -> tuple:
        """
        Get parameter count per layer by layer_id.
        Returns tuple: (params_per_layer, prefill_params_per_layer, decode_params_per_layer)

        根据 layer_id 获取每层的参数量
        返回三元组: (params_per_layer, prefill_params_per_layer, decode_params_per_layer)
        """
        # 初始化变量 | Initialize variables
        params_per_layer_per_gpu = 0
        prefill_params_per_layer_per_gpu = 0
        decode_params_per_layer_per_gpu = 0
            
        if self._replica_config.model_name == 'deepseek-671B':
            # 仅在首次调用时打印调试信息 | Only print debug info on first call
            if not self._debug_printed:
                logger.info("{s:{c}^{n}}".format(s="[ParamCounter] DeepSeek-671B Model Weights", n=60, c="-"))
                attn_params_bytes = self.get_attn_params_size(self._model_config, self.use_fp8)
                expert_params_bytes = self.get_expert_params_size(self._model_config, self.use_fp8)
                logger.info(f"[ParamCounter] One MLA params size (MB): {attn_params_bytes / 1024 / 1024:.2f}")
                logger.info(f"[ParamCounter] One expert params size (MB): {expert_params_bytes / 1024 / 1024:.2f}")
                logger.info(f"[ParamCounter] use_fp8={self.use_fp8}, tp={self.tp}, ep={self.ep}")
                self._debug_printed = True
                
            if layer_id >= 0 and layer_id <= 2:
                # 前 3 层是 dense，每层固定激活8个路由专家和1个共享专家
                # First 3 layers are dense, each layer activates 8 routed experts + 1 shared expert
                params_per_layer_per_gpu = (self.get_mla_params_size(self._model_config, self.use_fp8)/self.tp + 
                                           self.get_expert_params_size(self._model_config, self.use_fp8) * (8 + 1) / self.tp)
                prefill_params_per_layer_per_gpu = params_per_layer_per_gpu 
                decode_params_per_layer_per_gpu = params_per_layer_per_gpu
                    
            elif layer_id >= 3 and layer_id <= 60:
                # 后 58 层是 MoE 稀疏激活专家
                # Remaining 58 layers are MoE sparse activated experts
                mla_params = self.get_mla_params_size(self._model_config, self.use_fp8) / self.tp
                expert_params = self.get_expert_params_size(self._model_config, self.use_fp8)
                    
                params_per_layer_per_gpu = mla_params + expert_params * (256/self.ep + 1)
                prefill_params_per_layer_per_gpu = mla_params + expert_params * (256/self._replica_config.prefill_world_size + 1)
                decode_params_per_layer_per_gpu = mla_params + expert_params * (256/self._replica_config.decode_world_size + 1)
                    
        elif self._replica_config.model_name == 'qwen3-next-80B':
            # 仅在首次调用时打印调试信息 | Only print debug info on first call
            if not self._debug_printed:
                logger.info("{s:{c}^{n}}".format(s="[ParamCounter] Qwen3-Next-80B Model Weights", n=60, c="-"))
                full_attn_params_bytes = self.get_attn_params_size(self._model_config, self.use_fp8)
                linear_attn_params_bytes = self.get_linear_attn_params_size(self._model_config, self.use_fp8)
                expert_params_bytes = self.get_expert_params_size(self._model_config, self.use_fp8)
                logger.info(f"[ParamCounter] One full attn params size (MB): {full_attn_params_bytes / 1024 / 1024:.2f}")
                logger.info(f"[ParamCounter] One linear attn params size (MB): {linear_attn_params_bytes / 1024 / 1024:.2f}")
                logger.info(f"[ParamCounter] One expert params size (MB): {expert_params_bytes / 1024 / 1024:.2f}")
                logger.info(f"[ParamCounter] use_fp8={self.use_fp8}, tp={self.tp}, ep={self.ep}")
                self._debug_printed = True
                    
            full_attn_params_bytes = self.get_attn_params_size(self._model_config, self.use_fp8)
            linear_attn_params_bytes = self.get_linear_attn_params_size(self._model_config, self.use_fp8)
            expert_params_bytes = self.get_expert_params_size(self._model_config, self.use_fp8)
                
            # 基础参数: 专家网络部分 | Base params: expert network part
            params_per_layer_per_gpu = expert_params_bytes * (
                self.config.num_shared_experts + self.config.num_routed_experts / self._replica_config.world_size
            )
            prefill_params_per_layer_per_gpu = expert_params_bytes * (
                self.config.num_shared_experts + self.config.num_routed_experts / self._replica_config.prefill_world_size
            )
            decode_params_per_layer_per_gpu = expert_params_bytes * (
                self.config.num_shared_experts + self.config.num_routed_experts / self._replica_config.decode_world_size
            )
                
            # 根据 layer_id 添加注意力层参数 | Add attention layer params based on layer_id
            if layer_id % 4 == 3:  # Full attention layers (e.g., layer 3, 7, 11...)
                params_per_layer_per_gpu += full_attn_params_bytes / self.tp
                prefill_params_per_layer_per_gpu += full_attn_params_bytes / self.tp
                decode_params_per_layer_per_gpu += full_attn_params_bytes / self.tp
            else:  # Linear attention layers (e.g., layer 0, 1, 2, 4, 5, 6...)
                params_per_layer_per_gpu += linear_attn_params_bytes / self.tp
                prefill_params_per_layer_per_gpu += linear_attn_params_bytes / self.tp
                decode_params_per_layer_per_gpu += linear_attn_params_bytes / self.tp
                    
        elif self._replica_config.model_name == 'qwen3-moe-235B':
            # 仅在首次调用时打印调试信息 | Only print debug info on first call
            if not self._debug_printed:
                logger.info("{s:{c}^{n}}".format(s="[ParamCounter] Qwen3-MoE-235B Model Weights", n=60, c="-"))
                attn_params_bytes = self.get_mha_params_size(self._model_config, self.use_fp8)
                expert_params_bytes = self.get_expert_params_size(self._model_config, self.use_fp8)
                logger.info(f"[ParamCounter] One MHA params size (MB): {attn_params_bytes / 1024 / 1024:.2f}")
                logger.info(f"[ParamCounter] One expert params size (MB): {expert_params_bytes / 1024 / 1024:.2f}")
                logger.info(f"[ParamCounter] use_fp8={self.use_fp8}, tp={self.tp}, ep={self.ep}")
                self._debug_printed = True
                
            # Qwen3-MoE-235B: 128个路由专家, 0个共享专家, MHA/GQA注意力, 没有dense层
            # Qwen3-MoE-235B: 128 routed experts, 0 shared experts, MHA/GQA attention, no dense layers
            mha_params = self.get_mha_params_size(self._model_config, self.use_fp8)
            expert_params = self.get_expert_params_size(self._model_config, self.use_fp8)
                
            params_per_layer_per_gpu = mha_params + expert_params * 128
            prefill_params_per_layer_per_gpu = mha_params/self.tp + expert_params * (128/self._replica_config.prefill_world_size)
            decode_params_per_layer_per_gpu = mha_params/self.tp + expert_params * (128/self._replica_config.decode_world_size)
                
        return params_per_layer_per_gpu, prefill_params_per_layer_per_gpu, decode_params_per_layer_per_gpu

    def get_num_parameters_per_device(self) -> int:
        # TODO(tianhao909): refactor per-layer param calculation with layer_id support
        # TODO(tianhao909): 重构 get_num_parameters_per_device 支持按 layer_id 计算
        if self._replica_config.model_name in ['deepseek-671B', 'qwen3-moe-235B', 'qwen3-next-80B']:
            # Reference: see ExecutionTime._get_block_execution_time_by_layer_id
            # Need to get start/end layer_id for the current pipeline stage
            # 参考 ExecutionTime._get_block_execution_time_by_layer_id 的实现
            # 需要获取当前pipeline stage的起始和结束layer id
            # try:
            
            pipeline_stage_id = getattr(self, '_pipeline_stage_id', 0)
            start_layer = pipeline_stage_id * self._num_layers_per_pipeline_stage
            end_layer = start_layer + self._num_layers_per_pipeline_stage
            logger.debug(f"pipeline_stage_id={pipeline_stage_id} num_layers_per_pipeline_stage={self._num_layers_per_pipeline_stage} start_layer={start_layer} end_layer={end_layer}")
            
            params_per_gpu = 0
            prefill_params_per_gpu = 0  # 修正变量名 | Fixed variable name
            decode_params_per_gpu = 0   # 修正变量名 | Fixed variable name
            for layer_id in range(start_layer, end_layer):
                params_per_layer, prefill_params_per_layer, decode_params_per_layer = self.get_num_parameters_per_layer_by_layer_id(layer_id)
                params_per_gpu += params_per_layer
                prefill_params_per_gpu += prefill_params_per_layer
                decode_params_per_gpu += decode_params_per_layer
                
            # params_per_gpu 单位是B | Unit is Bytes
            params_per_gpu_gb = params_per_gpu / 1024 / 1024 / 1024  # Convert to GB / 转换为GB
            prefill_params_per_gpu_gb = prefill_params_per_gpu / 1024 / 1024 / 1024  # Convert to GB / 转换为GB
            decode_params_per_gpu_gb = decode_params_per_gpu / 1024 / 1024 / 1024  # Convert to GB / 转换为GB
            logger.info("{:<40} {:<10.2f}".format("Per GPU params size (GB):", params_per_gpu_gb))
            logger.info("{:<40} {:<10.2f}".format("Prefill Per GPU params size (GB):", prefill_params_per_gpu_gb))
            logger.info("{:<40} {:<10.2f}".format("Decode Per GPU params size (GB) :", decode_params_per_gpu_gb))
            logger.info(f"Prefill: tp={self.tp} dp={self._replica_config._num_prefill_replicas} ep={self._replica_config.prefill_world_size} prefill_params_per_gpu_gb={prefill_params_per_gpu_gb} (GB)")
            logger.info(f"Decode: tp={self.tp} dp={self._replica_config._num_decode_replicas} ep={self._replica_config.decode_world_size} decode_params_per_gpu_gb={decode_params_per_gpu_gb} (GB)")
            assert self._replica_config._num_prefill_replicas % 1 == 0 and self._replica_config._num_decode_replicas % 1 == 0, "Prefill and Decode replicas must be integer"
                
            # # 计算每张GPU上的模型参数总量（包括共享专家和路由专家）
            # params_per_gpu = attn_params_bytes + expert_params_bytes * (
            #     self._model_config.num_shared_experts
            #     + self._model_config.num_routed_experts / self.ep
            # )
            
            # params_per_gpu = params_per_gpu / 1024 / 1024 / 1024  # 转换为GB
            # params_per_gpu *= self._model_config.num_hidden_layers  # 乘以层数得到总参数量
            # # 计算可用KV缓存内存（总显存减去模型参数、运行时开销和编码器预留）
            # self.kvcache_mem = (
            #     self.gpu.mem - params_per_gpu - 15 - 5
            # )  # 15GB for runtime, 5GB for encoder（15GB用于运行时，5GB用于编码器）
            # print("{:<40} {:<10.2f}".format("Per GPU params size (GB):", params_per_gpu))  # 打印每GPU参数大小（GB）
            
            # Return tuple: (total params, prefill params, decode params)
            # 返回三元组: (总参数量, prefill参数量, decode参数量)
            return params_per_gpu, prefill_params_per_gpu, decode_params_per_gpu
            
            
            # except AttributeError:
            #     # 如果无法获取_pipeline_stage_id，则回退到原来的实现
            #     num_parameters_per_layer = self.get_num_parameters_per_layer()
            #     return num_parameters_per_layer * self._num_layers_per_pipeline_stage
        else:
            num_parameters_per_layer = self.get_num_parameters_per_layer()
            return num_parameters_per_layer * self._num_layers_per_pipeline_stage

    def get_attn_params_size(self, config, use_fp8):
        if config.attn_type == "MHA/GQA":  # MHA or GQA attention type / MHA或GQA注意力类型
            return get_mha_params_size(self, config, use_fp8)
        elif config.attn_type == "MLA":  # MLA architecture / MLA结构
            return get_mla_params_size(self, config, use_fp8)


    # Reference: /InferSim/params/params.py
    # 参考自 /InferSim/params/params.py
    # def get_mha_params_size(config: ModelConfig, use_fp8: bool):
    def get_mha_params_size(self, config, use_fp8):
        wq = config.hidden_size * config.num_attention_heads * config.head_dim  # Q weight: hidden * heads * head_dim
        wk = config.hidden_size * config.num_key_value_heads * config.head_dim  # K weight: hidden * kv_heads * head_dim
        wv = config.hidden_size * config.num_key_value_heads * config.head_dim  # V weight: hidden * kv_heads * head_dim
        wo = config.hidden_size * config.num_attention_heads * config.head_dim  # Output weight: hidden * heads * head_dim
        if use_fp8:  # FP8 quantization / FP8量化
            return wq + wk + wv + wo  # Single precision storage / 单精度存储
        return 2 * (wq + wk + wv + wo)  # Full precision (e.g. FP16, 2 bytes per param) / 全精度

    # MLA (suited for DeepSeek) / MLA（适合 DeepSeek）
        # DeepSeek V3 parameter derivation references:
        # dpsk v3 参数推导参考：
            # https://zhuanlan.zhihu.com/p/21455638257 
            # https://yangwenbo.com/articles/deepseek-v3-parameter-size.html
        # "hidden_size": 7168,
        # "num_key_value_heads": 128,
        # "v_head_dim": 128,
        # "kv_lora_rank": 512,

        # "num_attention_heads": 128,
        # "q_lora_rank": 1536,

        # "qk_nope_head_dim": 128,
        # "qk_rope_head_dim": 64,

        # "num_hidden_layers": 61,
    # def get_mla_params_size(config: ModelConfig, use_fp8: bool):
    def get_mla_params_size(self, config, use_fp8):
        # Per-layer MLA Q LoRA params:
        # 单层 MLA 中 Q 的 LoRA 参数量是：
            # = 7168 * 1536 + 1536 + 1536 * 128 * (128 + 64) = 48,760,320
            # = wq_down + wq_up
            # = (config.hidden_size * config.q_lora_rank) + (config.q_lora_rank * config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim))
            # = (config.hidden_size * config.q_lora_rank) + (config.q_lora_rank * config.num_attention_heads * (config.qk_head_dim))
        wq_down = config.hidden_size * config.q_lora_rank  # Q LoRA down-projection / Q的LoRA下投影矩阵参数量
        wq_up = config.q_lora_rank * config.num_attention_heads * config.qk_head_dim  # Q LoRA up-projection / Q的LoRA上投影矩阵参数量
        # Per-layer MLA KV LoRA params:
        # 单层 MLA 中 KV 的 LoRA 参数量是：
            # = 7168 * (512 + 64) + 512 + 512 * 128 * (128 + 128) = 20,906,496
            # = wkv_down + 512 + wkv_up (TODO(tianhao909): clarify what the 512 constant represents)
            # = config.hidden_size *（config.kv_lora_rank + config.qk_rope_head_dim) + 512 + config.kv_lora_rank * config.num_attention_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim)
            # = (config.hidden_size * config.kv_lora_rank) + (config.kv_lora_rank * config.num_key_value_heads * (config.qk_nope_head_dim + config.qk_rope_head_dim))
        wkv_down = config.hidden_size * config.kv_lora_rank  # KV LoRA down-projection / KV的LoRA下投影矩阵参数量
        wkv_up = (  # KV LoRA up-projection / KV的LoRA上投影矩阵参数量
            config.kv_lora_rank
            * config.num_attention_heads
            * (config.qk_nope_head_dim + config.v_head_dim)
        )
        # Per-layer MLA output (WO) params:
        # 单层 MLA 中 WO 的参数量是
            # 128 * 128 * 7168 = 117,440,512
            # config.num_attention_heads * config.v_head_dim * config.hidden_size
        wo = config.hidden_size * config.num_attention_heads * config.v_head_dim  # Output weight / 输出权重参数量
        if use_fp8:  # FP8 quantization / FP8量化
            return wq_down + wq_up + wkv_down + wkv_up + wo  # Sum all params (single precision) / 返回所有参数之和
        # Unit: Bytes / 单位:B
        return 2 * (wq_down + wq_up + wkv_down + wkv_up + wo)  # FP16: multiply by 2 / 否则乘以2

        # Additionally: pre+post attention layernorm params = 7168*2 = 14,336
        # DeepSeek V3 MLA total across 61 layers:
        # 另外：pre+post attention layernorm 的参数量 = 7168*2 = 14,336
        # 所以 DeepSeek V3 的 MLA 部分共 61 层的总参数量是：
            # (48,760,320 + 20,906,496 + 117,440,512 + 14,336) * 61 = 11,414,421,504 (~11B)


    # def get_gdn_params_size(config: ModelConfig, use_fp8: bool):
    def get_gdn_params_size(self, config, use_fp8):
        wq = config.hidden_size * config.linear_num_key_heads * config.linear_key_head_dim  # Q linear attention weight
        wk = wq  # K weight same as Q
        wv = (  # V weight params
            config.hidden_size
            * config.linear_num_value_heads
            * config.linear_value_head_dim
        )
        wz = wv  # Z weight same as V
        wa = config.hidden_size * config.linear_num_value_heads  # A gate params
        wb = wa  # B gate same as A
        s = wq + wk + wv + wz + wa + wb  # Total primary weight params
        wconv = (  # Conv kernel weight part 1
            config.linear_num_key_heads
            * config.linear_key_head_dim
            * config.linear_conv_kernel_dim
        )
        wconv += (  # Conv kernel weight part 2
            config.linear_num_key_heads
            * config.linear_key_head_dim
            * config.linear_conv_kernel_dim
        )
        wconv += (  # Conv kernel weight part 3
            config.linear_num_value_heads
            * config.linear_value_head_dim
            * config.linear_conv_kernel_dim
        )
        if use_fp8:  # FP8 quantization
            return s + wconv  # Primary + conv params (single precision)
        return 2 * s + wconv  # Primary *2, conv stays single precision


    # def get_attn_params_size(config: ModelConfig, use_fp8: bool):
    def get_attn_params_size(self, config, use_fp8):
        if config.attn_type == "MHA/GQA":  # MHA or GQA attention type
            return self.get_mha_params_size(config, use_fp8)
        elif config.attn_type == "MLA":  # MLA architecture
            return self.get_mla_params_size(config, use_fp8)


    # def get_linear_attn_params_size(config: ModelConfig, use_fp8: bool):
    def get_linear_attn_params_size(self, config, use_fp8):
        return self.get_gdn_params_size(config, use_fp8)  # Get linear attention (GD-Nets style) params / 获取线性注意力参数总量

    # MoE (suited for DeepSeek) / MoE（适合 DeepSeek）
        # "num_hidden_layers": 61,
        # "hidden_size": 7168,
        # "moe_intermediate_size": 2048,  // Routed expert MLP intermediate dim / 路由专家 MLP 的中间维度
        # "n_shared_experts": 1,          // Shared expert count / 共享专家数量
        # "n_routed_experts": 256,        // Routed expert count / 路由专家数量
        # "first_k_dense_replace": 3,     // First K layers use dense instead of MoE / 前几层使用dense替换MoE
        # "intermediate_size": 18432,     // First 3 layers (9*moe_intermediate_size) / 前3层
        
        # Per-expert params: / 每个专家的参数量是：
            # 7168 * 2048 * 3 = 44,040,192
            # config.hidden_size * config.moe_intermediate_size * 3
        # Router gate params: / 路由 Gate 的参数量是：
            # 256 * 7168 + 256 = 1,835,264
        # First 3 dense layers (8 routed + 1 shared per layer): / 前 3 层 dense（固定激活 8 路由专家）：
            # 44,040,192 * 9 * 3 = 1,189,085,184
        # Last 58 sparse layers (dynamically activate 8 routed): / 后 58 层稀疏（动态激活 8 路由专家）：
            # (44,040,192 * 257 + 1,835,264) * 58 = 656,569,547,264
        # DeepSeek V3 MoE total params: / DeepSeek V3 MoE 部分总参数量：
            # 1,189,085,184 + 656,569,547,264 = 657,758,632,448 (~657B)
        # Active params per forward (1 shared + 8 routed): / 每次计算激活参数量（1共享 + 8路由）：
            # 44,040,192 * 9 * 61 + 1,835,264 * 58 = 24,284,510,720 (~24B)
    # def get_expert_params_size(config: ModelConfig, use_fp8: bool):
    def get_expert_params_size(self, config, use_fp8):
        if self._replica_config.model_name in [ 'qwen3-moe-235B']:
            # config.intermediate_size = 122888
            # config.moe_intermediate_size = 1536
            config.intermediate_size = config.moe_intermediate_size
            w = 3 * config.hidden_size * config.intermediate_size  # MoE expert FFN params (W1, W2, W3) / MoE专家前馈网络参数量
        else:
            w = 3 * config.hidden_size * config.intermediate_size  # MoE expert FFN params (W1, W2, W3) / MoE专家前馈网络参数量
        if not use_fp8:  # Not using FP8 / 不使用FP8量化
            w *= 2  # Double for FP16 / 参数量翻倍
        return w  # Return expert params total / 返回专家参数总量


    # def load_attn_weights_time(config: ModelConfig, use_fp8: bool, gpu: GPU):
    def load_attn_weights_time(self, config, use_fp8, gpu):
        size = self.get_attn_params_size(config, use_fp8)  # Get attention weights size (bytes) / 获取注意力模块权重总大小
        return size / 1024 / 1024 / 1024 / gpu.mem_bw  # Convert to GB / mem_bw = load time (s) / 转换为GB并除以GPU内存带宽


    # def load_moe_weights_time(config: ModelConfig, use_fp8: bool, gpu: GPU, num_gpus):
    def load_moe_weights_time(self, config, use_fp8, gpu, num_gpus):
        size = self.get_expert_params_size(config, use_fp8)  # Get single expert weights size / 获取单个专家权重大小
        size *= config.num_routed_experts / num_gpus  # Distribute across GPUs / 总专家数分配到多个GPU上
        return size / 1024 / 1024 / 1024 / gpu.mem_bw  # Load time in seconds / 加载时间（秒）
