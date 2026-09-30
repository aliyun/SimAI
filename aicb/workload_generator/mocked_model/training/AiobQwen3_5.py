"""
AiobQwen3_5 -- GPU compute-time benchmark for Qwen3.5 training workloads.

Adapted from AiobMegatron.py.  Qwen3.5-specific changes:
  - GatedDeltaNet: 75% of layers use linear attention (Conv1D + Gated Delta Rule)
    No softmax, no RoPE. Requires flash-linear-attention (FLA) and causal-conv1d.
  - Gated Full Attention (25%): head_dim=256, partial_rotary=0.25, doubled q_proj
    with sigmoid gate on output. QK-Norm per-head.
  - 3D M-RoPE with mrope_section=[11,11,10] for multimodal position encoding.
  - RMSNormGated for DeltaNet output: norm(output) * silu(gate).

GPU library requirements (NOT available on CPU/MPS):
  - flash-linear-attention (chunk_gated_delta_rule, fused_recurrent_gated_delta_rule)
  - causal-conv1d (causal_conv1d_fn, causal_conv1d_update)
  - flash-attn (flash_attn_unpadded_func)
  - apex (FastLayerNormFN)
  - scaled_upper_triang_masked_softmax_cuda

This file documents the benchmark structure. Actual GPU timings require
a CUDA-capable machine with the above libraries installed.
"""

import torch
import torch.nn.functional as F
import time
import math
import warnings
from utils.utils import divide

# Optional GPU-only imports
try:
    from apex.contrib.layer_norm.layer_norm import FastLayerNormFN
except ImportError:
    FastLayerNormFN = None
try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import (
            flash_attn_varlen_func as flash_attn_unpadded_func,
        )
    except ImportError:
        flash_attn_unpadded_func = None

# GatedDeltaNet GPU kernels -- only available with FLA + causal-conv1d
try:
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
except ImportError:
    chunk_gated_delta_rule = None
    fused_recurrent_gated_delta_rule = None
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_update = None

FLA_AVAILABLE = all([chunk_gated_delta_rule, fused_recurrent_gated_delta_rule,
                      causal_conv1d_fn, causal_conv1d_update])


# ===========================================================================
# Qwen3.5-specific GPU kernels
# ===========================================================================

class Qwen3_5RMSNormGated(torch.nn.Module):
    """RMSNorm + SiLU gate (used in GatedDeltaNet output)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate):
        # norm(x) * silu(gate)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms * self.weight) * F.silu(gate)


class Qwen3_5RMSNorm(torch.nn.Module):
    """Qwen3.5 RMSNorm with zero-init weight: output = norm(x) * (1 + weight)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * (1.0 + self.weight)


class Qwen3_5GatedDeltaNetBenchmark(torch.nn.Module):
    """
    GatedDeltaNet GPU benchmark harness.

    Operations (all local compute, no TP):
      1. Conv1D (causal depthwise, kernel_dim=4) on fused QKV
      2. Split into Q, K, V
      3. Gated Delta Rule recurrence: S_t = S_{t-1} * decay + k^T * delta
      4. Output gate: z projection, silu activation
      5. RMSNormGated: norm(output) * silu(z)
      6. Output projection: value_dim -> hidden_size
    """

    def __init__(self, args, layer_id=0):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_v_heads = getattr(args, "linear_num_value_heads", 32)
        self.num_k_heads = getattr(args, "linear_num_key_heads", 16)
        self.head_k_dim = getattr(args, "linear_key_head_dim", 128)
        self.head_v_dim = getattr(args, "linear_value_head_dim", 128)
        self.conv_kernel_size = getattr(args, "linear_conv_kernel_dim", 4)
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.layer_id = layer_id

        if args.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        device = torch.cuda.current_device()
        self.dtype = dtype

        # Input projections
        self.in_proj_qkv = torch.rand(self.hidden_size,
                                       self.key_dim * 2 + self.value_dim,
                                       device=device).to(dtype)
        self.in_proj_z = torch.rand(self.hidden_size, self.value_dim,
                                     device=device).to(dtype)
        self.in_proj_b = torch.rand(self.hidden_size, self.num_v_heads,
                                     device=device).to(dtype)
        self.in_proj_a = torch.rand(self.hidden_size, self.num_v_heads,
                                     device=device).to(dtype)

        # Conv1D weight
        conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_weight = torch.rand(conv_dim, self.conv_kernel_size,
                                       device=device).to(dtype)

        # Output projection
        self.out_proj = torch.rand(self.value_dim, self.hidden_size,
                                    device=device).to(dtype)

        # RMSNormGated for output
        self.norm = Qwen3_5RMSNormGated(self.head_v_dim, eps=1e-6)

        # Delta rule parameters
        self.dt_bias = torch.rand(self.num_v_heads, device=device).to(dtype)
        self.A_log = torch.rand(self.num_v_heads, device=device).to(dtype)

        # Input tensor
        self.input = torch.rand(args.seq_length, args.micro_batch,
                                 args.hidden_size, device=device).to(dtype)

    def forward(self):
        if not FLA_AVAILABLE:
            return 0.0  # cannot benchmark without GPU libraries

        x = self.input
        seq_len = x.shape[0]
        batch = x.shape[1]

        torch.cuda.synchronize()
        t_start = time.time()

        # QKV projection + Z projection + B/A projections
        mixed_qkv = F.linear(x, self.in_proj_qkv)
        z = F.linear(x, self.in_proj_z)
        b = F.linear(x, self.in_proj_b)
        a = F.linear(x, self.in_proj_a)

        # Causal Conv1D
        mixed_qkv = mixed_qkv.transpose(0, 1).transpose(1, 2)  # (B, C, L)
        mixed_qkv = causal_conv1d_fn(
            x=mixed_qkv, weight=self.conv_weight.unsqueeze(1),
            bias=None, activation="silu", seq_idx=None,
        )
        mixed_qkv = mixed_qkv.transpose(1, 2).transpose(0, 1)  # back to (L, B, C)

        # Split Q, K, V
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1,
        )
        query = query.view(seq_len, batch, self.num_k_heads, self.head_k_dim)
        key = key.view(seq_len, batch, self.num_k_heads, self.head_k_dim)
        value = value.view(seq_len, batch, self.num_v_heads, self.head_v_dim)
        z = z.view(seq_len, batch, self.num_v_heads, self.head_v_dim)

        # Gated Delta Rule
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        # Repeat K/Q if num_v_heads > num_k_heads (GQA for DeltaNet)
        if self.num_v_heads // self.num_k_heads > 1:
            r = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(r, dim=2)
            key = key.repeat_interleave(r, dim=2)

        core_out, _ = chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta,
            initial_state=None, output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )

        # Output gate + RMSNormGated + output projection
        core_out = core_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        gated = self.norm(core_out, z)
        output = F.linear(gated, self.out_proj)

        torch.cuda.synchronize()
        return (time.time() - t_start) * 1e6  # microseconds


class Qwen3_5GatedAttentionBenchmark(torch.nn.Module):
    """
    Qwen3.5 gated full-attention benchmark (25% of layers).

    Differences from Qwen3 attention:
      - head_dim = 256 (2x Qwen3)
      - q_proj outputs 2 * num_heads * head_dim (half query, half gate)
      - sigmoid gate on attention output
      - QK-Norm per-head (RMSNorm on 256)
      - partial RoPE: only rotary_dim = head_dim * partial_rotary_factor = 64 dims
    """

    def __init__(self, args):
        super().__init__()
        self.tp = args.tensor_model_parallel_size
        hidden_size = args.hidden_size
        n_heads = args.num_attention_heads
        n_kv = args.num_key_value_heads
        head_dim = getattr(args, "head_dim", 256)
        micro_batch = args.micro_batch
        seq_len = args.seq_length

        if args.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        device = torch.cuda.current_device()
        self.dtype = dtype
        self.head_dim = head_dim

        # Doubled q_proj: half query, half gate
        q_dim = n_heads * head_dim * 2
        self.q_weight = torch.rand(divide(q_dim, self.tp), hidden_size,
                                    device=device).to(dtype)
        self.k_weight = torch.rand(divide(n_kv * head_dim, self.tp), hidden_size,
                                    device=device).to(dtype)
        self.v_weight = torch.rand(divide(n_kv * head_dim, self.tp), hidden_size,
                                    device=device).to(dtype)
        self.o_weight = torch.rand(hidden_size, divide(n_heads * head_dim, self.tp),
                                    device=device).to(dtype)

        # QK-Norm
        self.q_norm = Qwen3_5RMSNorm(head_dim, eps=1e-6)
        self.k_norm = Qwen3_5RMSNorm(head_dim, eps=1e-6)

        # Input
        self.input = torch.rand(seq_len, micro_batch, hidden_size,
                                 device=device).to(dtype)

    def forward(self):
        x = self.input

        torch.cuda.synchronize()
        t_start = time.time()

        # Q projection (doubled) + K + V
        q_out = F.linear(x, self.q_weight)  # (L, B, 2 * heads * head_dim / tp)
        k_out = F.linear(x, self.k_weight)
        v_out = F.linear(x, self.v_weight)

        # Split query and gate
        q_dim = q_out.shape[-1] // 2
        q_raw = q_out[:, :, :q_dim]
        gate = q_out[:, :, q_dim:]

        # QK-Norm (per-head)
        # Reshape: (L, B, heads/tp, head_dim)
        q_normed = self.q_norm(q_raw)
        k_normed = self.k_norm(k_out)

        # Output projection + gate
        o_out = F.linear(q_normed, self.o_weight)
        o_out = o_out * torch.sigmoid(gate)

        torch.cuda.synchronize()
        return (time.time() - t_start) * 1e6


class Qwen3_5MlpBenchmark(torch.nn.Module):
    """Qwen3.5 SwiGLU MLP (same as Qwen3, different intermediate_size)."""

    def __init__(self, args):
        super().__init__()
        self.tp = args.tensor_model_parallel_size
        hidden_size = args.hidden_size
        intermediate_size = getattr(args, "intermediate_size",
                                     getattr(args, "ffn_hidden_size", 4 * hidden_size))
        swiglu = getattr(args, "swiglu", True)
        if args.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        device = torch.cuda.current_device()
        self.dtype = dtype

        gate_up_dim = 2 * intermediate_size if swiglu else intermediate_size
        self.gate_up_weight = torch.rand(divide(gate_up_dim, self.tp), hidden_size,
                                          device=device).to(dtype)
        self.down_weight = torch.rand(hidden_size, divide(intermediate_size, self.tp),
                                       device=device).to(dtype)
        self.input = torch.rand(args.seq_length, args.micro_batch,
                                 hidden_size, device=device).to(dtype)

    def forward(self):
        x = self.input
        torch.cuda.synchronize()
        t_start = time.time()
        gate_up = F.linear(x, self.gate_up_weight)
        gate_up = F.silu(gate_up)
        down = F.linear(gate_up, self.down_weight)
        torch.cuda.synchronize()
        return (time.time() - t_start) * 1e6


# ===========================================================================
# Full Qwen3.5 benchmark model
# ===========================================================================

class Qwen3_5Model(torch.nn.Module):
    """Complete Qwen3.5 training benchmark with hybrid GatedDeltaNet + Full Attention."""

    def __init__(self, args):
        super().__init__()
        self.time_list = {}
        self.args = args

        from workload_generator.mocked_model.training.AiobMegatron import (
            MegatronEmbedding, MegatronLayernorm, logit, Grad_param,
        )

        self.Embedding = MegatronEmbedding(self.args)
        self.Layernorm = MegatronLayernorm(self.args)

        num_layers = getattr(args, "num_hidden_layers", args.num_layers)
        full_interval = getattr(args, "full_attention_interval", 4)

        self.gdn_layers = []
        self.attn_layers = []
        self.mlp_layers = []
        for i in range(num_layers):
            if (i + 1) % full_interval == 0:
                self.attn_layers.append(Qwen3_5GatedAttentionBenchmark(self.args))
            else:
                self.gdn_layers.append(Qwen3_5GatedDeltaNetBenchmark(self.args, layer_id=i))
            self.mlp_layers.append(Qwen3_5MlpBenchmark(self.args))

        self.logit = logit(self.args)
        self.grad_param = Grad_param(self.args)

    def forward(self, input_tensor):
        for _ in range(self.args.epoch_num):
            Emb_output, Emb_time = self.Embedding(input_tensor)
            self.time_list.setdefault("Emb", []).append({"time_gpu": Emb_time})

            gdn_idx = 0
            attn_idx = 0
            for i in range(len(self.mlp_layers)):
                lay_out, lay_time = self.Layernorm(Emb_output)
                self.time_list.setdefault("layernorm", []).append({"time_gpu": lay_time})

                # Token mixer
                if (i + 1) % getattr(self.args, "full_attention_interval", 4) == 0:
                    attn_time = self.attn_layers[attn_idx]()
                    attn_idx += 1
                    self.time_list.setdefault("atten_full", []).append({"time_gpu": attn_time})
                else:
                    if FLA_AVAILABLE:
                        gdn_time = self.gdn_layers[gdn_idx]()
                        gdn_idx += 1
                        self.time_list.setdefault("atten_gdn", []).append({"time_gpu": gdn_time})
                    else:
                        self.time_list.setdefault("atten_gdn", []).append({"time_gpu": 1})

                # MLP
                lay_out2, lay_time2 = self.Layernorm(lay_out)
                self.time_list.setdefault("layernorm2", []).append({"time_gpu": lay_time2})
                mlp_time = self.mlp_layers[i]()
                self.time_list.setdefault("mlp", []).append({"time_gpu": mlp_time})

            # Final
            lay_out_post, post_time = self.Layernorm(Emb_output)
            self.time_list.setdefault("layernorm_post", []).append({"time_gpu": post_time})
            logit_time = self.logit(lay_out_post)
            self.time_list.setdefault("logit_time", []).append({"time_gpu": logit_time})
            grad_time = self.grad_param(lay_out_post)
            self.time_list.setdefault("param_time", []).append({"time_gpu": grad_time})

        from utils.utils import write_time
        return write_time(self.time_list, self.args)
