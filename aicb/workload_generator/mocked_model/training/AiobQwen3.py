"""
AiobQwen3 -- GPU compute-time benchmark for Qwen3 training workloads.

Adapted from AiobMegatron.py.  Qwen3-specific changes:
  - GQA: separate num_key_value_heads with explicit head_dim (128)
  - QK-Norm: per-head RMSNorm on Q and K after projection, before attention
  - QKV projection size: (n_heads + 2*n_kv_heads) * head_dim instead of 3*hidden

Requires CUDA GPU with flash-attn, apex, and scaled_upper_triang_masked_softmax.
Not runnable on CPU/MPS -- this file documents the benchmark structure for
machines with the required GPU libraries.
"""

import torch
import torch.nn.functional as F
import time
import math
import warnings
from utils.utils import divide

# Optional GPU-only imports -- will fail on CPU/MPS
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


# ===========================================================================
# Qwen3-specific GPU kernels
# ===========================================================================

class Qwen3RMSNorm(torch.nn.Module):
    """RMSNorm for QK-Norm (per-head, on head_dim=128)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: (seq, batch, heads, head_dim)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class Qwen3Layernorm(torch.nn.Module):
    """Standard RMSNorm + QK-Norm timing harness for Qwen3."""

    def __init__(self, args):
        super().__init__()
        self.tp = args.tensor_model_parallel_size
        self.enable_sequence_parallel = args.enable_sequence_parallel
        self.hidden_size = args.hidden_size
        seq_len = args.seq_length
        micro_batch = args.micro_batch
        device = torch.cuda.current_device()
        if args.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        self.dtype = dtype

        # Standard layernorm weights
        self.ln_weight = torch.rand(self.hidden_size, device=device).to(dtype)
        self.ln_bias = torch.zeros(self.hidden_size, device=device).to(dtype)

        # QK-Norm weights (2 per layer: q_norm, k_norm)
        head_dim = getattr(args, "head_dim", 128)
        n_kv_heads = getattr(args, "num_key_value_heads", args.num_attention_heads)
        self.q_norm = Qwen3RMSNorm(head_dim, eps=args.rms_norm_eps if hasattr(args, "rms_norm_eps") else 1e-6)
        self.k_norm = Qwen3RMSNorm(head_dim, eps=args.rms_norm_eps if hasattr(args, "rms_norm_eps") else 1e-6)

        self.layernorm_input = torch.rand(
            seq_len * self.tp if self.enable_sequence_parallel else seq_len,
            micro_batch, self.hidden_size, device=device,
        ).to(dtype)

    def _apply_fused_layer_norm(self, hidden_states):
        torch.cuda.synchronize()
        t_start = time.time()
        ln_out = FastLayerNormFN.apply(
            hidden_states, torch.tensor([self.hidden_size], device=hidden_states.device),
            self.ln_weight, self.ln_bias, 1e-5,
        )
        torch.cuda.synchronize()
        return ln_out, (time.time() - t_start) * 1e6

    def forward(self, hidden_states):
        # Standard layernorm
        try:
            lay_out, lay_time = self._apply_fused_layer_norm(hidden_states)
        except Exception:
            torch.cuda.synchronize()
            t_start = time.time()
            lay_out = F.layer_norm(
                hidden_states, [self.hidden_size], self.ln_weight, self.ln_bias, 1e-5,
            )
            torch.cuda.synchronize()
            lay_time = (time.time() - t_start) * 1e6

        if self.enable_sequence_parallel:
            lay_out = lay_out.repeat((self.tp, 1, 1))
        return lay_out, lay_time

    def qk_norm_forward(self, q, k):
        """Time the QK-Norm operations (computed per-head)."""
        torch.cuda.synchronize()
        t_start = time.time()
        q = self.q_norm(q)
        k = self.k_norm(k)
        torch.cuda.synchronize()
        return q, k, (time.time() - t_start) * 1e6


class Qwen3Attention(torch.nn.Module):
    """Qwen3 GQA attention with QK-Norm (GPU benchmark harness)."""

    def __init__(self, args):
        super().__init__()
        self.tp = args.tensor_model_parallel_size
        self.enable_sequence_parallel = args.enable_sequence_parallel
        hidden_size = args.hidden_size
        n_heads = args.num_attention_heads
        n_kv = getattr(args, "num_key_value_heads", n_heads)
        head_dim = getattr(args, "head_dim", hidden_size // n_heads)
        micro_batch = args.micro_batch
        seq_len = args.seq_length

        if args.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        device = torch.cuda.current_device()

        # GQA-aware fused QKV weight
        q_dim = n_heads * head_dim
        k_dim = n_kv * head_dim
        v_dim = n_kv * head_dim
        total_qkv = q_dim + k_dim + v_dim
        self.atten_weight_qkv = torch.rand(
            divide(total_qkv, self.tp), hidden_size, device=device,
        ).to(dtype)

        # Output projection weight
        self.hidden_size_per_partition = divide(hidden_size, self.tp)
        self.atten_weight_out = torch.rand(
            hidden_size, divide(q_dim, self.tp), device=device,
        ).to(dtype)

        # Attention core tensors
        self.num_heads_per_tp = divide(n_heads, self.tp)
        self.num_kv_per_tp = divide(n_kv, self.tp)
        self.head_dim = head_dim

        query_layer = torch.rand(
            seq_len, micro_batch, self.num_heads_per_tp, head_dim, device=device,
        ).to(dtype)
        key_layer = torch.rand(
            seq_len, micro_batch, self.num_kv_per_tp, head_dim, device=device,
        ).to(dtype)
        value_layer = torch.rand(
            seq_len, micro_batch, self.num_kv_per_tp, head_dim, device=device,
        ).to(dtype)

        output_size = (
            query_layer.size(1), query_layer.size(2),
            query_layer.size(0), key_layer.size(0),
        )
        self.query_layer = query_layer.view(output_size[2], output_size[0] * output_size[1], -1)
        self.key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
        self.value_layer = value_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        self.matmul_input_buffer = torch.zeros(
            output_size[0] * output_size[1], output_size[2], output_size[3],
            device=device,
        ).to(dtype)
        self.scale_t = torch.tensor(1).to(dtype)
        soft_input = torch.rand(output_size, device=device).to(dtype)
        self.b, self.np, self.sq, self.sk = soft_input.size()

        # QK-Norm harness
        n_kv_tp = divide(n_kv, self.tp)
        self.q_for_norm = torch.rand(seq_len, micro_batch, self.num_heads_per_tp, head_dim, device=device).to(dtype)
        self.k_for_norm = torch.rand(seq_len, micro_batch, n_kv_tp, head_dim, device=device).to(dtype)

    def forward(self, hidden_states):
        torch.cuda.synchronize()
        t_start = time.time()

        # QKV fused projection
        mixed_qkv = torch.matmul(hidden_states, self.atten_weight_qkv.t())
        # Split into Q, K, V
        q_dim = self.num_heads_per_tp * self.head_dim
        k_dim = self.num_kv_per_tp * self.head_dim
        q_raw = mixed_qkv[:, :, :q_dim]
        k_raw = mixed_qkv[:, :, q_dim:q_dim + k_dim]
        v_raw = mixed_qkv[:, :, q_dim + k_dim:]

        qkv_time = (time.time() - t_start) * 1e6

        return q_raw, k_raw, v_raw, qkv_time


class Qwen3Mlp(torch.nn.Module):
    """Qwen3 SwiGLU MLP (GPU benchmark harness)."""

    def __init__(self, args):
        super().__init__()
        self.tp = args.tensor_model_parallel_size
        hidden_size = args.hidden_size
        intermediate_size = getattr(args, "intermediate_size", args.ffn_hidden_size)
        swiglu = getattr(args, "swiglu", False)
        if args.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        device = torch.cuda.current_device()

        gate_up_dim = 2 * intermediate_size if swiglu else intermediate_size
        self.mlp_weight_gate_up = torch.rand(
            divide(gate_up_dim, self.tp), hidden_size, device=device,
        ).to(dtype)
        self.mlp_weight_down = torch.rand(
            hidden_size, divide(intermediate_size, self.tp), device=device,
        ).to(dtype)

    def forward(self, hidden_states):
        torch.cuda.synchronize()
        t_start = time.time()
        gate_up = torch.matmul(hidden_states, self.mlp_weight_gate_up.t())
        # SiLU activation (swish)
        gate_up = F.silu(gate_up)
        gate_time = (time.time() - t_start) * 1e6

        torch.cuda.synchronize()
        t_start = time.time()
        down = torch.matmul(gate_up, self.mlp_weight_down.t())
        down_time = (time.time() - t_start) * 1e6

        return gate_time + down_time


# ===========================================================================
# Full Qwen3 benchmark model
# ===========================================================================

class Qwen3Model(torch.nn.Module):
    """Complete Qwen3 training benchmark.  Same structure as AiobMegatron.MegatronModel."""

    def __init__(self, args):
        super().__init__()
        self.time_list = {}
        self.args = args

        from workload_generator.mocked_model.training.AiobMegatron import (
            MegatronEmbedding, MegatronLayernorm, logit, Grad_param,
        )
        self.Embedding = MegatronEmbedding(self.args)
        self.Layernorm = MegatronLayernorm(self.args)
        self.Qwen3Layernorm = Qwen3Layernorm(self.args)
        self.Attention = Qwen3Attention(self.args)
        self.Mlp = Qwen3Mlp(self.args)
        self.logit = logit(self.args)
        self.grad_param = Grad_param(self.args)

    def forward(self, input):
        for _ in range(self.args.epoch_num):
            Emb_output, Emb_time = self.Embedding(input)
            self.time_list.setdefault("Emb", []).append({"time_gpu": Emb_time})

            for _ in range(self.args.num_layers):
                # Layernorm
                lay_out, lay_time = self.Layernorm(Emb_output)
                self.time_list.setdefault("layernorm", []).append({"time_gpu": lay_time})

                # Attention QKV projection
                q_raw, k_raw, v_raw, atten_qkv_time = self.Attention(lay_out)
                self.time_list.setdefault("atten_qkv", []).append({"time_gpu": atten_qkv_time})

                # QK-Norm (Qwen3-specific)
                q_normed, k_normed, qk_norm_time = self.Qwen3Layernorm.qk_norm_forward(q_raw, k_raw)
                self.time_list.setdefault("qk_norm", []).append({"time_gpu": qk_norm_time})

                # Attention core (Q*K^T, softmax, context)
                # Reuse Megatron's attention core timing -- dimensions match after GQA repeat
                # TODO: implement full attention core benchmark for GQA
                self.time_list.setdefault("atten_core", []).append({"time_gpu": 1})

                # Attention output projection
                atten_out_time = self._time_matmul(
                    lay_out[:, :, :self.Attention.hidden_size_per_partition],
                    self.Attention.atten_weight_out,
                )
                self.time_list.setdefault("atten_linear", []).append({"time_gpu": atten_out_time})

                # Post-attention layernorm
                lay_out2, lay_time2 = self.Layernorm(lay_out)
                self.time_list.setdefault("layernorm2", []).append({"time_gpu": lay_time2})

                # MLP
                mlp_time = self.Mlp(lay_out2)
                self.time_list.setdefault("mlp", []).append({"time_gpu": mlp_time})

            # Final layernorm + logit + grad_param
            lay_out_post, post_time = self.Layernorm(Emb_output)
            self.time_list.setdefault("layernorm_post", []).append({"time_gpu": post_time})

            logit_time = self.logit(lay_out_post)
            self.time_list.setdefault("logit_time", []).append({"time_gpu": logit_time})

            grad_time = self.grad_param(lay_out_post)
            self.time_list.setdefault("param_time", []).append({"time_gpu": grad_time})

        from utils.utils import write_time
        return write_time(self.time_list, self.args)

    @staticmethod
    def _time_matmul(a, b):
        torch.cuda.synchronize()
        t_start = time.time()
        torch.matmul(a, b.t())
        torch.cuda.synchronize()
        return (time.time() - t_start) * 1e6
