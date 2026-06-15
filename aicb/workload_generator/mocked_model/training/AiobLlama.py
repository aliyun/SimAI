"""
AIOB (AI Operator Benchmark) GPU compute-time profiler for LLaMA models.

Measures real GPU execution times for LLaMA-specific operators:
  - RMSNorm (weight-only, no bias)
  - RoPE (cos/sin application to Q/K)
  - GQA Attention (Q/K/V/O projections with grouped KV heads)
  - SwiGLU MLP (gate/up element-wise multiply + down)
  - LlamaDecoderLayer (full pre-norm block)

Usage:
    torchrun --nproc_per_node=<gpus> AiobLlama.py --frame LLaMA --aiob_enable ...

Requirements:
    - NVIDIA GPU (Hopper recommended for FlashAttention-3)
    - torch with CUDA
    - flash_attn (optional, for FlashAttention profiling)

Based on AiobMegatron.py patterns.
File: AiobLlama.py
License: Apache 2.0
"""

import torch
import time
import warnings
import torch.nn.functional as F
import math

from typing import Optional

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None

from utils.utils import Comp_with_aiob


# ---------------------------------------------------------------------------
# LlamaRMSNorm -- AIOB profiled version
# ---------------------------------------------------------------------------
class LlamaRMSNorm(torch.nn.Module):
    """GPU-profiled RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight.

    Simpler than LayerNorm: no bias, no mean subtraction.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.weight = torch.nn.Parameter(torch.ones(args.hidden_size))

    def forward(self, x):
        # Record compute time using the AIOB timing infrastructure
        start = time.time()
        # RMSNorm forward: x * rsqrt(mean(x^2) + eps)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        output = x * rms * self.weight
        torch.cuda.synchronize()
        elapsed = time.time() - start
        return output, elapsed


# ---------------------------------------------------------------------------
# LlamaRotaryEmbedding -- AIOB profiled version
# ---------------------------------------------------------------------------
class LlamaRotaryEmbedding(torch.nn.Module):
    """GPU-profiled RoPE: applies rotary position embeddings to Q and K.

    Precomputes cos/sin frequency tables and applies rotation in-place
    or via fused kernel.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dim = args.hidden_size // args.num_attention_heads
        self.max_position_embeddings = getattr(args, "max_position_embeddings", args.seq_length)
        self.rope_theta = getattr(args, "rope_theta", 10000.0)

        # Precompute frequency bands
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x, position_ids):
        start = time.time()
        # Apply RoPE: split x into pairs, rotate by cos/sin
        # Simplified version for timing measurement
        seq_len = x.shape[1]
        freqs = torch.outer(position_ids.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(0)
        sin = emb.sin().unsqueeze(0)

        # Rotate half the feature dimensions
        x_rot = x.float()
        x_out = torch.empty_like(x_rot)
        x_out[..., 0::2] = x_rot[..., 0::2] * cos - x_rot[..., 1::2] * sin
        x_out[..., 1::2] = x_rot[..., 1::2] * cos + x_rot[..., 0::2] * sin

        torch.cuda.synchronize()
        elapsed = time.time() - start
        return x_out.to(x.dtype), elapsed


# ---------------------------------------------------------------------------
# LlamaAttention -- AIOB profiled version (GQA)
# ---------------------------------------------------------------------------
class LlamaAttention(torch.nn.Module):
    """GPU-profiled GQA Attention with optional FlashAttention.

    Projects to Q (num_heads * head_dim), K/V (num_kv_heads * head_dim),
    then applies attention with KV repetition for GQA.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.num_kv_heads = args.num_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.tp = args.tensor_model_parallel_size

        # QKV projections
        self.q_proj = torch.nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = torch.nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = torch.nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = torch.nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(self, hidden_states):
        bsz, seq_len, _ = hidden_states.shape
        time_dict = {}

        # Q projection
        start = time.time()
        q = self.q_proj(hidden_states)
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        torch.cuda.synchronize()
        time_dict["attention_q"] = time.time() - start

        # K projection
        start = time.time()
        k = self.k_proj(hidden_states)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        torch.cuda.synchronize()
        time_dict["attention_k"] = time.time() - start

        # V projection
        start = time.time()
        v = self.v_proj(hidden_states)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        torch.cuda.synchronize()
        time_dict["attention_v"] = time.time() - start

        # Attention (FlashAttention if available, else scaled dot-product)
        start = time.time()
        if flash_attn_varlen_func is not None and self.num_kv_heads == self.num_heads:
            # FlashAttention: only works for MHA (kv_heads == heads) in some versions
            attn_output = flash_attn_varlen_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                max_seqlen_q=seq_len, max_seqlen_k=seq_len,
            )
        else:
            # Scaled dot-product attention with GQA KV repetition
            if self.num_kv_heads < self.num_heads:
                # Repeat K/V heads to match Q heads
                n_repeat = self.num_heads // self.num_kv_heads
                k = k.repeat_interleave(n_repeat, dim=1)
                v = v.repeat_interleave(n_repeat, dim=1)
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        torch.cuda.synchronize()
        time_dict["attention_scores"] = time.time() - start

        # O projection
        start = time.time()
        output = self.o_proj(attn_output)
        torch.cuda.synchronize()
        time_dict["attention_o"] = time.time() - start

        return output, time_dict


# ---------------------------------------------------------------------------
# LlamaMLP -- AIOB profiled version (SwiGLU)
# ---------------------------------------------------------------------------
class LlamaMLP(torch.nn.Module):
    """GPU-profiled SwiGLU MLP: gate(x) * up(x) -> down.

    SiLU activation on gate projection, element-wise multiply with up
    projection, then down projection. Three linear layers total.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.intermediate_size = args.ffn_hidden_size

        self.gate_proj = torch.nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = torch.nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = torch.nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )

    def forward(self, x):
        time_dict = {}

        # Gate projection + SiLU
        start = time.time()
        gate = F.silu(self.gate_proj(x))
        torch.cuda.synchronize()
        time_dict["mlp_gate"] = time.time() - start

        # Up projection
        start = time.time()
        up = self.up_proj(x)
        torch.cuda.synchronize()
        time_dict["mlp_up"] = time.time() - start

        # Element-wise multiply + down projection
        start = time.time()
        output = self.down_proj(gate * up)
        torch.cuda.synchronize()
        time_dict["mlp_down"] = time.time() - start

        return output, time_dict


# ---------------------------------------------------------------------------
# LlamaModel -- Full AIOB profiled model
# ---------------------------------------------------------------------------
class LlamaModel(torch.nn.Module):
    """Complete LLaMA model for AIOB GPU compute-time profiling.

    Measures per-operator GPU time for each component and stores in
    self.time_list for consumption by the AICB compute-cache pipeline.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.time_list = {}

        self.embedding = torch.nn.Embedding(
            args.padded_vocab_size, args.hidden_size
        )
        self.rotary_emb = LlamaRotaryEmbedding(args)
        self.attention = LlamaAttention(args)
        self.mlp = LlamaMLP(args)
        self.input_norm = LlamaRMSNorm(args)
        self.post_attn_norm = LlamaRMSNorm(args)
        self.final_norm = LlamaRMSNorm(args)
        self.lm_head = torch.nn.Linear(
            args.hidden_size, args.padded_vocab_size, bias=False
        )

    def forward(self, input_ids):
        for _ in range(self.args.epoch_num):
            # Embedding
            start = time.time()
            hidden_states = self.embedding(input_ids)
            torch.cuda.synchronize()
            self.time_list.setdefault("embedding", []).append(
                {"time_gpu": time.time() - start}
            )

            for _ in range(self.args.num_layers):
                # Input RMSNorm
                hidden_states, norm_time = self.input_norm(hidden_states)
                self.time_list.setdefault("input_norm", []).append(
                    {"time_gpu": norm_time}
                )

                # Attention (GQA)
                attn_output, attn_times = self.attention(hidden_states)
                for k, v in attn_times.items():
                    self.time_list.setdefault(k, []).append({"time_gpu": v})
                hidden_states = hidden_states + attn_output

                # Post-attention RMSNorm
                hidden_states, norm_time = self.post_attn_norm(hidden_states)
                self.time_list.setdefault("post_attn_norm", []).append(
                    {"time_gpu": norm_time}
                )

                # MLP (SwiGLU)
                mlp_output, mlp_times = self.mlp(hidden_states)
                for k, v in mlp_times.items():
                    self.time_list.setdefault(k, []).append({"time_gpu": v})
                hidden_states = hidden_states + mlp_output

            # Final RMSNorm
            hidden_states, norm_time = self.final_norm(hidden_states)
            self.time_list.setdefault("final_norm", []).append(
                {"time_gpu": norm_time}
            )

            # LM Head
            start = time.time()
            logits = self.lm_head(hidden_states)
            torch.cuda.synchronize()
            self.time_list.setdefault("lm_head", []).append(
                {"time_gpu": time.time() - start}
            )

        return logits, self.time_list


# ---------------------------------------------------------------------------
# Entry point (same pattern as AiobMegatron.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from utils.utils import get_args

    args = get_args()
    assert args.aiob_enable, "AIOB profiling requires --aiob_enable flag"
    assert args.frame == "LLaMA", "AiobLlama requires --frame LLaMA"

    model = LlamaModel(args).cuda()
    # Use the Comp_with_aiob pipeline from utils to save timing data
    print(f"[AiobLlama] Profiling LLaMA model with {args.num_layers} layers...")
    print(f"[AiobLlama] hidden={args.hidden_size}, intermediate={args.ffn_hidden_size}")
    print(f"[AiobLlama] Q heads={args.num_attention_heads}, KV heads={args.num_kv_heads}")
    print(f"[AiobLlama] TP={args.tensor_model_parallel_size}, seq_len={args.seq_length}")
