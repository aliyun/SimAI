"""Provides DeepSeek implementation for Aiob compute time calculations.

Based on https://github.com/deepseek-ai/DeepSeek-V3/tree/f6e34dd26772dd4a216be94a8899276c5dca9e43
and https://github.com/deepseek-ai/DeepGEMM/tree/391755ada0ffefa9a6a52b6f14dcaf22d1a463e0

Note: DeepGEMM sadly only works with SM90, SM100 i.e. Hopper and Blackwell

Todo:
- add FlashMLA from https://github.com/deepseek-ai/FlashMLA/ for attentions
    as of https://github.com/deepseek-ai/FlashMLA/commit/41b611f7d7561790a2f5040ff89212e08c7b0011 
    FlashMLA's kernel supports backwards :)
- add DeepEP from https://github.com/deepseek-ai/DeepEP/

@misc{deepseekai2024deepseekv3technicalreport,
      title={DeepSeek-V3 Technical Report},
      author={DeepSeek-AI},
      year={2024},
      eprint={2412.19437},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.19437},
}


File: AiobDeepSeek.py
License: Apache 2.0

"""

import random
from dataclasses import dataclass
from typing import Optional
import math

import torch
import torch.nn.functional as F
import torch.nn as nn

# from  workload_generator.mocked_model.training.AiobMegatron import linear_with_grad_accumulation_and_async_allreduce
# can't import because of circular imports
# FIXME: maybe move this to utils

from torch.cuda.amp import custom_bwd, custom_fwd
class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel,
        tp,
    ):
        ctx.save_for_backward(input, weight)  #
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.async_grad_allreduce = async_grad_allreduce
        ctx.sequence_parallel = sequence_parallel

        if sequence_parallel:

            total_input = input
        else:
            total_input = input

        output = torch.matmul(total_input, weight.t())

        if bias is not None:
            output = output + bias
        return output


def linear_with_grad_accumulation_and_async_allreduce(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    gradient_accumulation_fusion: bool,
    async_grad_allreduce: bool,
    sequence_parallel: bool,
    tp,
) -> torch.Tensor:
    """Linear layer execution with asynchronous communication and
    gradient accumulation fusion in backprop.

    This has the option to accumulate the result of backprop
    calculation into an existing gradient buffer, preventing the need
    to do an additional addition kernel after the gradient
    calculation.

    Additionally, the tensor parallel all reduce of the input
    gradients can be done asynchronously with the calculation of
    the weight gradients.

    In the case of sequence parallelism, the reduce scatter of the
    input gradients is done asynchronously with the calcluation of the
    weight gradients.

    Use of this module requires that the environment variable
    CUDA_DEVICE_MAX_CONNECTIONS=1. There are a few collective
    operations, noted in the code, that should be scheduled before
    compute kernels to overlap the communication with the computation,
    which is necessary for a speedup but not for correctness so that
    ordering isn't imposed by the scheduler. Setting
    CUDA_DEVICE_MAX_CONNECTIONS=1 forces the kernels to be scheduled
    in the order they are called.

    Arguments:

    input (torch.Tensor required): input like torch.nn.functional.linear

    weight (torch.Tensor required): weight like torch.nn.functional.linear

    bias (torch.Tensor optional): bias like torch.nn.functional.linear

    gradient_accumulation_fusion (bool required): Perform the gradient
        accumulation fusion, requires the custom CUDA extension
        fused_weight_gradient_mlp_cuda module. To use
        gradient_accumulation_fusion you must install APEX with
        --cpp_ext and --cuda_ext. For example: "pip install
        --global-option=\"--cpp_ext\" --global-option=\"--cuda_ext .\"
        " Note that the extension requires CUDA>=11. Otherwise, you
        must turn off gradient accumulation fusion."

    async_grad_allreduce (bool required): Do the allreduce of input
        gradients asyncronously with the computation of weight
        gradients. If sequence_parallel is True, this must be
        False, as no all reduce is performed.

    sequence_parallel (bool required): Indicates that sequence
        parallelism is used and thus in the forward pass the input is
        all gathered, and the backward pass the input gradients are
        reduce scattered.
    """
    args = [
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        async_grad_allreduce,
        sequence_parallel,
        tp,
    ]

    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)


linear_with_grad_accumulation_and_async_allreduce.warned = False

from utils.utils import cuda_timing_decorator, divide

# needs deep_gemm from https://github.com/deepseek-ai/DeepGEMM
try:
    import deep_gemm
    from deep_gemm import ceil_div
except ImportError as e:
    print("""
        Needs deep_gemm from https://github.com/deepseek-ai/DeepGEMM
        to install,
           git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
           cd DeepGEMM
           ./install.sh
    """)
    raise e


# from deepgemm/tests
def per_token_cast_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    pad_size = (128 - (n % 128)) % 128
    x = torch.nn.functional.pad(
        x, (0, pad_size), value=0) if pad_size > 0 else x
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    fp8_data = (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn)
    return fp8_data.view(m, n + pad_size)[:, :n], (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_div(m, 128) * 128, ceil_div(n, 128) * 128),
        dtype=x.dtype,
        device=x.device,
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(
        x_view.size(0),
        x_view.size(2),
    )


@dataclass
class Args:
    hidden_size: int
    ffn_hidden_size: int
    micro_batch: int
    expert_model_parallel_size: int
    num_experts: int
    moe_router_topk: int
    seq_length: int
    n_shared_expert: int
    qk_nope_dim: int
    qk_rope_dim: int
    v_head_dim: int
    q_lora_rank: int
    kv_lora_rank: int
    num_attention_heads: int
    enable_sequence_parallel: bool
    tensor_model_parallel_size: int

# from deepseek (without quantization part)
# https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py
def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.
    """
    # sequence parallel, and tp are no-op in linear_with_grad_accumulation_and_async_allreduce -> LinearWithGradAccumulationAndAsyncCommunication
    return linear_with_grad_accumulation_and_async_allreduce(input=x,
                                                      weight=weight,
                                                      bias=bias,
                                                      gradient_accumulation_fusion=True,
                                                      async_grad_allreduce=False,
                                                      sequence_parallel=True,
                                                      tp=1)


class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))
        if self.weight.element_size() == 1:
            pass
            # skip scaling part
            # scale_out_features = (out_features + block_size - 1) // block_size
            # scale_in_features = (in_features + block_size - 1) // block_size
            # self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None, world_size: int = 1):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None, world_size: int = 1):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight)
        # if world_size > 1:
        #     dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y
class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6, dtype = torch.bfloat16):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)

def precompute_freqs_cis(args: Args) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    dim = args.qk_rope_dim
    seqlen = args.seq_length
    beta_fast = 32
    beta_slow = 1
    base = 10000.0
    factor = 40

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seqlen > 4096:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, 4096)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

#############################
# End of copy from DeepSeek #
#############################
class DeepSeekMLA(torch.nn.Module):
    """DeepSeekMLA layer for AIOB

    Args:
        torch (torch.Tensor): input tensor

    forward() return timing in format of dict,
    i.e.
        layer_time_map = {
            "attention_linear_q_lora": q_lora_time,
            "attention_q_column": q_time,
            "attention_linear_kv_lora": kv_lora_time,
            "attention_kv_column": kv_time,
            "attention_o_row": o_time
        }
    # FIXME: create a generic/structured way to return timings
    # FIXME: fix apply_pe functions, right now they're commented out, should be trivial compute overhead 
    """
    def __init__(self, args: Args):
        super().__init__()
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)
        self.args = args
        self.enable_sequence_parallel = args.enable_sequence_parallel
        self.tp = args.tensor_model_parallel_size
        self.qk_head_dim = self.args.qk_nope_dim + self.args.qk_rope_dim
        # init torch
        self.device = torch.cuda.current_device()
        self.dtype = torch.bfloat16
        self.hidden_size = self.args.hidden_size
        self.n_local_heads = self.args.num_attention_heads // self.tp


        # Q down projection
        self.wq_a = Linear(self.args.hidden_size, self.args.q_lora_rank, dtype=self.dtype)
        self.q_norm = RMSNorm(self.args.q_lora_rank, dtype=self.dtype)

        self.wq_b = ColumnParallelLinear(
            self.args.q_lora_rank, self.args.num_attention_heads * self.qk_head_dim,
            dtype=self.dtype,
            world_size=self.tp
        )

        # # KV down projection
        self.wkv_a = Linear(
            self.args.hidden_size, self.args.kv_lora_rank + self.args.qk_rope_dim, dtype=self.dtype
        )

        self.kv_norm = RMSNorm(self.args.kv_lora_rank, dtype=self.dtype)
        self.wkv_b = ColumnParallelLinear(
            self.args.kv_lora_rank,
            self.args.num_attention_heads
            * (self.args.qk_nope_dim + self.args.v_head_dim),
            dtype=self.dtype,
            world_size=self.tp
        )

        self.wo = RowParallelLinear(
            self.args.num_attention_heads * self.args.v_head_dim, self.args.hidden_size,
            dtype=self.dtype,
            world_size=self.tp
        )

        self.softmax_scale = self.qk_head_dim**-0.5

    @cuda_timing_decorator
    def _apply_linear_q_lora(self, x):
        return self.q_norm(self.wq_a(x))

    @cuda_timing_decorator
    def _apply_q(self, x):
        bsz, seqlen, _ = x.size()
        q = self.wq_b(x)
        q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.args.qk_nope_dim, self.args.qk_rope_dim], dim=-1)
        # q_pe = apply_rotary_emb(q_pe, self.freqs_cis)
        return (q_nope, q_pe)

    @cuda_timing_decorator
    def _apply_linear_kv_lora(self, x):
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.args.kv_lora_rank, self.args.qk_rope_dim], dim=-1)
        kv = self.kv_norm(kv)
        return (kv, k_pe)

    @cuda_timing_decorator
    def _apply_kv(self, x, q_nope, q_pe, kv, k_pe):

        # k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        k_pe.unsqueeze(2)

        wkv_b = self.wkv_b.weight
        wkv_b = wkv_b.view(self.n_local_heads, -1, self.args.kv_lora_rank)

        q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.args.qk_nope_dim])

        scores = (torch.einsum("bshc,btc->bsht", q_nope, kv) +
                      torch.einsum("bshr,btr->bsht", q_pe, k_pe.squeeze(2))) * self.softmax_scale

        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        x = torch.einsum("bsht,btc->bshc", scores, kv)
        x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.args.v_head_dim:])
        return x

    @cuda_timing_decorator
    def _apply_kv_naive(self, x, q_nope, q_pe):
        bsz, seqlen, _ = x.size()
        # compress kv
        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.args.kv_lora_rank, self.args.qk_rope_dim], dim=-1)
        # rotate
        # k_pe = apply_rotary_emb(k_pe.unsqueeze(2), self.freqs_cis)
        k_pe = k_pe.unsqueeze(2)

        q = torch.cat([q_nope, q_pe], dim=-1)
        kv = self.wkv_b(self.kv_norm(kv))
        kv = kv.view(bsz, seqlen, self.n_local_heads, self.args.qk_nope_dim + self.args.v_head_dim)
        k_nope, v = torch.split(kv, [self.args.qk_nope_dim, self.args.v_head_dim], dim=-1)

        print(f"kv {kv.shape} k_pe: {k_pe.shape}")
        k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
        scores = torch.einsum("bshd,bthd->bsht", q, k) * self.softmax_scale
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        x = torch.einsum("bsht,bthd->bshd", scores, v)
        return x

    @cuda_timing_decorator
    def _apply_wo(self, x):
        x = self.wo(x.flatten(2))
        return x

    def forward(self, x):
        q, q_lora_time = self._apply_linear_q_lora(x)
        (q_nope, q_pe), q_time = self._apply_q(q)
        (kv, k_pe), kv_lora_time = self._apply_linear_kv_lora(x)
        x, kv_time = self._apply_kv(x, q_nope, q_pe, kv, k_pe)
        x, o_time = self._apply_wo(x)
        layer_time_map = {
            "attention_linear_q_lora": q_lora_time,
            "attention_q_column": q_time,
            "attention_linear_kv_lora": kv_lora_time,
            "attention_kv_column": kv_time,
            "attention_o_row": o_time
        }
        return x, layer_time_map

def test_deepseek_mla() -> torch.Tensor:
    x = torch.randn(5120, 1, 12288, dtype=torch.bfloat16)
    args = Args(
        expert_model_parallel_size=1,
        ffn_hidden_size=4096,
        hidden_size=12288,
        num_experts=8,
        micro_batch=1,
        moe_router_topk=4,
        n_shared_expert=2,
        seq_length=5120,
        enable_sequence_parallel=True,
        kv_lora_rank=512,
        num_attention_heads= 128,
        q_lora_rank=1536,
        qk_nope_dim=128,
        qk_rope_dim=64,
        tensor_model_parallel_size=1,
        v_head_dim=128
    )
    model = DeepSeekMLA(args)
    y = model(x)
    return y

class DeepSeekExpert(torch.nn.Module):
    """DeepSeekExpert"""

    def __init__(self, args: Args):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.ffn_hidden_size = args.ffn_hidden_size
        self.dev = torch.cuda.current_device()
        self.w1 = torch.randn(
            args.ffn_hidden_size,
            args.hidden_size,
            device=self.dev,
        ).to(torch.bfloat16)
        self.w1_fp8 = per_block_cast_to_fp8(self.w1)

        self.w2 = torch.randn(
            args.hidden_size,
            args.ffn_hidden_size,
            device=self.dev,
        ).to(torch.bfloat16)

        self.w2_fp8 = per_block_cast_to_fp8(self.w2)

        self.w3 = torch.randn(
            args.ffn_hidden_size,
            args.hidden_size,
            device=self.dev,
        ).to(torch.bfloat16)

        self.w3_fp8 = per_block_cast_to_fp8(self.w3)

        self.o0 = torch.empty(
            (args.seq_length, self.ffn_hidden_size),
            device=self.dev,
            dtype=torch.bfloat16,
        )
        self.o1 = torch.empty(
            (args.seq_length, self.ffn_hidden_size),
            device=self.dev,
            dtype=torch.bfloat16,
        )
        self.out = torch.empty(
            (args.seq_length, self.hidden_size),
            device=self.dev,
            dtype=torch.bfloat16,
        )

    @cuda_timing_decorator
    def _apply_linear1(self, x: torch.Tensor, x_scale: torch.Tensor):
        deep_gemm.gemm_fp8_fp8_bf16_nt((x, x_scale), self.w1_fp8, self.o0)
        deep_gemm.gemm_fp8_fp8_bf16_nt((x, x_scale), self.w3_fp8, self.o1)
        return (self.o0, self.o1)

    @cuda_timing_decorator
    def _apply_linear2(self, x: torch.Tensor):
        self.out = torch.matmul(x, self.w2.T)
        # or one can scale it down to fp8 and apply deepgemm
        # x_fp8 = per_token_cast_to_fp8(x)
        # deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, self.w2_fp8, self.out)
        return self.out

    @cuda_timing_decorator
    def _apply_activation(self, x: torch.Tensor):
        return F.silu(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, float, float, float]:
        # based on https://github.com/deepseek-ai/DeepSeek-V3/blob/f6e34dd26772dd4a216be94a8899276c5dca9e43/inference/model.py#L630
        # i.e. out = self.w2(F.silu(self.w1(x)) * self.w3(x))
        x, x_scale = per_token_cast_to_fp8(x.view(-1, self.hidden_size))
        l1_out, l1_time = self._apply_linear1(x, x_scale)
        act_out, act_time = self._apply_activation(l1_out[0])
        l2_out, l2_time = self._apply_linear2(act_out * l1_out[0])
        return l2_out, l1_time, act_time, l2_time


class DeepSeekMoE(torch.nn.Module):
    """DeepSeekMoE"""

    def __init__(self, args: Args):
        super().__init__()
        self.n_local_experts = (
            args.n_shared_expert + args.num_experts // args.expert_model_parallel_size
        )
        self.hidden_size = args.hidden_size
        self.dev = torch.cuda.current_device()
        in_dim = self.hidden_size * args.micro_batch
        self.in_dim = in_dim
        self.inter_dim = args.ffn_hidden_size * self.n_local_experts
        # yoinked from AiobMegatron.py->MoELayer
        ep = args.expert_model_parallel_size
        num_experts = args.num_experts
        micro_batch = args.micro_batch
        seq_len = args.seq_length
        topk = args.moe_router_topk
        self.topk = topk
        hidden_size = args.hidden_size
        self.dispatched_input = torch.rand(
            int(seq_len * micro_batch * topk * ep /
                num_experts * self.n_local_experts),
            hidden_size,
            device=self.dev,
        ).to(torch.bfloat16)
        self.dispatched_input_fp8 = per_token_cast_to_fp8(
            self.dispatched_input)

        self.tokens_per_expert = torch.full(
            (self.n_local_experts,),
            int(seq_len * micro_batch * topk * ep / num_experts),
        )

        self.local_experts = torch.nn.ModuleList()
        for _ in range(self.n_local_experts):
            self.local_experts.append(DeepSeekExpert(args))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        mlp_l1_all, mlp_act_all, mlp_l2_all = 0, 0, 0
        o = torch.zeros(
            self.n_local_experts,
            x.shape[0],
            self.hidden_size,
            device=self.dev,
        ).to(torch.bfloat16)
        for i, expert in enumerate(self.local_experts):
            out, mlp_l1, mlp_act, mlp_l2 = expert(x.view(-1, self.in_dim))
            mlp_l1_all += mlp_l1
            mlp_act_all += mlp_act
            mlp_l2_all += mlp_l2
            o[i] = out
        return o, mlp_l1_all, mlp_act_all, mlp_l2_all


# some util tests to measure timings
def test_deepseek_expert() -> torch.Tensor:
    x = torch.randn(2048, 1, 1024, dtype=torch.bfloat16)
    args = Args(
        expert_model_parallel_size=1,
        ffn_hidden_size=4096,
        hidden_size=1024,
        num_experts=8,
        micro_batch=1,
        moe_router_topk=4,
        n_shared_expert=2,
        seq_length=2048,
        enable_sequence_parallel=True,
        kv_lora_rank=512,
        num_attention_heads= 128 + 64,
        q_lora_rank=0,
        qk_nope_dim=128,
        qk_rope_dim=64,
        tensor_model_parallel_size=1,
        v_head_dim=128
    )
    model = DeepSeekExpert(args)
    y = model(x)
    print(y)


def test_deepseek_mlp():
    x = torch.randn(2048, 1, 1024, dtype=torch.bfloat16)
    args = Args(
        expert_model_parallel_size=1,
        ffn_hidden_size=4096,
        hidden_size=1024,
        num_experts=8,
        micro_batch=1,
        moe_router_topk=4,
        n_shared_expert=2,
        seq_length=2048,
        enable_sequence_parallel=True,
        kv_lora_rank=512,
        num_attention_heads= 128 + 64,
        q_lora_rank=16,
        qk_nope_dim=128,
        qk_rope_dim=64,
        tensor_model_parallel_size=1,
        v_head_dim=128
    )
    model = DeepSeekMoE(args)
    y = model(x)
    return y


def test_expert(
    m: int = 2048 * 32,
    n: int = 1024,
    k: int = 2048 * 32,
) -> tuple[int, int]:
    """Test DeepSeek's expert MLP.

    i.e. self.w2(F.silu(self.w1(x)) * self.w3(x))
    with torch.matmul (BF16) and deep_gemm (FP8).
    """
    x = torch.randn(m, k).to(torch.bfloat16)
    w = torch.randn(k, n).to(torch.bfloat16)
    w = w.t()
    w2 = torch.randn(k, n).to(torch.bfloat16)
    w2 = w2.t()
    o = torch.randn(m, n).to(torch.bfloat16)
    o2 = torch.randn(m, n).to(torch.bfloat16)

    w3 = torch.randn(n, k).to(torch.bfloat16)
    w3_fp8 = per_block_cast_to_fp8(w3.t())

    out = torch.randn(m, k).to(torch.bfloat16)
    w_fp8 = per_block_cast_to_fp8(w)
    w2_fp8 = per_block_cast_to_fp8(w2)
    x_fp8 = per_token_cast_to_fp8(x)

    _, t0 = do_deepgemm(x_fp8, w_fp8, w2_fp8, w3_fp8, o, o2, out)
    _, t1 = do_matmul(x, w, w3, o, out)
    return t0, t1


@cuda_timing_decorator
def do_deepgemm(
    x: tuple[torch.Tensor, torch.Tensor],
    w: tuple[torch.Tensor, torch.Tensor],
    w2: tuple[torch.Tensor, torch.Tensor],
    w3: torch.Tensor,
    o: torch.Tensor,
    o2: torch.Tensor,
    out: torch.Tensor,
) -> tuple[int, int]:
    deep_gemm.gemm_fp8_fp8_bf16_nt(x, w, o)
    deep_gemm.gemm_fp8_fp8_bf16_nt(x, w2, o2)
    o2 = F.silu(o) + o2
    o2_fp8 = per_token_cast_to_fp8(o2)
    deep_gemm.gemm_fp8_fp8_bf16_nt(o2_fp8, w3, out)
    return out


@cuda_timing_decorator
def do_matmul(
    x: torch.Tensor,
    w: torch.Tensor,
    w3: torch.Tensor,
    o: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    torch.matmul(x, w.t(), out=o)
    return torch.matmul(F.silu(o), w3, out=out)

device = torch.device("cuda:0")
torch.set_default_device(device)

if __name__ == "__main__":
    # torch.set_default_dtype(torch.bfloat16)
    device = torch.device("cuda:0")
    torch.set_default_device(device)
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    random.seed(0)

    # run once to load/cache kernels
    test_deepseek_mlp()
    test_expert()
    test_deepseek_mla()

    # for kineto traces
    import torch.profiler

    with torch.profiler.profile(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
        record_shapes=True,
        with_flops=True,
    ) as prof:
        test_deepseek_mlp()
        t0, t1 = test_expert()
    print(f"matmul: deepgemm: {t0} matmul: {t1} diff: {t1 - t0}")

    with torch.profiler.profile(
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log"),
        record_shapes=True,
        with_flops=True,
    ) as prof:
        _, t = test_deepseek_mla()
    print(f"test_deepseek_mla: compute time {t}")
