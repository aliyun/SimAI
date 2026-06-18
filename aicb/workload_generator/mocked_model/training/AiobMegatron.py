"""
Copyright (c) 2021, Alibaba Group;
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import time
import warnings
import torch.nn.functional as F
from apex.contrib.layer_norm.layer_norm import FastLayerNormFN
import math
import scaled_upper_triang_masked_softmax_cuda
from torch.cuda.amp import custom_bwd, custom_fwd
from utils.utils import *
from core import grouped_gemm_util as gg
try:
    from einops import rearrange
except ImportError as e:
    rearrange = None
    print("Failed to import 'einops'. Functions using 'rearrange' might not work.")
from typing import Callable, Optional

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_func
except ImportError:
    try:
        from flash_attn.flash_attn_interface import (
            flash_attn_varlen_func as flash_attn_unpadded_func,
        )
    except ImportError:
        flash_attn_unpadded_func = None

from workload_generator.mocked_model.training.AiobDeepSeek import DeepSeekMoE, DeepSeekMLA

class MegatronModel(torch.nn.Module):
    def __init__(self, args=None):
        super(MegatronModel, self).__init__()
        self.time_list = {}
        self.args = args

        self.Embedding = MegatronEmbedding(self.args)
        self.Layernorm = MegatronLayernorm(self.args)
        if args.frame == "DeepSeek":
            self.Attention = DeepSeekMLA(self.args)
        if args.frame == "Megatron":
            if self.args.use_flash_attn:
                self.Attention = MegatronFlashAtten(self.args)
            else:
                self.Attention = MegatronAtten(self.args)
        if self.args.moe_enable:
            if self.args.frame == "DeepSeek":
                self.Mlp = DeepSeekMoE(self.args)
            else:
                self.Mlp = MoELayer(self.args)
        else:
            self.Mlp = MegatronMlp(self.args)
        self.logit = logit(self.args)
        self.grad_param = Grad_param(self.args)

    def forward(self, input):
        for _ in range(self.args.epoch_num):
            # #Embedding
            Emb_output, Emb_time = self.Embedding(input)
            self.time_list.setdefault("Emb", []).append({"time_gpu": Emb_time})

            for _ in range(self.args.num_layers):
                # #layernorm
                lay_out, layernorm = self.Layernorm(Emb_output)
                self.time_list.setdefault("layernorm", []).append(
                    {"time_gpu": layernorm}
                )
                if self.args.frame == "DeepSeek":
                    atten_output, time_map = self.Attention(lay_out)
                    for k, v in time_map.items():
                        self.time_list.setdefault(k, []).append(
                            {"time_gpu": v}
                        )
                if self.args.frame == "Megatron":
                    if self.args.use_flash_attn:
                        atten_output, atten_qkv, atten_core, atten_linear = self.Attention(
                            lay_out
                        )
                        self.time_list.setdefault("atten_qkv", []).append(
                            {"time_gpu": atten_qkv}
                        )
                        self.time_list.setdefault("atten_flash", []).append(
                            {"time_gpu": atten_core}
                        )
                        self.time_list.setdefault("atten_linear", []).append(
                            {"time_gpu": atten_linear}
                        )
                    else:
                        (
                            atten_output,
                            atten_qkv,
                            atten_core_qk,
                            atten_core_softmax,
                            atten_core_contex,
                            atten_linear,
                        ) = self.Attention(lay_out)
                        self.time_list.setdefault("atten_qkv", []).append(
                            {"time_gpu": atten_qkv}
                        )
                        self.time_list.setdefault("atten_core_qk", []).append(
                            {"time_gpu": atten_core_qk}
                        )
                        self.time_list.setdefault("atten_core_softmax", []).append(
                            {"time_gpu": atten_core_softmax}
                        )
                        self.time_list.setdefault("atten_core_contex", []).append(
                            {"time_gpu": atten_core_contex}
                        )
                        self.time_list.setdefault("atten_linear", []).append(
                            {"time_gpu": atten_linear}
                        )
                # layernorm
                lay2_out, layernorm2 = self.Layernorm(atten_output)

                # mlp layer
                mlp_out, mlp_linear_1, mlp_gelu, mlp_linear_2 = self.Mlp(lay2_out)
                self.time_list.setdefault("layernorm2", []).append(
                    {"time_gpu": layernorm2}
                )
                self.time_list.setdefault("mlp_linear_1", []).append(
                    {"time_gpu": mlp_linear_1}
                )
                self.time_list.setdefault("mlp_gelu", []).append({"time_gpu": mlp_gelu})
                self.time_list.setdefault("mlp_linear_2", []).append(
                    {"time_gpu": mlp_linear_2}
                )

            lay_post__out, layernorm_post = self.Layernorm(mlp_out)
            self.time_list.setdefault("layernorm_post", []).append(
                {"time_gpu": layernorm_post}
            )
            print(f"lay_post__out.shape: {lay_post__out.shape}")
            logit_out, logit_time = self.logit(lay_post__out)
            self.time_list.setdefault("logit_time", []).append({"time_gpu": logit_time})
            _, param_time = self.grad_param._apply()

            self.time_list.setdefault("param_time", []).append({"time_gpu": param_time})
        
        filepath = write_op(self.time_list, self.args)
        process_all_keys(filepath)
        
        return filepath


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


# Embedding
class MegatronEmbedding(torch.nn.Module):
    def __init__(self, args=None):
        super(MegatronEmbedding, self).__init__()
        self.tp = args.tensor_model_parallel_size
        micro_batch = args.micro_batch
        seq_len = args.seq_length
        hidden_size = args.hidden_size
        max_position_embeddings = args.max_position_embeddings
        self.vocab_size = args.padded_vocab_size
        device = torch.cuda.current_device()
        if args.dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif args.dtype == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        # self.masked_input = torch.randint(0,math.ceil(self.vocab_size/self.tp),
        #                                   (micro_batch,seq_len),
        #                                   device=device, dtype=torch.int64)
        self.weight = torch.randint(
            0,
            1,
            (math.ceil(self.vocab_size / self.tp), hidden_size),
            device=device,
            dtype=torch.int64,
        )
        self.position_embeddings = torch.nn.Embedding(
            max_position_embeddings, hidden_size
        ).to(device)
        self.position_ids = torch.randint(
            0, seq_len - 1, (1, seq_len), device=device, dtype=torch.int64
        )

    @cuda_timing_decorator
    def _apply(self, input):
        if self.tp > 1:
            # Build the mask.
            input_mask = (input < 0) | (input >= math.ceil(self.vocab_size / self.tp))
            # Mask the input.
            masked_input = input.clone() - 0
            masked_input[input_mask] = 0

        else:
            masked_input = input
        words_embeddings = self.weight[masked_input]

        position_embeddings_i = self.position_embeddings(self.position_ids)

        embeddings = words_embeddings + position_embeddings_i
        embeddings = embeddings.transpose(0, 1).contiguous()

        return embeddings

    def forward(self, input):

        result, emb_time = self._apply(input)

        result = result.to(self.dtype)

        return result, emb_time


class MegatronLayernorm(torch.nn.Module):
    def __init__(self, args=None):
        super(MegatronLayernorm, self).__init__()
        self.tp = args.tensor_model_parallel_size
        self.enable_sequence_parallel = args.enable_sequence_parallel
        self.hidden_size = args.hidden_size
        device = torch.cuda.current_device()
        if args.dtype == "bfloat16":
            self.dtype = torch.bfloat16
        elif args.dtype == "float16":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        # self.input_l = torch.rand(seq_len,
        #                           micro_batch,
        #                           hidden_size,
        #                           device=device).to(dtype)
        self.lay_weight = torch.rand(self.hidden_size, device=device).to(self.dtype)
        self.bias = torch.zeros(self.hidden_size, device=device).to(self.dtype)

    @cuda_timing_decorator
    def _apply_fused_layer_norm(self, hidden_states):
        output_lay = FastLayerNormFN.apply(
            hidden_states, self.lay_weight, self.bias, 1e-05
        )
        return output_lay

    @cuda_timing_decorator
    def _apply_torch_layernorm(self, hidden_states):
        output_lay = torch.nn.functional.layer_norm(
            hidden_states, [self.hidden_size] , self.lay_weight, self.bias, 1e-05
        )
        return output_lay

    def forward(self, hidden_states):
        # hidden_states = hidden_states.to(self.dtype)
        if self.enable_sequence_parallel:
            chunks = torch.chunk(hidden_states, self.tp, 0)
            hidden_states = chunks[0]

        # in case of DeepSeek16B, for Hidden size 10944, FastLayerNormFN fails with
        # FWD: Unsupported hidden_size or types: 10944BFloat16BFloat16BFloat16Float
        # because https://github.com/NVIDIA/apex/blob/4bdecd06b3c4b2c0a8fb6603829a8f9f05a42b49/apex/contrib/csrc/layer_norm/ln_fwd_cuda_kernel.cu#L73-L227
        # thus, use torch's layer_norm

        # try Fused, if exception, use torch
        try:
            lay_out, lay_time = self._apply_fused_layer_norm(hidden_states)
        except Exception as e:
            print(f"FastLayerNormFN failed with error {e}, using torch.nn.functional.layer_norm")
            lay_out, lay_time = self._apply_torch_layernorm(hidden_states)


        if self.enable_sequence_parallel:
            lay_out = lay_out.repeat((self.tp, 1, 1))

        return lay_out, lay_time


class MegatronAtten(torch.nn.Module):
    def __init__(self, args=None):
        super(MegatronAtten, self).__init__()
        self.enable_sequence_parallel = args.enable_sequence_parallel
        self.tp = args.tensor_model_parallel_size
        micro_batch = args.micro_batch
        seq_len = args.seq_length
        hidden_size = args.hidden_size
        num_attention_heads = args.num_attention_heads
        self.input_in_float16 = False
        if args.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.dtype == "float16":
            dtype = torch.float16
            self.input_in_float16 = True
        else:
            dtype = torch.float32
        device = torch.cuda.current_device()
        # self.atten_total_input_1 = torch.rand(seq_len,
        #                                       micro_batch,
        #                                       hidden_size,
        #                                       device=device).to(dtype)
        self.atten_weight_1 = torch.rand(
            divide((3 * hidden_size), self.tp), hidden_size, device=device
        ).to(dtype)
        self.hidden_size_per_partition = divide(hidden_size, self.tp)
        self.num_attention_heads_per_partition = divide(num_attention_heads, self.tp)
        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        self.num_query_groups_per_partition = self.num_attention_heads_per_partition
        query_layer = torch.rand(
            seq_len,
            micro_batch,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            device=device,
        ).to(dtype)
        key_layer = query_layer
        value_layer = query_layer
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )
        self.query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        self.key_layer = key_layer.view(
            output_size[3], output_size[0] * output_size[1], -1
        )
        self.matmul_input_buffer = torch.zeros(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            device=device,
        ).to(dtype)
        self.scale_t = torch.tensor(1).to(dtype)
        soft_input = torch.rand(output_size, device=device).to(dtype)
        self.b, self.np, self.sq, self.sk = soft_input.size()
        self.soft_input_1 = soft_input.view(-1, self.sq, self.sk)
        self.output_size_2 = (
            value_layer.size(1),
            value_layer.size(2),
            self.query_layer.size(0),
            value_layer.size(3),
        )
        self.value_layer = value_layer.view(
            value_layer.size(0), self.output_size_2[0] * self.output_size_2[1], -1
        )
        self.atten_linear_weight = torch.rand(
            hidden_size, self.hidden_size_per_partition, device=device
        ).to(dtype)
        # self.linear_function = LinearWithGradAccumulationAndAsyncCommunication.apply
    def get_batch_per_block(self, sq, sk, b, np):
        import scaled_masked_softmax_cuda

        return scaled_masked_softmax_cuda.get_batch_per_block(sq, sk, b, np)
    def is_kernel_available(self, b, np, sq, sk):
        attn_batches = b * np

        if (
            self.input_in_float16  # input must be fp16
            and 16 < sk <= 16384  # sk must be 16 ~ 16384
            and sq % 4 == 0  # sq must be divisor of 4
            and sk % 4 == 0  # sk must be divisor of 4
            and attn_batches % 4 == 0  # np * b must be divisor of 4
        ):
            if 0 <= sk <= 16384:
                batch_per_block = self.get_batch_per_block(sq, sk, b, np)

                if attn_batches % batch_per_block == 0:
                    return True

        return False
    @cuda_timing_decorator
    def _apply_attenqkv(self, hideen_states):
        _forward_impl = linear_with_grad_accumulation_and_async_allreduce
        output = _forward_impl(
            input=hideen_states,
            weight=self.atten_weight_1,
            bias=None,
            gradient_accumulation_fusion=True,
            async_grad_allreduce=False,
            sequence_parallel=self.enable_sequence_parallel,
            tp=self.tp,
        )
        # linear_function = LinearWithGradAccumulationAndAsyncCommunication.apply
        # output = linear_function(self.atten_total_input_1,self.atten_weight_1,None,None,False,False)
        # # output = torch.matmul(self.atten_total_input_1, self.atten_weight_1)
        new_tensor_shape = output.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    + 2
                )
                * self.hidden_size_per_attention_head
            ),
        )
        output = output.view(*new_tensor_shape)
        (query_layer, key_layer, value_layer) = torch.split(
            output,
            [
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    * self.hidden_size_per_attention_head
                ),
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ],
            dim=3,
        )
        return query_layer, key_layer, value_layer

    @cuda_timing_decorator
    def _apply_QK(self, q, k):
        matmul_result = torch.baddbmm(
            self.matmul_input_buffer,
            q.transpose(0, 1),  # [b * np, sq, hn]
            k.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / 11.313708498984761),
        )
        return matmul_result

    @cuda_timing_decorator
    def _apply_Softmax(self,attention_scores):
        if self.is_kernel_available(*attention_scores.size()):
            b, np, sq, sk = attention_scores.size()
            attention_scores = attention_scores.view(-1, sq, sk)
            softmax_results = scaled_upper_triang_masked_softmax_cuda.forward(
                    attention_scores, self.scale_t
                )
            prob = softmax_results.view(self.b, self.np, self.sq, self.sk)
        else:
            if self.scale_t is not None:
                attention_scores = attention_scores * self.scale_t
            prob = torch.nn.Softmax(dim=-1)(attention_scores)
        
        return prob

    @cuda_timing_decorator
    def _apply_Contex(self, prob, value_layer):
        value_layer = value_layer.view(
            value_layer.size(0), self.output_size_2[0] * self.output_size_2[1], -1
        )
        attention_probs = prob.view(
            self.output_size_2[0] * self.output_size_2[1], self.output_size_2[2], -1
        )
        output = torch.bmm(attention_probs, value_layer.transpose(0, 1))
        context_layer = (
            output.view(*self.output_size_2).permute(2, 0, 1, 3).contiguous()
        )
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

    @cuda_timing_decorator
    def _apply_Linear(self, context_layer):
        _forward_impl = linear_with_grad_accumulation_and_async_allreduce
        output_parallel = _forward_impl(
            input=context_layer,
            weight=self.atten_linear_weight,
            bias=None,
            gradient_accumulation_fusion=True,
            async_grad_allreduce=False,
            sequence_parallel=self.enable_sequence_parallel,
            tp=self.tp,
        )
        return output_parallel

    def forward(self, hideen_states):
        qkv_out, qkv_time = self._apply_attenqkv(hideen_states)

        query_layer, key_layer, value_layer = qkv_out
        
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )
        query_layer = query_layer.reshape(
            output_size[2], output_size[0] * output_size[1], -1
        )
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)
        matmul_result, qk_time = self._apply_QK(query_layer, key_layer)
        attention_scores = matmul_result.view(*output_size)
        softmax_results, softmax_time = self._apply_Softmax(attention_scores)
        context_layer, contex_time = self._apply_Contex(softmax_results, value_layer)
        output, attrn_linear_time = self._apply_Linear(context_layer)
        return output, qkv_time, qk_time, softmax_time, contex_time, attrn_linear_time


class MegatronFlashAtten(torch.nn.Module):
    def __init__(self, args=None):
        super(MegatronFlashAtten, self).__init__()
        self.enable_sequence_parallel = args.enable_sequence_parallel
        self.tp = args.tensor_model_parallel_size
        micro_batch = args.micro_batch
        seq_len = args.seq_length

        hidden_size = args.hidden_size
        num_attention_heads = args.num_attention_heads
        if args.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        device = torch.cuda.current_device()

        self.atten_weight_1 = torch.rand(
            divide((3 * hidden_size), self.tp), hidden_size, device=device
        ).to(dtype)

        self.hidden_size_per_partition = divide(hidden_size, self.tp)
        self.num_attention_heads_per_partition = divide(num_attention_heads, self.tp)
        self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        self.num_query_groups_per_partition = self.num_attention_heads_per_partition

        self.atten_linear_weight = torch.rand(
            hidden_size, self.hidden_size_per_partition, device=device
        ).to(dtype)

    @cuda_timing_decorator
    def _apply_attenqkv(self, hideen_states):
        _forward_impl = linear_with_grad_accumulation_and_async_allreduce

        output = _forward_impl(
            input=hideen_states,
            weight=self.atten_weight_1,
            bias=None,
            gradient_accumulation_fusion=True,
            async_grad_allreduce=False,
            sequence_parallel=self.enable_sequence_parallel,
            tp=self.tp,
        )

        new_tensor_shape = output.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    + 2
                )
                * self.hidden_size_per_attention_head
            ),
        )
        output = output.view(*new_tensor_shape)
        (query_layer, key_layer, value_layer) = torch.split(
            output,
            [
                (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    * self.hidden_size_per_attention_head
                ),
                self.hidden_size_per_attention_head,
                self.hidden_size_per_attention_head,
            ],
            dim=3,
        )

        return query_layer, key_layer, value_layer

    @cuda_timing_decorator
    def _apply_flash_atten(self, q, k, v):

        output = flash_attn_unpadded_func(
            q,
            k,
            v,
            self.cu_seqlens_q,
            self.cu_seqlens_k,
            self.seqlen_q,
            self.seqlen_k,
            0.0,
            softmax_scale=None,
            causal=True,
        )

        context_layer = rearrange(output, "(b s) ... -> b s ...", b=self.micro_batch)

        return context_layer

    @cuda_timing_decorator
    def _apply_Linear(self, context_layer):
        _forward_impl = linear_with_grad_accumulation_and_async_allreduce
        context_layer = rearrange(context_layer, "b s h d -> s b (h d)").contiguous()
        output_parallel = _forward_impl(
            input=context_layer,
            weight=self.atten_linear_weight,
            bias=None,
            gradient_accumulation_fusion=True,
            async_grad_allreduce=False,
            sequence_parallel=self.enable_sequence_parallel,
            tp=self.tp,
        )
        return output_parallel

    def forward(self, hidden_state):
        if rearrange is None:
            raise ImportError(
                "The function 'rearrange' from 'einops' is required but not available."
            )
        result, qkv_time = self._apply_attenqkv(hidden_state)
        q, k, v = result
        q, k, v = [rearrange(x, "s b ... -> b s ...").contiguous() for x in (q, k, v)]
        self.micro_batch, self.seqlen_q = q.shape[0], q.shape[1]
        self.seqlen_k = k.shape[1]

        q, k, v = [rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v]]
        self.cu_seqlens_q = torch.arange(
            0,
            (self.micro_batch + 1) * self.seqlen_q,
            step=self.seqlen_q,
            dtype=torch.int32,
            device=q.device,
        )

        if self.training:
            assert self.seqlen_k == self.seqlen_q
            self.cu_seqlens_k = self.cu_seqlens_q
        context_layer, flash_time = self._apply_flash_atten(q, k, v)
        output, attrn_linear_time = self._apply_Linear(context_layer)
        return output, qkv_time, flash_time, attrn_linear_time


class MegatronMlp(torch.nn.Module):
    def __init__(self, args=None):
        super(MegatronMlp, self).__init__()
        self.tp = args.tensor_model_parallel_size
        self.enable_sequence_parallel = args.enable_sequence_parallel
        micro_batch = args.micro_batch
        seq_len = args.seq_length
        self.add_bias_linear = False
        if args.add_bias_linear:
            self.add_bias_linear = True

        hidden_size = args.hidden_size
        ffn_hidden_size = args.ffn_hidden_size
        if args.gated_linear_unit:
            ffn_hidden_size *= 2
        num_attention_heads = args.num_attention_heads
        device = torch.cuda.current_device()
        if args.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        # activation
        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu
        elif args.swiglu:

            def swiglu(x):
                x = torch.chunk(x, 2, dim=-1)

                return F.silu(x[0]) * x[1]

            self.activation_func = swiglu
        elif args.squared_relu:

            def squared_relu(x):
                return torch.pow(F.relu(x), 2)

            self.activation_func = squared_relu
        else:
            self.bias_gelu_fusion = args.bias_gelu_fusion
            self.activation_func = F.gelu
        output_size_per_partition = divide(ffn_hidden_size, self.tp)

        self.weight_1 = torch.rand(
            output_size_per_partition, hidden_size, device=device
        ).to(dtype)
        self.bias = torch.empty(output_size_per_partition, device=device).to(dtype)

        self.weight_2 = torch.rand(
            hidden_size, args.ffn_hidden_size // self.tp, device=device
        ).to(dtype)

    @cuda_timing_decorator
    def _apply_Linear1(self, hidden_state):
        _forward_impl = linear_with_grad_accumulation_and_async_allreduce
        output_parallel = _forward_impl(
            input=hidden_state,
            weight=self.weight_1,
            bias=None,
            gradient_accumulation_fusion=True,
            async_grad_allreduce=False,
            sequence_parallel=self.enable_sequence_parallel,
            tp=self.tp,
        )
        return output_parallel

    @cuda_timing_decorator
    def _apply_activation(self, hidden_state):
        if self.add_bias_linear:
            intermediate_parallel = self.activation_func(hidden_state + self.bias)
        else:
            intermediate_parallel = self.activation_func(hidden_state)

        return intermediate_parallel

    @cuda_timing_decorator
    def _apply_Linear2(self, hidden_state):
        _forward_impl = linear_with_grad_accumulation_and_async_allreduce
        output_parallel = _forward_impl(
            input=hidden_state,
            weight=self.weight_2,
            bias=None,
            gradient_accumulation_fusion=True,
            async_grad_allreduce=False,
            sequence_parallel=self.enable_sequence_parallel,
            tp=self.tp,
        )
        return output_parallel

    def forward(self, hidden_state):
        l1_out, l1_time = self._apply_Linear1(hidden_state)
        act_out, act_time = self._apply_activation(l1_out)
        l2_out, l2_time = self._apply_Linear2(act_out)
        return l2_out, l1_time, act_time, l2_time


class logit(torch.nn.Module):
    def __init__(self, args=None):
        super(logit, self).__init__()
        self.enable_sequence_parallel = args.enable_sequence_parallel
        self.tp = args.tensor_model_parallel_size
        micro_batch = args.micro_batch
        seq_len = args.seq_length
        vocab_size = args.padded_vocab_size
        hidden_size = args.hidden_size
        ffn_hidden_size = args.ffn_hidden_size
        if args.gated_linear_unit:
            ffn_hidden_size *= 2
        num_attention_heads = args.num_attention_heads
        device = torch.cuda.current_device()
        if args.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        output_size_per_partition = divide(vocab_size, self.tp)
        self.word_embeddings_weight = torch.rand(
            output_size_per_partition, hidden_size, device=device, requires_grad=True
        ).to(dtype)

    @cuda_timing_decorator
    def _apply(self, hidden_state):
        _forward_impl = linear_with_grad_accumulation_and_async_allreduce
        output_parallel = _forward_impl(
            input=hidden_state,
            weight=self.word_embeddings_weight,
            bias=None,
            gradient_accumulation_fusion=True,
            async_grad_allreduce=True,
            sequence_parallel=self.enable_sequence_parallel,
            tp=self.tp,
        )
        return output_parallel

    def forward(self, input):
        log_out, log_time = self._apply(input)
        return log_out, log_time


class Grad_param:
    def __init__(self, args=None):
        tp = args.tensor_model_parallel_size
        param = args.model_param
        self.dp = args.dp_num

        device = torch.cuda.current_device()
        dtype = torch.float32
        self.data = torch.rand(param//tp, device=device).to(dtype)

    @cuda_timing_decorator
    def _apply(self):

        self.data /= self.dp
        # assert self.data.numel() % self.dp == 0
        shard_size = self.data.numel() // self.dp
        sharded_buffer = [
            self.data[(r * shard_size) : ((r + 1) * shard_size)] for r in range(self.dp)
        ]
        return sharded_buffer
class SequentialMLP(torch.nn.Module):
    """An implementation of the Experts layer using a sequence of MLP layers.

    This class executes each expert sequentially.
    """

    def __init__(self, num_local_experts,args=None):
        super(SequentialMLP,self).__init__()
        tp = args.tensor_model_parallel_size
        ep = args.expert_model_parallel_size
        num_experts = args.num_experts
        micro_batch = args.micro_batch
        seq_len = args.seq_length
        topk = args.moe_router_topk
        hidden_size = args.hidden_size
        num_attention_heads = args.num_attention_heads
        self.add_bias = False
        # self.moe_extended_tp = config.moe_extended_tp
        self.num_local_experts = num_local_experts
        self.local_experts = torch.nn.ModuleList()
        for _ in range(self.num_local_experts):
            expert = MegatronMlp(args)
            self.local_experts.append(expert)

    def forward(self, permuted_local_hidden_states, tokens_per_expert):

        output_local = torch.zeros_like(permuted_local_hidden_states)
        output_bias_local = None
        if self.add_bias:
            output_bias_local = torch.zeros_like(permuted_local_hidden_states)

        cumsum_num_tokens = torch.cumsum(tokens_per_expert, dim=0)
        # Insert zero at the begining for offset index's convenience
        zero_tensor = torch.zeros(1, dtype=torch.long, device=cumsum_num_tokens.device)
        cumsum_num_tokens = torch.cat((zero_tensor, cumsum_num_tokens))
        mlp_linear_1_all , mlp_gelu_all ,mlp_linear_2_all = 0 ,0 ,0
        for expert_num, expert in enumerate(self.local_experts):
            start = cumsum_num_tokens[expert_num]
            end = cumsum_num_tokens[expert_num + 1]
            
            hidden = permuted_local_hidden_states[start:end]
            # output, output_bias = expert(hidden)
            output,mlp_linear_1,mlp_gelu,mlp_linear_2 = expert(hidden)

            output_local[start:end] = output
            mlp_linear_1_all += mlp_linear_1
            mlp_gelu_all += mlp_gelu
            mlp_linear_2_all += mlp_linear_2
            if self.add_bias:
                output_bias = output_bias.expand_as(output)
                output_bias_local[start:end, :] = output_bias

        return output_local, mlp_linear_1_all,mlp_gelu_all,mlp_linear_2_all
class GroupedMLP(torch.nn.Module):
    """An efficient implementation of the Experts layer using CUTLASS GroupedGEMM.

    This class is designed to execute multiple experts in parallel, thereby maximizing computational efficiency.
    """

    def __init__(self, num_local_experts,args=None):
        super(GroupedMLP,self).__init__()
        self.num_local_experts = num_local_experts
        gg.assert_grouped_gemm_is_available()
        tp = args.tensor_model_parallel_size
        self.hidden_size = args.hidden_size
        self.expert_parallel = args.expert_model_parallel_size > 1
        device = torch.cuda.current_device()
        if args.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu
        elif args.swiglu:
            def swiglu(x):
                x = torch.chunk(x, 2, dim=-1)
                
                return F.silu(x[0]) * x[1]
            self.activation_func = swiglu
        elif args.squared_relu:
            def squared_relu(x):
                return torch.pow(F.relu(x), 2)
            self.activation_func = squared_relu
        else:
            self.bias_gelu_fusion = args.bias_gelu_fusion
            self.activation_func = F.gelu
        # if args.gated_linear_unit:
        #     def glu(x):
        #         x = torch.chunk(x, 2, dim=-1)
        #         return self.config.activation_func(x[0]) * x[1]

        #     self.activation_func = glu
        # else:
        #     self.activation_func = self.activation_func

        # How many feature each rank holds for fc1 and fc2, respectively.
        # self.moe_extended_tp = config.moe_extended_tp
        # if config.moe_extended_tp:
        #     tp_size = parallel_state.get_tensor_and_expert_parallel_world_size()
        # else:
        #     tp_size = parallel_state.get_tensor_model_parallel_world_size()

        fc1_output_size = args.ffn_hidden_size * self.num_local_experts
        if args.gated_linear_unit:
            # Project to 4h. If using swiglu double the output width,
            # see https://arxiv.org/pdf/2002.05202.pdf
            fc1_output_size *= 2
        fc1_output_size_per_partition = divide(fc1_output_size, tp)

        fc2_input_size = args.ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp)

        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # (https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
        # and as a result we avoid allocate the transpose of weights.
        # Initialize weight.
        self.weight1 = torch.rand(self.hidden_size ,
                                   fc1_output_size_per_partition,
                                   device=device).to(dtype)
        self.weight2 = torch.rand(fc2_input_size_per_partition, 
                                   self.hidden_size ,
                                   device=device).to(dtype)
    @cuda_timing_decorator 
    def _apply_Linear1(self,permuted_local_hidden_states,tokens_per_expert,w1):
        
        

        fc1_output = gg.ops.gmm(
            permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False
        )
        return fc1_output
    
    @cuda_timing_decorator
    def _apply_activation(self,fc1_output):

        intermediate_parallel = self.activation_func(fc1_output)

        return intermediate_parallel
    
    @cuda_timing_decorator
    def _apply_Linear2(self,intermediate_parallel,tokens_per_expert,w2):

        
        fc2_output = gg.ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)

        return fc2_output
    def forward(self, permuted_local_hidden_states, tokens_per_expert):
        w1 = self.weight1.view(self.num_local_experts, self.hidden_size, -1)
        w2 = self.weight2.view(self.num_local_experts, -1, self.hidden_size )
        l1_out,l1_time = self._apply_Linear1(permuted_local_hidden_states,tokens_per_expert,w1)
        act_out,act_time = self._apply_activation(l1_out)
        l2_out,l2_time = self._apply_Linear2(act_out,tokens_per_expert,w2)

        return l2_out,l1_time,act_time,l2_time
class MoELayer(torch.nn.Module):
    def __init__(self,args=None):
        super(MoELayer,self).__init__()
        
        ep = args.expert_model_parallel_size
        num_experts = args.num_experts
        micro_batch = args.micro_batch
        seq_len = args.seq_length
        topk = args.moe_router_topk
        hidden_size = args.hidden_size
        num_attention_heads = args.num_attention_heads
        self.num_local_experts = int(num_experts / ep)
        if args.dtype == "bfloat16":
            dtype = torch.bfloat16
        elif args.dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        device = torch.cuda.current_device()
        if args.moe_grouped_gemm:
            self.experts = GroupedMLP(self.num_local_experts, args)
        else:
            self.experts = SequentialMLP(self.num_local_experts, args)
        # print("aa",seq_len*micro_batch*topk*dp/num_experts*self.num_local_experts)
        self.dispatched_input = torch.rand(int(seq_len*micro_batch*topk*ep/num_experts*self.num_local_experts), hidden_size
                                  ,device = device).to(dtype)
        temp_val = int(seq_len*micro_batch*topk*ep/num_experts)
        # self.tokens_per_expert = torch.tensor([temp,temp],device = device)
                                  
        self.tokens_per_expert = torch.full((self.num_local_experts,), temp_val)
        # print('aa',self.tokens_per_expert)
    def forward(self, hidden_states: torch.Tensor):
        # probs, indices = self.router(hidden_states)
        # (dispatched_input, tokens_per_expert) = self.token_dispatcher.token_permutation(
        #     hidden_states, probs, indices
        # )
        
        expert_output, mlp_linear_1,mlp_gelu,mlp_linear_2 = self.experts(self.dispatched_input, self.tokens_per_expert)
        # output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
        return expert_output,mlp_linear_1,mlp_gelu,mlp_linear_2
