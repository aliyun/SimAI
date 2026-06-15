import torch
# torch.set_default_device("cuda")
import workload_generator.mocked_model.inference.MockedQwen3Next as MockedQwen3Next
from workload_generator.mocked_model.MockedModel import InferencePhase
from utils.utils import *
from utils.deepgemm_utils import *
import torch.nn.functional as F
import random
from typing import Optional, Union, Tuple
# import triton
import numpy as np
from vllm.model_executor.layers.quantization.utils.fp8_utils import per_token_group_quant_fp8
from vllm import _custom_ops as vllm_ops
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding, get_rope
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
import flashinfer
from vllm.model_executor.layers.fla.ops import (RMSNormGated, fused_recurrent_gated_delta_rule)
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn, causal_conv1d_update)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

def GemmaRMSNormTest(
        hidden_size,
        variance_epsilon: float,
        x: torch.Tensor,
        residual: Optional[torch.Tensor]):
    gemmaRMSNormImpl = GemmaRMSNorm(hidden_size, variance_epsilon)
    gemmaRMSNormImpl = gemmaRMSNormImpl.to(x.device)
    def test_func():
        gemmaRMSNormImpl.forward_cuda(x, residual)
    t = bench_kineto(test_func, "triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0", suppress_kineto_output=True)
    return t

# modified from test_causal_conv1d.py
def test_causal_conv1d_update(batch, dim, width, seqlen, has_bias, silu_activation,
                              itype):
    device = "cuda"
    rtol, atol = (3e-4, 1e-3) if itype == torch.float32 else (3e-3, 5e-3)
    if itype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    # set seed
    current_platform.seed_everything(0)
    x = torch.randn(batch, dim, seqlen, device=device, dtype=itype)
    x_ref = x.clone()
    conv_state = torch.randn(batch, dim, width - 1, device=device, dtype=itype)

    weight = torch.randn(dim, width, device=device, dtype=itype)
    bias = torch.randn(dim, device=device, dtype=itype) if has_bias else None
    activation = None if not silu_activation else "silu"
    out = causal_conv1d_update(x,
                               conv_state,
                               weight,
                               bias,
                               activation=activation)
    return out

# modified from qwen3_next.py
# g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
@triton.jit
def fused_gdn_gating_kernel(
    g,
    A_log,
    a,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr = 1.0,
    threshold: tl.constexpr = 20.0,
    BLK_HEADS: tl.constexpr = 8
):
    i_b, i_s, i_d = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS
    blk_A_log = tl.load(A_log + head_off, mask=mask)
    blk_a = tl.load(a + off, mask=mask)
    blk_bias = tl.load(dt_bias + head_off, mask=mask)
    # If the model is loaded in fp16, without the .float() here, A might be -inf
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = (1 / beta) * tl.log(1 + tl.exp(beta * x))
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)


class Qwen3NextGatedDeltaNet(torch.nn.Module):
    def __init__(self, args=None):
        super(Qwen3NextGatedDeltaNet, self).__init__()
        self.args = args
        self.tp = self.args.tensor_model_parallel_size

        self.norm = RMSNormGated(
            self.args.linear_value_head_dim,
            eps=self.args.rms_norm_eps,
            group_size=None,
            norm_before_gate=True,
            device='cuda', dtype=torch.bfloat16
        )

    def _attn_quant(self, qk_or_o, m):
        if qk_or_o:
            n = self.args.hidden_size
        else:
            n = self.args.num_attention_heads * self.args.head_dim // self.tp
        input = torch.randn(m, n, device='cuda', dtype=torch.bfloat16)
        def test_func():
            per_token_group_quant_fp8(input, 128, column_major_scales=True)

        candidates = [
            'per_token_group_quant_fp8', # old version
            'per_token_group_quant_8bit', # new version
        ]
        last_err = None
        for names in candidates:
            try:
                t = bench_kineto(test_func, names, suppress_kineto_output=True)
                return t
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"bench_kineto failed for all candidates: {last_err}")
    
    def _attn_proj(self, k, n, m):
        x_fp8, y_fp8, c, out, ref_out = construct(m, k, n)
        # deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
        deep_gemm.fp8_gemm_nt(x_fp8, y_fp8, out, c=c, disable_ue8m0_cast=True, recipe = None)
        diff = calc_diff(out, ref_out)
        assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'
        x_fp8, y_fp8, c, out, ref_out = construct(m, k, n)
        def test_func():
            # deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
            deep_gemm.fp8_gemm_nt(x_fp8, y_fp8, out, c=c, disable_ue8m0_cast=True, recipe = None)
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        return t

    def _causal_conv(self, m):
        n = (self.args.num_attention_heads * self.args.head_dim + self.args.linear_num_key_heads*self.args.linear_key_head_dim + self.args.linear_num_value_heads*self.args.linear_value_head_dim) // self.tp

        def test_func():
            test_causal_conv1d_update(batch=m, 
                                      dim=n, 
                                      width=self.args.linear_conv_kernel_dim,
                                      seqlen=1, # TODO decoding self.args.seq_length
                                      has_bias=False, silu_activation=True, itype=torch.bfloat16)
        t = bench_kineto(test_func, "_causal_conv1d_update_kernel", suppress_kineto_output=True)
        return t

    def _fused_gdn_gating(self, m):
        A_log = torch.empty(
                divide(self.args.linear_num_value_heads, self.tp),
                dtype=torch.float32,
            )
        a = torch.randn(m, self.args.linear_num_value_heads//self.tp, device='cuda', dtype=torch.bfloat16)
        dt_bias = torch.ones(self.args.linear_num_value_heads // self.tp)

        batch, num_heads = a.shape
        seq_len=1 # TODO decoding self.args.seq_length
        grid = (batch, seq_len, triton.cdiv(num_heads, 8))
        g = torch.empty_like(a, dtype=torch.float32)

        def test_func():
            fused_gdn_gating_kernel[grid](g,
                                  A_log,
                                  a,
                                  dt_bias,
                                  seq_len,
                                  num_heads,
                                  1.0,
                                  20.0,
                                  8,
                                  num_warps=1)
        t = bench_kineto(test_func, "fused_gdn_gating_kernel", suppress_kineto_output=True)
        return t
    
    def _fused_gdn_core(self):
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        if phase == InferencePhase.DECODE.value:
            B = self.args.micro_batch
            T = 1
        elif phase == InferencePhase.PREFILL.value:
            B = 1
            T = self.args.seq_length

        H = self.args.linear_num_key_heads // self.tp
        HV = self.args.linear_num_value_heads // self.tp
        K = self.args.linear_key_head_dim
        V = self.args.linear_value_head_dim

        # 1 is for decoding
        q = torch.randn(B, 1, H, K, device='cuda', dtype=torch.bfloat16).contiguous()
        k = F.normalize(torch.randn(B, 1, H, K, device='cuda', dtype=torch.bfloat16), p=2, dim=-1).contiguous()
        v = torch.randn(B, 1, HV, V, device='cuda', dtype=torch.bfloat16).contiguous()
        g = F.logsigmoid(torch.rand(B, 1, HV, device='cuda', dtype=torch.bfloat16)).contiguous()
        beta = torch.rand(B, 1, HV, device='cuda', dtype=torch.bfloat16).sigmoid().contiguous()
        # h0 = torch.empty(B, HV, V, device=q.device, dtype=v.dtype)
        h0 = torch.randn(B, HV, K, V, device='cuda', dtype=torch.bfloat16).contiguous()

        ssm = torch.zeros(B*T, device='cuda', dtype=torch.int32).contiguous()

        # print('q', q.shape, q.stride())
        # print('k', k.shape, k.stride())
        # print('v', v.shape, v.stride())
        # print('g', g.shape, g.stride())
        # print('beta', None if beta is None else beta.shape)
        # print('init_ht', None if h0 is None else h0.shape)
        # print('ssm', None if ssm is None else ssm.shape)

        def test_func():
            o, ht = fused_recurrent_gated_delta_rule(
                q, k, v, g, beta,
                initial_state=h0,
                inplace_final_state=True,
                # inplace_final_state=False,
                ssm_state_indices=ssm,
                # ssm_state_indices=None,
                # use_qk_l2norm_in_kernel=False,
                use_qk_l2norm_in_kernel=True,
            )
        t = bench_kineto(test_func, "fused_recurrent_gated_delta_rule_fwd_kernel", suppress_kineto_output=True)
        return t
        
    def _norm(self, m):
        k = self.args.linear_num_value_heads // self.tp
        n = self.args.linear_value_head_dim

        core_attn_out = torch.randn(m*k, n, device='cuda', dtype=torch.bfloat16)
        z = torch.randn(m*k, n, device='cuda', dtype=torch.bfloat16)

        def test_func():
            self.norm(core_attn_out, z)
        t = bench_kineto(test_func, "layer_norm_fwd_kernel", suppress_kineto_output=True)
        return t
    
    def forward(self):
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        if phase == InferencePhase.DECODE.value:
            m = self.args.micro_batch
        elif phase == InferencePhase.PREFILL.value:
            m = self.args.seq_length

        # 1. qkv
        qkv_quant = self._attn_quant(True, m)
        
        qkvz_proj_k = self.args.hidden_size
        qkvz_proj_n = (self.args.linear_value_head_dim+self.args.linear_value_head_dim)*2 // self.tp
        qkvz_proj = self._attn_proj(qkvz_proj_k, qkvz_proj_n, m)

        ba_proj_k = self.args.hidden_size
        ba_proj_n = self.args.linear_num_value_heads*2 // self.tp
        ba_proj = self._attn_proj(ba_proj_k, ba_proj_n, m)
        
        attn_qkv = qkv_quant+qkvz_proj+ba_proj

        # 2. Causal Convolution
        attn_causal_conv = self._causal_conv(m)

        # 3. Recurrent Attention
        gdn_gating = self._fused_gdn_gating(m)
        #fused_recurrent_gated_delta_rule
        gdn_core = self._fused_gdn_core()

        attn_gdn = gdn_gating + gdn_core

        # 4. output
        # attn_out_norm=RMSNormGated

        o_norm = self._norm(m)
        o_quant = self._attn_quant(False, m)
        o_proj_k = self.args.linear_value_head_dim // self.tp
        o_proj_n = self.args.hidden_size
        o_proj = self._attn_proj(o_proj_k,o_proj_n, m)
        
        attn_o = o_norm + o_quant + o_proj

        return attn_qkv, attn_causal_conv, attn_gdn, attn_o

        
class Qwen3NextAttention(torch.nn.Module):
    def __init__(self, args=None):
        super(Qwen3NextAttention, self).__init__()
        self.args = args
        self.tp = self.args.tensor_model_parallel_size
    
    def _attn_quant(self, qk_or_o, m):
        if qk_or_o:
            n = self.args.hidden_size
        else:
            n = self.args.num_attention_heads * self.args.head_dim // self.tp
        input = torch.randn(m, n, device='cuda', dtype=torch.bfloat16)
        def test_func():
            per_token_group_quant_fp8(input, 128, column_major_scales=True)

        candidates = [
            'per_token_group_quant_fp8', # old version
            'per_token_group_quant_8bit', # new version
        ]
        last_err = None
        for names in candidates:
            try:
                t = bench_kineto(test_func, names, suppress_kineto_output=True)
                return t
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"bench_kineto failed for all candidates: {last_err}")

    def _attn_proj(self, qk_or_o, m):
        if qk_or_o:
            n = (self.args.num_attention_heads + self.args.num_key_value_heads * 2) * self.args.head_dim // self.tp
            k = self.args.hidden_size
        else:
            n = self.args.hidden_size
            k = self.args.num_attention_heads  * self.args.head_dim // self.tp
        x_fp8, y_fp8, c, out, ref_out = construct(m, k, n)
        # deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
        deep_gemm.fp8_gemm_nt(x_fp8, y_fp8, out, c=c, disable_ue8m0_cast=True, recipe = None)
        diff = calc_diff(out, ref_out)
        assert diff < 0.001, f'{m=}, {k=}, {n=}, {diff:.5f}'
        x_fp8, y_fp8, c, out, ref_out = construct(m, k, n)
        def test_func():
            # deep_gemm.gemm_fp8_fp8_bf16_nt(x_fp8, y_fp8, out)
            deep_gemm.fp8_gemm_nt(x_fp8, y_fp8, out, c=c, disable_ue8m0_cast=True, recipe = None)
        t = bench_kineto(test_func, 'fp8_gemm', suppress_kineto_output=True)
        return t
    def _qk_norm(self, q_or_k, m):
        head_dim = self.args.head_dim
        if q_or_k:
            n = self.args.num_attention_heads * head_dim // self.tp
        else:
            n = self.args.num_key_value_heads * head_dim // self.tp
        x = torch.randn(m,n, device='cuda', dtype=torch.bfloat16)
        x = x.view(*x.shape[:-1], x.shape[-1] // head_dim, head_dim)
        residual = torch.rand_like(x, device='cuda', dtype=torch.bfloat16)
        return GemmaRMSNormTest(head_dim,self.args.rms_norm_eps,x, residual)
            
    def _rotary_emb(self, m):
        head_dim = self.args.head_dim
        n_q = self.args.num_attention_heads * head_dim // self.tp
        n_k = self.args.num_key_value_heads * head_dim // self.tp

        q = torch.randn(m, n_q, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(m, n_k, device='cuda', dtype=torch.bfloat16)
        pos = torch.arange(0, m, device='cuda')
        rotary_emb = get_rope(
                head_dim,
                rotary_dim=head_dim,
                max_position=self.args.max_position_embeddings,
                base=self.args.rope_theta,
                rope_scaling=self.args.rope_scaling,
                dual_chunk_attention_config = getattr(self.args, 
                                        "dual_chunk_attention_config",
                                        None),
            )
        def test_func():
            rotary_emb(pos, q, k)

        t = bench_kineto(test_func, 'rotary_embedding_kernel', suppress_kineto_output=True)
        return t

    def _attn_core(self, kv_lens):
        block_size = 32
        num_blocks = 32768 
        
        head_size = self.args.head_dim
        num_query_heads = self.args.num_attention_heads // self.tp
        num_kv_heads = self.args.num_key_value_heads// self.tp
        dtype = torch.bfloat16


        # based on vllm:test_flashinfer_decode_with_paged_kv
        torch.set_default_device("cuda")
        num_seqs = len(kv_lens)
        assert num_query_heads % num_kv_heads == 0
        max_kv_len = max(kv_lens)
        scale = head_size**-0.5
        soft_cap = None

        query = torch.randn(num_seqs, num_query_heads, head_size, dtype=dtype)

        key_value_cache = torch.randn(num_blocks,
                                    2,
                                    block_size,
                                    num_kv_heads,
                                    head_size,
                                    dtype=dtype)
        # key_cache = key_value_cache[:, 0, :, :, :].squeeze(1)
        # value_cache = key_value_cache[:, 1, :, :, :].squeeze(1)

        max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
        block_tables = torch.randint(0,
                                    num_blocks,
                                    (num_seqs, max_num_blocks_per_seq),
                                    dtype=torch.int32)

        kv_indptr = [0]
        kv_indices = []
        kv_last_page_lens = []
        for i in range(num_seqs):
            seq_len = kv_lens[i]
            assert seq_len > 0
            num_blocks = (seq_len + block_size - 1) // block_size
            kv_indices.extend(block_tables[i, :num_blocks])
            kv_indptr.append(kv_indptr[-1] + num_blocks)
            kv_last_page_len = seq_len % block_size
            if kv_last_page_len == 0:
                kv_last_page_len = block_size
            kv_last_page_lens.append(kv_last_page_len)

        kv_indptr = torch.tensor(kv_indptr, dtype=torch.int32)
        kv_indices = torch.tensor(kv_indices, dtype=torch.int32)
        kv_last_page_lens = torch.tensor(kv_last_page_lens, dtype=torch.int32)
        
        # 128 MB 
        workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8)
        wrapper = flashinfer.\
            BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD",
                    use_tensor_cores=(
                        (num_query_heads//num_kv_heads) > 4)
                    )
        wrapper.plan(kv_indptr,
                    kv_indices,
                    kv_last_page_lens,
                    num_query_heads,
                    num_kv_heads,
                    head_size,
                    block_size,
                    "NONE",
                    q_data_type=dtype,
                    kv_data_type=dtype,
                    logits_soft_cap=soft_cap)
        
        def test_func():
            wrapper.run(query, key_value_cache)
        t = bench_kineto(test_func, 'BatchPrefillWithPagedKVCacheKernel', suppress_kineto_output=True)
        return t

    def forward(self):
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        if phase == InferencePhase.DECODE.value:
            m = self.args.micro_batch
            kv_lens = [self.args.seq_length] * m
        elif phase == InferencePhase.PREFILL.value:
            m = self.args.seq_length
            kv_lens = [self.args.seq_length] * 1

        # qkv
        qkv_quant = self._attn_quant(True,m)
        qkv_proj = self._attn_proj(True,m)
        q_norm = self._qk_norm(True,m)
        k_norm = self._qk_norm(False,m)
        attn_qkv = qkv_quant + qkv_proj + q_norm + k_norm

        # rotary_embedding
        rotary_emb = self._rotary_emb(m)

        # core
        attn = self._attn_core(kv_lens)

        # o
        o_quant = self._attn_quant(False,m)
        o_proj = self._attn_proj(False,m)
        attn_o = o_quant + o_proj

        return attn_qkv, rotary_emb, attn, attn_o

class Qwen3NextSparseMoeBlock(torch.nn.Module):
    def __init__(self, args):
        super(Qwen3NextSparseMoeBlock, self).__init__()
        self.args = args
        self.batch_size = args.micro_batch
        self.hidden_size = args.hidden_size
        self.num_experts = args.num_experts
        self.tp = args.tensor_model_parallel_size
        self.dp = args.world_size // self.tp // args.pipeline_model_parallel
        self.ep = self.tp * self.dp
        self.topk = args.num_experts_per_tok
    
    def _route_gate(self, m):
        # ReplicatedLinear
        x = torch.randn(m, self.hidden_size, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(self.num_experts, self.hidden_size, device='cuda', dtype=torch.bfloat16)
        # print(x.shape, w.shape)

        def test_func():
            F.linear(x, w)

        candidates = [
            ("cutlass", "cublasLt::splitK"),
            ("nvjet_tst", "cublasLt::splitK"),
            "cutlass",
            "nvjet_tst",
        ]
        last_err = None
        for names in candidates:
            try:
                res = bench_kineto(test_func, names, suppress_kineto_output=True)
                return sum(res) if isinstance(res, (list, tuple)) else float(res)
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"bench_kineto failed for all candidates: {last_err}")      

    def _select_experts(self, total_tokens):
        # FusedMoE.select_experts
        hidden_states = torch.randn(total_tokens, self.hidden_size, device='cuda', dtype=torch.bfloat16)
        router_logits = torch.randn(total_tokens, self.num_experts, device='cuda', dtype=torch.bfloat16)

        def test_func():
            fused_topk(hidden_states=hidden_states,
                            gating_output=router_logits,
                            topk=self.topk,
                            renormalize=True)
        t = bench_kineto(test_func, 'moe::topkGatingSoftmax', suppress_kineto_output=True)
        return t

    def _moe_gate_proj(self, up_or_down, m):
        num_groups = self.num_experts // self.ep
        strategy = getattr(self.args, "moe_routing_strategy", Strategy.RoundRobin)
        expected_m_per_group = get_ep_expected_m_per_group(
            m, num_groups, self.topk, self.ep, strategy
        )
        if up_or_down:
            n = 2 * self.args.moe_intermediate_size
            k = self.hidden_size
        else:
            n = self.hidden_size
            k = self.args.moe_intermediate_size

        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)

        if phase == InferencePhase.DECODE.value:
            t = bench_masked(num_groups, expected_m_per_group, k, n)
        elif phase == InferencePhase.PREFILL.value:
            t = bench_contiguous(num_groups, expected_m_per_group, k, n)

        return t

    def _moe_act(self, m):
        # SiluAndMul
        num_groups=self.num_experts // self.ep
        total = m * self.dp * self.topk // self.num_experts
        n = 2 * self.args.moe_intermediate_size

        total_max = total # TODO: maybe should set in config
        intermediate_cache1 = torch.randn((num_groups * total_max, n),
                                        device='cuda',
                                        dtype=torch.bfloat16)
        intermediate_cache2 = torch.randn((num_groups * total_max, n // 2),
                                        device='cuda',
                                        dtype=torch.bfloat16)
        
        def test_func():
            torch.ops._C.silu_and_mul(
                intermediate_cache2.unflatten(0, (num_groups, total_max)),
                intermediate_cache1.unflatten(0, (num_groups, total_max)))
            

        candidates = [
            "vllm::silu",
            "vectorized_elementwise_kernel",
        ]
        last_err = None
        for names in candidates:
            try:
                res = bench_kineto(test_func, names, suppress_kineto_output=True)
                return sum(res) if isinstance(res, (list, tuple)) else float(res)
            except Exception as e:
                last_err = e
                continue

        raise RuntimeError(f"bench_kineto failed for all candidates: {last_err}")


    def forward(self):
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        if phase == InferencePhase.DECODE.value:
            m = self.args.micro_batch
            total_tokens = m  # TODO: maybe should set output seq_length
        elif phase == InferencePhase.PREFILL.value:
            m = self.args.seq_length
            total_tokens = m

        route_gate = self._route_gate(m)
        route_select_experts = self._select_experts(total_tokens)
        moe_up = self._moe_gate_proj(True, m)
        moe_act = self._moe_act(m)
        moe_down = self._moe_gate_proj(False, m)

        return route_gate, route_select_experts, moe_up, moe_act, moe_down


class Qwen3NextModel(torch.nn.Module):
    def __init__(self, args=None):
        super(Qwen3NextModel, self).__init__()
        self.time_list = {}
        self.args = args
        # self.Embedding = Qwen3Embedding(self.args)
        self.FullAttention = Qwen3NextAttention(self.args)
        self.GDN = Qwen3NextGatedDeltaNet(self.args)
        # self.MLP = Qwen3NextMLP(self.args) # TODO add MLP only
        self.MoE = Qwen3NextSparseMoeBlock(self.args)
        self.aiob_forward_loops = getattr(args, "aiob_forward_loops", 10)
    
    def _norm(self):
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        if phase == InferencePhase.DECODE.value:
            m = self.args.micro_batch
        elif phase == InferencePhase.PREFILL.value:
            m = self.args.seq_length
        hidden_size = self.args.hidden_size
        x = torch.randn(m,hidden_size, device='cuda', dtype=torch.bfloat16)
        residual = torch.randn(m,hidden_size, device='cuda', dtype=torch.bfloat16)
        return GemmaRMSNormTest(hidden_size, self.args.rms_norm_eps, x, residual)
    
    def forward(self):
        for i in range(self.aiob_forward_loops):
            # print(f'======================{i}======================')
            # 1. input_layernorm
            pre_norm = self._norm()
            self.time_list.setdefault("atten_norm", []).append(
                        {"time_gpu": pre_norm* 1e6}
                    )
            # 2.A. self_attn
            atten_qkv, atten_rotary_emb, atten_core, atten_o = self.FullAttention()
            self.time_list.setdefault("atten_qkv", []).append(
                        {"time_gpu": atten_qkv* 1e6}
                    )
            self.time_list.setdefault("atten_rotary_emb", []).append(
                        {"time_gpu": atten_rotary_emb* 1e6}
                    )
            self.time_list.setdefault("atten_flash", []).append(
                        {"time_gpu": atten_core* 1e6}
                    )
            self.time_list.setdefault("atten_o", []).append(
                        {"time_gpu": atten_o* 1e6}
                    )
            
            # 2.B. linear_attn
            gdn_qkv, gdn_causal_conv, gdn_core, gdn_o = self.GDN()
            self.time_list.setdefault("gdn_qkv", []).append(
                    {"time_gpu": gdn_qkv * 1e6}
                )
            self.time_list.setdefault("gdn_causal_conv", []).append(
                        {"time_gpu": gdn_causal_conv * 1e6}
                    )
            self.time_list.setdefault("gdn_core", []).append(
                        {"time_gpu": gdn_core * 1e6}
                    )
            self.time_list.setdefault("gdn_o", []).append(
                        {"time_gpu": gdn_o * 1e6}
                    )

            # 3. post_attention_layernorm
            post_norm = self._norm()
            self.time_list.setdefault("moe_norm", []).append(
                        {"time_gpu": post_norm* 1e6}
                    )

            # 4. mlp
            moe_route_gate, moe_route_select_experts, moe_up, moe_act, moe_down = self.MoE()
            self.time_list.setdefault("moe_route_gate", []).append(
                        {"time_gpu": moe_route_gate* 1e6}
                    )
            self.time_list.setdefault("moe_route_select_experts", []).append(
                        {"time_gpu": moe_route_select_experts* 1e6}
                    )
            self.time_list.setdefault("moe_expert_up", []).append(
                        {"time_gpu": moe_up* 1e6}
                    )
            self.time_list.setdefault("moe_expert_act", []).append(
                        {"time_gpu": moe_act* 1e6}
                    )
            self.time_list.setdefault("moe_expert_down", []).append(
                        {"time_gpu": moe_down* 1e6}
                    )
        
        print(self.time_list)
        result_dir = "./results/aiob_outputs"
        if not os.path.isdir(result_dir):
            os.makedirs(result_dir)
        filename = f"{self.args.model_name}_time_list_stats.txt"
        file_name = os.path.join(result_dir, filename)
        calculate_stats(self.time_list, file_name)
        filepath = write_time(self.time_list, self.args)
        process_all_keys(filepath)
        return filepath
    

if __name__ == "__main__":
    args = MockedQwen3Next.Qwen3NextParams()
    x = torch.randint(0, args.vocab_size, (2, 1))
    model = Qwen3NextModel(args)
    filepath = model(x)
    print(filepath)     