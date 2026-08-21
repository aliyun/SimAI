import torch
import workload_generator.mocked_model.inference.MockedDeepSeek as MockedDeepSeek
from workload_generator.mocked_model.MockedModel import InferencePhase
from utils.utils import *
from utils.deepgemm_utils import *
import torch.nn.functional as F
import random
from typing import Tuple
import triton
import numpy as np
from flash_mla import flash_mla_with_kvcache, get_mla_metadata, flash_mla_sparse_fwd

def to_float8(x, dtype=torch.float8_e4m3fn):
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()
def construct_bmm(b: int, m: int, k: int, n: int) -> \
        Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor],
              Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    x = torch.randn((b, m, k), device='cuda:0', dtype=torch.bfloat16)
    y = torch.randn((b, n, k), device='cuda:0', dtype=torch.bfloat16)
    out = torch.empty((b, m, n), device='cuda:0', dtype=torch.bfloat16)

    y = y.transpose(-2, -1)
    ref_out = torch.bmm(x, y)

    x_fp8 = to_float8(x)
    y_fp8 = to_float8(y)
    return x, y, x_fp8, y_fp8, out, ref_out

class DeepSeekEmbedding(torch.nn.Module):
    def __init__(self, args=None):
        super(DeepSeekEmbedding, self).__init__()
        self.tp = args.tensor_model_parallel_size
        hidden_size = args.hidden_size
        # max_position_embeddings = args.max_position_embeddings
        self.vocab_size = args.vocab_size
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        self.device = device
        self.dtype = torch.bfloat16
        self.weight = torch.randint(
            low=0,
            high=1,  # 设置权重范围内的随机值，假设权重值在 0 到 1之间
            size=(self.vocab_size, hidden_size),
            device=device,
            dtype=torch.bfloat16 
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
        
        embeddings = F.embedding(masked_input,self.weight)
        return embeddings
    def forward(self,input):
        input = input.to(self.device) 
        result, emb_time = self._apply(input)

        result = result.to(self.dtype)

        return result, emb_time

class DeepSeekRMSNorm(torch.nn.Module):
    def __init__(self, args=None):
        super(DeepSeekRMSNorm, self).__init__()
        self.hidden_size = args.hidden_size
        
        self.tp = args.tensor_model_parallel_size
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        self.dtype = torch.bfloat16
        self.norm_weight = torch.rand(self.hidden_size, device=device).to(self.dtype)
    
    @cuda_timing_decorator
    def _apply(self, hidden_states):
        output_norm = F.rms_norm(hidden_states, (self.hidden_size,), self.norm_weight, 1e-6)
        return output_norm

    def forward(self, hidden_states):
        norm_output, norm_time = self._apply(hidden_states)

        return norm_output, norm_time



class DeepSeekAtten(torch.nn.Module):
    def __init__(self, args=None):
        super(DeepSeekAtten, self).__init__()
        self.tp = args.tensor_model_parallel_size
        self.hidden_size = args.hidden_size
        self.num_heads = args.head_num
        self.batch_size = args.micro_batch
        self.d_kv_c =  args.d_kv_c
        self.d_q_c = args.d_q_c
        self.d_r = args.d_r
        self.d_q = args.d_q
        self.d_kv = args.d_kv
        self.args = args

    def _qkv_compression(self, m):
        k = self.hidden_size #7168
        n = self.d_kv_c + self.d_q_c + self.d_r  #2112
        print(f'm={m}, k={k}, n={n}')
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

    def _q_uncrompression(self, m):
        k = self.d_q_c #1536
        n = self.num_heads *(self.d_q + self.d_r) // self.tp
        print(f'm={m}, k={k}, n={n}')
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
    def _q_bmm(self, m):
        b = self.num_heads //self.tp
        k = self.d_q
        n = self.d_kv_c #512
        print(f'm={m}, b={b}, k={k}, n={n}')
        x_bf16, y_bf16, x_fp8, y_fp8, out, ref_out = construct_bmm(
                            b, m, k, n)
        def test_func():
                        x_bf16, y_bf16, x_fp8, y_fp8, out, ref_out = construct_bmm(
                            b, m, k, n)
                        # with nvtx.range("matmul"):
                        out = torch.bmm(x_bf16, y_bf16)
        
        try:
            t = bench_kineto(test_func, "gemm_bf16", suppress_kineto_output=True)
        except:
            try:
                t = bench_kineto(test_func, "gemm_relu_bf16", suppress_kineto_output=True)
            except:
                # H20 use a strange kernel named "nvjet_tst_xxx_v_bz_TNN" to perform
                t = bench_kineto(test_func, "nvjet_tst", suppress_kineto_output=True)
        return t

    def _attention(self):
        device = torch.device("cuda:0")
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device(device)
        torch.cuda.set_device(device)
        torch.manual_seed(0)
        random.seed(0)

        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        if phase == InferencePhase.DECODE.value:
            b = self.batch_size
            s_q = 1
            mean_sk = self.args.seq_length + 1 # TODO not a good solution.
        elif phase == InferencePhase.PREFILL.value:
            b = 1
            s_q = self.args.seq_length
            mean_sk = self.args.seq_length
        h_q = self.num_heads //self.tp
        d = self.d_kv_c + self.d_r
        q = torch.randn(b, s_q, h_q, d)
        h_kv = 1

        if phase == InferencePhase.DECODE.value:
            #From DeepSeek-simulator
            cache_seqlens = torch.full((b,), mean_sk, dtype=torch.int32)
            for i in range(b):
                cache_seqlens[i] = max(
                    random.normalvariate(mean_sk, mean_sk / 2), s_q)
            # total_seqlens = cache_seqlens.sum().item()
            # mean_seqlens = cache_seqlens.float().mean().int().item()
            max_seqlen = cache_seqlens.max().item()
            max_seqlen_pad = triton.cdiv(max_seqlen, 256) * 256
            block_size = 64

            block_table = torch.arange(
                b * max_seqlen_pad // block_size, dtype=torch.int32
                ).view(b, max_seqlen_pad // block_size)
            blocked_k = torch.randn(block_table.numel(), block_size, h_kv, d)
            for i in range(b):
                blocked_k.view(b, max_seqlen_pad, h_kv, d)[i, cache_seqlens[i].item():] = (
                    float("nan")
                )
            # blocked_v = blocked_k[..., :self.d_kv_c]
            
            tile_scheduler_metadata, num_splits = get_mla_metadata(
                cache_seqlens, s_q * h_q // h_kv, h_kv
            )
            
            def flash_mla():
                return flash_mla_with_kvcache(
                    q,
                    blocked_k,
                    block_table,
                    cache_seqlens,
                    self.d_kv_c,
                    tile_scheduler_metadata,
                    num_splits,
                    causal=True,
                    )
            t = triton.testing.do_bench(flash_mla)
            return t
        elif phase == InferencePhase.PREFILL.value:
            # from https://github.com/deepseek-ai/FlashMLA/blob/main/tests/test_flash_mla_prefill.py
            s_kv = max(s_q,4096) #TODO
            topk = 2048 #TODO

            kv = torch.randn(b, s_kv, h_kv, d)
            indices = torch.full((b, s_q, h_kv, topk), s_kv, dtype=torch.int32)
            sm_scale = 1 / math.sqrt(d)
            def flash_mla():
                return flash_mla_sparse_fwd(q.squeeze(0), kv.squeeze(0), indices.squeeze(0), sm_scale)
            t = triton.testing.do_bench(flash_mla)
            return t

    def _O_bmm(self, m):
        b = self.num_heads //self.tp
        k = self.d_kv_c #512
        n = self.d_kv #128
        print(f'm={m}, b={b}, k={k}, n={n}')
        x_bf16, y_bf16, x_fp8, y_fp8, out, ref_out = construct_bmm(
                            b, m, k, n)
        def test_func():
                        x_bf16, y_bf16, x_fp8, y_fp8, out, ref_out = construct_bmm(
                            b, m, k, n)
                        # with nvtx.range("matmul"):
                        out = torch.bmm(x_bf16, y_bf16)
        
        try:
            t = bench_kineto(test_func, "gemm_bf16", suppress_kineto_output=True)
        except:
            try:
                t = bench_kineto(test_func, "gemm_relu_bf16", suppress_kineto_output=True)
            except:
                # H20 uses a strange kernel named "nvjet_tst_xxx_v_bz_TNN"
                t = bench_kineto(test_func, "nvjet_tst", suppress_kineto_output=True)
        return t

    def _O_proj(self, m):
        k = self.num_heads * self.d_kv //self.tp
        n = self.hidden_size
        print(f'm={m}, k={k}, n={n}')
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
         
    def forward(self):
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        if phase == InferencePhase.DECODE.value:
            m = self.batch_size
        elif phase == InferencePhase.PREFILL.value:
            m = self.args.seq_length
        qkv_compression_time = self._qkv_compression(m)
        print("matrix 1 is:" , qkv_compression_time*1e6)
        q_uncrompression_time = self._q_uncrompression(m)
        print("matrix 2 is:" , q_uncrompression_time*1e6)
        q_bemm = self._q_bmm(m)
        print("matrix 3 is:" , q_bemm*1e6)
        atten_qkv = qkv_compression_time + q_uncrompression_time + q_bemm
        atten_core = self._attention()
        linear_bmm = self._O_bmm(m)
        print("matrix 9 is:" , linear_bmm*1e6)
        linear_proj = self._O_proj(m)
        print("matrix 4 is:" , linear_proj*1e6)
        atten_linear = linear_bmm + linear_proj
        
        return atten_qkv, atten_core, atten_linear


        
class DeepSeekMLP(torch.nn.Module):
    def __init__(self, args=None):
        super(DeepSeekMLP, self).__init__()
        # self.tp = 1
        self.batch_size = args.micro_batch
        self.hidden_size = args.hidden_size
        self.expert_dim = args.expert_dim
        self.seq_length = args.seq_length
        self.args = args
    
    def _up_gate(self, m):
        k = self.hidden_size #7168
        n = self.expert_dim * 2   #不开TP
        print(f'm={m}, k={k}, n={n}')
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

    def _down(self, m):
        k = self.expert_dim  #不开TP
        n = self.hidden_size
        print(f'm={m}, k={k}, n={n}')
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

    def forward(self):
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        if phase == InferencePhase.DECODE.value:
            m = self.batch_size
        elif phase == InferencePhase.PREFILL.value:
            m = self.seq_length
        up_gate = self._up_gate(m)
        print("MLP:UP: ", up_gate*1e6)
        down = self._down(m)
        print("MLP:DOWN: ", down*1e6)
        return up_gate , down

class DeepSeekMOE(torch.nn.Module):
    def __init__(self, args=None):
        super(DeepSeekMOE, self).__init__()
        self.batch_size = args.micro_batch
        self.hidden_size = args.hidden_size
        self.expert_dim = args.expert_dim
        self.num_experts = args.router_expert
        self.tp = args.tensor_model_parallel_size
        self.dp = args.world_size // self.tp // args.pipeline_model_parallel
        self.ep = self.tp * self.dp
        self.topk = args.moe_router_topk
        self.seq_length = args.seq_length
        self.args = args
    
    def _up_gate(self, m):
        num_groups = self.num_experts // self.ep
        strategy = getattr(self.args, "moe_routing_strategy", Strategy.RoundRobin)
        expected_m_per_group = get_ep_expected_m_per_group(
            m, num_groups, self.topk, self.ep, strategy
        )

        n = self.expert_dim * 2
        k = self.hidden_size

        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        
        if phase == InferencePhase.DECODE.value:
            t = bench_masked(num_groups, expected_m_per_group, k, n)
        elif phase == InferencePhase.PREFILL.value:
            t = bench_contiguous(num_groups, expected_m_per_group, k, n)
        
        return t
    
    def _down(self, m):
        num_groups = self.num_experts // self.ep
        strategy = getattr(self.args, "moe_routing_strategy", Strategy.RoundRobin)
        expected_m_per_group = get_ep_expected_m_per_group(
            m, num_groups, self.topk, self.ep, strategy
        )

        n = self.hidden_size
        k = self.expert_dim

        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        
        if phase == InferencePhase.DECODE.value:
            t = bench_masked(num_groups, expected_m_per_group, k, n)
        elif phase == InferencePhase.PREFILL.value:
            t = bench_contiguous(num_groups, expected_m_per_group, k, n)
        
        return t

    def forward(self):
        phase = getattr(self.args, "phase", InferencePhase.DECODE.value)
        if phase == InferencePhase.DECODE.value:
            m = self.batch_size
        elif phase == InferencePhase.PREFILL.value:
            m = self.args.seq_length
        up_gate = self._up_gate(m)
        down = self._down(m)

        return up_gate, down


class DeepSeekModel(torch.nn.Module):
    def __init__(self, args=None):
        super(DeepSeekModel, self).__init__()
        self.time_list = {}
        self.args = args
        self.Embedding = DeepSeekEmbedding(self.args)
        # self.transformer = DeepSeekTransformer(self.args)
        self.RMSNorm = DeepSeekRMSNorm(self.args)
        self.Attention = DeepSeekAtten(self.args)
        self.MLP = DeepSeekMLP(self.args)
        self.MOE = DeepSeekMOE(self.args)
        self.aiob_forward_loops = getattr(args, "aiob_forward_loops", 10)


    def forward(self):
        # Emb_output, Emb_time = self.Embedding(input)
        # print("Embedding time: ", Emb_time)
        # self.time_list.setdefault("Emb", []).append({"time_gpu": Emb_time})
        for i in range(self.aiob_forward_loops):
            # norm_output, norm_time1 = self.RMSNorm(Emb_output)
            # self.time_list.setdefault("RMSNorm1", []).append(
            #         {"time_gpu": norm_time1}
            #     )
            atten_qkv, atten_core, atten_linear = self.Attention()
            self.time_list.setdefault("atten_qkv", []).append(
                        {"time_gpu": atten_qkv* 1e6}
                    )
            self.time_list.setdefault("atten_flash", []).append(
                        {"time_gpu": atten_core* 1e3}
                    )
            self.time_list.setdefault("atten_linear", []).append(
                        {"time_gpu": atten_linear* 1e6}
                    )
            # print("norm_time time: ", norm_time1)
            print(f'qkv_core time: , {atten_qkv * 1e6}')
            print(f'qkv_attention time: , {atten_core* 1e3}')
            print(f'atten_linear time: , {atten_linear * 1e6}')
            # self.time_list.setdefault("RMSNorm2", []).append(
            #         {"time_gpu": norm_time2}
            #     )
            # print("norm_time time: ", norm_time2)
            mlp_up, mlp_down = self.MLP()
            self.time_list.setdefault("mlp_up", []).append(
                {"time_gpu": mlp_up* 1e6}
                )
            self.time_list.setdefault("mlp_down", []).append(
                {"time_gpu": mlp_down* 1e6}
            )
            print(f'mlp_time time: , {mlp_up * 1e6}')
            print(f'mlp_down time: , {mlp_down * 1e6}')
            up_gate_moe, down_moe = self.MOE()
            self.time_list.setdefault("moe_up_gate", []).append(
                    {"time_gpu": up_gate_moe* 1e6}
                )
            self.time_list.setdefault("moe_down", []).append(
                    {"time_gpu": down_moe* 1e6}
                )
            print(f'up_gate_moe time: , {up_gate_moe * 1e6}')  
            print(f'down_moe time: , {down_moe * 1e6}')  
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
    args = MockedDeepSeek.DeepSeekParams()
    x = torch.randint(0, args.vocab_size, (2, 1))
    model = DeepSeekModel(args)
    filepath = model(x)
    print(filepath)