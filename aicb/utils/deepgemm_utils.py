import torch
from typing import Tuple, Generator
import deep_gemm
import enum
import random
from deep_gemm.testing import bench_kineto, calc_diff, count_bytes
from deep_gemm.utils import (align, ceil_div,
                             get_mk_alignment_for_contiguous_layout,
                             per_block_cast_to_fp8, per_token_cast_to_fp8)
# from deep_gemm.jit_kernels.utils import get_m_alignment_for_contiguous_layout

# From DeepGEMM/tests/generators.py
class KernelType(enum.Enum):
    Kernel1D1D = 0
    Kernel1D2D = 1
    KernelNoSF = 2

    def is_1d1d(self):
        return self.value == 0

    def is_1d2d(self):
        return self.value == 1

    def is_nosf(self):
        return self.value == 2


class MajorTypeAB(enum.Enum):
    KMajor = 0
    MNMajor = 1

    def is_k_major(self):
        return self.value == 0

    def is_mn_major(self):
        return self.value == 1


# === Common ===
def generate_normal(
    m: int,
    n: int,
    k: int,
    major_a: MajorTypeAB,
    major_b: MajorTypeAB,
    accumulate: bool,
    out_dtype: torch.dtype,
    kernel_type: KernelType,
    use_ue8m0: bool = False,
    use_bf16: bool = False,
):
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    d = (
        torch.randn((m, n), device="cuda", dtype=out_dtype) * 32
        if accumulate
        else torch.empty((m, n), device="cuda", dtype=out_dtype)
    )
    c = d if accumulate else None
    ref_d = (a.float() @ b.float().t() + (c if accumulate else 0)).to(out_dtype)

    if use_bf16:
        a = a if major_a.is_k_major() else a.T.contiguous().T
        b = b if major_b.is_k_major() else b.T.contiguous().T
        return a, b, c, d, ref_d

    a_fp8 = per_token_cast_to_fp8(a, use_ue8m0=use_ue8m0)
    b_fp8 = (
        per_token_cast_to_fp8(b, use_ue8m0=use_ue8m0)
        if kernel_type.is_1d1d() and accumulate
        else per_block_cast_to_fp8(b, use_ue8m0=use_ue8m0)
    )
    a_fp8 = a_fp8 if major_a.is_k_major() else (a_fp8[0].T.contiguous().T, a_fp8[1])
    b_fp8 = b_fp8 if major_b.is_k_major() else (b_fp8[0].T.contiguous().T, b_fp8[1])
    return a_fp8, b_fp8, c, d, ref_d


# === Masked ===
def enumerate_m_grouped_masked() -> Generator:
    max_m = 128#4096
    yield KernelType.Kernel1D2D, max_m

def generate_m_grouped_masked(
    num_groups: int,
    max_m: int,
    expected_m_per_group: int,
    n: int,
    k: int,
    use_ue8m0: bool = False,
    use_bf16: bool = False,
):
    a = torch.randn((num_groups, max_m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
    d = torch.empty((num_groups, max_m, n), device="cuda", dtype=torch.bfloat16)
    ref_d = torch.einsum("gmk,gnk->gmn", a, b)

    masked_m = torch.empty((num_groups,), device="cuda", dtype=torch.int)
    for j in range(num_groups):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
    assert masked_m.amax().item() <= max_m

    if use_bf16:
        return a, b, masked_m, d, ref_d

    a_fp8 = (
        torch.empty_like(a, dtype=torch.float8_e4m3fn),
        torch.empty(
            (num_groups, max_m, ceil_div(k, 128)), device="cuda", dtype=torch.float
        ),
    )
    b_fp8 = (
        torch.empty_like(b, dtype=torch.float8_e4m3fn),
        torch.empty(
            (num_groups, ceil_div(n, 128), ceil_div(k, 128)),
            device="cuda",
            dtype=torch.float,
        ),
    )
    for i in range(num_groups):
        a_fp8[0][i], a_fp8[1][i] = per_token_cast_to_fp8(a[i], use_ue8m0=use_ue8m0)
        b_fp8[0][i], b_fp8[1][i] = per_block_cast_to_fp8(b[i], use_ue8m0=use_ue8m0)

    return a_fp8, b_fp8, masked_m, d, ref_d


# === Contiguous ===
def enumerate_m_grouped_contiguous() -> Generator:
    yield KernelType.Kernel1D2D, MajorTypeAB.KMajor, MajorTypeAB.KMajor

def generate_m_grouped_contiguous(
    num_groups: int,
    expected_m_per_group: int,
    n: int,
    k: int,
    major_a: MajorTypeAB,
    major_b: MajorTypeAB,
    use_ue8m0: bool = False,
    use_bf16: bool = False,
):
    actual_ms = [int(expected_m_per_group * random.uniform(0.7, 1.3)) for _ in range(num_groups)]
    aligned_ms = [align(actual_m, get_mk_alignment_for_contiguous_layout()) for actual_m in actual_ms]
    m = sum(aligned_ms)

    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((num_groups, n, k), device="cuda", dtype=torch.bfloat16)
    m_indices = torch.empty(m, device="cuda", dtype=torch.int32)
    d = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    ref_d = torch.randn((m, n), device="cuda", dtype=torch.bfloat16)

    start = 0
    for i, (actual_m, aligned_m) in enumerate(zip(actual_ms, aligned_ms)):
        actual_end = start + actual_m
        aligned_end = start + aligned_m
        m_indices[start:actual_end] = i
        m_indices[actual_end:aligned_end] = -1
        ref_d[start:aligned_end] = a[start:aligned_end] @ b[i].t()
        start = aligned_end
    ref_d = torch.where((m_indices == -1).unsqueeze(1), torch.zeros_like(ref_d), ref_d)

    if use_bf16:
        b = b if major_b.is_k_major() else b.mT.contiguous().mT
        return m, a, b, m_indices, d, ref_d

    assert major_a.is_k_major()
    a_fp8 = per_token_cast_to_fp8(a, use_ue8m0=use_ue8m0)
    b_fp8 = (
        torch.empty_like(b, dtype=torch.float8_e4m3fn),
        torch.empty(
            (num_groups, ceil_div(n, 128), ceil_div(k, 128)),
            device="cuda",
            dtype=torch.float,
        ),
    )
    for i in range(num_groups):
        b_fp8[0][i], b_fp8[1][i] = per_block_cast_to_fp8(b[i], use_ue8m0=use_ue8m0)
    b_fp8 = b_fp8 if major_b.is_k_major() else (b_fp8[0].mT.contiguous().mT, b_fp8[1])
    return m, a_fp8, b_fp8, m_indices, d, ref_d

# This function uses generate_normal to construct x_fp8, y_fp8, out, ref_out
def construct(m: int, k: int, n: int) -> \
        Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
    a_fp8, b_fp8, c, d, ref_d = generate_normal(m, n, k, MajorTypeAB.KMajor, MajorTypeAB.KMajor, False, torch.bfloat16, KernelType.Kernel1D2D)
    return a_fp8, b_fp8, c, d, ref_d

# This function uses generate_m_grouped_masked and tests m_grouped_fp8_gemm_nt_masked
def test_func_masked(num_groups, expected_m_per_group, k, n) -> None:
    [(kerneltype, max_m)] = enumerate_m_grouped_masked()
    a_fp8, b_fp8, masked_m, d, ref_d = generate_m_grouped_masked(
        num_groups, max_m, expected_m_per_group, n, k
    )
    deep_gemm.m_grouped_fp8_gemm_nt_masked(
        a_fp8,
        b_fp8,
        d,
        masked_m,
        expected_m_per_group,
        disable_ue8m0_cast=True,
    )

# This function uses generate_m_grouped_contiguous and tests m_grouped_fp8_gemm_nt_contiguous
def test_func_contiguous(num_groups, expected_m_per_group, k, n) -> None:
    [(kerneltype, major_a, major_b)] = enumerate_m_grouped_contiguous()
    m, a_fp8, b_fp8, m_indices, d, ref_d = generate_m_grouped_contiguous(
        num_groups, expected_m_per_group, n, k, major_a, major_b
    )
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        a_fp8, b_fp8, d, m_indices, disable_ue8m0_cast=True
    )

def bench_masked(num_groups, expected_m_per_group, k, n) -> float:
    [(kerneltype, max_m)] = enumerate_m_grouped_masked()
    a_fp8, b_fp8, masked_m, d, ref_d = generate_m_grouped_masked(
        num_groups, max_m, expected_m_per_group, n, k
    )
    def test_func():
        deep_gemm.m_grouped_fp8_gemm_nt_masked(
            a_fp8,
            b_fp8,
            d,
            masked_m,
            expected_m_per_group,
            disable_ue8m0_cast=True,
        )
    return bench_kineto(test_func, "fp8_gemm", suppress_kineto_output=True)

def bench_contiguous(num_groups, expected_m_per_group, k, n) -> float:
    [(kerneltype, major_a, major_b)] = enumerate_m_grouped_contiguous()
    m, a_fp8, b_fp8, m_indices, d, ref_d = generate_m_grouped_contiguous(
        num_groups, expected_m_per_group, n, k, major_a, major_b
    )
    def test_func():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            a_fp8, b_fp8, d, m_indices, disable_ue8m0_cast=True
        )
    return bench_kineto(test_func, "fp8_gemm", suppress_kineto_output=True)
