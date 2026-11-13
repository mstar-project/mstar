# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's ``model_executor/layers/quantization/awq_triton.py`` and the int4 (``use_int4_w4a16``) branch
# of SGLang's / vLLM's ``fused_moe_kernel_gptq_awq`` (both Apache-2.0, Copyright the vLLM and SGLang authors): the
# K-split dequant GEMM, reading the packed weight byte per K pair. Changed here: the weight is the compressed-tensors
# layout (``[N, K/2]`` bytes, low nibble first) rather than AWQ's interleaved ``[K, N/8]``, scales and packed zero
# points are ``[N, K/g]`` / ``[N/8, K/g]``, and an e2m1 + e8m0 (MXFP4) decode sits beside the int4 one.
"""The CUDA kernel of ``mstar::w4a16_linear`` (``model.components.int4_formats``): ``x @ W.T`` for a 4-bit ``W``.

Up to ``DEQUANT_M`` tokens it is one Triton GEMM that dequantizes each ``[BLOCK_K, BLOCK_N]`` weight tile in
registers to ``x``'s dtype (exact: an int4 times its bf16/fp16 scale, or an e2m1 times a power of two, rounded once)
and accumulates in fp32, split along K when the output tiles alone can't fill the GPU. From ``DEQUANT_M`` tokens up,
where the GEMM is compute-bound, the weight is dequantized once into a bf16 scratch and cuBLAS runs it (vLLM's AWQ
does the same past 256 tokens; the threshold is a hypothesis until ``bench_linears`` sweeps it).
"""

from __future__ import annotations

from functools import lru_cache

import torch
import triton
import triton.language as tl

from mstar.model.components.int4_formats import KIND_E2M1, KIND_INT4_ZP, w4a16_linear

DEQUANT_M = 256


@triton.jit
def _decode_tile(b, k, offs_n, mask, s_ptr, z_ptr, stride_sn, stride_sk, stride_zn, stride_zk,
                 KIND: tl.constexpr, GROUP: tl.constexpr):
    """The fp32 values of a ``[BLOCK_K, BLOCK_N]`` tile of packed bytes ``b`` at rows ``k`` (K indices)."""
    code = ((b >> ((k[:, None] % 2) * 4)) & 0xF).to(tl.int32)
    group = k[:, None] // GROUP
    if KIND == 2:
        sign = tl.where((code & 8) != 0, -1.0, 1.0)
        e = (code >> 1) & 3
        m = (code & 1).to(tl.float32)
        mag = tl.where(e == 0, 0.5 * m, (1.0 + 0.5 * m) * tl.exp2((e - 1).to(tl.float32)))
        s = tl.load(s_ptr + offs_n[None, :] * stride_sn + group * stride_sk, mask=mask, other=127)
        return sign * mag * tl.exp2(s.to(tl.float32) - 127.0)
    s = tl.load(s_ptr + offs_n[None, :] * stride_sn + group * stride_sk, mask=mask, other=0.0).to(tl.float32)
    if KIND == 1:
        zw = tl.load(z_ptr + (offs_n[None, :] // 8) * stride_zn + group * stride_zk, mask=mask, other=0)
        z = (zw >> ((offs_n[None, :] % 8) * 4)) & 0xF
    else:
        z = 8
    return (code - z).to(tl.float32) * s


@triton.jit
def _w4a16_gemm_kernel(
    x_ptr, w_ptr, s_ptr, z_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk, stride_wn, stride_wk, stride_sn, stride_sk, stride_zn, stride_zk,
    stride_os, stride_om, stride_on,
    KIND: tl.constexpr, GROUP: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # split ``pid_k`` takes K tiles pid_k, pid_k + SPLIT_K, ...
    for k0 in range(0, K, BLOCK_K * SPLIT_K):
        k = k0 + pid_k * BLOCK_K + offs_k
        mask_k = k < K
        a = tl.load(x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        wmask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(w_ptr + offs_n[None, :] * stride_wn + (k[:, None] // 2) * stride_wk, mask=wmask, other=0)
        w = _decode_tile(b, k, offs_n, wmask, s_ptr, z_ptr, stride_sn, stride_sk, stride_zn, stride_zk, KIND, GROUP)
        acc += tl.dot(a, w.to(a.dtype))
    out = out_ptr + pid_k * stride_os + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out, acc.to(out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def _w4a16_dequant_kernel(
    w_ptr, s_ptr, z_ptr, out_ptr, N, K,
    stride_wn, stride_wk, stride_sn, stride_sk, stride_zn, stride_zk, stride_on, stride_ok,
    KIND: tl.constexpr, GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (k[:, None] < K) & (offs_n[None, :] < N)
    b = tl.load(w_ptr + offs_n[None, :] * stride_wn + (k[:, None] // 2) * stride_wk, mask=mask, other=0)
    w = _decode_tile(b, k, offs_n, mask, s_ptr, z_ptr, stride_sn, stride_sk, stride_zn, stride_zk, KIND, GROUP)
    tl.store(out_ptr + offs_n[None, :] * stride_on + k[:, None] * stride_ok, w.to(out_ptr.dtype.element_ty),
             mask=mask)


@lru_cache(maxsize=None)
def _sm_count(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def _config(m: int, n: int, k: int, group: int, sms: int) -> tuple[int, int, int, int]:
    """``(BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K)``: the smallest M tile that holds the batch, and a K split that
    brings the CTA count to about two waves (hypothesis: decode is weight-bandwidth-bound, so occupancy is what
    counts; ``bench_linears`` is where it gets tuned)."""
    block_m = 16 if m <= 16 else 32 if m <= 32 else 64
    block_n = 64 if m <= 32 else 128
    block_k = 128 if group % 128 == 0 or group >= 128 else 64
    tiles = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    k_tiles = triton.cdiv(k, block_k)
    split = max(1, min(8, k_tiles, (2 * sms) // max(tiles, 1)))
    return block_m, block_n, block_k, split


def _strides(t: torch.Tensor | None) -> tuple[int, int]:
    return (0, 0) if t is None else (t.stride(0), t.stride(1))


def dequantize(weight: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor | None, group: int, kind: int,
               dtype: torch.dtype) -> torch.Tensor:
    """The ``[N, K]`` weight in ``dtype`` (the large-M path's scratch, and tests)."""
    n, k = weight.shape[0], weight.shape[1] * 2
    out = torch.empty((n, k), dtype=dtype, device=weight.device)
    bn, bk = 64, 128
    _w4a16_dequant_kernel[(triton.cdiv(n, bn), triton.cdiv(k, bk))](
        weight, scale, zero_point if zero_point is not None else scale, out, n, k,
        weight.stride(0), weight.stride(1), *_strides(scale), *_strides(zero_point), out.stride(0), out.stride(1),
        KIND=kind, GROUP=group, BLOCK_N=bn, BLOCK_K=bk,
    )
    return out


def gemm(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor | None, group: int,
         kind: int) -> torch.Tensor:
    m, k = x.shape
    n = weight.shape[0]
    assert weight.shape[1] * 2 == k and weight.dtype == torch.uint8, (tuple(weight.shape), k, weight.dtype)
    assert kind != KIND_INT4_ZP or zero_point is not None
    if m >= DEQUANT_M:
        return torch.nn.functional.linear(x, dequantize(weight, scale, zero_point, group, kind, x.dtype))
    bm, bn, bk, split = _config(m, n, k, group, _sm_count(x.device.index or 0))
    out = torch.empty((split, m, n), dtype=torch.float32 if split > 1 else x.dtype, device=x.device)
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn), split)
    _w4a16_gemm_kernel[grid](
        x, weight, scale, zero_point if zero_point is not None else scale, out, m, n, k,
        x.stride(0), x.stride(1), weight.stride(0), weight.stride(1), *_strides(scale), *_strides(zero_point),
        out.stride(0), out.stride(1), out.stride(2),
        KIND=kind, GROUP=group, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split, GROUP_M=8,
        num_warps=4, num_stages=3 if kind != KIND_E2M1 else 2,
    )
    return out[0] if split == 1 else out.sum(0).to(x.dtype)


@w4a16_linear.register_kernel("cuda")
def _(x, weight, scale, zero_point, group, kind):
    return gemm(x, weight, scale, zero_point, group, kind)
