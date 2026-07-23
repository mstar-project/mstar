"""Block-scaled fp8 (w8a8) grouped-GEMM MoE for the Qwen3-Omni Thinker.

Re-adds the fp8 path that this Triton kernel's sglang origin had (and which
``kernels.py`` explicitly stripped). Weights are quantized per (128, 128)
block; activations per (row, 128) group along K — the DeepSeek/sglang blockwise
w8a8 scheme. At decode the MoE is memory-bandwidth-bound on the expert weights,
so halving weight bytes (bf16 -> fp8_e4m3) is a ~2x lever on the MoE slice.

Numerically NOT identical to bf16 (fp8 quant); validate cosine >= ~0.99. Gated
by ``MSTAR_MOE_FP8`` at the call site in ``moe.py``; this module is pure kernel.
"""
from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from mstar.utils.fused_moe.align import moe_align_block_size
from mstar.utils.fused_moe.kernels import act_and_mul_triton, moe_sum_reduce_triton

_FP8 = torch.float8_e4m3fn
_FP8_MAX = 448.0
_BLK = 128  # scale block size (K and N granularity)


# --------------------------------------------------------------------------
# Quantization helpers (torch ops; cheap at decode M, run outside the graph
# hot loop for weights — weights are quantized once and cached).
# --------------------------------------------------------------------------
def per_block_cast_to_fp8_weight(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize expert weights ``(E, N, K)`` per (128,128) block.

    Returns ``(w_fp8 (E,N,K), scales (E, N//128, K//128) float32)``.
    """
    E, N, K = w.shape
    assert N % _BLK == 0 and K % _BLK == 0, f"N={N} K={K} must be /{_BLK}"
    wv = w.reshape(E, N // _BLK, _BLK, K // _BLK, _BLK).float()
    amax = wv.abs().amax(dim=(2, 4), keepdim=True).clamp_(1e-4)
    scale = amax / _FP8_MAX
    wq = (wv / scale).clamp_(-_FP8_MAX, _FP8_MAX).to(_FP8).reshape(E, N, K)
    return wq.contiguous(), scale.reshape(E, N // _BLK, K // _BLK).contiguous()


@triton.jit
def _per_token_group_quant_kernel(a_ptr, aq_ptr, s_ptr, M, K, stride_am,
                                  GROUP: tl.constexpr, NG: tl.constexpr, FP8_MAX: tl.constexpr):
    """One program per (row, 128-group): read bf16, compute group amax, write
    fp8 + fp32 scale. Replaces ~7 torch ops (0.16ms) with one launch."""
    row = tl.program_id(0)
    g = tl.program_id(1)
    offs = tl.arange(0, GROUP)
    ptr = a_ptr + row * stride_am + g * GROUP + offs
    x = tl.load(ptr).to(tl.float32)
    amax = tl.max(tl.abs(x), axis=0)
    amax = tl.maximum(amax, 1e-4)
    scale = amax / FP8_MAX
    xq = (x / scale)
    xq = tl.minimum(tl.maximum(xq, -FP8_MAX), FP8_MAX)
    tl.store(aq_ptr + row * K + g * GROUP + offs, xq.to(aq_ptr.dtype.element_ty))
    tl.store(s_ptr + row * NG + g, scale)


def per_token_group_quant_fp8(a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Quantize activations ``(M, K)`` per (row, 128-group along K).

    Returns ``(a_fp8 (M,K), scales (M, K//128) float32)``. Fused single-kernel.
    """
    M, K = a.shape
    assert K % _BLK == 0, f"K={K} must be /{_BLK}"
    NG = K // _BLK
    aq = torch.empty((M, K), device=a.device, dtype=_FP8)
    scale = torch.empty((M, NG), device=a.device, dtype=torch.float32)
    _per_token_group_quant_kernel[(M, NG)](a, aq, scale, M, K, a.stride(0),
                                           GROUP=_BLK, NG=NG, FP8_MAX=_FP8_MAX, num_warps=4)
    return aq, scale


@triton.jit
def _fused_moe_fp8_kernel(
    a_ptr, b_ptr, c_ptr,
    a_scale_ptr, b_scale_ptr,
    topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr, num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak,
    stride_be, stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_asm, stride_ask,          # a_scale: (M_rows, K//128)
    stride_bse, stride_bsn, stride_bsk,  # b_scale: (E, N//128, K//128)
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr,
    compute_type: tl.constexpr,
):
    """Block-scaled fp8 w8a8 MoE GEMM. Requires BLOCK_SIZE_K == 128 and
    BLOCK_SIZE_N <= 128 so each k-step maps to exactly one (row,K) activation
    scale and one (N,K) weight scale block."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_row = offs_token[:, None] // top_k
    a_ptrs = a_ptr + (a_row * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # scale pointers: a_scale per row, b_scale per (expert, n_block=pid_n, k_block)
    a_scale_row = (offs_token // top_k) * stride_asm
    n_block = pid_n  # BLOCK_SIZE_N==128 => pid_n indexes the 128-wide N scale block

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    num_k = tl.cdiv(K, BLOCK_SIZE_K)
    for k_idx in range(0, num_k):
        a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        a_s = tl.load(a_scale_ptr + a_scale_row + k_idx * stride_ask, mask=token_mask, other=0.0)
        b_s = tl.load(b_scale_ptr + off_experts * stride_bse + n_block * stride_bsn + k_idx * stride_bsk)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def _invoke_fp8(A, A_s, B, B_s, C, topk_weights, sorted_token_ids, expert_ids,
                num_tokens_post_padded, mul_routed_weight, top_k, config, compute_type):
    def grid(META):
        return (triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
                * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),)
    _fused_moe_fp8_kernel[grid](
        A, B, C, A_s, B_s,
        topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        B.shape[1], A.shape[1], sorted_token_ids.shape[0],
        topk_weights.numel() if mul_routed_weight else A.shape[0] * top_k,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(2), B.stride(1),
        C.stride(-2), C.stride(-1),
        A_s.stride(0), A_s.stride(1),
        B_s.stride(0), B_s.stride(1), B_s.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k, compute_type=compute_type,
        **config,
    )


def fused_experts_fp8(
    hidden_states: torch.Tensor,
    w1_fp8: torch.Tensor, w1_scale: torch.Tensor,
    w2_fp8: torch.Tensor, w2_scale: torch.Tensor,
    topk_weights: torch.Tensor, topk_ids: torch.Tensor,
    activation: str = "silu", reduce_results: bool = True,
) -> torch.Tensor:
    """Block-fp8 w8a8 grouped-GEMM MoE. Mirrors runner.fused_experts but with
    fp8 expert weights (pre-quantized + cached) and fp8-quantized activations."""
    assert hidden_states.dtype in (torch.bfloat16, torch.float16)
    num_tokens, hidden = hidden_states.shape
    E, two_inter, k_in = w1_fp8.shape
    _, w2_hidden, inter = w2_fp8.shape
    assert k_in == hidden and w2_hidden == hidden and two_inter == 2 * inter
    top_k = topk_ids.shape[1]
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.contiguous()
    compute_type = tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16

    # Small M (decode): a narrow M-tile keeps SM occupancy high without the
    # overhead a wider tile would add at this size.
    if num_tokens <= E:
        config = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
                  "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3}
    else:
        # Large M (prefill): a wider M-tile with more L2-reuse grouping
        # amortizes launch overhead better than the decode tile above.
        config = {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
                  "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 3}
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], E)

    m_topk = num_tokens * top_k
    cache1 = torch.empty((m_topk, two_inter), device=hidden_states.device, dtype=hidden_states.dtype)
    cache2 = torch.empty((m_topk, inter), device=hidden_states.device, dtype=hidden_states.dtype)
    cache3 = torch.empty((num_tokens, top_k, hidden), device=hidden_states.device, dtype=hidden_states.dtype)

    # GEMM1: quantize hidden per-token-group, fp8 grouped-GEMM against w1.
    a1_fp8, a1_scale = per_token_group_quant_fp8(hidden_states)
    _invoke_fp8(a1_fp8, a1_scale, w1_fp8, w1_scale, cache1, topk_weights, sorted_token_ids,
                expert_ids, num_tokens_post_padded, False, top_k, config, compute_type)

    act_and_mul_triton(cache1, cache2, activation=activation)

    # GEMM2: quantize SwiGLU output, fp8 grouped-GEMM against w2 (top_k=1).
    a2_fp8, a2_scale = per_token_group_quant_fp8(cache2)
    _invoke_fp8(a2_fp8, a2_scale, w2_fp8, w2_scale, cache3.view(m_topk, hidden), topk_weights,
                sorted_token_ids, expert_ids, num_tokens_post_padded, True, 1, config, compute_type)

    if not reduce_results:
        return cache3
    output = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(cache3, output, routed_scaling_factor=1.0)
    return output
