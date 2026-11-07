# SPDX-License-Identifier: Apache-2.0
# Adapted from the int4 (``use_int4_w4a16``) branch of SGLang's / vLLM's ``fused_moe_kernel_gptq_awq`` (Apache-2.0,
# Copyright the SGLang and vLLM authors), on the grouped GEMM of ``kernels.fused_moe_kernel``: the weight tile is
# read a byte per K pair and dequantized in registers. Changed here: the compressed-tensors layout (``[E, N, K/2]``
# bytes, scales ``[E, N, K/g]``, packed zero points ``[E, N/8, K/g]``) and an MXFP4 decode beside the int4 one
# (``utils.w4a16._decode_tile``).
"""Routed experts held as 4-bit weights (``w4a16``, ``mxfp4``): gate/up GEMM, SwiGLU, down GEMM, top-k sum.

The plan is :func:`runner.fused_experts`'s (token permute and per-expert block alignment, the same SwiGLU and
reduce kernels); only the two GEMMs differ, dequantizing each ``[BLOCK_K, BLOCK_N]`` weight tile to the
activation's dtype in registers, so the weights stay at 4.25-4.5 bits in memory (a Kimi K3 rank's experts would
not fit in bf16) and the numerics are the bf16 kernel's on the dequantized weights.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from mstar.model.components.int4_formats import KIND_E2M1, KIND_INT4_SYM, KIND_INT4_ZP  # noqa: F401
from mstar.utils.fused_moe.align import moe_align_block_size
from mstar.utils.fused_moe.kernels import _grid_rows, act_and_mul_triton, get_default_config, moe_sum_reduce_triton
from mstar.utils.w4a16 import _decode_tile


@triton.jit
def fused_moe_w4a16_kernel(
    a_ptr, b_ptr, s_ptr, z_ptr, c_ptr, topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_ak, stride_be, stride_bn, stride_bk, stride_se, stride_sn, stride_sk,
    stride_ze, stride_zn, stride_zk, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr,
    compute_type: tl.constexpr, KIND: tl.constexpr, GROUP: tl.constexpr,
):
    """One ``[BLOCK_SIZE_M, BLOCK_SIZE_N]`` tile of ``C`` (``(M * top_k, N)``) from expert-aligned rows of ``A``
    and the expert's packed ``B`` (``(E, N, K/2)`` bytes); see ``kernels.fused_moe_kernel`` for the plan."""
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
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    b_base = b_ptr + off_experts * stride_be + offs_bn[None, :] * stride_bn
    s_base = s_ptr + off_experts * stride_se
    z_base = z_ptr + off_experts * stride_ze

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_SIZE_K):
        k = k_start + offs_k
        mask_k = k < K
        a = tl.load(a_ptrs, mask=token_mask[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_base + (k[:, None] // 2) * stride_bk, mask=mask_k[:, None], other=0)
        w = _decode_tile(b, k, offs_bn, mask_k[:, None], s_base, z_base, stride_sn, stride_sk, stride_zn,
                         stride_zk, KIND, GROUP)
        accumulator += tl.dot(a, w.to(compute_type))
        a_ptrs += BLOCK_SIZE_K * stride_ak

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    tl.store(c_ptrs, accumulator, mask=token_mask[:, None] & (offs_cn[None, :] < N))


def _launch(a, b, s, z, c, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_ids,
            mul_routed_weight, top_k, config, compute_type, kind, group):
    n, k = b.shape[1], b.shape[2] * 2
    em = _grid_rows(sorted_token_ids, topk_ids, config["BLOCK_SIZE_M"])
    grid = (triton.cdiv(em, config["BLOCK_SIZE_M"]) * triton.cdiv(n, config["BLOCK_SIZE_N"]),)
    zs = (0, 0, 0) if z is None else (z.stride(0), z.stride(1), z.stride(2))
    fused_moe_w4a16_kernel[grid](
        a, b, s, z if z is not None else s, c, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        n, k, em, topk_ids.numel(),
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), b.stride(2), s.stride(0), s.stride(1), s.stride(2),
        *zs, c.stride(-2), c.stride(-1),
        MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k, compute_type=compute_type, KIND=kind, GROUP=group,
        **config,
    )


def _compute_type(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    return tl.float32


def fused_experts_w4a16_impl(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor,
    z1: torch.Tensor | None, z2: torch.Tensor | None, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
    kind: int, group: int, activation: str = "silu", skip_invalid: bool = False,
) -> torch.Tensor:
    """``(tokens, hidden)``: the routed experts' weighted sum. ``w1`` ``(E, 2I, H/2)`` / ``w2`` ``(E, H, I/2)``
    uint8 (an int4 checkpoint's int32 words viewed as bytes), ``s*`` their group scales (bf16/fp16 for int4, uint8
    e8m0 for MXFP4), ``z*`` packed zero points for ``KIND_INT4_ZP``. With ``skip_invalid``, slots routed to the
    sentinel id ``E`` run no GEMM."""
    assert hidden_states.is_contiguous() and hidden_states.dim() == 2
    num_tokens, hidden = hidden_states.shape
    E, two_inter, half_hidden = w1.shape
    inter = two_inter // 2
    assert half_hidden * 2 == hidden and tuple(w2.shape) == (E, hidden, inter // 2), (w1.shape, w2.shape)
    top_k = topk_ids.shape[1]
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.contiguous()

    config = get_default_config(M=num_tokens, E=E, N=two_inter, K=hidden, top_k=top_k)
    # the dequant keeps more registers live than the bf16 kernel's tiles assume
    config["num_stages"] = min(config.get("num_stages", 3), 3)
    compute_type = _compute_type(hidden_states.dtype)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], E)

    m_topk = num_tokens * top_k
    device, dtype = hidden_states.device, hidden_states.dtype
    cache1 = torch.empty((m_topk, two_inter), device=device, dtype=dtype)
    cache2 = torch.empty((m_topk, inter), device=device, dtype=dtype)
    # skipped slots are never written by the down GEMM, so they must start at zero
    cache3 = (torch.zeros if skip_invalid else torch.empty)((num_tokens, top_k, hidden), device=device, dtype=dtype)

    _launch(hidden_states, w1, s1, z1, cache1, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
            topk_ids, False, top_k, config, compute_type, kind, group)
    act_and_mul_triton(cache1, cache2, activation=activation)
    _launch(cache2, w2, s2, z2, cache3.view(m_topk, hidden), topk_weights, sorted_token_ids, expert_ids,
            num_tokens_post_padded, topk_ids, True, 1, config, compute_type, kind, group)
    output = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(cache3, output, routed_scaling_factor=1.0)
    return output


@torch.library.custom_op("mstar::fused_experts_w4a16", mutates_args=())
def fused_experts_w4a16(
    hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, s1: torch.Tensor, s2: torch.Tensor,
    z1: torch.Tensor | None, z2: torch.Tensor | None, topk_weights: torch.Tensor, topk_ids: torch.Tensor,
    kind: int, group: int, activation: str, skip_invalid: bool,
) -> torch.Tensor:
    """One opaque op to dynamo (the launch path's host logic would break the graph); see the impl."""
    return fused_experts_w4a16_impl(hidden_states, w1, w2, s1, s2, z1, z2, topk_weights, topk_ids, kind, group,
                                    activation, skip_invalid)


@fused_experts_w4a16.register_fake
def _(hidden_states, w1, w2, s1, s2, z1, z2, topk_weights, topk_ids, kind, group, activation, skip_invalid):
    return torch.empty_like(hidden_states)
