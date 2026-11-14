# SPDX-License-Identifier: Apache-2.0
# The int4 decode follows the int4 (``use_int4_w4a16``) branch of SGLang's / vLLM's ``fused_moe_kernel_gptq_awq``
# (Apache-2.0, Copyright the SGLang and vLLM authors), on the grouped GEMM of ``kernels.fused_moe_kernel``. Changed:
# the compressed-tensors layout (``[E, N, K/8]`` int32 words, scales ``[E, N, K/g]``, packed zero points
# ``[E, N/8, K/g]``) read in its own orientation, weight rows on the MMA's M side, one dot per scale group, and an MXFP4
# decode beside the int4 one.
"""The grouped GEMM over routed experts held as 4-bit weights (``w4a16``, ``mxfp4``).

The plan is :func:`runner.fused_experts`'s (token permute and per-expert block alignment); the GEMM differs: it loads
an expert's ``[BLOCK_N, BLOCK_K / 8]`` words along K, decodes them in registers (``_codes`` / ``_values``) and runs
``W_tile @ A_tile.T`` with the expert's rows on the MMA's M side and the block's tokens on its N side, one dot per
scale group with the scale applied to the partial. The weights stay at 4.25-4.5 bits in memory.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from mstar.utils.fused_moe.kernels import _grid_rows, get_default_config

# the weight decodes the kernel's ``KIND`` selects
KIND_INT4_SYM, KIND_INT4_ZP, KIND_E2M1 = 0, 1, 2


@triton.jit
def _codes(words, ROWS: tl.constexpr, COLS: tl.constexpr):
    """int32 words ``[ROWS, COLS // 8]`` -> their 4-bit codes ``[ROWS, COLS]``: word i's nibble j is column 8i + j."""
    shifts = tl.arange(0, 8) * 4
    return tl.reshape((words[:, :, None] >> shifts[None, None, :]) & 0xF, (ROWS, COLS))


@triton.jit
def _e2m1(codes):
    """fp32 values of e2m1 codes: magnitude e:m (3 bits) as an fp32 exponent / mantissa, (e + 126) << 23 | m << 22,
    exact for e >= 1; e = 0 is 0 or 0.5."""
    v = ((((codes & 7) << 22) + (126 << 23)) | ((codes & 8) << 28)).to(tl.float32, bitcast=True)
    small = (codes & 1).to(tl.float32) * 0.5
    small = tl.where((codes & 8) != 0, -small, small)
    return tl.where((codes & 6) == 0, small, v)


@triton.jit
def _values(codes, zf, KIND: tl.constexpr):
    """fp32 values of 4-bit codes: an e2m1 code's value, or an int4 code minus its zero point, given as
    ``zf = 2**23 + zero`` (a float or a tensor broadcasting against ``codes``)."""
    if KIND == 2:
        return _e2m1(codes)
    return (codes | 0x4B000000).to(tl.float32, bitcast=True) - zf


@triton.jit
def _group_scale(s_ptr, n, live, g, stride_sn, KIND: tl.constexpr):
    """Rows ``n``'s fp32 scale at group ``g`` (shapes broadcast): e8m0 for MXFP4, the checkpoint's float for int4."""
    if KIND == 2:
        return tl.exp2(tl.load(s_ptr + n * stride_sn + g, mask=live, other=127).to(tl.float32) - 127.0)
    return tl.load(s_ptr + n * stride_sn + g, mask=live, other=0.0).to(tl.float32)


@triton.jit
def _group_zero(z_ptr, n, live, g, stride_zn):
    """Rows ``n``'s zero point at group ``g`` as ``2**23 + zero`` (fp32), from the words packing 8 rows each."""
    zw = tl.load(z_ptr + (n // 8) * stride_zn + g, mask=live, other=0)
    return 8388608.0 + ((zw >> ((n % 8) * 4)) & 0xF).to(tl.float32)


@triton.jit
def fused_moe_w4a16_kernel(
    a_ptr, b_ptr, s_ptr, z_ptr, c_ptr, topk_weights_ptr, sorted_token_ids_ptr, expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N, K, EM, num_valid_tokens,
    stride_am, stride_be, stride_bn, stride_se, stride_sn, stride_ze, stride_zn, stride_cm,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, SB: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr, MUL_ROUTED_WEIGHT: tl.constexpr, top_k: tl.constexpr,
    compute_type: tl.constexpr, KIND: tl.constexpr, GROUP: tl.constexpr, STAGES: tl.constexpr,
):
    """One ``[BLOCK_SIZE_M, BLOCK_SIZE_N]`` tile of ``C`` (``(M * top_k, N)``) from expert-aligned rows of ``A`` and
    the expert's words ``B`` (``(E, N, K/8)`` int32); see ``kernels.fused_moe_kernel`` for the plan. Steps of
    BLOCK_SIZE_K columns are BLOCK_SIZE_K / SB dots of one scale group each (SB = min(BLOCK_SIZE_K, GROUP))."""
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

    n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    live_n = n < N
    w_rows = b_ptr + off_experts * stride_be + n[:, None].to(tl.int64) * stride_bn
    s_base = s_ptr + off_experts * stride_se
    z_base = z_ptr + off_experts * stride_ze
    # offs_token // top_k recovers the source row in A for (token, slot) pairs; the down GEMM passes top_k=1
    a_rows = a_ptr + (offs_token // top_k)[:, None] * stride_am
    sub = tl.arange(0, SB)

    acc = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_M), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_SIZE_K, num_stages=STAGES):
        for j in tl.static_range(BLOCK_SIZE_K // SB):
            k = k0 + j * SB
            live_k = k < K
            words = tl.load(w_rows + (k // 8 + tl.arange(0, SB // 8))[None, :], mask=live_n[:, None] & live_k,
                            other=0)
            scale = _group_scale(s_base, n, live_n & live_k, k // GROUP, stride_sn, KIND)
            if KIND == 1:
                zf = _group_zero(z_base, n, live_n & live_k, k // GROUP, stride_zn)
                v = _values(_codes(words, BLOCK_SIZE_N, SB), zf[:, None], KIND)
            else:
                v = _values(_codes(words, BLOCK_SIZE_N, SB), 8388616.0, KIND)
            a = tl.load(a_rows + (k + sub)[None, :], mask=token_mask[:, None] & live_k, other=0.0)
            acc += tl.dot(v.to(compute_type), tl.trans(a)) * scale[:, None]

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        acc = acc * moe_weight[None, :]
    c_ptrs = c_ptr + stride_cm * offs_token[None, :] + n[:, None]
    tl.store(c_ptrs, acc.to(compute_type), mask=token_mask[None, :] & live_n[:, None])


def _launch(a, b, s, z, c, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_ids,
            mul_routed_weight, top_k, config, compute_type, kind, group):
    words = b.view(torch.int32) if b.dtype == torch.uint8 else b
    n, k = words.shape[1], words.shape[2] * 8
    assert a.stride(1) == 1 and c.stride(-1) == 1 and words.is_contiguous()
    s = s.contiguous()
    z = z.contiguous() if z is not None else None
    bk = config["BLOCK_SIZE_K"]
    em = _grid_rows(sorted_token_ids, topk_ids, config["BLOCK_SIZE_M"])
    grid = (triton.cdiv(em, config["BLOCK_SIZE_M"]) * triton.cdiv(n, config["BLOCK_SIZE_N"]),)
    fused_moe_w4a16_kernel[grid](
        a, words, s, z if z is not None else s, c, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
        n, k, em, topk_ids.numel(),
        a.stride(0), words.stride(0), words.stride(1), s.stride(0), s.stride(1),
        z.stride(0) if z is not None else 0, z.stride(1) if z is not None else 0, c.stride(-2),
        BLOCK_SIZE_M=config["BLOCK_SIZE_M"], BLOCK_SIZE_N=config["BLOCK_SIZE_N"], BLOCK_SIZE_K=bk,
        SB=min(bk, group), GROUP_SIZE_M=config["GROUP_SIZE_M"], MUL_ROUTED_WEIGHT=mul_routed_weight, top_k=top_k,
        compute_type=compute_type, KIND=kind, GROUP=group, STAGES=config["num_stages"],
        num_warps=config["num_warps"],
    )


def moe_config(num_tokens: int, E: int, N: int, K: int, top_k: int, group: int) -> dict:
    """The fused MoE's shape-picked BLOCK_SIZE_M (both GEMMs read one alignment) with 4-bit tiles: 64 weight rows,
    K steps of 128 (or the group, when it is larger and 128 doesn't divide it). A hypothesis until bench_experts
    sweeps it."""
    config = dict(get_default_config(M=num_tokens, E=E, N=N, K=K, top_k=top_k))
    bk = 128 if (128 % group == 0 or group % 128 == 0) else group
    config.update(BLOCK_SIZE_N=64, BLOCK_SIZE_K=bk, num_stages=3, num_warps=4)
    return config


def _compute_type(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    return tl.float32
