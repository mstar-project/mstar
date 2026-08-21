"""Triton kernels for fused MoE dispatch.

Ported from sglang's ``fused_moe_triton_kernels.py`` and
``fused_moe_triton_config.py`` (Apache-2.0).  The int8 quantization
branches (int8_w8a8, int8_w8a16) and the TMA / swap_ab / fused-all-reduce /
expert-parallel paths are stripped; the int4_w4a16 branch was ported back
for Kimi's packed experts and the block-scale fp8_w8a8 branch for
GLM-5.2's fp8-resident experts.

The kernels mirror sglang's layout:

* :func:`fused_moe_kernel` -- grouped GEMM; the same kernel is used for
  both the gate+up GEMM and the down GEMM.
* :func:`fused_moe_kernel_w4a16` / :func:`fused_moe_kernel_fp8_w8a8` --
  the same grouped GEMM with in-kernel dequant for packed INT4 /
  block-scaled e4m3 weights.
* :func:`per_token_group_quant_fp8_kernel` -- per-token per-group e4m3
  activation quant feeding the W8A8 GEMMs.
* :func:`act_and_mul_kernel` -- per-slot SwiGLU activation on the
  ``(M*topk, 2*inter)`` intermediate.
* :func:`moe_sum_reduce_kernel` -- weight-free sum over the top-k
  dimension of the ``(M, topk, hidden)`` down-GEMM output.

The Python wrappers :func:`invoke_fused_moe_kernel`,
:func:`act_and_mul_triton`, :func:`moe_sum_reduce_triton` set up launch
grids and keep the Triton-specific boilerplate out of the runner.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import triton
import triton.language as tl

# Same value as mstar.model.glm52.quantization.FP8_DTYPE; redeclared here so
# utils/ keeps no dependency on model/.
FP8_DTYPE = torch.float8_e4m3fn

# ---------------------------------------------------------------------------
# Main grouped-GEMM kernel (used for both gate_up and down projections)
# ---------------------------------------------------------------------------


@triton.jit
def fused_moe_kernel(
    # Pointers
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # Strides
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Block sizes (compile-time)
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """Compute one ``[BLOCK_SIZE_M, BLOCK_SIZE_N]`` tile of the MoE output.

    ``A`` holds the input rows (``M`` rows, ``K`` cols); at the first GEMM
    ``A`` is the hidden states tensor and at the second GEMM it is the
    SwiGLU intermediate.  ``B`` is the stacked expert weight tensor of
    shape ``(E, N, K)``.  ``C`` is the output cache of shape
    ``(M*topk, N)``.

    Tokens are permuted into expert-aligned blocks by
    ``moe_align_block_size`` before the launch.  ``sorted_token_ids``
    holds the permuted slot indices (< ``num_valid_tokens`` for real
    tokens, >= for padding) and ``expert_ids`` maps each ``BLOCK_SIZE_M``
    block to the expert index.
    """
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Skip blocks past the padded token count entirely.
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # offs_token // top_k recovers the source row in A for (token, slot) pairs.
    # For the second GEMM we pass top_k=1 so the divide is a no-op and the
    # kernel reads intermediate-cache rows directly.
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        if even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)
        accumulator += tl.dot(a, b)
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


@triton.jit
def fused_moe_kernel_w4a16(
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions (K is the LOGICAL contraction dim, not the packed width)
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    group_size: tl.constexpr,
    PACK_FACTOR: tl.constexpr,
    HAS_ZP: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """Fused MoE tile for packed W4A16 weights.

    ``K`` is logical; weights are int32-packed low-order-first along K. K tiles
    must span whole int32s and whole quant groups.
    """
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
    # Packed B is indexed by logical K, then shifted to the right INT4 nibble.
    b_ptrs = (
        b_ptr + off_experts * stride_be
        + (offs_k[:, None] // PACK_FACTOR) * stride_bk
        + offs_bn[None, :] * stride_bn
    )
    b_shifter = (offs_k[:, None] % PACK_FACTOR) * 4

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        # Group scales are indexed by logical K, not packed K.
        offs_ks = (offs_k[:, None] + k_start) // group_size
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[None, :] * stride_bsn + offs_ks * stride_bsk
        if even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b_packed = tl.load(b_ptrs)
            b_scale = tl.load(b_scale_ptrs).to(tl.float32)
        else:
            k_mask = offs_k[:, None] < K - k_start
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0.0,
            )
            b_packed = tl.load(b_ptrs, mask=k_mask, other=0)
            b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=1.0).to(tl.float32)
        # Mask after arithmetic shift so the top nibble is exact when bit 31 is set.
        b_nib = ((b_packed >> b_shifter) & 0xF).to(tl.float32)
        if HAS_ZP:
            # Optional asymmetric zero point; Kimi uses symmetric INT4.
            b_zp_ptrs = b_zp_ptr + off_experts * stride_bze + offs_bn[None, :] * stride_bzn + offs_ks * stride_bzk
            if even_Ks:
                b_zp = tl.load(b_zp_ptrs).to(tl.float32)
            else:
                b_zp = tl.load(b_zp_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0).to(tl.float32)
            b = ((b_nib - b_zp) * b_scale).to(compute_type)
        else:
            b = ((b_nib - 8.0) * b_scale).to(compute_type)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += (BLOCK_SIZE_K // PACK_FACTOR) * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_fp8_w8a8(
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """Fused MoE tile for block-scaled FP8 W8A8 (sglang's fp8_w8a8 branch).

    ``A`` is e4m3 with per-token per-``group_k`` fp32 scales of shape
    ``(M, K // group_k)``; ``B`` is e4m3 ``(E, N, K)`` with fp32
    ``weight_scale_inv`` blocks ``(E, ceil(N/group_n), ceil(K/group_k))``
    (dequant = value * scale).  The fp8 dot accumulates in fp32 and each K
    tile is rescaled with ``a_scale[:, None] * b_scale[None, :]`` -- exact
    only when the tile lies inside one quant group, hence the static assert.
    """
    tl.static_assert(BLOCK_SIZE_K == group_k, "K tiles must cover exactly one quant group")

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
    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    # A scales are per source row (same offs_token // top_k as the A loads);
    # B scales are per output-column group.  The K-group index advances with
    # the loop below.
    a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
    offs_bsn = offs_bn // group_n
    b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bsn * stride_bsn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_SIZE_K):
        if even_Ks:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k_start),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k_start, other=0.0)
        offs_ks = k_start // group_k
        a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask, mask=token_mask, other=0.0)
        b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)
        accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
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


def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
) -> None:
    """Launch :func:`fused_moe_kernel` with the right grid size.

    Parameters
    ----------
    A : torch.Tensor
        Input tensor, shape ``(M, K)`` (hidden states or SwiGLU intermediate).
    B : torch.Tensor
        Stacked expert weights, shape ``(E, N, K)``.
    C : torch.Tensor
        Output buffer, shape ``(M*topk, N)`` for GEMM-1 or reshape-viewed
        ``(M*topk, hidden)`` for GEMM-2.
    topk_weights, topk_ids : torch.Tensor
        Router output, shapes ``(M, top_k)``.  ``topk_ids`` must be int32.
    sorted_token_ids, expert_ids, num_tokens_post_padded : torch.Tensor
        Outputs of :func:`moe_align_block_size`.
    mul_routed_weight : bool
        If ``True``, multiply the accumulator by ``topk_weights`` before
        writing -- used for the down GEMM to fold the routing weight into
        the output rows that ``moe_sum_reduce_triton`` then sums.
    top_k : int
        ``topk`` for GEMM-1; pass ``1`` for GEMM-2 so the in-kernel
        ``offs_token // top_k`` becomes an identity.
    config : dict
        Tile sizes; output of :func:`get_default_config`.
    compute_type : triton.language.dtype
        Dtype for the accumulator store.
    """
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    def grid(META):
        return (
            triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
            * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
        )

    K = B.shape[2]
    even_Ks = (K % config["BLOCK_SIZE_K"]) == 0

    fused_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        K,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(-2),
        C.stride(-1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        even_Ks=even_Ks,
        **config,
    )


def invoke_fused_moe_kernel_w4a16(
    A: torch.Tensor,
    B_packed: torch.Tensor,
    C: torch.Tensor,
    B_scale: torch.Tensor,
    B_zp: torch.Tensor | None,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    K: int,
    pack_factor: int,
    group_size: int,
) -> None:
    """Launch the packed-W4A16 MoE kernel.

    ``K`` is logical, while ``B_packed.shape[2]`` is ``K // pack_factor``. When
    ``B_zp`` is absent, ``HAS_ZP`` gates all loads from the stand-in tensor.
    """
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    N = B_packed.shape[1]

    def grid(META):
        return (
            triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    even_Ks = (K % config["BLOCK_SIZE_K"]) == 0
    has_zp = B_zp is not None
    zp = B_zp if has_zp else B_scale  # stand-in; every zp load is gated by HAS_ZP

    fused_moe_kernel_w4a16[grid](
        A,
        B_packed,
        C,
        B_scale,
        zp,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B_packed.stride(0),
        B_packed.stride(2),
        B_packed.stride(1),
        C.stride(-2),
        C.stride(-1),
        B_scale.stride(0),
        B_scale.stride(2),
        B_scale.stride(1),
        zp.stride(0),
        zp.stride(2),
        zp.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        group_size=group_size,
        PACK_FACTOR=pack_factor,
        HAS_ZP=has_zp,
        even_Ks=even_Ks,
        **config,
    )


def invoke_fused_moe_kernel_fp8_w8a8(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: tl.dtype,
    block_shape: tuple[int, int],
) -> None:
    """Launch the block-scaled FP8 W8A8 MoE kernel.

    ``A`` and ``B`` must already be e4m3 (the runner re-views the uint8
    weight containers).  ``block_shape`` is ``(group_n, group_k)``;
    ``config["BLOCK_SIZE_K"]`` must equal ``group_k`` so the per-K-tile
    rescale is exact.
    """
    assert topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1
    assert A.dtype == FP8_DTYPE and B.dtype == FP8_DTYPE
    assert A_scale.dtype == torch.float32 and B_scale.dtype == torch.float32

    group_n, group_k = block_shape
    assert config["BLOCK_SIZE_K"] == group_k, "per-K-tile rescale needs BLOCK_SIZE_K == group_k"

    N = B.shape[1]
    K = B.shape[2]

    def grid(META):
        return (
            triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
            * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    even_Ks = (K % config["BLOCK_SIZE_K"]) == 0

    fused_moe_kernel_fp8_w8a8[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(-2),
        C.stride(-1),
        A_scale.stride(0),
        A_scale.stride(1),
        B_scale.stride(0),
        B_scale.stride(2),
        B_scale.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        top_k=top_k,
        compute_type=compute_type,
        group_n=group_n,
        group_k=group_k,
        even_Ks=even_Ks,
        **config,
    )


# ---------------------------------------------------------------------------
# Per-token-group FP8 activation quantization (for the W8A8 path)
# ---------------------------------------------------------------------------


@triton.jit
def per_token_group_quant_fp8_kernel(
    y_ptr,
    y_q_ptr,
    y_s_ptr,
    group_size,
    eps,
    fp8_min,
    fp8_max,
    BLOCK: tl.constexpr,
):
    """Quantize one contiguous ``group_size`` slice to e4m3 with an fp32 scale.

    One program per group; groups tile the rows of a contiguous 2-D tensor,
    so program ``g`` covers ``y.view(-1)[g*group_size:(g+1)*group_size]`` and
    writes scale slot ``g`` of the row-major ``(M, K // group_size)`` scales.
    """
    g_id = tl.program_id(0).to(tl.int64)
    y_ptr += g_id * group_size
    y_q_ptr += g_id * group_size
    y_s_ptr += g_id

    cols = tl.arange(0, BLOCK)
    mask = cols < group_size
    y = tl.load(y_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    # amax / e4m3-max with an eps floor so all-zero groups get a finite scale.
    y_s = tl.maximum(tl.max(tl.abs(y)), eps) / fp8_max
    y_q = tl.minimum(tl.maximum(y / y_s, fp8_min), fp8_max).to(y_q_ptr.dtype.element_ty)
    tl.store(y_q_ptr + cols, y_q, mask=mask)
    tl.store(y_s_ptr, y_s)


@torch.compiler.disable
def per_token_group_quant_fp8(
    x: torch.Tensor,
    group_size: int,
    eps: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` to e4m3, one fp32 scale per ``group_size`` K-slice per row.

    Returns ``(x_q, x_s)`` with ``x_q`` e4m3 of ``x.shape`` and ``x_s`` fp32 of
    shape ``(M, K // group_size)``; dequant is ``x_q * x_s`` (the same
    multiply-back convention as the checkpoint's ``weight_scale_inv``).

    ``torch.compiler.disable``: Inductor's (re)compile of THIS kernel dies
    with "PassManager::run failed" in Triton make_llir — in-process and
    subprocess alike (2026-08-07; the kernel's own JIT path is fine, it ran
    3,264+ tokens eager). The graph break keeps Inductor out of the launch
    while stream capture still records it, so compiled-forward capture and
    this kernel coexist. Remove when the toolchain bug is fixed upstream.
    """
    assert x.dim() == 2 and x.is_contiguous()
    assert x.shape[-1] % group_size == 0, f"last dim {x.shape[-1]} must be a multiple of group_size {group_size}"

    M, K = x.shape
    finfo = torch.finfo(FP8_DTYPE)
    x_q = torch.empty_like(x, dtype=FP8_DTYPE)
    x_s = torch.empty((M, K // group_size), dtype=torch.float32, device=x.device)

    num_groups = M * (K // group_size)
    per_token_group_quant_fp8_kernel[(num_groups,)](
        x,
        x_q,
        x_s,
        group_size,
        eps,
        finfo.min,
        finfo.max,
        BLOCK=triton.next_power_of_2(group_size),
    )
    return x_q, x_s


# ---------------------------------------------------------------------------
# Activation (SwiGLU / GeGLU) kernel
# ---------------------------------------------------------------------------


@triton.jit
def _tanh(x):
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _apply_activation(x, ACTIVATION_TYPE: tl.constexpr):
    x = x.to(tl.float32)
    if ACTIVATION_TYPE == "silu":
        return x * tl.sigmoid(x)
    elif ACTIVATION_TYPE == "gelu":
        k = 0.7978845608028654  # sqrt(2/pi)
        return 0.5 * x * (1.0 + _tanh(k * (x + 0.044715 * x * x * x)))
    else:
        tl.static_assert(False, "Unsupported activation")
        return x


@triton.jit
def act_and_mul_kernel(
    gateup_output_ptr,
    down_input_ptr,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION_TYPE: tl.constexpr,
):
    """Per-slot SwiGLU activation.

    Input ``gateup_output`` has layout ``(M*topk, 2*inter)`` with the
    gate half in columns ``[0:inter]`` and the up half in columns
    ``[inter:2*inter]``.  Writes ``act(gate) * up`` to ``down_input`` of
    shape ``(M*topk, inter)``.
    """
    in_dtype = gateup_output_ptr.dtype.element_ty
    out_dtype = down_input_ptr.dtype.element_ty

    half = hidden_size // 2
    pid = tl.program_id(0)

    gate_row = gateup_output_ptr + pid * hidden_size
    up_row = gate_row + half
    out_row = down_input_ptr + pid * half

    for start_offset in tl.range(0, half, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < half
        gate = tl.load(gate_row + offset, mask=mask)
        up = tl.load(up_row + offset, mask=mask)
        activated = _apply_activation(gate, ACTIVATION_TYPE).to(in_dtype)
        out = activated * up
        tl.store(out_row + offset, out.to(out_dtype), mask=mask)


def act_and_mul_triton(
    gateup_output: torch.Tensor,
    down_input: torch.Tensor,
    activation: str = "silu",
) -> None:
    """Wrapper launching :func:`act_and_mul_kernel` per intermediate slot."""
    assert gateup_output.is_contiguous()
    assert down_input.is_contiguous()
    assert gateup_output.shape[0] == down_input.shape[0]
    assert gateup_output.shape[1] == 2 * down_input.shape[1]

    grid = (down_input.shape[0],)
    hidden_size = gateup_output.shape[1]
    act_and_mul_kernel[grid](
        gateup_output,
        down_input,
        hidden_size,
        BLOCK_SIZE=512,
        ACTIVATION_TYPE=activation,
    )


# ---------------------------------------------------------------------------
# Sum-reduce over the top-k dimension
# ---------------------------------------------------------------------------


@triton.jit
def moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    routed_scaling_factor: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    """Sum ``input`` over its ``topk`` dim, optionally scaling the result.

    ``input`` is ``(token_num, topk_num, hidden_dim)``; ``output`` is
    ``(token_num, hidden_dim)``.  The output dtype matches ``input``.
    """
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    offs_token = token_block_id * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_dim = dim_block_id * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    mask_token = offs_token < token_num
    mask_dim = offs_dim < hidden_dim

    base_ptrs = input_ptr + offs_token[:, None] * input_stride_0 + offs_dim[None, :]
    accumulator = tl.zeros((BLOCK_M, BLOCK_DIM), dtype=tl.float32)
    for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
        tile = tl.load(
            base_ptrs + i * input_stride_1,
            mask=mask_token[:, None] & mask_dim[None, :],
            other=0.0,
        )
        accumulator += tile.to(tl.float32)
    accumulator *= routed_scaling_factor

    store_ptrs = output_ptr + offs_token[:, None] * output_stride_0 + offs_dim[None, :]
    tl.store(
        store_ptrs,
        accumulator.to(input_ptr.dtype.element_ty),
        mask=mask_token[:, None] & mask_dim[None, :],
    )


def moe_sum_reduce_triton(
    input: torch.Tensor,
    output: torch.Tensor,
    routed_scaling_factor: float = 1.0,
) -> None:
    """Launch :func:`moe_sum_reduce_kernel`."""
    assert input.is_contiguous()
    assert output.is_contiguous()

    token_num, topk_num, hidden_dim = input.shape
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    BLOCK_M = 1
    BLOCK_DIM = 2048
    NUM_STAGE = 1
    num_warps = 16

    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )
    moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        routed_scaling_factor=routed_scaling_factor,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# Block-size / warp-count configuration (bf16 only; no quant, no marlin)
# ---------------------------------------------------------------------------


def get_default_config(
    M: int, E: int, N: int, K: int, top_k: int, group_size: int | None = None,
) -> Dict[str, int]:
    """Pick Triton tile sizes based on problem shape.

    Mirrors sglang's ``get_default_config`` for the unquantized path.
    For decode batch sizes (``M`` on the order of 1--64) we always fall
    into the ``M <= E`` branch since Qwen3-Omni has ``E == 128``.

    When ``group_size`` is set for W4A16, ``BLOCK_SIZE_K`` is clamped until it is
    divisible by both INT4 pack factor and group size. ``None`` preserves the
    historical config.
    """
    if M <= E:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
    if group_size is not None:
        pack_factor = 8  # INT4: 32 // 4
        bk = config["BLOCK_SIZE_K"]
        while bk % pack_factor != 0 or bk % group_size != 0:
            bk //= 2
            if bk < pack_factor:
                raise ValueError(
                    f"cannot pick a BLOCK_SIZE_K divisible by pack_factor={pack_factor} "
                    f"and group_size={group_size}; got down to {bk}"
                )
        config["BLOCK_SIZE_K"] = bk
        assert config["BLOCK_SIZE_K"] % pack_factor == 0
        assert config["BLOCK_SIZE_K"] % group_size == 0
    return config
