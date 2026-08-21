"""Fused MoE entry point: plan permute, two GEMMs, activation, reduce.

Drop-in replacement for
:func:`mstar.model.components.moe.dispatch_experts_fused` for bf16 /
fp16 routed-expert dispatch (:func:`fused_experts`, which also covers the
packed-W4A16 weights) and for GLM-5.2's fp8-resident experts
(:func:`fused_experts_fp8`). The shared expert (when present) stays
outside and the router stays outside — these functions handle only the
gate_up GEMM, SwiGLU, down GEMM, and weighted sum over top-k.
"""
from __future__ import annotations

import torch
import triton.language as tl

from mstar.utils.fused_moe.align import moe_align_block_size
from mstar.utils.fused_moe.kernels import (
    FP8_DTYPE,
    act_and_mul_triton,
    get_default_config,
    invoke_fused_moe_kernel,
    invoke_fused_moe_kernel_fp8_w8a8,
    invoke_fused_moe_kernel_w4a16,
    moe_sum_reduce_triton,
    per_token_group_quant_fp8,
)


def _tl_compute_type(dtype: torch.dtype) -> tl.dtype:
    if dtype == torch.bfloat16:
        return tl.bfloat16
    if dtype == torch.float16:
        return tl.float16
    raise ValueError(f"fused_experts: unsupported dtype {dtype}; use bf16 or fp16")


def fused_experts(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    reduce_results: bool = True,
    w1_scale: torch.Tensor | None = None,
    w2_scale: torch.Tensor | None = None,
    w1_zp: torch.Tensor | None = None,
    w2_zp: torch.Tensor | None = None,
    group_size: int | None = None,
    pack_factor: int | None = None,
) -> torch.Tensor:
    """Grouped-GEMM Triton MoE dispatch.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Shape ``(tokens, hidden)``, bf16/fp16, contiguous.
    w1 : torch.Tensor
        Fused gate+up projection weights, shape
        ``(num_experts, 2 * moe_intermediate_size, hidden)``.  Matches the
        ``experts.gate_up_proj`` parameter the WeightConverter already
        produces in :mod:`mstar.model.qwen3_omni.qwen3_omni_model`.
        Packed int32 on the W4A16 path.
    w2 : torch.Tensor
        Down projection weights, shape
        ``(num_experts, hidden, moe_intermediate_size)``.  Matches
        ``experts.down_proj``. Packed int32 on the W4A16 path.
    topk_weights : torch.Tensor
        ``(tokens, top_k)``, routing probabilities (possibly
        renormalized).  Dtype matches ``hidden_states``.
    topk_ids : torch.Tensor
        ``(tokens, top_k)`` int; will be cast to int32 if not already.
    activation : str
        ``"silu"`` (default) or ``"gelu"``.  Qwen3-Omni always uses silu.
    reduce_results : bool
        If True (default), sum-reduce over the top-k dimension and return
        ``(tokens, hidden)``. If False, skip the sum-reduce and return
        ``(tokens, top_k, hidden)`` — the caller is responsible for the
        reduce (e.g. after an all-reduce for TP).
    w1_scale, w2_scale : torch.Tensor | None
        W4A16 group scales; both ``None`` keeps the historical bf16 path.
    w1_zp, w2_zp : torch.Tensor | None
        Optional asymmetric zero points (unused for Kimi's symmetric INT4).
    group_size, pack_factor : int | None
        Required on the W4A16 path.

    Returns
    -------
    torch.Tensor
        Shape ``(tokens, hidden)`` if ``reduce_results`` else
        ``(tokens, top_k, hidden)``.
    """
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert hidden_states.dim() == 2
    assert topk_weights.shape == topk_ids.shape
    # Only weights are quantized; activations stay bf16/fp16.
    assert hidden_states.dtype in (torch.bfloat16, torch.float16)

    quantized = w1_scale is not None
    num_tokens, hidden = hidden_states.shape
    if quantized:
        assert pack_factor is not None and group_size is not None, (
            "W4A16 path requires pack_factor and group_size"
        )
        assert w2_scale is not None, "W4A16 path requires both w1_scale and w2_scale"
        assert w1.dtype == torch.int32 and w2.dtype == torch.int32, (
            "W4A16 path expects packed int32 weights"
        )
        assert w1.is_contiguous(), "w1 (packed) must be contiguous"
        assert w2.is_contiguous(), "w2 (packed) must be contiguous"
        E, two_inter, k1_packed = w1.shape
        assert k1_packed == hidden // pack_factor, (
            f"w1 packed last dim {k1_packed} != hidden//pack_factor {hidden // pack_factor}"
        )
        _, w2_hidden, k2_packed = w2.shape
        assert w2_hidden == hidden, f"w2 dim[1] {w2_hidden} != hidden {hidden}"
        inter = two_inter // 2
        assert k2_packed == inter // pack_factor, (
            f"w2 packed last dim {k2_packed} != inter//pack_factor {inter // pack_factor}"
        )
    else:
        assert w1.is_contiguous(), "w1 must be contiguous"
        assert w2.is_contiguous(), "w2 must be contiguous"
        E, two_inter, k_in = w1.shape
        assert k_in == hidden, f"w1 last dim {k_in} != hidden {hidden}"
        _, w2_hidden, inter = w2.shape
        assert w2_hidden == hidden, f"w2 dim[1] {w2_hidden} != hidden {hidden}"
        assert two_inter == 2 * inter, f"w1 dim[1] {two_inter} != 2 * w2 dim[2] {2 * inter}"

    top_k = topk_ids.shape[1]
    # moe_align_block_size expects int32; torch.topk returns int64.
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.contiguous()

    config = get_default_config(
        M=num_tokens, E=E, N=two_inter, K=hidden, top_k=top_k, group_size=group_size,
    )
    compute_type = _tl_compute_type(hidden_states.dtype)

    # 1. Token permute + per-expert block alignment.
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], E)

    # 2. Scratch buffers (all sized from static inputs -- no data-dependent shapes).
    m_topk = num_tokens * top_k
    cache1 = torch.empty(
        (m_topk, two_inter),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    cache2 = torch.empty(
        (m_topk, inter),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    cache3 = torch.empty(
        (num_tokens, top_k, hidden),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    # 3. Gate+up GEMM: cache1[slot] = hidden[slot // top_k] @ w1[expert].T
    if quantized:
        invoke_fused_moe_kernel_w4a16(
            A=hidden_states,
            B_packed=w1,
            C=cache1,
            B_scale=w1_scale,
            B_zp=w1_zp,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=top_k,
            config=config,
            compute_type=compute_type,
            K=hidden,
            pack_factor=pack_factor,
            group_size=group_size,
        )
    else:
        invoke_fused_moe_kernel(
            A=hidden_states,
            B=w1,
            C=cache1,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=False,
            top_k=top_k,
            config=config,
            compute_type=compute_type,
        )

    # 4. SwiGLU: cache2[slot] = silu(gate) * up
    act_and_mul_triton(cache1, cache2, activation=activation)

    # 5. Down GEMM (weighted): cache3[slot] = topk_weight[slot] * (cache2[slot] @ w2[expert].T)
    # top_k=1 for this GEMM so the kernel's offs_token // top_k is identity
    # -- it reads cache2 rows directly instead of the (slot // top_k)-th source row.
    if quantized:
        invoke_fused_moe_kernel_w4a16(
            A=cache2,
            B_packed=w2,
            C=cache3.view(m_topk, hidden),
            B_scale=w2_scale,
            B_zp=w2_zp,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=True,
            top_k=1,
            config=config,
            compute_type=compute_type,
            K=inter,
            pack_factor=pack_factor,
            group_size=group_size,
        )
    else:
        invoke_fused_moe_kernel(
            A=cache2,
            B=w2,
            C=cache3.view(m_topk, hidden),
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            sorted_token_ids=sorted_token_ids,
            expert_ids=expert_ids,
            num_tokens_post_padded=num_tokens_post_padded,
            mul_routed_weight=True,
            top_k=1,
            config=config,
            compute_type=compute_type,
        )

    # 6. Sum over the top-k slots -> (tokens, hidden).
    if not reduce_results:
        return cache3
    output = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(cache3, output, routed_scaling_factor=1.0)
    return output


def fused_experts_fp8(
    hidden_states: torch.Tensor,
    w1_fp8: torch.Tensor,
    w2_fp8: torch.Tensor,
    w1_scale_inv: torch.Tensor,
    w2_scale_inv: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
    activation: str = "silu",
    reduce_results: bool = True,
) -> torch.Tensor:
    """Grouped-GEMM MoE dispatch for block-scaled FP8 experts (W8A8).

    FP8 twin of :func:`fused_experts` for GLM-5.2's fp8-resident routed
    experts: weights stay e4m3 (the uint8 containers are re-viewed here) and
    activations are quantized per-token per-``block_size[1]``-group on the
    fly, so both GEMMs run fp8 x fp8 with fp32 accumulation.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Shape ``(tokens, hidden)``, bf16/fp16, contiguous.
    w1_fp8 : torch.Tensor
        Fused gate+up weights, shape ``(num_experts, 2 * inter, hidden)``,
        e4m3 or its uint8 byte view -- pass ``experts.gate_up_proj_fp8``
        from :class:`~mstar.model.glm52.components.moe.Glm52SparseMoeBlock`
        directly.
    w2_fp8 : torch.Tensor
        Down weights, shape ``(num_experts, hidden, inter)`` --
        ``experts.down_proj_fp8``.
    w1_scale_inv, w2_scale_inv : torch.Tensor
        fp32 block scales, shape ``(E, ceil(N/block_n), ceil(K/block_k))``
        per weight, in the checkpoint's multiply-back convention
        (dequant = w * scale) -- ``experts.*_scale_inv``.
    topk_weights, topk_ids : torch.Tensor
        Router output, shapes ``(tokens, top_k)``; weights in
        ``hidden_states.dtype``.
    block_size : tuple[int, int]
        ``(block_n, block_k)`` weight scale blocks.  ``block_k`` doubles as
        the activation quant group and the kernel's ``BLOCK_SIZE_K``, so
        both GEMM contraction dims must be multiples of it.
    activation, reduce_results
        As in :func:`fused_experts`.

    Returns
    -------
    torch.Tensor
        Shape ``(tokens, hidden)`` in ``hidden_states.dtype`` if
        ``reduce_results`` else ``(tokens, top_k, hidden)``.
    """
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert hidden_states.dim() == 2
    assert topk_weights.shape == topk_ids.shape
    assert hidden_states.dtype in (torch.bfloat16, torch.float16)

    block_n, block_k = block_size
    num_tokens, hidden = hidden_states.shape
    w1 = w1_fp8.view(FP8_DTYPE) if w1_fp8.dtype == torch.uint8 else w1_fp8
    w2 = w2_fp8.view(FP8_DTYPE) if w2_fp8.dtype == torch.uint8 else w2_fp8
    assert w1.dtype == FP8_DTYPE and w2.dtype == FP8_DTYPE, "weights must be e4m3 or its uint8 view"
    assert w1.is_contiguous(), "w1 must be contiguous"
    assert w2.is_contiguous(), "w2 must be contiguous"
    E, two_inter, k_in = w1.shape
    assert k_in == hidden, f"w1 last dim {k_in} != hidden {hidden}"
    _, w2_hidden, inter = w2.shape
    assert w2_hidden == hidden, f"w2 dim[1] {w2_hidden} != hidden {hidden}"
    assert two_inter == 2 * inter, f"w1 dim[1] {two_inter} != 2 * w2 dim[2] {2 * inter}"
    # Both contraction dims are activation-quantized in whole K-groups.
    assert hidden % block_k == 0 and inter % block_k == 0, (
        f"hidden {hidden} and inter {inter} must be multiples of block_k {block_k}"
    )
    expect1 = (E, -(-two_inter // block_n), hidden // block_k)
    expect2 = (E, -(-hidden // block_n), inter // block_k)
    assert tuple(w1_scale_inv.shape) == expect1, f"w1_scale_inv {tuple(w1_scale_inv.shape)} != {expect1}"
    assert tuple(w2_scale_inv.shape) == expect2, f"w2_scale_inv {tuple(w2_scale_inv.shape)} != {expect2}"

    top_k = topk_ids.shape[1]
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.contiguous()

    config = get_default_config(M=num_tokens, E=E, N=two_inter, K=hidden, top_k=top_k)
    # The per-K-tile rescale in the kernel is exact only when each K tile is
    # exactly one quant group, so the shape-picked BLOCK_SIZE_K is overridden.
    config["BLOCK_SIZE_K"] = block_k
    compute_type = _tl_compute_type(hidden_states.dtype)

    # 1. Token permute + per-expert block alignment.
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(topk_ids, config["BLOCK_SIZE_M"], E)

    # 2. Scratch buffers (all sized from static inputs -- no data-dependent shapes).
    m_topk = num_tokens * top_k
    cache1 = torch.empty(
        (m_topk, two_inter),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    cache2 = torch.empty(
        (m_topk, inter),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    cache3 = torch.empty(
        (num_tokens, top_k, hidden),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    # 3. Gate+up GEMM on quantized activations.
    a_q, a_s = per_token_group_quant_fp8(hidden_states, block_k)
    invoke_fused_moe_kernel_fp8_w8a8(
        A=a_q,
        B=w1,
        C=cache1,
        A_scale=a_s,
        B_scale=w1_scale_inv,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=False,
        top_k=top_k,
        config=config,
        compute_type=compute_type,
        block_shape=block_size,
    )

    # 4. SwiGLU in bf16, then requantize the intermediate for GEMM-2.
    act_and_mul_triton(cache1, cache2, activation=activation)
    a2_q, a2_s = per_token_group_quant_fp8(cache2, block_k)

    # 5. Down GEMM (weighted); top_k=1 exactly as in fused_experts so the
    # kernel reads intermediate rows (and their scales) by slot index.
    invoke_fused_moe_kernel_fp8_w8a8(
        A=a2_q,
        B=w2,
        C=cache3.view(m_topk, hidden),
        A_scale=a2_s,
        B_scale=w2_scale_inv,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=True,
        top_k=1,
        config=config,
        compute_type=compute_type,
        block_shape=block_size,
    )

    # 6. Sum over the top-k slots -> (tokens, hidden).
    if not reduce_results:
        return cache3
    output = torch.empty_like(hidden_states)
    moe_sum_reduce_triton(cache3, output, routed_scaling_factor=1.0)
    return output
