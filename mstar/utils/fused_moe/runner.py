"""Fused MoE entry point: plan permute, two GEMMs, activation, reduce.

Drop-in replacement for
:func:`mstar.model.components.moe.dispatch_experts_fused` for bf16 /
fp16 routed-expert dispatch. The shared expert (when present) stays
outside and the router stays outside — this function handles only the
gate_up GEMM, SwiGLU, down GEMM, and weighted sum over top-k.
"""
from __future__ import annotations

import torch
import triton.language as tl

from mstar.utils.fused_moe.align import moe_align_block_size
from mstar.utils.fused_moe.kernels import (
    act_and_mul_triton,
    get_default_config,
    invoke_fused_moe_kernel,
    moe_sum_reduce_triton,
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
    w2 : torch.Tensor
        Down projection weights, shape
        ``(num_experts, hidden, moe_intermediate_size)``.  Matches
        ``experts.down_proj``.
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

    Returns
    -------
    torch.Tensor
        Shape ``(tokens, hidden)`` if ``reduce_results`` else
        ``(tokens, top_k, hidden)``.
    """
    assert hidden_states.is_contiguous(), "hidden_states must be contiguous"
    assert w1.is_contiguous(), "w1 must be contiguous"
    assert w2.is_contiguous(), "w2 must be contiguous"
    assert hidden_states.dim() == 2
    assert topk_weights.shape == topk_ids.shape
    assert hidden_states.dtype in (torch.bfloat16, torch.float16)

    num_tokens, hidden = hidden_states.shape
    E, two_inter, k_in = w1.shape
    assert k_in == hidden, f"w1 last dim {k_in} != hidden {hidden}"
    _, w2_hidden, inter = w2.shape
    assert w2_hidden == hidden, f"w2 dim[1] {w2_hidden} != hidden {hidden}"
    assert two_inter == 2 * inter, f"w1 dim[1] {two_inter} != 2 * w2 dim[2] {2 * inter}"

    top_k = topk_ids.shape[1]
    # sgl_kernel's moe_align_block_size expects int32; torch.topk returns int64.
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.contiguous()

    config = get_default_config(M=num_tokens, E=E, N=two_inter, K=hidden, top_k=top_k)
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
