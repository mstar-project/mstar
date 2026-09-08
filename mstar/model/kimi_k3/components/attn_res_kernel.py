"""Fused Block Attention Residual read (spec B.6) for CUDA, in two Triton launches.

Pass 1 scores every candidate row independently (``grid = (T, m + 1)``): the RMS-normalized
dot product with the folded score weight. Pass 2 (``grid = (T,)``) turns the scores into softmax
weights, mixes the unnormalized candidates in fp32 and optionally applies the layer's RMSNorm
to the result with ``KimiRMSNorm``'s exact arithmetic. The earlier single-launch version walked
the candidates sequentially inside one program, which cost ~1 µs per candidate at K3 width
(8-13 µs per read); the two passes take about 2 + 3 µs at any batch size, and folding the norm
in saves the separate norm launch the decoder layer would otherwise issue.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

MAX_CANDIDATES = 16  # the residual stack holds at most 93 / 12 + 1 entries


@triton.jit
def attn_res_scores_kernel(
    prefix_ptr, blocks_ptr, w_ptr, scores_ptr,
    stride_pt, stride_bt, stride_bm,
    D, num_blocks, eps,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    is_prefix = i == num_blocks
    blk_ptr = blocks_ptr + t * stride_bt + i * stride_bm + offs
    pre_ptr = prefix_ptr + t * stride_pt + offs
    v = tl.load(tl.where(is_prefix, pre_ptr, blk_ptr), mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(v * v, axis=0) / D + eps)
    tl.store(scores_ptr + t * (num_blocks + 1) + i, tl.sum(v * w, axis=0) * rstd)


@triton.jit
def attn_res_mix_kernel(
    prefix_ptr, blocks_ptr, scores_ptr, out_ptr, out_norm_ptr,
    stride_pt, stride_bt, stride_bm, stride_ot,
    D, num_blocks, out_eps,
    BLOCK_D: tl.constexpr, BLOCK_M: tl.constexpr, APPLY_OUT_NORM: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    n_cand = num_blocks + 1
    cand = tl.arange(0, BLOCK_M)
    s = tl.load(scores_ptr + t * n_cand + cand, mask=cand < n_cand, other=-float("inf"))
    p = tl.exp(s - tl.max(s, axis=0))
    p = p / tl.sum(p, axis=0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for i in range(0, n_cand):
        is_prefix = i == num_blocks
        blk_ptr = blocks_ptr + t * stride_bt + i * stride_bm + offs
        pre_ptr = prefix_ptr + t * stride_pt + offs
        v = tl.load(tl.where(is_prefix, pre_ptr, blk_ptr), mask=mask, other=0.0).to(tl.float32)
        p_i = tl.sum(tl.where(cand == i, p, 0.0), axis=0)
        acc += p_i * v
    out = acc
    if APPLY_OUT_NORM:
        ow = tl.load(out_norm_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        rstd_o = 1.0 / tl.sqrt(tl.sum(out * out, axis=0) / D + out_eps)
        # KimiRMSNorm: normalize in fp32, cast, multiply by the weight in the activation dtype
        out = (out * rstd_o).to(out_ptr.dtype.element_ty).to(tl.float32) * ow
    tl.store(out_ptr + t * stride_ot + offs, out.to(out_ptr.dtype.element_ty), mask=mask)


@torch.compiler.disable
def attn_res_read_triton(
    prefix: torch.Tensor,
    blocks: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float = 1e-5,
    out_norm_weight: torch.Tensor | None = None,
    out_eps: float = 1e-5,
) -> torch.Tensor:
    """``prefix [T, D]``, ``blocks [T, m, D]`` (m >= 0), ``score_weight [D]`` fp32 ->
    ``[T, D]`` in ``prefix.dtype`` (optionally followed by the layer RMSNorm)."""
    t, d = prefix.shape
    m = blocks.shape[1] if blocks is not None else 0
    if m == 0 and out_norm_weight is None:
        return prefix
    if m == 0:
        blocks = prefix.unsqueeze(1)  # unused (num_blocks=0), keeps strides valid
    assert m + 1 <= MAX_CANDIDATES, m
    out = torch.empty_like(prefix)
    block_d = triton.next_power_of_2(d)
    num_warps = 8 if block_d >= 4096 else 4
    scores = torch.empty(t, m + 1, dtype=torch.float32, device=prefix.device)
    w = score_weight.contiguous()
    attn_res_scores_kernel[(t, m + 1)](
        prefix, blocks, w, scores,
        prefix.stride(0), blocks.stride(0), blocks.stride(1),
        d, m, float(eps), BLOCK_D=block_d, num_warps=num_warps,
    )
    attn_res_mix_kernel[(t,)](
        prefix, blocks, scores, out, out_norm_weight if out_norm_weight is not None else w,
        prefix.stride(0), blocks.stride(0), blocks.stride(1), out.stride(0),
        d, m, float(out_eps),
        BLOCK_D=block_d, BLOCK_M=MAX_CANDIDATES, APPLY_OUT_NORM=out_norm_weight is not None,
        num_warps=num_warps,
    )
    return out
