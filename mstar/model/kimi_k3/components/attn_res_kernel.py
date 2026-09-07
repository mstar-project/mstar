"""Fused Block-Attention-Residual read in Triton (spec B.6).

One program per token row: it streams the ``num_blocks`` residual-stack entries and the
running prefix, scores each by ``<RMSNorm_noweight(v), w>`` (fp32, no ``1/sqrt(d)``), and
accumulates the softmax mixture of the *unnormalized* candidates with an online softmax,
so the whole read is a single launch that touches each candidate once. Optionally the
layer's own RMSNorm is applied to the result in the same pass (``out_norm_w``).

Numerics match ``reference.attn_res.attn_res_read`` up to the online-softmax reassociation
(fp32 throughout; output cast to the activation dtype).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def attn_res_kernel(
    prefix_ptr, blocks_ptr, w_ptr, out_ptr, out_norm_ptr,
    stride_pt, stride_bt, stride_bm, stride_ot,
    D, num_blocks, eps, out_eps,
    BLOCK_D: tl.constexpr, APPLY_OUT_NORM: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m_run = -float("inf")
    denom = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    n_cand = num_blocks + 1
    for i in range(0, n_cand):
        is_prefix = i == num_blocks
        blk_ptr = blocks_ptr + t * stride_bt + i * stride_bm + offs
        pre_ptr = prefix_ptr + t * stride_pt + offs
        v = tl.load(tl.where(is_prefix, pre_ptr, blk_ptr), mask=mask, other=0.0).to(tl.float32)
        rstd = 1.0 / tl.sqrt(tl.sum(v * v, axis=0) / D + eps)
        score = tl.sum(v * w, axis=0) * rstd
        m_new = tl.maximum(m_run, score)
        alpha = tl.exp(m_run - m_new)
        p = tl.exp(score - m_new)
        acc = acc * alpha + p * v
        denom = denom * alpha + p
        m_run = m_new
    out = acc / denom
    if APPLY_OUT_NORM:
        ow = tl.load(out_norm_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        rstd_o = 1.0 / tl.sqrt(tl.sum(out * out, axis=0) / D + out_eps)
        # KimiRMSNorm: normalize in fp32, cast, multiply by the weight in the activation dtype
        out = ((out * rstd_o).to(out_ptr.dtype.element_ty).to(tl.float32) * ow)
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
    out = torch.empty_like(prefix)
    block_d = triton.next_power_of_2(d)
    attn_res_kernel[(t,)](
        prefix, blocks, score_weight.contiguous(), out,
        out_norm_weight if out_norm_weight is not None else score_weight,
        prefix.stride(0), blocks.stride(0), blocks.stride(1), out.stride(0),
        d, m, float(eps), float(out_eps),
        BLOCK_D=block_d, APPLY_OUT_NORM=out_norm_weight is not None,
        num_warps=8 if block_d >= 4096 else 4,
    )
    return out
