"""One Triton kernel for what a chunked GDN prefill needs from its inputs.

FlashInfer's chunked kernel takes contiguous q/k/v, L2-normalized q and k, the
decay and learning rate already formed, room for the tokens a padding row
borrows, and a bucket's padded tail blanked. As eager torch ops that is some 40
kernels per layer, most of the layer's non-GEMM time at a few thousand tokens.
Here it is one launch, over the same values at bf16 tolerance.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from mstar.engine.resources.recurrent.config import DeltaNetGeometry
from mstar.platforms import current_platform

_BLOCK_TOKENS = 16
# `tl.load` over a head dim past this needs more registers than a warp keeps
_MAX_HEAD_DIM = 512
_NORM_EPS = 1e-12
# `F.softplus`'s default: past it the linear branch is exact in fp32
_SOFTPLUS_THRESHOLD = 20.0


class GDNStagedInputs(NamedTuple):
    """Each ``[tokens + room, ...]``, contiguous, with zeros (``alpha`` ones)
    on every position ``token_mask`` drops."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor


def gdn_stage_applies(device: torch.device, geometry: DeltaNetGeometry) -> bool:
    """Whether ``stage_gdn_prefill_inputs`` can stand in for the eager path on
    this pool: one head dim for q, k and v, and a GPU it has been measured on.
    Nothing in the kernel is specific to Blackwell; older ones keep the eager
    path until someone measures them."""
    return (
        device.type == "cuda"
        and current_platform.has_cuda_capability(10, 0, device)
        and geometry.head_k_dim == geometry.head_v_dim
        and geometry.head_k_dim <= _MAX_HEAD_DIM
    )


@triton.jit(do_not_specialize=["tokens", "padded"])
def _gdn_prefill_stage_kernel(
    q, k, v, a, b, a_log, dt_bias, token_mask,
    q_out, k_out, v_out, alpha_out, beta_out,
    stride_qt, stride_qh, stride_qd,
    stride_kt, stride_kh, stride_kd,
    stride_vt, stride_vh, stride_vd,
    stride_at, stride_ah,
    stride_bt, stride_bh,
    tokens, padded,
    HEADS: tl.constexpr,
    V_HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    NORMALIZE: tl.constexpr,
    NORM_EPS: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
):
    token = tl.program_id(0) * BT
    head = tl.program_id(1)
    rows = token + tl.arange(0, BT)
    dims = tl.arange(0, BD)
    # the padded tail reads as zeros, and a position past `tokens` is not read
    in_range = rows < tokens
    keep = tl.load(token_mask + rows, mask=rows < padded, other=0.0) != 0.0
    keep = keep & in_range
    keep_2d = keep[:, None] & (dims[None, :] < DIM)
    writes = (rows[:, None] < padded) & (dims[None, :] < DIM)

    if head < HEADS:
        q_vals = tl.load(
            q + rows[:, None] * stride_qt + head * stride_qh + dims[None, :] * stride_qd,
            mask=keep_2d, other=0.0,
        ).to(tl.float32)
        k_vals = tl.load(
            k + rows[:, None] * stride_kt + head * stride_kh + dims[None, :] * stride_kd,
            mask=keep_2d, other=0.0,
        ).to(tl.float32)
        if NORMALIZE:
            q_vals = q_vals / tl.maximum(tl.sqrt(tl.sum(q_vals * q_vals, axis=1)), NORM_EPS)[:, None]
            k_vals = k_vals / tl.maximum(tl.sqrt(tl.sum(k_vals * k_vals, axis=1)), NORM_EPS)[:, None]
        at = rows[:, None] * (HEADS * DIM) + head * DIM + dims[None, :]
        tl.store(q_out + at, q_vals.to(q_out.dtype.element_ty), mask=writes)
        tl.store(k_out + at, k_vals.to(k_out.dtype.element_ty), mask=writes)
    else:
        v_head = head - HEADS
        v_vals = tl.load(
            v + rows[:, None] * stride_vt + v_head * stride_vh + dims[None, :] * stride_vd,
            mask=keep_2d, other=0.0,
        )
        at = rows[:, None] * (V_HEADS * DIM) + v_head * DIM + dims[None, :]
        tl.store(v_out + at, v_vals, mask=writes)

        a_vals = tl.load(a + rows * stride_at + v_head * stride_ah, mask=keep, other=0.0).to(tl.float32)
        b_vals = tl.load(b + rows * stride_bt + v_head * stride_bh, mask=keep, other=0.0).to(tl.float32)
        x = a_vals + tl.load(dt_bias + v_head).to(tl.float32)
        softplus = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
        alpha = tl.exp(-tl.exp(tl.load(a_log + v_head).to(tl.float32)) * softplus)
        at_gate = rows * V_HEADS + v_head
        in_pad = rows < padded
        tl.store(alpha_out + at_gate, tl.where(keep, alpha, 1.0), mask=in_pad)
        tl.store(beta_out + at_gate, tl.where(keep, tl.sigmoid(b_vals), 0.0), mask=in_pad)


def stage_gdn_prefill_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    token_mask: torch.Tensor,
    room: int,
    normalize: bool,
) -> GDNStagedInputs:
    """``q``/``k`` are ``[tokens, H, D]``, ``v`` ``[tokens, HV, D]``, ``a``/``b``
    ``[tokens, HV]``, any of them strided, and ``token_mask`` covers at least
    ``tokens + room`` positions (fp32, nonzero to keep). See
    ``gdn_stage_applies``."""
    tokens, heads, dim = q.shape
    v_heads = v.shape[1]
    padded = tokens + room
    assert token_mask.shape[0] >= padded, (
        f"gdn stage: mask holds {token_mask.shape[0]} positions, the step {padded}"
    )
    device = q.device
    q_out = torch.empty((padded, heads, dim), dtype=q.dtype, device=device)
    k_out = torch.empty((padded, heads, dim), dtype=q.dtype, device=device)
    v_out = torch.empty((padded, v_heads, dim), dtype=q.dtype, device=device)
    alpha = torch.empty((padded, v_heads), dtype=torch.float32, device=device)
    beta = torch.empty((padded, v_heads), dtype=torch.float32, device=device)
    grid = (triton.cdiv(padded, _BLOCK_TOKENS), heads + v_heads)
    _gdn_prefill_stage_kernel[grid](
        q, k, v, a, b, a_log, dt_bias, token_mask,
        q_out, k_out, v_out, alpha, beta,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        tokens, padded,
        HEADS=heads,
        V_HEADS=v_heads,
        DIM=dim,
        BT=_BLOCK_TOKENS,
        BD=triton.next_power_of_2(dim),
        NORMALIZE=normalize,
        NORM_EPS=_NORM_EPS,
        SOFTPLUS_THRESHOLD=_SOFTPLUS_THRESHOLD,
        num_warps=4,
        num_stages=2,
    )
    return GDNStagedInputs(q_out, k_out, v_out, alpha, beta)
