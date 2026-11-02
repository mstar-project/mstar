"""H3's per-head q/k RMSNorm and partial rope in one pass over q and k, for the compiled DiT.

Inductor runs the two as a looped reduction that reads every head twice and writes the
roped heads at ~1.4 TB/s; this kernel reads each head once and rounds like the eager ops
(the norm's output to bf16, then ``x * cos`` and ``partner * sin`` to bf16, then their
sum). Eager steps keep the torch ops, so they stay bit-exact with the reference."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _qk_norm_rope_kernel(QKV, W, COS, SIN, OUT, eps, row_stride, table_stride,
                         H: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr, ROT: tl.constexpr,
                         HB: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1)
    heads = block * HB + tl.arange(0, HB)  # q heads 0..H-1, then k heads; q and k lie that way in QKV
    which = (block * HB) // H  # 0 q, 1 k: HB divides H
    d = tl.arange(0, D)
    partner = tl.where(d < HALF, d + HALF, tl.where(d < ROT, d - HALF, d))
    base = QKV + row * row_stride + heads[:, None] * D
    x = tl.load(base + d[None, :]).to(tl.float32)
    xp = tl.load(base + partner[None, :]).to(tl.float32)
    w = tl.load(W + which * D + d).to(tl.float32)
    wp = tl.load(W + which * D + partner).to(tl.float32)
    rstd = libdevice.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    n = (x * rstd[:, None] * w[None, :]).to(tl.bfloat16).to(tl.float32)
    n_partner = (xp * rstd[:, None] * wp[None, :]).to(tl.bfloat16).to(tl.float32)
    cos = tl.load(COS + row * table_stride + d).to(tl.float32)
    sin = tl.load(SIN + row * table_stride + d).to(tl.float32)
    a = (n * cos[None, :]).to(tl.bfloat16).to(tl.float32)
    b = (n_partner * sin[None, :]).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + (row * 2 * H + heads[:, None]) * D + d[None, :], (a + b).to(tl.bfloat16))


@torch.library.custom_op("mstar::h3_qk_norm_rope", mutates_args=())
def qk_norm_rope(qkv: torch.Tensor, weights: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float,
                 heads: int, rotary_dim: int) -> torch.Tensor:
    """``qkv`` ``[S, >= 2 * heads * D]`` rows of q heads then k heads (head dim contiguous), ``weights``
    ``[2, D]`` (q's norm, k's), ``cos`` / ``sin`` ``[S, D]`` a ``RopeTable``'s -> roped ``[S, 2 * heads, D]``."""
    rows, d = qkv.shape[0], weights.shape[1]
    out = qkv.new_empty(rows, 2 * heads, d)
    block = 8
    _qk_norm_rope_kernel[(rows, 2 * heads // block)](
        qkv, weights, cos, sin, out, eps, qkv.stride(0), cos.stride(0),
        H=heads, D=d, HALF=rotary_dim // 2, ROT=rotary_dim, HB=block, num_warps=4, enable_fp_fusion=False,
    )
    return out


@qk_norm_rope.register_fake
def _qk_norm_rope_fake(qkv, weights, cos, sin, eps, heads, rotary_dim):
    return qkv.new_empty(qkv.shape[0], 2 * heads, weights.shape[1])
