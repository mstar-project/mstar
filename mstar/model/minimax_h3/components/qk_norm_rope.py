"""H3's per-head q/k RMSNorm and partial rope in one pass over q and k, for the compiled DiT.

Inductor runs the two as a looped reduction over each head and writes the roped heads at
~1.4 TB/s. This kernel loads each head twice too, as ``x`` and in its rotate-half order,
from the same row (the second load is probably a cache hit; not measured), and rounds like
the eager ops (the norm's output to bf16, then ``x * cos`` and ``partner * sin`` to bf16,
then their sum). Eager steps keep the torch ops, so they stay bit-exact with the reference.

``QKNormRope.into`` runs the same kernel with each head's output wherever a table puts it, which the Ulysses
pipeline uses to have it write q and k straight into its exchange buffers."""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _qk_norm_rope_kernel(QKV, W, COS, SIN, OUT, HEAD_OFFSET, eps, row_stride, table_stride, out_row_stride,
                         H: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr, ROT: tl.constexpr,
                         HB: tl.constexpr, SCATTER: tl.constexpr):
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
    if SCATTER:  # head h's row r at OUT + HEAD_OFFSET[h] + r * out_row_stride
        offset = tl.load(HEAD_OFFSET + heads)
    else:  # [S, 2 * H, D]
        offset = heads.to(tl.int64) * D
    tl.store(OUT + row.to(tl.int64) * out_row_stride + offset[:, None] + d[None, :], (a + b).to(tl.bfloat16))


@triton.jit
def _qk_norm_rope_heads_kernel(QKV, W, COS, SIN, OUT, HEAD_OFFSET, HEAD_INDEX, num_heads, num_rows, eps, row_stride,
                               table_stride, out_row_stride, H: tl.constexpr, D: tl.constexpr, HALF: tl.constexpr,
                               ROT: tl.constexpr, HB: tl.constexpr, R: tl.constexpr):
    """The kernel above over the heads HEAD_INDEX lists (q's 0..H-1, k's H..2H-1, in any mix), the i-th's row r to
    OUT + HEAD_OFFSET[i] + r * out_row_stride; R rows a program, each row's arithmetic as above."""
    d = tl.arange(0, D)
    partner = tl.where(d < HALF, d + HALF, tl.where(d < ROT, d - HALF, d))
    slot = tl.program_id(1) * HB + tl.arange(0, HB)
    valid = slot < num_heads
    heads = tl.load(HEAD_INDEX + slot, mask=valid, other=0)
    which = (heads // H)[:, None] * D
    w = tl.load(W + which + d[None, :]).to(tl.float32)
    wp = tl.load(W + which + partner[None, :]).to(tl.float32)
    offset = tl.load(HEAD_OFFSET + slot, mask=valid, other=0)
    for i in tl.static_range(R):
        row = tl.program_id(0) * R + i
        live = valid & (row < num_rows)
        base = QKV + row * row_stride + heads[:, None] * D
        x = tl.load(base + d[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        xp = tl.load(base + partner[None, :], mask=live[:, None], other=0.0).to(tl.float32)
        rstd = libdevice.rsqrt(tl.sum(x * x, axis=1) / D + eps)
        n = (x * rstd[:, None] * w).to(tl.bfloat16).to(tl.float32)
        n_partner = (xp * rstd[:, None] * wp).to(tl.bfloat16).to(tl.float32)
        in_rows = d < D * (row < num_rows)
        cos = tl.load(COS + row * table_stride + d, mask=in_rows, other=0.0).to(tl.float32)
        sin = tl.load(SIN + row * table_stride + d, mask=in_rows, other=0.0).to(tl.float32)
        a = (n * cos[None, :]).to(tl.bfloat16).to(tl.float32)
        b = (n_partner * sin[None, :]).to(tl.bfloat16).to(tl.float32)
        tl.store(OUT + row.to(tl.int64) * out_row_stride + offset[:, None] + d[None, :], (a + b).to(tl.bfloat16),
                 mask=live[:, None])


@torch.library.custom_op("mstar::h3_qk_norm_rope", mutates_args=())
def qk_norm_rope(qkv: torch.Tensor, weights: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, eps: float,
                 heads: int, rotary_dim: int) -> torch.Tensor:
    """``qkv`` ``[S, >= 2 * heads * D]`` rows of q heads then k heads (head dim contiguous), ``weights``
    ``[2, D]`` (q's norm, k's), ``cos`` / ``sin`` ``[S, D]`` a ``RopeTable``'s -> roped ``[S, 2 * heads, D]``."""
    rows, d = qkv.shape[0], weights.shape[1]
    out = qkv.new_empty(rows, 2 * heads, d)
    _check_heads(heads)
    _launch(qkv, weights, cos, sin, eps, heads, rotary_dim, out, None, None, 2 * heads * d)
    return out


_BLOCK = 8


_ROWS = 4  # rows a program over a head list: ~10% faster than 1 on B300, bit-identical (8 warps is not)


def _launch(qkv, weights, cos, sin, eps, heads, rotary_dim, out, head_offset, head_index, out_row_stride):
    rows, d = qkv.shape[0], weights.shape[1]
    if head_index is not None:
        listed = head_index.numel()
        _qk_norm_rope_heads_kernel[(triton.cdiv(rows, _ROWS), triton.cdiv(listed, _BLOCK))](
            qkv, weights, cos, sin, out, head_offset, head_index, listed, rows, eps, qkv.stride(0), cos.stride(0),
            out_row_stride, H=heads, D=d, HALF=rotary_dim // 2, ROT=rotary_dim, HB=_BLOCK, R=_ROWS, num_warps=4,
            enable_fp_fusion=False,
        )
        return
    _qk_norm_rope_kernel[(rows, 2 * heads // _BLOCK)](
        qkv, weights, cos, sin, out, out if head_offset is None else head_offset, eps, qkv.stride(0), cos.stride(0),
        out_row_stride, H=heads, D=d, HALF=rotary_dim // 2, ROT=rotary_dim, HB=_BLOCK,
        SCATTER=head_offset is not None, num_warps=4, enable_fp_fusion=False,
    )


@qk_norm_rope.register_fake
def _qk_norm_rope_fake(qkv, weights, cos, sin, eps, heads, rotary_dim):
    return qkv.new_empty(qkv.shape[0], 2 * heads, weights.shape[1])


def _check_heads(heads: int) -> None:
    # a program's heads must all be q or all k, or it norms some with the other's weight
    # (28 local heads under TP2 would)
    if heads % _BLOCK:
        raise ValueError(f"qk_norm_rope needs a multiple of {_BLOCK} heads, got {heads}")


class QKNormRope(NamedTuple):
    """One ``qk_norm_rope`` call, not yet run: ``standard()`` returns its q and k, ``into()`` writes them where
    the caller says."""

    qkv: torch.Tensor      # [S, >= 2 * heads * D]
    weights: torch.Tensor  # [2, D]
    cos: torch.Tensor      # [S, D]
    sin: torch.Tensor
    eps: float
    heads: int
    rotary_dim: int

    def standard(self) -> tuple[torch.Tensor, torch.Tensor]:
        """q and k, ``[S, heads, D]`` each."""
        qk = qk_norm_rope(*self)
        return qk[:, :self.heads], qk[:, self.heads:]

    def into(self, out: torch.Tensor, head_offset: torch.Tensor, row_stride: int,
             head_index: torch.Tensor | None = None) -> None:
        """Row r of head h (q's heads, then k's) to ``out.view(-1)[head_offset[h] + r * row_stride:][:D]``;
        with ``head_index`` (int32), only the heads it lists, the i-th to ``head_offset[i]``. Offsets int64, both
        on the device."""
        if head_index is None:
            _check_heads(self.heads)
        _launch(self.qkv, self.weights, self.cos, self.sin, self.eps, self.heads, self.rotary_dim, out,
                head_offset, head_index, row_stride)
