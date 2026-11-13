"""Query/key RMSNorm followed by the interleaved-pair rotary embedding, for DiTs
whose q/k norm spans the whole projection (one RMSNorm across all heads, as in
Wan's ``rms_norm_across_heads``).

Eager, this is the model's own ``nn.RMSNorm`` and ``rope.apply_rotary_interleaved``.
Compiled, Inductor lowers the pair to several passes over each row (the norm's
reduction, then the rotated copy of the row, then the rotation), so inside a
compiled region one kernel does both q and k: each program takes a row,
normalizes it, rounds the normalized row to the activation dtype where the eager
norm's output does, and rotates it in fp32. A row of up to four power-of-two
tiles of pairs (every Wan2.2 width) is read once and held in registers; a wider
split takes it in chunked passes (the mean square, then the output). Its mean
square is reduced in a different order than the eager norm's, so the two agree
to fp32 rounding rather than bitwise.
"""

from __future__ import annotations

import torch
from torch import nn

from mstar.model.components.diffusion.rope import apply_rotary_interleaved

try:
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton
except ImportError:  # CPU-only installs
    triton = None


if triton is not None:

    @triton.jit
    def _rms_norm_rope_kernel(
        q_ptr, k_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, q_out_ptr, k_out_ptr,
        seq, width, q_eps, k_eps, HEAD_DIM: tl.constexpr, CHUNK_PAIRS: tl.constexpr, NCHUNK: tl.constexpr,
    ):
        # one row of q or k per program, in passes of CHUNK_PAIRS pairs: the mean square, then
        # the normalized, rotated output (the second read hits cache)
        row = tl.program_id(0).to(tl.int64)
        if tl.program_id(1) == 0:
            x_ptr, w_ptr, out_ptr, eps = q_ptr, qw_ptr, q_out_ptr, q_eps
        else:
            x_ptr, w_ptr, out_ptr, eps = k_ptr, kw_ptr, k_out_ptr, k_eps
        pos = row % seq
        base = row * width
        acc = tl.zeros([CHUNK_PAIRS, 2], dtype=tl.float32)
        for c in tl.static_range(NCHUNK):
            pairs = c * CHUNK_PAIRS + tl.arange(0, CHUNK_PAIRS)
            pmask = pairs < width // 2
            offs = pairs[:, None] * 2 + tl.arange(0, 2)[None, :]
            x = tl.load(x_ptr + base + offs, mask=pmask[:, None], other=0.0).to(tl.float32)
            acc += x * x
        mean_sq = tl.sum(tl.sum(acc, axis=1), axis=0) / width
        # eps arrives as fp64 from a compiled caller; keep the row in fp32
        rstd = tl.math.rsqrt(mean_sq + eps.to(tl.float32))
        for c in tl.static_range(NCHUNK):
            pairs = c * CHUNK_PAIRS + tl.arange(0, CHUNK_PAIRS)
            pmask = pairs < width // 2
            offs = pairs[:, None] * 2 + tl.arange(0, 2)[None, :]
            mask = pmask[:, None]
            x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            x = (x * rstd * w).to(out_ptr.dtype.element_ty).to(tl.float32)
            even, odd = tl.split(x)
            # the tables repeat each frequency over its pair, so the even column serves both
            col = pos * HEAD_DIM + (pairs % (HEAD_DIM // 2)) * 2
            cos = tl.load(cos_ptr + col, mask=pmask, other=0.0)
            sin = tl.load(sin_ptr + col, mask=pmask, other=0.0)
            out = tl.join(even * cos - odd * sin, odd * cos + even * sin)
            tl.store(out_ptr + base + offs, out.to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _load_tile(x_ptr, base, OFF: tl.constexpr, P: tl.constexpr):
        offs = (OFF + tl.arange(0, P))[:, None] * 2 + tl.arange(0, 2)[None, :]
        return tl.load(x_ptr + base + offs).to(tl.float32)

    @triton.jit
    def _store_tile(x, w_ptr, cos_ptr, sin_ptr, out_ptr, base, pos, rstd, OFF: tl.constexpr, P: tl.constexpr,
                    HEAD_DIM: tl.constexpr):
        pairs = OFF + tl.arange(0, P)
        offs = pairs[:, None] * 2 + tl.arange(0, 2)[None, :]
        x = (x * rstd * tl.load(w_ptr + offs).to(tl.float32)).to(out_ptr.dtype.element_ty).to(tl.float32)
        even, odd = tl.split(x)
        col = pos * HEAD_DIM + (pairs % (HEAD_DIM // 2)) * 2
        cos = tl.load(cos_ptr + col)
        sin = tl.load(sin_ptr + col)
        out = tl.join(even * cos - odd * sin, odd * cos + even * sin)
        tl.store(out_ptr + base + offs, out.to(out_ptr.dtype.element_ty))

    @triton.jit
    def _rms_norm_rope_tiles_kernel(
        q_ptr, k_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, q_out_ptr, k_out_ptr,
        seq, width, q_eps, k_eps, HEAD_DIM: tl.constexpr,
        P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
    ):
        # one row of q or k per program, as tiles of P0 + P1 + P2 + P3 = width / 2 pairs (powers of two, or 0):
        # read once and held in registers through the mean square, so HBM sees one read and one write
        row = tl.program_id(0).to(tl.int64)
        if tl.program_id(1) == 0:
            x_ptr, w_ptr, out_ptr, eps = q_ptr, qw_ptr, q_out_ptr, q_eps
        else:
            x_ptr, w_ptr, out_ptr, eps = k_ptr, kw_ptr, k_out_ptr, k_eps
        pos = row % seq
        base = row * width
        x0 = _load_tile(x_ptr, base, 0, P0)
        sum_sq = tl.sum(tl.sum(x0 * x0, axis=1), axis=0)
        if P1 > 0:
            x1 = _load_tile(x_ptr, base, P0, P1)
            sum_sq += tl.sum(tl.sum(x1 * x1, axis=1), axis=0)
        if P2 > 0:
            x2 = _load_tile(x_ptr, base, P0 + P1, P2)
            sum_sq += tl.sum(tl.sum(x2 * x2, axis=1), axis=0)
        if P3 > 0:
            x3 = _load_tile(x_ptr, base, P0 + P1 + P2, P3)
            sum_sq += tl.sum(tl.sum(x3 * x3, axis=1), axis=0)
        # eps arrives as fp64 from a compiled caller; keep the row in fp32
        rstd = tl.math.rsqrt(sum_sq / width + eps.to(tl.float32))
        _store_tile(x0, w_ptr, cos_ptr, sin_ptr, out_ptr, base, pos, rstd, 0, P0, HEAD_DIM)
        if P1 > 0:
            _store_tile(x1, w_ptr, cos_ptr, sin_ptr, out_ptr, base, pos, rstd, P0, P1, HEAD_DIM)
        if P2 > 0:
            _store_tile(x2, w_ptr, cos_ptr, sin_ptr, out_ptr, base, pos, rstd, P0 + P1, P2, HEAD_DIM)
        if P3 > 0:
            _store_tile(x3, w_ptr, cos_ptr, sin_ptr, out_ptr, base, pos, rstd, P0 + P1 + P2, P3, HEAD_DIM)

    def _pair_tiles(pairs: int) -> list[int]:
        """``pairs`` as descending powers of two (its set bits)."""
        return [1 << bit for bit in reversed(range(pairs.bit_length())) if pairs >> bit & 1]

    # pairs per pass of the chunked kernel, measured on a B300 at widths 3072 and 5120
    _CHUNK_PAIRS = 512
    _MAX_TILES = 4

    @triton_op("mstar::rms_norm_rope_pair", mutates_args=())
    def _rms_norm_rope_op(
        q: torch.Tensor, k: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
        cos: torch.Tensor, sin: torch.Tensor, q_eps: float, k_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq, width = q.shape
        q_out, k_out = torch.empty_like(q), torch.empty_like(k)
        tiles = _pair_tiles(width // 2)
        if len(tiles) <= _MAX_TILES:
            # H200, Wan2.2-T2V-A14B 480p q and k: 758 us at 4 warps vs 1021 for the chunked passes (a copy of
            # both is 631); 8 or 16 warps per row were slower
            tiles += [0] * (_MAX_TILES - len(tiles))
            wrap_triton(_rms_norm_rope_tiles_kernel)[(batch * seq, 2)](
                q, k, q_weight, k_weight, cos, sin, q_out, k_out, seq, width, q_eps, k_eps,
                HEAD_DIM=cos.shape[-1], P0=tiles[0], P1=tiles[1], P2=tiles[2], P3=tiles[3], num_warps=4,
            )
            return q_out, k_out
        wrap_triton(_rms_norm_rope_kernel)[(batch * seq, 2)](
            q, k, q_weight, k_weight, cos, sin, q_out, k_out, seq, width, q_eps, k_eps,
            HEAD_DIM=cos.shape[-1], CHUNK_PAIRS=_CHUNK_PAIRS, NCHUNK=triton.cdiv(width // 2, _CHUNK_PAIRS),
            num_warps=8,
        )
        return q_out, k_out


def rms_norm_rope(
    q: torch.Tensor, k: torch.Tensor, q_norm: nn.RMSNorm, k_norm: nn.RMSNorm,
    cos: torch.Tensor, sin: torch.Tensor, num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``rope(q_norm(q)), rope(k_norm(k))`` for ``q, k [B, S, H * D]`` and fp32
    tables ``cos, sin [S, D]`` (each frequency repeated over its pair), returned
    as ``[B, S, H, D]``. Both norms span the whole ``H * D`` row."""

    def eager(x: torch.Tensor, norm: nn.RMSNorm) -> torch.Tensor:
        return apply_rotary_interleaved(norm(x).unflatten(2, (num_heads, -1)), cos, sin)

    if triton is not None and q.is_cuda and torch.compiler.is_compiling():
        assert q.shape == k.shape and cos.shape == (q.shape[1], q.shape[2] // num_heads)
        q_out, k_out = _rms_norm_rope_op(
            q.contiguous(), k.contiguous(), q_norm.weight, k_norm.weight, cos.contiguous(), sin.contiguous(),
            q_norm.eps, k_norm.eps,
        )
        return q_out.unflatten(2, (num_heads, -1)), k_out.unflatten(2, (num_heads, -1))
    return eager(q, q_norm), eager(k, k_norm)
