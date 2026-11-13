"""Query/key RMSNorm followed by the interleaved-pair rotary embedding, for DiTs
whose q/k norm spans the whole projection (one RMSNorm across all heads, as in
Wan's ``rms_norm_across_heads``).

Eager, this is the model's own ``nn.RMSNorm`` and ``rope.apply_rotary_interleaved``.
Compiled, Inductor lowers the pair to several passes over each row (the norm's
reduction, then the rotated copy of the row, then the rotation), so inside a
compiled region one kernel does both q and k: each program holds a row in
registers, normalizes it, rounds the normalized row to the activation dtype
where the eager norm's output does, and rotates it in fp32. Its mean square is
reduced in a different order than the eager norm's, so the two agree to fp32
rounding rather than bitwise.
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
        seq, width, q_stride, k_stride, q_eps, k_eps, HEAD_DIM: tl.constexpr, BLOCK_PAIRS: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        if tl.program_id(1) == 0:
            x_ptr, w_ptr, out_ptr, eps, stride = q_ptr, qw_ptr, q_out_ptr, q_eps, q_stride
        else:
            x_ptr, w_ptr, out_ptr, eps, stride = k_ptr, kw_ptr, k_out_ptr, k_eps, k_stride
        pos = row % seq
        pairs = tl.arange(0, BLOCK_PAIRS)
        pmask = pairs < width // 2
        offs = pairs[:, None] * 2 + tl.arange(0, 2)[None, :]
        mask = pmask[:, None]

        x = tl.load(x_ptr + row * stride + offs, mask=mask, other=0.0).to(tl.float32)
        mean_sq = tl.sum(tl.sum(x * x, axis=1), axis=0) / width
        # eps arrives as fp64 from a compiled caller; keep the row in fp32
        rstd = tl.math.rsqrt(mean_sq + eps.to(tl.float32))
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x = (x * rstd * w).to(out_ptr.dtype.element_ty).to(tl.float32)

        even, odd = tl.split(x)
        # the tables repeat each frequency over its pair, so the even column serves both
        col = pos * HEAD_DIM + (pairs % (HEAD_DIM // 2)) * 2
        cos = tl.load(cos_ptr + col, mask=pmask, other=0.0)
        sin = tl.load(sin_ptr + col, mask=pmask, other=0.0)
        out = tl.join(even * cos - odd * sin, odd * cos + even * sin)
        tl.store(out_ptr + row * width + offs, out.to(out_ptr.dtype.element_ty), mask=mask)

    @triton_op("mstar::rms_norm_rope_pair", mutates_args=())
    def _rms_norm_rope_op(
        q: torch.Tensor, k: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
        cos: torch.Tensor, sin: torch.Tensor, q_eps: float, k_eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq, width = q.shape
        # q and k may be views into a fused projection: rows at their own stride; the outputs are dense
        q_out = torch.empty((batch, seq, width), dtype=q.dtype, device=q.device)
        k_out = torch.empty((batch, seq, width), dtype=k.dtype, device=k.device)
        wrap_triton(_rms_norm_rope_kernel)[(batch * seq, 2)](
            q, k, q_weight, k_weight, cos, sin, q_out, k_out, seq, width, q.stride(1), k.stride(1), q_eps, k_eps,
            HEAD_DIM=cos.shape[-1], BLOCK_PAIRS=triton.next_power_of_2(width // 2), num_warps=8,
        )
        return q_out, k_out


def _uniform_rows(x: torch.Tensor) -> torch.Tensor:
    """``x [B, S, W]`` as is when its ``B * S`` rows sit at one stride with dense columns (a slice of a fused
    projection's output), else a dense copy."""
    if x.stride(2) == 1 and (x.shape[0] == 1 or x.stride(0) == x.shape[1] * x.stride(1)):
        return x
    return x.contiguous()


def rms_norm_rope(
    q: torch.Tensor, k: torch.Tensor, q_norm: nn.RMSNorm, k_norm: nn.RMSNorm,
    cos: torch.Tensor, sin: torch.Tensor, num_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``rope(q_norm(q)), rope(k_norm(k))`` for ``q, k [B, S, H * D]`` and fp32
    tables ``cos, sin [S, D]`` (each frequency repeated over its pair), returned
    as ``[B, S, H, D]``. Both norms span the whole ``H * D`` row. ``q`` and ``k``
    may be column slices of one ``[B, S, 3 * H * D]`` projection."""

    def eager(x: torch.Tensor, norm: nn.RMSNorm) -> torch.Tensor:
        return apply_rotary_interleaved(norm(x).unflatten(2, (num_heads, -1)), cos, sin)

    if triton is not None and q.is_cuda and torch.compiler.is_compiling():
        assert q.shape == k.shape and cos.shape == (q.shape[1], q.shape[2] // num_heads)
        q_out, k_out = _rms_norm_rope_op(
            _uniform_rows(q), _uniform_rows(k), q_norm.weight, k_norm.weight, cos.contiguous(), sin.contiguous(),
            q_norm.eps, k_norm.eps,
        )
        return q_out.unflatten(2, (num_heads, -1)), k_out.unflatten(2, (num_heads, -1))
    return eager(q, q_norm), eager(k, k_norm)
