"""Fused kernels for a DiT block over a packed step.

A captured solve replays 20 evaluations of 22 blocks, so its small ops, a few
microseconds of launch each, are a large part of a short solve. One launch here does
a block's residual add and modulated norm, and another its partial rotary.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_BLOCK_N = 16
_BLOCK_D = 128


@triton.jit
def _residual_norm_kernel(
    X, Y, gate, shift, scale, X_OUT, NORM_OUT, n, eps,
    D: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, RESIDUAL: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = rows < n
    mask2 = mask[:, None]
    total = tl.zeros((BLOCK_N,), dtype=tl.float32)
    squares = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for d in tl.static_range(0, D, BLOCK_D):
        cols = d + tl.arange(0, BLOCK_D)
        offs = rows[:, None] * D + cols[None, :]
        x = tl.load(X + offs, mask=mask2, other=0.0).to(tl.float32)
        if RESIDUAL:
            y = tl.load(Y + offs, mask=mask2, other=0.0).to(tl.float32)
            out = (x + tl.load(gate + cols).to(tl.float32)[None, :] * y).to(X_OUT.dtype.element_ty)
            tl.store(X_OUT + offs, out, mask=mask2)
            x = out.to(tl.float32)
        total += tl.sum(x, 1)
        squares += tl.sum(x * x, 1)
    mean = total / D
    inverse = 1.0 / tl.sqrt(squares / D - mean * mean + eps)
    for d in tl.static_range(0, D, BLOCK_D):
        cols = d + tl.arange(0, BLOCK_D)
        offs = rows[:, None] * D + cols[None, :]
        x = tl.load((X_OUT if RESIDUAL else X) + offs, mask=mask2, other=0.0).to(tl.float32)
        scales = 1.0 + tl.load(scale + cols).to(tl.float32)
        shifts = tl.load(shift + cols).to(tl.float32)
        norm = (x - mean[:, None]) * inverse[:, None] * scales[None, :] + shifts[None, :]
        tl.store(NORM_OUT + offs, norm.to(NORM_OUT.dtype.element_ty), mask=mask2)


def modulated_norm(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, eps: float) -> torch.Tensor:
    """``LayerNorm(x) * (1 + scale) + shift``: ``x`` ``[N, D]``, the rest ``[D]``."""
    n, d = x.shape
    norm = torch.empty_like(x)
    grid = (triton.cdiv(n, _BLOCK_N),)
    _residual_norm_kernel[grid](
        x, x, shift, shift, scale, x, norm, n, eps,
        D=d, BLOCK_N=_BLOCK_N, BLOCK_D=_BLOCK_D, RESIDUAL=False, num_warps=4, num_stages=1,
    )
    return norm


def residual_modulated_norm(
    x: torch.Tensor, y: torch.Tensor, gate: torch.Tensor,
    shift: torch.Tensor, scale: torch.Tensor, eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``x + gate * y`` and that sum's ``modulated_norm``, in one launch."""
    n, d = x.shape
    out = torch.empty_like(x)
    norm = torch.empty_like(x)
    grid = (triton.cdiv(n, _BLOCK_N),)
    _residual_norm_kernel[grid](
        x, y, gate, shift, scale, out, norm, n, eps,
        D=d, BLOCK_N=_BLOCK_N, BLOCK_D=_BLOCK_D, RESIDUAL=True, num_warps=4, num_stages=1,
    )
    return out, norm


@triton.jit
def _rotate_pairs(first, c, s, mask):
    a = tl.load(first, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(first + 1, mask=mask, other=0.0).to(tl.float32)
    tl.store(first, (a * c - b * s).to(first.dtype.element_ty), mask=mask)
    tl.store(first + 1, (b * c + a * s).to(first.dtype.element_ty), mask=mask)


@triton.jit
def _partial_rotary_kernel(
    Q, K, COS, SIN, n, s_q, s_k,
    ROT: tl.constexpr, BLOCK_N: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rows < n)[:, None]
    pairs = tl.arange(0, ROT // 2) * 2
    c = tl.load(COS + rows[:, None] * ROT + pairs[None, :], mask=mask, other=1.0)
    s = tl.load(SIN + rows[:, None] * ROT + pairs[None, :], mask=mask, other=0.0)
    _rotate_pairs(Q + rows[:, None] * s_q + pairs[None, :], c, s, mask)
    _rotate_pairs(K + rows[:, None] * s_k + pairs[None, :], c, s, mask)


def partial_rotary_(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
    """Rotate the first ``cos.shape[-1]`` channels of ``q`` and ``k`` ``[N, C]`` in place,
    in float32 (interleaved pairs, each frequency given twice): ``apply_partial_rotary``."""
    n, rot = cos.shape
    assert q.stride(-1) == 1 and k.stride(-1) == 1
    _partial_rotary_kernel[(triton.cdiv(n, 256),)](
        q, k, cos, sin, n, q.stride(0), k.stride(0), ROT=rot, BLOCK_N=256, num_warps=4, num_stages=1,
    )
