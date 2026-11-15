"""Fused kernels for a DiT block over a packed step.

A captured solve replays 20 evaluations of 22 blocks, so its small ops, a few
microseconds of launch each, are a large part of a short solve. One launch here does
a block's partial rotary; its residual add and modulated norm are the DiT scaffold's
``mstar::ada_layer_norm`` (``mstar/model/components/diffusion/modulation.py``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


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
