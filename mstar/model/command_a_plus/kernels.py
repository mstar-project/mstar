"""Fused Triton kernels for the Command A+ decoder layer.

Each kernel replaces a chain of small PyTorch ops in the eager layer. The
fused layer keeps sums in fp32 and rounds to bf16 fewer times than the unfused
one (``moe_combine`` rounds once, after the sum), so the two can produce
different greedy outputs.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from mstar.utils.fused_moe.route import ROUTE_ALIGN_MAX_SLOTS, route_align  # noqa: F401  (re-export)


@triton.jit
def _silu_mul_kernel(
    gate_up_ptr, out_ptr, stride_in, stride_out, I, scale,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < I
    g = tl.load(gate_up_ptr + row * stride_in + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gate_up_ptr + row * stride_in + I + offs, mask=mask, other=0.0)
    dtype = out_ptr.dtype.element_ty
    a = (g * tl.sigmoid(g)).to(dtype)
    y = (a.to(tl.float32) * u.to(tl.float32)).to(dtype)
    tl.store(out_ptr + row * stride_out + offs, (y.to(tl.float32) * scale).to(dtype), mask=mask)


def silu_mul_into(gate_up: torch.Tensor, out: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """``out = scale * (silu(gate) * up)`` with ``gate_up = [gate | up]`` on the last dim.

    Rounds like ``F.silu(gate) * up`` in bf16; ``scale`` should be a power of
    two so it adds no rounding. Both tensors may be row-strided views.
    """
    tokens, two_i = gate_up.shape
    inter = two_i // 2
    assert out.shape == (tokens, inter) and gate_up.stride(1) == 1 and out.stride(1) == 1
    block = 1024
    if tokens:
        _silu_mul_kernel[(tokens, triton.cdiv(inter, block))](
            gate_up, out, gate_up.stride(0), out.stride(0), inter, scale, BLOCK=block, num_warps=4,
        )
    return out


@triton.jit
def _splitk_linear_kernel(
    x0_ptr, x1_ptr, w_ptr, out_ptr, M, N, K0, stride_x0, stride_x1, stride_wn, stride_os, stride_om,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, K_PER_SPLIT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_m = tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    k0 = pid_s * K_PER_SPLIT
    # Columns [0, K0) of the logical input come from x0, the rest from x1.
    if k0 < K0:
        x_rows = x0_ptr + offs_m[:, None] * stride_x0 + k0
    else:
        x_rows = x1_ptr + offs_m[:, None] * stride_x1 + (k0 - K0)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, BK):
        a = tl.load(x_rows + kk + offs_k[None, :], mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(w_ptr + offs_n[None, :] * stride_wn + (k0 + kk + offs_k)[:, None],
                    mask=offs_n[None, :] < N, other=0.0)
        acc += tl.dot(a, b)
    tl.store(
        out_ptr + pid_s * stride_os + offs_m[:, None] * stride_om + offs_n[None, :], acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


SPLITK_MAX_TOKENS = 16
_SPLITK = dict(BN=128, BK=128, SPLIT=8, num_warps=4, num_stages=3)


def splitk_supported(xs: tuple[torch.Tensor, ...], w: torch.Tensor) -> bool:
    k = sum(x.shape[1] for x in xs)
    k_per_split = k // _SPLITK["SPLIT"]
    return (
        len(xs) in (1, 2) and xs[0].is_cuda and 0 < xs[0].shape[0] <= SPLITK_MAX_TOKENS
        and all(x.shape[0] == xs[0].shape[0] and x.stride(1) == 1 for x in xs) and w.stride(1) == 1 and w.shape[1] == k
        and k % (_SPLITK["SPLIT"] * _SPLITK["BK"]) == 0 and xs[0].shape[1] % k_per_split == 0
    )


def splitk_linear(xs: tuple[torch.Tensor, ...], w: torch.Tensor) -> torch.Tensor:
    """fp32 split-K partials of ``cat(xs, -1) @ w.T`` with shape ``[SPLIT, M, N]``; sum over dim 0.

    cuBLAS picks a non-split kernel for these skinny decode shapes and leaves
    HBM bandwidth idle; splitting K fills the GPU. ``xs`` is one or two
    inputs read in place, so the caller needn't concatenate them. Callers
    fold the partial sum into their next elementwise kernel (see ``moe_combine``).
    """
    x0, x1 = xs[0], xs[-1]
    m, n = x0.shape[0], w.shape[0]
    k0 = x0.shape[1] if len(xs) == 2 else w.shape[1]
    split = _SPLITK["SPLIT"]
    out = torch.empty((split, m, n), dtype=torch.float32, device=x0.device)
    _splitk_linear_kernel[(triton.cdiv(n, _SPLITK["BN"]), split)](
        x0, x1, w, out, m, n, k0, x0.stride(0), x1.stride(0), w.stride(0), out.stride(0), out.stride(1),
        BM=max(16, triton.next_power_of_2(m)), BN=_SPLITK["BN"], BK=_SPLITK["BK"],
        K_PER_SPLIT=w.shape[1] // split, num_warps=_SPLITK["num_warps"], num_stages=_SPLITK["num_stages"],
    )
    return out


@triton.jit
def _moe_combine_kernel(
    base_ptr, cache_ptr, out_ptr, stride_bs, stride_b, stride_c0, stride_c1, stride_o, H, scale,
    TOPK: tl.constexpr, BASE_SPLITS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < H
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        acc += tl.load(cache_ptr + row * stride_c0 + k * stride_c1 + offs, mask=mask, other=0.0).to(tl.float32)
    base = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(BASE_SPLITS):
        base += tl.load(base_ptr + s * stride_bs + row * stride_b + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * stride_o + offs, (base + scale * acc).to(out_ptr.dtype.element_ty), mask=mask)


def moe_combine(base: torch.Tensor, cache3: torch.Tensor, scale: float) -> torch.Tensor:
    """``out = base + scale * cache3.sum(1)``, accumulated in fp32 and rounded once.

    ``base`` is ``[tokens, hidden]``, or ``[splits, tokens, hidden]`` split-K
    partials that are summed first.
    """
    tokens, topk, hidden = cache3.shape
    base3 = base if base.dim() == 3 else base.unsqueeze(0)
    assert base3.shape[1:] == (tokens, hidden) and cache3.stride(2) == 1 and base3.stride(2) == 1
    out = torch.empty((tokens, hidden), dtype=cache3.dtype, device=cache3.device)
    block = 1024
    if tokens:
        _moe_combine_kernel[(tokens, triton.cdiv(hidden, block))](
            base3, cache3, out, base3.stride(0), base3.stride(1),
            cache3.stride(0), cache3.stride(1), out.stride(0),
            hidden, scale, TOPK=topk, BASE_SPLITS=base3.shape[0], BLOCK=block, num_warps=4,
        )
    return out
