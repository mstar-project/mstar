"""Fused NoAux-TC routing for CUDA: after the gate GEMV (``gate_logits``), one Triton program per
token computes the sigmoid scores, adds the correction bias, selects the top-k experts and writes
the renormalized, scaled weights. Replaces the eight-kernel torch chain (sigmoid, add, topk,
gather, sum, div, mul, casts) with one launch; the expert sets and weights match the reference
(``reference.router._route_fp32``) to fp32 rounding. Single expert group only (Kimi K3).

The per-k argmax loop below was measured against a rank-counting formulation (every candidate
counts how many beat it; scatter by rank): 7 µs vs 28 µs at one token on H100, so the loop stays.
One warp per token: the 256-wide reductions of each of the k rounds then stay inside the warp
(5.7 µs against 7.8 with four warps at 1 to 64 tokens, same outputs; ``bench/kernels/router_warps_bench.py``).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _route_kernel(
    logits_ptr, bias_ptr, idx_ptr, w_ptr, E, scale,
    SIGMOID: tl.constexpr, RENORM: tl.constexpr, K: tl.constexpr, BLOCK_E: tl.constexpr,
):
    t = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_E)
    mask = offs < E
    logits = tl.load(logits_ptr + t * E + offs, mask=mask, other=-float("inf")).to(tl.float32)
    if SIGMOID:
        scores = tl.sigmoid(logits)
    else:
        ex = tl.exp(logits - tl.max(logits, axis=0))
        scores = ex / tl.sum(ex, axis=0)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    choice = tl.where(mask, scores + bias, -float("inf"))
    total = 0.0
    for j in tl.static_range(K):
        m = tl.max(choice, axis=0)
        sel = tl.min(tl.where(choice == m, offs, BLOCK_E), axis=0)  # lowest index among ties
        w = tl.sum(tl.where(offs == sel, scores, 0.0), axis=0)
        tl.store(idx_ptr + t * K + j, sel.to(tl.int32))
        tl.store(w_ptr + t * K + j, w)
        total += w
        choice = tl.where(offs == sel, -float("inf"), choice)
    ks = tl.arange(0, K)
    ws = tl.load(w_ptr + t * K + ks)
    if RENORM:
        ws = ws / (total + 1e-20)
    tl.store(w_ptr + t * K + ks, ws * scale)


def fused_route_supported(top_k: int, num_expert_group: int, topk_group: int, scoring: str) -> bool:
    return (num_expert_group <= 1 or num_expert_group <= topk_group) and top_k in (1, 2, 4, 8, 16, 32) \
        and scoring in ("sigmoid", "softmax")


@torch.compiler.disable
def fused_route(
    logits: torch.Tensor, bias: torch.Tensor, top_k: int, *, scoring: str, renormalize: bool, scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``logits [T, E]`` fp32 -> ``(topk_idx [T, k] int32, topk_weight [T, k] fp32)``."""
    t, e = logits.shape
    logits = logits.contiguous()
    idx = torch.empty(t, top_k, dtype=torch.int32, device=logits.device)
    w = torch.empty(t, top_k, dtype=torch.float32, device=logits.device)
    _route_kernel[(t,)](
        logits, bias, idx, w, e, float(scale),
        SIGMOID=scoring == "sigmoid", RENORM=bool(renormalize and top_k > 1), K=top_k,
        BLOCK_E=triton.next_power_of_2(e), num_warps=1,
    )
    return idx, w


def gate_logits(x: torch.Tensor, weight: torch.Tensor, weight_fp32) -> torch.Tensor:
    """The router's fp32 logits ``[T, E]``.

    The reference computes ``F.linear(x.float(), weight.float())``. With a bf16 gate weight the
    products of the two bf16 values are exact in fp32 either way, so a bf16-input GEMM that
    accumulates and *outputs* in fp32 (``torch.mm(..., out_dtype=float32)``) gives the same logits
    up to summation order -- without the ``x.float()`` launch and reading half the weight bytes:
    6 µs flat at 1..64 tokens on H100 against 7.4 / 19.7 / 20.9 µs for the cast + fp32 GEMV.
    Other weight dtypes (fp32 test models) keep the reference arithmetic; ``weight_fp32`` supplies
    the cached fp32 copy for that path.
    """
    if x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16 and _mm_out_dtype_supported():
        return torch.mm(x, weight.t(), out_dtype=torch.float32)
    return torch.nn.functional.linear(x.float(), weight_fp32())


_MM_OUT_DTYPE: bool | None = None


def _mm_out_dtype_supported() -> bool:
    global _MM_OUT_DTYPE
    if _MM_OUT_DTYPE is None:
        try:
            a = torch.zeros(1, 2, dtype=torch.bfloat16, device="cuda" if torch.cuda.is_available() else "cpu")
            _MM_OUT_DTYPE = torch.mm(a, a.t(), out_dtype=torch.float32).dtype == torch.float32
        except (TypeError, RuntimeError):
            _MM_OUT_DTYPE = False
    return _MM_OUT_DTYPE
