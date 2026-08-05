"""Fused Triton-kernel MoE path.

Replaces the naive per-expert Python loop in
:func:`mstar.model.components.moe.dispatch_experts_fused` with a
grouped-GEMM implementation adapted from sglang's ``fused_moe_triton``.

:func:`fused_experts` covers the bf16 / fp16 and packed-W4A16 paths;
:func:`fused_experts_fp8` covers block-scaled FP8 W8A8 (GLM-5.2's
fp8-resident experts).  If triton is not installed the import fails and
callers fall back to the naive dispatch.  The token-alignment step uses a
vendored CUDA kernel (JIT-built on first use) with a torch fallback.
"""
from __future__ import annotations

from mstar.utils.fused_moe.kernels import moe_sum_reduce_triton
from mstar.utils.fused_moe.runner import fused_experts, fused_experts_fp8

__all__ = ["fused_experts", "fused_experts_fp8", "moe_sum_reduce_triton"]
