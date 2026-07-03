"""Fused Triton-kernel MoE path.

Replaces the naive per-expert Python loop in
:func:`mstar.model.components.moe.dispatch_experts_fused` with a
grouped-GEMM implementation adapted from sglang's ``fused_moe_triton``.

Only the bf16 / fp16 unquantized path is provided.  The entry point is
:func:`fused_experts`; if its dependency ``sgl_kernel`` is not installed
the import fails and callers fall back to the naive dispatch.
"""
from __future__ import annotations

from mstar.utils.fused_moe.kernels import moe_sum_reduce_triton
from mstar.utils.fused_moe.runner import fused_experts

__all__ = ["fused_experts", "moe_sum_reduce_triton"]
