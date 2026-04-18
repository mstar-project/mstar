"""Fused Triton-kernel MoE path for Qwen3-Omni (Thinker + Talker).

Replaces the naive per-expert Python loop in
:func:`mminf.model.qwen3_omni.components.moe._dispatch_experts_fused`
with a grouped-GEMM implementation adapted from sglang's
``fused_moe_triton``.

Only the bf16 / fp16 unquantized path is provided.  The entry point is
:func:`fused_experts`; if its dependency ``sgl_kernel`` is not installed
the import fails and callers fall back to the naive dispatch.
"""

from __future__ import annotations

from mminf.model.qwen3_omni.components.fused_moe.runner import fused_experts

__all__ = ["fused_experts"]
