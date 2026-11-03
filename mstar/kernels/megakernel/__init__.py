"""Persistent decode kernels for Qwen3.5's layer stack at small batch (rows <= 4).

One launch per stage (``MlpMegakernel``, ``GdnMegakernel``, ``AttnMegakernel``) or per 4-layer block
(``BlockMegakernel``): grid = SM count, stages ordered by global counters instead of kernel
boundaries. Inputs are the engine's own buffers (the recurrent pool's state and conv blocks with the
step's slots, the paged KV layer view with the fmha wrapper's block table and lengths, the step's
cos/sin), so a caller hands over device tables and plans nothing itself.

Qwen3.5 runs the block kernel for its captured decode buckets of 1-2 rows (``FusedDecodeConfig``, see
``mstar/model/qwen3_5/fused_decode.py``); ``benchmark/megakernel/`` measures each kernel against the
captured decode graph.
"""
from __future__ import annotations

import torch

from mstar.platforms import current_platform

# Measured faster than the captured graph at 1 and 2 rows; 3-4 rows run but lose (per-row reductions).
MAX_FAST_ROWS = 2
MAX_ROWS = 4


def megakernel_applies(device: torch.device, rows: int) -> bool:
    """Compute capability 10.x (the only GPUs these were built and measured on) and a decode batch the
    kernels are faster for; everything else keeps the captured graph."""
    if device.type != "cuda" or not 1 <= rows <= MAX_FAST_ROWS:
        return False
    capability = current_platform.cuda_capability(device)
    return capability is not None and capability[0] == 10


# What the block kernel is written for: 128-token pages, 4-tap convs, the GDN state in fp32, head
# sizes as below, and register tiles as wide as the hidden size (a power of two).
PAGE_SIZE = 128
GDN_HEAD_DIM = 128
ATTN_HEAD_DIM = 256
KV_HEADS = 2
CONV_TAPS = 4
MAX_HIDDEN = 2048


def block_refusal(
    device: torch.device, *, hidden_size: int, gdn_head_dims: tuple[int, int], attn_head_dim: int,
    kv_heads: int, page_size: int, state_dtype: torch.dtype, conv_taps: int, gated_attention: bool,
    adapters: bool,
) -> str | None:
    """Why the block kernel can't run a stack of these sizes on ``device``, or None if it can."""
    if device.type != "cuda":
        return f"{device.type} is not a CUDA device"
    capability = current_platform.cuda_capability(device)
    if capability is None or capability[0] != 10:
        return f"compute capability {capability} (the kernels are built and measured on 10.x only)"
    try:
        import triton  # noqa: F401
    except ImportError:
        return "Triton is not installed"
    checks = (
        (page_size == PAGE_SIZE, f"KV page size {page_size}, not {PAGE_SIZE}"),
        (gdn_head_dims == (GDN_HEAD_DIM, GDN_HEAD_DIM), f"GDN head dims {gdn_head_dims}, not {GDN_HEAD_DIM}"),
        (attn_head_dim == ATTN_HEAD_DIM, f"attention head dim {attn_head_dim}, not {ATTN_HEAD_DIM}"),
        (kv_heads == KV_HEADS, f"{kv_heads} KV heads, not {KV_HEADS}"),
        (state_dtype == torch.float32, f"GDN state in {state_dtype}, not float32"),
        (conv_taps == CONV_TAPS, f"a {conv_taps}-tap conv, not {CONV_TAPS}"),
        (gated_attention, "attention without its output gate"),
        (not adapters, "LoRA adapters on the projections"),
        (hidden_size & (hidden_size - 1) == 0 and hidden_size <= MAX_HIDDEN,
         f"hidden size {hidden_size} (a power of two up to {MAX_HIDDEN})"),
    )
    return next((why for ok, why in checks if not ok), None)


__all__ = ["MAX_FAST_ROWS", "MAX_ROWS", "block_refusal", "megakernel_applies"]
