"""Persistent decode kernels for Qwen3.5's layer stack at small batch (rows <= 4).

One launch per stage (``MlpMegakernel``, ``GdnMegakernel``, ``AttnMegakernel``) or per 4-layer block
(``BlockMegakernel``): grid = SM count, stages ordered by global counters instead of kernel
boundaries. Inputs are the engine's own buffers (the recurrent pool's state and conv blocks with the
step's slots, the paged KV layer view with the fmha wrapper's block table and lengths, the step's
cos/sin), so a caller hands over device tables and plans nothing itself.

Not wired into the engine: ``benchmark/megakernel/`` measures them against the captured decode graph.
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


__all__ = ["MAX_FAST_ROWS", "MAX_ROWS", "megakernel_applies"]
