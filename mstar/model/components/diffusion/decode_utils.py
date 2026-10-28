"""VAE decode for the DiT scaffold: the compiled decode and the batch splitting
that keeps it on shapes it was warmed with.

``VAE_DECODE_BATCH_SIZES`` is the scaffold's default ladder; a model may pass its
own to the decoder submodule.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch
from torch import nn

from mstar.engine.accelerator_graph_runner import compile_level
from mstar.model.components.diffusion.compile_utils import with_inductor_rounding

logger = logging.getLogger(__name__)

# torch.compile's mode for the VAE decode per compile level (``mstar serve -O``), not the
# captured forwards' COMPILE_LEVELS: on an H100 at 1024^2, coordinate descent decoded slower
# than max-autotune (38.0 vs 31.1 ms at batch 1) and warmed almost as long, so -O1 keeps
# max-autotune. -O0 warms in 90-100 s instead of 520-610 s and decodes in 40.6 ms.
VAE_COMPILE_MODES = {0: "default", 1: "max-autotune-no-cudagraphs", 2: "max-autotune-no-cudagraphs"}

VAE_DECODE_BATCH_SIZES = (1, 2, 4, 8)


def compile_vae_decode(vae: nn.Module):
    """
    ``vae.decode`` compiled per static shape in the compile level's ``VAE_COMPILE_MODES`` mode
    (no cudagraphs: the engine's runner owns capture). Static, not symbolic: a symbolic batch
    dimension decoded batch 8 in 346 ms against 223 ms for the per-size graph. A fresh compile
    costs tens of seconds per shape under max-autotune, so the decoder warms every batch size it
    will ever call at load and splits larger batches into those sizes (see ``decode_in_chunks``).

    Numerics (H100, 1024^2, measured 2026-09-21): the compiled decode lands ~56 dB from the eager
    one (inductor's GroupNorm / SiLU decompositions), and because the autotuner benchmarks
    candidate conv kernels, two server processes can pick different ones (64 dB apart on the same
    seeds). An "exact" compile that keeps GroupNorm and SiLU on the eager kernels is bit-exact but
    slower than eager (111 vs 89 ms), so the exactness knob for the VAE is ``vae_compile: false``.
    """
    return with_inductor_rounding(
        torch.compile(vae.decode, fullgraph=False, dynamic=False, mode=VAE_COMPILE_MODES[compile_level()]),
        eager_rounding=True,
    )


def decode_in_chunks(decode, latents: torch.Tensor, chunk_sizes: Sequence[int]) -> torch.Tensor:
    """Decode ``latents`` in slices whose batch sizes are all in ``chunk_sizes`` (largest
    first), so a compiled ``decode`` only ever sees the shapes it was warmed with."""
    sizes = sorted(set(int(s) for s in chunk_sizes), reverse=True)
    if not sizes or latents.shape[0] in sizes:
        return decode(latents)
    outputs, start, remaining = [], 0, latents.shape[0]
    while remaining:
        size = next((s for s in sizes if s <= remaining), sizes[-1])
        if size > remaining:
            raise ValueError(f"cannot split a batch of {latents.shape[0]} into chunks of {sizes}")
        outputs.append(decode(latents[start:start + size]))
        start, remaining = start + size, remaining - size
    return torch.cat(outputs)
