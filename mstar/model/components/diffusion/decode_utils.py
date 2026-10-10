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

from mstar.model.components.diffusion.compile_utils import with_inductor_rounding

logger = logging.getLogger(__name__)

VAE_COMPILE_MODE = "max-autotune-no-cudagraphs"

VAE_DECODE_BATCH_SIZES = (1, 2, 4, 8)


def compile_vae_decode(vae: nn.Module):
    """
    ``vae.decode`` compiled per static shape with inductor autotuning (no cudagraphs: the
    engine's runner owns capture). Static, not symbolic: a symbolic batch dimension decoded
    batch 8 in 346 ms against 223 ms for the per-size graph. A fresh max-autotune compile costs
    tens of seconds, so the decoder warms every batch size it will ever call at load and splits
    larger batches into those sizes (see ``decode_in_chunks``).

    Numerics (H100, 1024^2, measured 2026-09-21): the compiled decode lands ~56 dB from the eager
    one (inductor's GroupNorm / SiLU decompositions), and because the autotuner benchmarks
    candidate conv kernels, two server processes can pick different ones (64 dB apart on the same
    seeds). An "exact" compile that keeps GroupNorm and SiLU on the eager kernels is bit-exact but
    slower than eager (111 vs 89 ms), so the exactness knob for the VAE is ``vae_compile: false``.
    """
    return with_inductor_rounding(
        torch.compile(vae.decode, fullgraph=False, dynamic=False, mode=VAE_COMPILE_MODE),
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
