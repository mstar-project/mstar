"""Wan2.2-TI2V-5B component modules.

Native components: the video DiT (``dit.py`` — patchify, 3D-RoPE attention
blocks, adaLN head, loaded through ``wan22.weight_loader``) and the inline
UniPC solver (``unipc.py``). The UMT5 text encoder and the Wan2.2-VAE stay
thin diffusers wrappers, constructed in ``Wan22Model.get_submodule``.
"""

from mstar.model.wan22.components.dit import (
    Wan22DiT,
    Wan22DiTAttention,
    Wan22DiTBlock,
    Wan22RoPE3D,
    Wan22TimeTextEmbedding,
)

__all__ = [
    "Wan22DiT",
    "Wan22DiTAttention",
    "Wan22DiTBlock",
    "Wan22RoPE3D",
    "Wan22TimeTextEmbedding",
]
