"""GR00T's DT: the action head's transformer, and the pieces around it.

The DT is 32 blocks at 32 heads x 48, conditioned on the denoising timestep
through AdaLN, alternating cross-attention over the bridge's rows with
self-attention over its own 41 tokens.

Attention arrives as a callable, so nothing here knows whether it is a CPU
stand-in or one of M\\*'s ragged resources (C07 supplies the second).
"""
from __future__ import annotations

import torch
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch import nn

# The sinusoid's width. Not a config field: the checkpoint's
# `timestep_encoder.timestep_embedder.linear_1` is 1536x256, so 256 is what
# the weights were trained against.
_TIME_PROJ_CHANNELS = 256


class TimestepEncoder(nn.Module):
    """The denoising timestep as a conditioning vector, ``[B] -> [B, dim]``.

    Both halves are diffusers' own classes, which is what the reference
    imports too, so this is an import rather than a port -- and
    ``timestep_embedder.linear_1`` / ``linear_2`` are the checkpoint's keys.
    Note the flip: ``flip_sin_to_cos=True`` here, where the action encoder's
    own sinusoid does not flip (see ``SinusoidalPositionalEncoding``).
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=_TIME_PROJ_CHANNELS,
            flip_sin_to_cos=True,
            downscale_freq_shift=1,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=_TIME_PROJ_CHANNELS, time_embed_dim=embedding_dim,
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        # the reference casts the projection to the weights' dtype before the
        # embedder, which matters once the model is bf16
        dtype = next(self.parameters()).dtype
        return self.timestep_embedder(self.time_proj(timesteps).to(dtype))
