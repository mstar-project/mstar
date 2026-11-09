"""The decoder half of MiniMax-H3's audio VAE: a 1x1 projection into a BigVGAN.

Mono 32 kHz, 800 samples per latent; the pipeline carries stereo as two batch items.
fp32, as the reference keeps it (its BigVGAN degrades audibly in bf16).
"""

from __future__ import annotations

import torch
from torch import nn

from mstar.model.components.bigvgan import BigVGANGenerator
from mstar.model.minimax_h3.config import H3AudioVAEConfig


class H3AudioDecoder(nn.Module):
    """Denormalized latents ``[batch, 32, frames]`` -> waveform ``[batch, 1, frames * 800]``
    in ``[-1, 1]``."""

    def __init__(self, config: H3AudioVAEConfig):
        super().__init__()
        self.dec_in_proj = nn.Conv1d(config.latent_channels, config.latent_dim, 1)
        self.decoder = BigVGANGenerator(
            config.latent_dim, config.decoder_dim, config.upsample_rates, config.upsample_kernel_sizes,
            config.resblock_kernel_sizes, config.resblock_dilations, final_bias=False, final_tanh=False,
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.dec_in_proj(latents))
