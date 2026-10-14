"""FLUX.2 VAE: the shared KL autoencoder plus the checkpoint's BatchNorm latent statistics.

``AutoencoderKLFlux2`` is the SD-style KL autoencoder with a 32-channel latent; the only
FLUX.2-specific part is that the pipelines normalize the 2x2-patchified latents
(``[B, 128, h, w]``) with the ``BatchNorm2d`` running statistics stored in the checkpoint
(``bn.running_mean`` / ``bn.running_var``), applied explicitly with the reference's op order
(:meth:`normalize_latents` / :meth:`denormalize_latents`). They are kept in the checkpoint
dtype as frozen parameters so the streaming loader fills and completeness-checks them.
"""

from __future__ import annotations

import torch
from torch import nn

from mstar.model.components.diffusion.autoencoder_kl import AutoencoderKL, remap_autoencoder_kl_key
from mstar.model.flux2_klein.config import Flux2VaeConfig


class Flux2VAE(AutoencoderKL):
    def __init__(self, config: Flux2VaeConfig):
        super().__init__(config.autoencoder)
        self.flux2_config = config
        self.latent_mean = nn.Parameter(torch.zeros(config.patched_latent_channels), requires_grad=False)
        self.latent_var = nn.Parameter(torch.ones(config.patched_latent_channels), requires_grad=False)

    def _latent_std(self, like: torch.Tensor) -> torch.Tensor:
        # sqrt in the statistics' own dtype, then cast — the pipelines' order.
        return torch.sqrt(self.latent_var.view(1, -1, 1, 1) + self.flux2_config.batch_norm_eps).to(like.dtype)

    def normalize_latents(self, patched: torch.Tensor) -> torch.Tensor:
        """``(x - mean) / std`` over ``[B, 128, h, w]`` patchified latents (encode side)."""
        mean = self.latent_mean.view(1, -1, 1, 1).to(patched.dtype)
        return (patched - mean) / self._latent_std(patched)

    def denormalize_latents(self, patched: torch.Tensor) -> torch.Tensor:
        """``x * std + mean`` (decode side)."""
        mean = self.latent_mean.view(1, -1, 1, 1).to(patched.dtype)
        return patched * self._latent_std(patched) + mean


def remap_flux2_vae_key(name: str) -> str:
    """``AutoencoderKLFlux2`` checkpoint key -> native path (the KL remap + the BN statistics)."""
    name = remap_autoencoder_kl_key(name)
    return name.replace("bn.running_mean", "latent_mean").replace("bn.running_var", "latent_var")
