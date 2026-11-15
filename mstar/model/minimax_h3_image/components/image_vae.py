"""MiniMax-H3-Image's VAE decoder: ``post_quant_conv`` and H3's ViT decoder in its 2D form,
decoded in 256-px tiles overlapping by at least 64 and blended as the reference does
(``AutoencoderKLMiniMaxH3Image._decode``, tiling on, as released). The encoder is H3's
keyframe encoder (``H3KeyframeEncoder``). Ported from diffusers'
``autoencoder_kl_minimax_h3_image.py``; weights from ``vae/``.
"""

from __future__ import annotations

import torch
from torch import nn

from mstar.model.minimax_h3.components.video_vae import Attend, H3ViTDecoder, TiledViTDecoder, stitch_tiles
from mstar.model.minimax_h3.config import H3VideoVAEConfig


class H3ImageDecoder(TiledViTDecoder):
    """Denormalized latents ``[B, C, H, W]`` -> pixels ``[B, 3, 16H, 16W]`` in the VAE's
    ImageNet-normalized space."""

    def __init__(self, cfg: H3VideoVAEConfig):
        super().__init__()
        if not cfg.decoder_2d:
            raise ValueError("H3ImageDecoder decodes with the 2D ViT decoder (decoder_2d)")
        self.cfg = cfg
        self.post_quant_conv = nn.Conv2d(cfg.latent_channels, cfg.latent_channels, kernel_size=1)
        self.decoder = H3ViTDecoder(cfg)
        self.ratio = cfg.spatial_compression

    def decode_tiles(self, tiles: list[torch.Tensor], attend: Attend, batch: int) -> list[torch.Tensor]:
        """Decode same-shape latent tiles ``batch`` at a time; 1 is the reference's order of work."""
        out: list[torch.Tensor] = []
        for i in range(0, len(tiles), batch):
            chunk = tiles[i:i + batch]
            z = self.post_quant_conv(torch.cat(chunk))
            decoded = self.decoder(z[:, :, None], attend)[:, :, 0]
            out.extend(decoded.split([t.shape[0] for t in chunk]))
        return out

    def decode(self, z: torch.Tensor, attend: Attend, tile_batch: int = 1) -> torch.Tensor:
        z = z.to(self.post_quant_conv.weight.dtype)
        grid, y_overlaps, x_overlaps = self.tile_slices(z.shape[-2], z.shape[-1])
        tiles = [z[..., ys, xs] for row in grid for ys, xs in row]
        decoded = iter(self.decode_tiles(tiles, attend, max(1, int(tile_batch))))
        return stitch_tiles([[next(decoded) for _ in row] for row in grid], y_overlaps, x_overlaps)
