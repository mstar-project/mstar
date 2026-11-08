"""MiniMax-H3's video VAE encoder (a causal 3D CNN) on the keyframes: 256-px tiles,
moments blended in latent space, the posterior sampled under seed 42 on a CPU generator
and rounded to fp16, as the reference's ``encode_vae_condition``. Ported from diffusers'
``autoencoder_kl_minimax_h3.py`` (Apache-2.0), fp32.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.minimax_h3.components.video_vae import split_tiles, stitch_tiles
from mstar.model.minimax_h3.config import PIXEL_MEAN, PIXEL_STD, H3VideoEncoderConfig, H3VideoVAEConfig


class CausalConv3d(nn.Conv3d):
    """Symmetric spatial padding (``mode``), causal temporal padding (zeros before)."""

    def __init__(self, cin: int, cout: int, kernel: int, stride=1, spatial_padding: int = 0,
                 temporal_padding: int = 0, mode: str = "reflect"):
        super().__init__(cin, cout, kernel_size=kernel, stride=stride, padding=0)
        self.spatial_padding, self.temporal_padding, self.mode = spatial_padding, temporal_padding, mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_padding:
            p = self.spatial_padding
            x = F.pad(x, (p, p, p, p, 0, 0), mode=self.mode)
        if self.temporal_padding:
            x = F.pad(x, (0, 0, 0, 0, self.temporal_padding, 0))
        return F.conv3d(x, self.weight, self.bias, stride=self.stride)


class FrameGroupNorm(nn.GroupNorm):
    """GroupNorm with each frame its own sample: statistics never mix across time."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b * t, c, 1, h, w)
        x = super().forward(x)
        return x.view(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


class ResnetBlock(nn.Module):
    def __init__(self, cin: int, cout: int, cfg: H3VideoEncoderConfig):
        super().__init__()
        mode = cfg.padding_mode
        self.norm1 = FrameGroupNorm(cfg.norm_groups, cin, eps=cfg.norm_eps)
        self.conv1 = CausalConv3d(cin, cout, 3, spatial_padding=1, temporal_padding=2, mode=mode)
        self.norm2 = FrameGroupNorm(cfg.norm_groups, cout, eps=cfg.norm_eps)
        self.conv2 = CausalConv3d(cout, cout, 3, spatial_padding=1, temporal_padding=2, mode=mode)
        self.nin_shortcut = CausalConv3d(cin, cout, 1) if cin != cout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv2(F.silu(self.norm2(self.conv1(F.silu(self.norm1(x))))))
        return (x if self.nin_shortcut is None else self.nin_shortcut(x)) + h


class Downsample(nn.Module):
    """Strided 3x3x3 conv; a spatial stride of 2 pads bottom / right by one first."""

    def __init__(self, channels: int, temporal: int, spatial: int, mode: str):
        super().__init__()
        self.spatial, self.mode = spatial, mode
        self.conv = CausalConv3d(channels, channels, 3, stride=(temporal, spatial, spatial), temporal_padding=2,
                                 mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial == 2:
            x = F.pad(x, (0, 1, 0, 1, 0, 0), mode=self.mode)
        return self.conv(x)


class DownLevel(nn.Module):
    def __init__(self, cin: int, cout: int, temporal: int, spatial: int, cfg: H3VideoEncoderConfig):
        super().__init__()
        self.block = nn.ModuleList(
            ResnetBlock(cin if j == 0 else cout, cout, cfg) for j in range(cfg.num_res_blocks)
        )
        self.downsample = Downsample(cout, temporal, spatial, cfg.padding_mode) if temporal * spatial > 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block:
            x = block(x)
        return x if self.downsample is None else self.downsample(x)


class H3VideoEncoder(nn.Module):
    """Pixels ``[B, 3, T, H, W]`` -> moments ``[B, 2 * latent, T', H / 16, W / 16]`` (before
    ``quant_conv``), named as the original checkpoint's ``encoder.``."""

    def __init__(self, cfg: H3VideoEncoderConfig):
        super().__init__()
        chans = cfg.block_channels
        self.conv_in = CausalConv3d(cfg.in_channels, chans[0], 3, spatial_padding=1, temporal_padding=2,
                                    mode=cfg.padding_mode)
        ins = (chans[0],) + chans[:-1]
        self.down = nn.ModuleList(
            DownLevel(ins[i], chans[i], cfg.temporal_down[i], cfg.spatial_down[i], cfg) for i in range(len(chans))
        )
        self.norm_out = FrameGroupNorm(cfg.norm_groups, chans[-1], eps=cfg.norm_eps)
        self.conv_out = CausalConv3d(chans[-1], 2 * cfg.latent_channels, 3, spatial_padding=1, temporal_padding=2,
                                     mode=cfg.padding_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for level in self.down:
            x = level(x)
        return self.conv_out(F.silu(self.norm_out(x)))


class H3KeyframeEncoder(nn.Module):
    """``encoder`` + ``quant_conv``, and the reference's single-frame encode."""

    def __init__(self, cfg: H3VideoEncoderConfig, vae: H3VideoVAEConfig):
        super().__init__()
        self.vae = vae
        self.encoder = H3VideoEncoder(cfg)
        self.quant_conv = nn.Conv3d(2 * cfg.latent_channels, 2 * cfg.latent_channels, kernel_size=1)

    def moments(self, pixels: torch.Tensor) -> torch.Tensor:
        """ImageNet-normalized fp32 pixels ``[1, 3, 1, H, W]`` -> ``[1, 48, 1, h, w]``, tiled."""
        tile, overlap, ratio = self.vae.tile_size, self.vae.tile_overlap_min, self.vae.spatial_compression
        ys, y_lens, y_overlaps = split_tiles(pixels.shape[-2], tile, overlap, ratio)
        xs, x_lens, x_overlaps = split_tiles(pixels.shape[-1], tile, overlap, ratio)
        rows = [
            [self.quant_conv(self.encoder(pixels[..., y:y + ylen, x:x + xlen]))
             for x, xlen in zip(xs, x_lens, strict=True)]
            for y, ylen in zip(ys, y_lens, strict=True)
        ]
        return stitch_tiles(rows, [o // ratio for o in y_overlaps], [o // ratio for o in x_overlaps])

    @torch.no_grad()
    def encode(self, frame: torch.Tensor, seed: int) -> torch.Tensor:
        """uint8 ``[3, H, W]`` on the device -> normalized latents ``[1, 24, 1, H / 16, W / 16]``
        fp32 on that device (``encode_vae_condition``)."""
        device = frame.device
        # every host constant moved before the encoder runs: a pageable copy waits for the stream
        mean, std, latents_mean, latents_std = (
            torch.tensor(values, device=device).view(1, -1, 1, 1, 1)
            for values in (PIXEL_MEAN, PIXEL_STD, self.vae.latents_mean, self.vae.latents_std)
        )
        pixels = (frame[None, :, None].to(torch.float32).div(255.0) - mean) / std
        moments = self.moments(pixels.to(self.quant_conv.weight.dtype))
        # DiagonalGaussianDistribution.sample, its noise drawn on the CPU (pinned, so moving it
        # doesn't wait for the encoder either)
        posterior_mean, logvar = moments.chunk(2, dim=1)
        posterior_std = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
        noise = torch.randn(posterior_mean.shape, generator=torch.Generator().manual_seed(seed), dtype=moments.dtype,
                            pin_memory=device.type != "cpu").to(device, non_blocking=True)
        latents = (posterior_mean + posterior_std * noise).to(torch.float16).float()
        return (latents - latents_mean) / latents_std
