"""Native FLUX.2 VAE (exact port of diffusers ``AutoencoderKLFlux2`` for the klein
checkpoints' configuration: an SD-style KL autoencoder with a 32-channel latent).

    encode: conv_in -> 4 down stages (2 resnets each, strided 3x3 downsample between) -> mid
            (resnet, single-head attention, resnet) -> GroupNorm/SiLU/conv_out (64 ch) -> quant_conv
            -> mean half of the (mean, logvar) split
    decode: post_quant_conv -> conv_in -> mid -> 4 up stages (3 resnets, nearest x2 + conv) ->
            GroupNorm/SiLU/conv_out (3 ch)

The checkpoint's ``BatchNorm2d`` running statistics normalize the 2x2-patchified
latents (``[B, 128, h, w]``); the pipelines apply them explicitly, so they are kept
as plain buffers here (``latent_mean`` / ``latent_var``) in the checkpoint dtype
and applied with the reference's op order (:meth:`normalize_latents` /
:meth:`denormalize_latents`).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.flux2_klein.config import Flux2VaeConfig


class ResnetBlock(nn.Module):
    """GroupNorm -> SiLU -> conv3x3 -> GroupNorm -> SiLU -> conv3x3, plus a 1x1 shortcut when
    the channel count changes (diffusers ``ResnetBlock2D`` with no time embedding)."""

    def __init__(self, in_channels: int, out_channels: int, groups: int, eps: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return x + h


class MidAttention(nn.Module):
    """The mid block's spatial self-attention: one head over the channel dim, GroupNorm on the
    input, residual add (diffusers ``Attention(residual_connection=True)`` via SDPA)."""

    def __init__(self, channels: int, groups: int, eps: float):
        super().__init__()
        self.group_norm = nn.GroupNorm(groups, channels, eps=eps, affine=True)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        # Flatten first and normalize the transposed 3D view, in the reference's order: the
        # GroupNorm kernel's reduction differs between the 4D and the 3D layout in the last
        # bit for channels-last inputs.
        h = x.view(batch, channels, height * width).transpose(1, 2)  # [B, HW, C]
        h = self.group_norm(h.transpose(1, 2)).transpose(1, 2)
        q, k, v = (t.unsqueeze(1) for t in (self.to_q(h), self.to_k(h), self.to_v(h)))  # one head
        h = F.scaled_dot_product_attention(q, k, v, is_causal=False).squeeze(1).to(q.dtype)
        h = self.to_out(h).transpose(1, 2).reshape(batch, channels, height, width)
        return h + x


class MidBlock(nn.Module):
    def __init__(self, channels: int, groups: int, eps: float, add_attention: bool):
        super().__init__()
        self.resnet1 = ResnetBlock(channels, channels, groups, eps)
        self.attn = MidAttention(channels, groups, eps) if add_attention else None
        self.resnet2 = ResnetBlock(channels, channels, groups, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet1(x)
        if self.attn is not None:
            x = self.attn(x)
        return self.resnet2(x)


class DownStage(nn.Module):
    """``layers_per_block`` resnets then (all but the last stage) a stride-2 conv with the
    reference's asymmetric (0, 1, 0, 1) zero pad."""

    def __init__(self, in_channels: int, out_channels: int, num_layers: int, groups: int, eps: float, downsample: bool):
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(in_channels if i == 0 else out_channels, out_channels, groups, eps) for i in range(num_layers)
        )
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=0) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.downsample is not None:
            x = self.downsample(F.pad(x, (0, 1, 0, 1)))
        return x


class UpStage(nn.Module):
    """``layers_per_block + 1`` resnets then (all but the last stage) nearest x2 + conv3x3."""

    def __init__(self, in_channels: int, out_channels: int, num_layers: int, groups: int, eps: float, upsample: bool):
        super().__init__()
        self.resnets = nn.ModuleList(
            ResnetBlock(in_channels if i == 0 else out_channels, out_channels, groups, eps) for i in range(num_layers)
        )
        self.upsample = nn.Conv2d(out_channels, out_channels, 3, padding=1) if upsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        if self.upsample is not None:
            x = self.upsample(F.interpolate(x, scale_factor=2.0, mode="nearest"))
        return x


class Encoder(nn.Module):
    def __init__(self, config: Flux2VaeConfig):
        super().__init__()
        chans, groups, eps = config.block_out_channels, config.norm_num_groups, 1e-6
        self.conv_in = nn.Conv2d(config.in_channels, chans[0], 3, padding=1)
        self.down_stages = nn.ModuleList()
        prev = chans[0]
        for i, out in enumerate(chans):
            self.down_stages.append(
                DownStage(prev, out, config.layers_per_block, groups, eps, downsample=i < len(chans) - 1)
            )
            prev = out
        self.mid = MidBlock(chans[-1], groups, eps, config.mid_block_add_attention)
        self.norm_out = nn.GroupNorm(groups, chans[-1], eps=eps)
        self.conv_out = nn.Conv2d(chans[-1], 2 * config.latent_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for stage in self.down_stages:
            x = stage(x)
        x = self.mid(x)
        return self.conv_out(F.silu(self.norm_out(x)))


class Decoder(nn.Module):
    def __init__(self, config: Flux2VaeConfig):
        super().__init__()
        chans, groups, eps = config.block_out_channels, config.norm_num_groups, 1e-6
        rev = list(reversed(chans))
        self.conv_in = nn.Conv2d(config.latent_channels, rev[0], 3, padding=1)
        self.mid = MidBlock(rev[0], groups, eps, config.mid_block_add_attention)
        self.up_stages = nn.ModuleList()
        prev = rev[0]
        for i, out in enumerate(rev):
            self.up_stages.append(
                UpStage(prev, out, config.layers_per_block + 1, groups, eps, upsample=i < len(rev) - 1)
            )
            prev = out
        self.norm_out = nn.GroupNorm(groups, rev[-1], eps=eps)
        self.conv_out = nn.Conv2d(rev[-1], config.out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.mid(self.conv_in(z))
        for stage in self.up_stages:
            x = stage(x)
        return self.conv_out(F.silu(self.norm_out(x)))


class Flux2VAE(nn.Module):
    def __init__(self, config: Flux2VaeConfig):
        super().__init__()
        self.config = config
        z = config.latent_channels
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.quant_conv = nn.Conv2d(2 * z, 2 * z, 1) if config.use_quant_conv else None
        self.post_quant_conv = nn.Conv2d(z, z, 1) if config.use_post_quant_conv else None
        # BatchNorm running statistics over the patchified latent channels; loaded from the
        # checkpoint (``bn.running_mean`` / ``bn.running_var``), never updated. Frozen
        # parameters rather than buffers so the streaming loader (which fills parameters)
        # loads and completeness-checks them like any other tensor.
        self.latent_mean = nn.Parameter(torch.zeros(config.patched_latent_channels), requires_grad=False)
        self.latent_var = nn.Parameter(torch.ones(config.patched_latent_channels), requires_grad=False)

    @property
    def dtype(self) -> torch.dtype:
        return self.decoder.conv_in.weight.dtype

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Pixels in ``[-1, 1]`` -> the posterior mode (mean) ``[B, C, H/8, W/8]``
        (``retrieve_latents(..., sample_mode="argmax")``)."""
        moments = self.encoder(pixels)
        if self.quant_conv is not None:
            moments = self.quant_conv(moments)
        mean, _logvar = moments.chunk(2, dim=1)
        return mean

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.post_quant_conv is not None:
            latents = self.post_quant_conv(latents)
        return self.decoder(latents)

    def _latent_std(self, like: torch.Tensor) -> torch.Tensor:
        # sqrt in the statistics' own dtype, then cast — the pipelines' order.
        return torch.sqrt(self.latent_var.view(1, -1, 1, 1) + self.config.batch_norm_eps).to(like.dtype)

    def normalize_latents(self, patched: torch.Tensor) -> torch.Tensor:
        """``(x - mean) / std`` over ``[B, 128, h, w]`` patchified latents (encode side)."""
        mean = self.latent_mean.view(1, -1, 1, 1).to(patched.dtype)
        return (patched - mean) / self._latent_std(patched)

    def denormalize_latents(self, patched: torch.Tensor) -> torch.Tensor:
        """``x * std + mean`` (decode side)."""
        mean = self.latent_mean.view(1, -1, 1, 1).to(patched.dtype)
        return patched * self._latent_std(patched) + mean
