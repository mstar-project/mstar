"""Native KL autoencoder (the SD / FLUX family VAE) for the image DiTs.

An exact port of the diffusers ``AutoencoderKL`` / ``AutoencoderKLFlux2`` structure for
the configurations the served checkpoints use (``DownEncoderBlock2D`` /
``UpDecoderBlock2D`` stages, a mid block with one single-head attention, SiLU):

    encode: conv_in -> down stages (resnets, strided 3x3 downsample between) -> mid
            (resnet, attention, resnet) -> GroupNorm/SiLU/conv_out (2C ch) [-> quant_conv]
            -> mean half of the (mean, logvar) split
    decode: [post_quant_conv ->] conv_in -> mid -> up stages (resnets, nearest x2 + conv)
            -> GroupNorm/SiLU/conv_out (3 ch)

What differs between checkpoints is only the config (channels, quant convs) and how the
latents are normalized on the way in and out — a scaling/shift pair (FLUX.1 / Z-Image)
or BatchNorm running statistics over 2x2-patchified latents (FLUX.2). Those live in
the model packages (``Flux2VAE`` subclasses this) or in the pipeline math.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class AutoencoderKLConfig:
    """``vae/config.json`` of a diffusers ``AutoencoderKL``-family checkpoint."""

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    use_quant_conv: bool = False
    use_post_quant_conv: bool = False
    mid_block_add_attention: bool = True
    scaling_factor: float | None = None
    shift_factor: float | None = None

    @property
    def spatial_compression(self) -> int:
        """Pixels per latent cell along each axis (2 ** (#stages - 1))."""
        return 2 ** (len(self.block_out_channels) - 1)

    @classmethod
    def from_dict(cls, cfg: dict) -> "AutoencoderKLConfig":
        if cfg.get("act_fn", "silu") != "silu":
            raise NotImplementedError(f"AutoencoderKL port implements act_fn='silu', got {cfg['act_fn']!r}")
        if any(t != "DownEncoderBlock2D" for t in cfg.get("down_block_types", ())) or any(
            t != "UpDecoderBlock2D" for t in cfg.get("up_block_types", ())
        ):
            raise NotImplementedError("AutoencoderKL port implements DownEncoderBlock2D / UpDecoderBlock2D only")
        return cls(
            in_channels=int(cfg.get("in_channels", 3)),
            out_channels=int(cfg.get("out_channels", 3)),
            latent_channels=int(cfg["latent_channels"]),
            block_out_channels=tuple(int(c) for c in cfg["block_out_channels"]),
            layers_per_block=int(cfg.get("layers_per_block", 2)),
            norm_num_groups=int(cfg.get("norm_num_groups", 32)),
            use_quant_conv=bool(cfg.get("use_quant_conv", True)),
            use_post_quant_conv=bool(cfg.get("use_post_quant_conv", True)),
            mid_block_add_attention=bool(cfg.get("mid_block_add_attention", True)),
            scaling_factor=None if cfg.get("scaling_factor") is None else float(cfg["scaling_factor"]),
            shift_factor=None if cfg.get("shift_factor") is None else float(cfg["shift_factor"]),
        )


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
    def __init__(self, config: AutoencoderKLConfig):
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
    def __init__(self, config: AutoencoderKLConfig):
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


class AutoencoderKL(nn.Module):
    """Encoder + decoder with the optional 1x1 quant convs; ``encode`` returns the posterior
    mode (mean), which every pipeline here uses (``sample_mode="argmax"``)."""

    def __init__(self, config: AutoencoderKLConfig):
        super().__init__()
        self.config = config
        z = config.latent_channels
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.quant_conv = nn.Conv2d(2 * z, 2 * z, 1) if config.use_quant_conv else None
        self.post_quant_conv = nn.Conv2d(z, z, 1) if config.use_post_quant_conv else None

    @property
    def dtype(self) -> torch.dtype:
        return self.decoder.conv_in.weight.dtype

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """Pixels in ``[-1, 1]`` -> the posterior mode (mean) ``[B, C, H/8, W/8]``."""
        moments = self.encoder(pixels)
        if self.quant_conv is not None:
            moments = self.quant_conv(moments)
        mean, _logvar = moments.chunk(2, dim=1)
        return mean

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self.post_quant_conv is not None:
            latents = self.post_quant_conv(latents)
        return self.decoder(latents)

    # FLUX.1 / Z-Image style latent normalization (``scaling_factor`` / ``shift_factor``).
    def scale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Encode side: ``(z - shift) * scale``."""
        return (latents - self.config.shift_factor) * self.config.scaling_factor

    def unscale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode side: ``z / scale + shift``."""
        return latents / self.config.scaling_factor + self.config.shift_factor


def remap_autoencoder_kl_key(name: str) -> str:
    """diffusers ``AutoencoderKL`` checkpoint key -> native parameter path."""
    name = name.replace(".down_blocks.", ".down_stages.").replace(".up_blocks.", ".up_stages.")
    name = name.replace(".downsamplers.0.conv.", ".downsample.").replace(".upsamplers.0.conv.", ".upsample.")
    name = name.replace(".mid_block.resnets.0.", ".mid.resnet1.").replace(".mid_block.resnets.1.", ".mid.resnet2.")
    name = name.replace(".mid_block.attentions.0.", ".mid.attn.").replace(".mid.attn.to_out.0.", ".mid.attn.to_out.")
    return name.replace(".conv_norm_out.", ".norm_out.")
