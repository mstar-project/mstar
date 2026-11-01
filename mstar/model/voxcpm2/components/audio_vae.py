# Copyright (c) 2026 OpenBMB.
# SPDX-License-Identifier: Apache-2.0
#
# This file has been modified for M* in 2026: weight norm folded at load, and
# the decoder's receptive field derived for streamed windows.
#
# Original file (OpenBMB/VoxCPM, src/voxcpm/modules/audiovae/audio_vae_v2.py)
# was released under Apache-2.0, with the full license text available at
# https://github.com/OpenBMB/VoxCPM/blob/main/LICENSE.
#
# This modified file is released under the same license.

"""AudioVAE V2 of VoxCPM2: 16 kHz waveform -> 25 Hz latents -> 48 kHz waveform.

A port of the reference ``AudioVAEV2`` (a DAC-style causal conv VAE with an
asymmetric decoder and sample-rate conditioning). Every convolution pads on
the left only and every transposed convolution trims on the right, so a latent
frame's audio depends on a bounded span of earlier frames:
:func:`decoder_receptive_field_frames` derives that span from the modules,
which is how much left context a streamed window needs to reproduce a whole
decode.

Weight norm is folded into plain weights at load (``fold_weight_norm``); the
checkpoint's ``weight_g`` / ``weight_v`` pairs are combined once, in fp32.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.voxcpm2.config import AudioVAEConfig


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, padding: int = 0, output_padding: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.left_pad = padding * 2 - output_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_pad, 0)))


class CausalTransposeConv1d(nn.ConvTranspose1d):
    def __init__(self, *args, padding: int = 0, output_padding: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.right_trim = padding * 2 - output_padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x)[..., : -self.right_trim]


class Snake1d(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + (self.alpha + 1e-9).reciprocal() * torch.sin(self.alpha * x).pow(2)


class CausalResidualUnit(nn.Module):
    def __init__(self, dim: int, dilation: int = 1, kernel: int = 7, groups: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            CausalConv1d(dim, dim, kernel_size=kernel, dilation=dilation, padding=pad, groups=groups),
            Snake1d(dim),
            CausalConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CausalEncoderBlock(nn.Module):
    def __init__(self, output_dim: int, stride: int, groups: int = 1):
        super().__init__()
        input_dim = output_dim // 2
        self.block = nn.Sequential(
            CausalResidualUnit(input_dim, dilation=1, groups=groups),
            CausalResidualUnit(input_dim, dilation=3, groups=groups),
            CausalResidualUnit(input_dim, dilation=9, groups=groups),
            Snake1d(input_dim),
            CausalConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CausalEncoder(nn.Module):
    def __init__(self, d_model: int, latent_dim: int, strides: tuple[int, ...], depthwise: bool):
        super().__init__()
        blocks: list[nn.Module] = [CausalConv1d(1, d_model, kernel_size=7, padding=3)]
        for stride in strides:
            d_model *= 2
            blocks.append(CausalEncoderBlock(d_model, stride, groups=d_model // 2 if depthwise else 1))
        self.fc_mu = CausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)
        self.fc_logvar = CausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)
        self.block = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc_mu(self.block(x))


class CausalDecoderBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int, groups: int = 1):
        super().__init__()
        self.input_channels = input_dim
        self.block = nn.Sequential(
            Snake1d(input_dim),
            CausalTransposeConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
            CausalResidualUnit(output_dim, dilation=1, groups=groups),
            CausalResidualUnit(output_dim, dilation=3, groups=groups),
            CausalResidualUnit(output_dim, dilation=9, groups=groups),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SampleRateCondition(nn.Module):
    """``scale_bias`` conditioning: a per-channel affine picked by output-rate bucket."""

    def __init__(self, input_dim: int, buckets: int):
        super().__init__()
        self.scale_embed = nn.Embedding(buckets, input_dim)
        self.bias_embed = nn.Embedding(buckets, input_dim)

    def forward(self, x: torch.Tensor, sr_index: torch.Tensor) -> torch.Tensor:
        return x * self.scale_embed(sr_index).unsqueeze(-1) + self.bias_embed(sr_index).unsqueeze(-1)


class CausalDecoder(nn.Module):
    def __init__(self, config: AudioVAEConfig):
        super().__init__()
        channels, latent = config.decoder_dim, config.latent_dim
        if config.depthwise:
            layers: list[nn.Module] = [
                CausalConv1d(latent, latent, kernel_size=7, padding=3, groups=latent),
                CausalConv1d(latent, channels, kernel_size=1),
            ]
        else:
            layers = [CausalConv1d(latent, channels, kernel_size=7, padding=3)]
        for i, stride in enumerate(config.decoder_rates):
            input_dim, output_dim = channels // 2**i, channels // 2 ** (i + 1)
            layers.append(
                CausalDecoderBlock(input_dim, output_dim, stride, groups=output_dim if config.depthwise else 1)
            )
        layers += [Snake1d(output_dim), CausalConv1d(output_dim, 1, kernel_size=7, padding=3), nn.Tanh()]
        self.model = nn.ModuleList(layers)
        buckets = len(config.sr_bin_boundaries) + 1
        self.register_buffer(
            "sr_bin_boundaries",
            torch.tensor(config.sr_bin_boundaries, dtype=torch.int32),
            persistent=True,
        )
        self.sr_cond_model = nn.ModuleList(
            [
                SampleRateCondition(layer.input_channels, buckets) if isinstance(layer, CausalDecoderBlock) else None
                for layer in layers
            ]
        )

    def forward(self, z: torch.Tensor, sr_index: torch.Tensor) -> torch.Tensor:
        x = z
        for layer, cond in zip(self.model, self.sr_cond_model, strict=True):
            if cond is not None:
                x = cond(x, sr_index)
            x = layer(x)
        return x


class AudioVAE(nn.Module):
    """``encode``: ``[B, samples@16k]`` -> ``[B, 64, frames]``;
    ``decode``: ``[B, 64, frames]`` -> ``[B, frames * 1920]`` at 48 kHz."""

    def __init__(self, config: AudioVAEConfig):
        super().__init__()
        if config.use_noise_block or config.cond_type != "scale_bias" or config.cond_out_layer:
            raise ValueError("VoxCPM2 AudioVAE port supports the published V2 decoder only")
        if not config.sr_bin_boundaries:
            raise ValueError("VoxCPM2 AudioVAE port needs sample-rate conditioning")
        self.config = config
        self.encoder = CausalEncoder(config.encoder_dim, config.latent_dim, config.encoder_rates, config.depthwise)
        self.decoder = CausalDecoder(config)

    def output_rate_index(self, device) -> torch.Tensor:
        """Bucket of the default output rate, as ``AudioVAEV2.decode`` picks it."""
        rate = torch.full((1,), self.config.out_sample_rate, dtype=torch.int32, device=device)
        return torch.bucketize(rate, self.decoder.sr_bin_boundaries.to(device))

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        hop = self.config.encode_hop
        audio = F.pad(audio, (0, -audio.shape[-1] % hop))
        return self.encoder(audio.unsqueeze(1))

    def decode(self, z: torch.Tensor, sr_index: torch.Tensor | None = None) -> torch.Tensor:
        if sr_index is None:
            sr_index = self.output_rate_index(z.device)
        return self.decoder(z, sr_index.expand(z.shape[0])).squeeze(1)


def fold_weight_norm(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """``weight_g`` / ``weight_v`` pairs -> ``weight`` (``torch.nn.utils.weight_norm``, dim 0)."""
    out: dict[str, torch.Tensor] = {}
    for name, tensor in state.items():
        if name.endswith(".weight_v"):
            continue
        if name.endswith(".weight_g"):
            base = name[: -len("_g")]
            v = state[base + "_v"].float()
            norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
            out[base] = tensor.float() * v / norm
            continue
        out[name] = tensor
    return out


def _earliest_input(module: nn.Module, out_index: int) -> int:
    """First input position output position ``out_index`` of ``module`` reads."""
    if isinstance(module, (Snake1d, nn.Tanh)):
        return out_index
    if isinstance(module, CausalTransposeConv1d):
        kernel, stride = module.kernel_size[0], module.stride[0]
        # input i writes outputs i*stride .. i*stride + kernel - 1; the right trim drops none of o's
        return -((kernel - 1 - out_index) // stride)
    if isinstance(module, CausalConv1d):
        return out_index * module.stride[0] - module.left_pad
    if isinstance(module, CausalResidualUnit):
        index = out_index
        for block in reversed(module.block):
            index = _earliest_input(block, index)
        return min(index, out_index)
    if isinstance(module, (CausalDecoderBlock, nn.Sequential)):
        blocks = module.block if isinstance(module, CausalDecoderBlock) else module
        for block in reversed(blocks):
            out_index = _earliest_input(block, out_index)
        return out_index
    raise TypeError(f"no causal receptive-field rule for {type(module).__name__}")


def decoder_receptive_field_frames(decoder: CausalDecoder) -> int:
    """Latent frames of left context a frame's audio needs to equal a whole decode.

    Walks the decoder backwards from a frame's first output sample; exact for
    the modules modelled in ``_earliest_input`` (20 frames for VoxCPM2). The
    sample-rate conditioning is per channel, so it adds nothing.
    """
    upsample = math.prod(layer.block[1].stride[0] for layer in decoder.model if isinstance(layer, CausalDecoderBlock))
    frame = 10_000
    index = frame * upsample
    for module in reversed(list(decoder.model)):
        index = _earliest_input(module, index)
    return frame - index
