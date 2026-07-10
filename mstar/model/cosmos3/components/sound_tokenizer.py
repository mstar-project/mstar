"""Cosmos3 AVAE sound tokenizer (decoder-only), ported from the diffusers-format
``sound_tokenizer/`` checkpoint component that ships with the Cosmos3 models.

The checkpoint is an Oobleck-style 1D decoder (Snake activations, weight-normed
convs) that turns joint-denoised sound latents ``[B, C=64, T]`` into a waveform
``[B, channels=2, T * hop_size]`` at 48 kHz. The safetensors ``decoder.*`` keys
use the classic ``weight_g``/``weight_v`` weight-norm naming, which this module
reproduces one-to-one so loading is a plain name-matching stream with a
completeness check (same design as the transformer loader).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import weight_norm


class Snake1d(nn.Module):
    """1D Snake activation (``x + 1/b * sin^2(a x)``) with log-scale parameters."""

    def __init__(self, hidden_dim: int, logscale: bool = True):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.beta = nn.Parameter(torch.zeros(1, hidden_dim, 1))
        self.logscale = logscale

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        alpha = torch.exp(self.alpha) if self.logscale else self.alpha
        beta = torch.exp(self.beta) if self.logscale else self.beta
        hidden_states = hidden_states.reshape(shape[0], shape[1], -1)
        hidden_states = hidden_states + (beta + 1e-9).reciprocal() * torch.sin(alpha * hidden_states).pow(2)
        return hidden_states.reshape(shape)


class OobleckResidualUnit(nn.Module):
    """Dilated residual unit (snake -> conv k7 -> snake -> conv k1)."""

    def __init__(self, dimension: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dimension)
        self.conv1 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=7, dilation=dilation, padding=pad))
        self.snake2 = Snake1d(dimension)
        self.conv2 = weight_norm(nn.Conv1d(dimension, dimension, kernel_size=1))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        output = self.conv2(self.snake2(self.conv1(self.snake1(hidden_state))))
        padding = (hidden_state.shape[-1] - output.shape[-1]) // 2
        if padding > 0:
            hidden_state = hidden_state[..., padding:-padding]
        return hidden_state + output


class OobleckDecoderBlock(nn.Module):
    """Upsampling block: snake -> transposed conv (stride upsample) -> 3 residual units."""

    def __init__(self, input_dim: int, output_dim: int, stride: int = 1, output_padding: int = 0):
        super().__init__()
        self.snake1 = Snake1d(input_dim)
        self.conv_t1 = weight_norm(
            nn.ConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=output_padding,
            )
        )
        self.res_unit1 = OobleckResidualUnit(output_dim, dilation=1)
        self.res_unit2 = OobleckResidualUnit(output_dim, dilation=3)
        self.res_unit3 = OobleckResidualUnit(output_dim, dilation=9)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv_t1(self.snake1(hidden_state))
        hidden_state = self.res_unit1(hidden_state)
        hidden_state = self.res_unit2(hidden_state)
        return self.res_unit3(hidden_state)


class OobleckDecoder(nn.Module):
    """Latent [B, C_latent, T] -> waveform [B, audio_channels, T * prod(strides)]."""

    def __init__(
        self,
        channels: int,
        input_channels: int,
        audio_channels: int,
        upsampling_ratios: list[int],
        channel_multiples: list[int],
    ):
        super().__init__()
        strides = upsampling_ratios
        channel_multiples = [1] + channel_multiples

        self.conv1 = weight_norm(nn.Conv1d(input_channels, channels * channel_multiples[-1], kernel_size=7, padding=3))
        self.block = nn.ModuleList(
            OobleckDecoderBlock(
                input_dim=channels * channel_multiples[len(strides) - i],
                output_dim=channels * channel_multiples[len(strides) - i - 1],
                stride=stride,
                output_padding=stride % 2,
            )
            for i, stride in enumerate(strides)
        )
        self.snake1 = Snake1d(channels)
        self.conv2 = weight_norm(nn.Conv1d(channels, audio_channels, kernel_size=7, padding=3, bias=False))

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.conv1(hidden_state)
        for layer in self.block:
            hidden_state = layer(hidden_state)
        return self.conv2(self.snake1(hidden_state))


def _config_get(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        value = config.get(key)
        if value is not None:
            return value
    return default


class Cosmos3SoundTokenizer(nn.Module):
    """Decoder-only AVAE for Cosmos3 sound latents.

    Built from the checkpoint's ``sound_tokenizer/config.json`` (Oobleck decoder
    geometry, sample rate, channel count, hop size) and loaded strictly from
    ``sound_tokenizer/diffusion_pytorch_model.safetensors`` — every decoder
    parameter must be filled, so a layout drift surfaces as a load error rather
    than silently wrong audio.
    """

    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"

    def __init__(self, config: dict[str, Any]):
        super().__init__()
        self.sample_rate = int(_config_get(config, "sampling_rate", "sample_rate", default=48000))
        stereo_default = 2 if bool(config.get("stereo", True)) else 1
        self.audio_channels = int(_config_get(config, "dec_out_channels", "audio_channels", default=stereo_default))
        self.latent_ch = int(_config_get(config, "vocoder_input_dim", "io_channels", "latent_ch", default=64))
        dec_strides = [int(s) for s in _config_get(config, "dec_strides", default=[2, 4, 5, 6, 8])]
        self.hop_size = int(_config_get(config, "hop_size", default=math.prod(dec_strides)))
        if math.prod(dec_strides) != self.hop_size:
            raise ValueError(
                "Cosmos3 sound tokenizer dec_strides product must equal hop_size: "
                f"prod({dec_strides})={math.prod(dec_strides)}, hop_size={self.hop_size}"
            )
        self.latent_fps = float(self.sample_rate) / float(self.hop_size)

        # Optional latent (de)normalization; the published checkpoints use none
        # (``latent_mean``/``latent_std`` null), but the tanh variant is kept for
        # tokenizer revisions that set it.
        normalization_type = str(_config_get(config, "normalization_type", default="none"))
        if normalization_type == "none" and bool(_config_get(config, "normalize_latents", default=False)):
            normalization_type = "tanh"
        self.normalization_type = normalization_type
        self.tanh_input_scale = float(_config_get(config, "tanh_input_scale", default=1.5))
        self.tanh_output_scale = float(_config_get(config, "tanh_output_scale", default=3.5))
        self.tanh_clamp = float(_config_get(config, "tanh_clamp", default=0.995))

        self.decoder = OobleckDecoder(
            channels=int(_config_get(config, "dec_dim", default=320)),
            input_channels=self.latent_ch,
            audio_channels=self.audio_channels,
            upsampling_ratios=list(reversed(dec_strides)),
            channel_multiples=list(_config_get(config, "dec_c_mults", default=[1, 2, 4, 8, 16])),
        )

    @classmethod
    def from_pretrained(
        cls, checkpoint_dir: str | Path, device: str = "cpu", dtype: torch.dtype = torch.bfloat16
    ) -> "Cosmos3SoundTokenizer":
        tdir = Path(checkpoint_dir) / "sound_tokenizer"
        with open(tdir / cls.CONFIG_NAME) as f:
            config = json.load(f)
        model = cls(config)

        from safetensors.torch import load_file

        # Load on CPU and move with the module: safetensors' direct-to-cuda path
        # is unreliable inside freshly spawned worker processes.
        state_dict = load_file(str(tdir / cls.WEIGHTS_NAME))
        expected = set(model.state_dict().keys())
        # Published checkpoints ship the full AVAE (encoder + decoder); we build
        # and run only the decoder (sound is decode-only — there is no
        # sound-as-input path), so keep the decoder tensors and ignore any
        # encoder tensors the checkpoint carries.
        state_dict = {k: v for k, v in state_dict.items() if k in expected}
        missing = sorted(expected - set(state_dict))
        if missing:
            raise KeyError(
                f"Cosmos3 sound tokenizer missing keys under {tdir}: {missing[:10]}"
            )
        model.load_state_dict(state_dict, strict=True)
        model.eval().requires_grad_(False)
        return model.to(device=device, dtype=dtype)

    def get_audio_num_samples(self, num_latent_frames: int) -> int:
        return int(num_latent_frames) * self.hop_size

    def get_latent_num_frames(self, num_audio_samples: int) -> int:
        return max(1, math.ceil(max(1, int(num_audio_samples)) / self.hop_size))

    def _denormalize(self, latent: torch.Tensor) -> torch.Tensor:
        if self.normalization_type == "none":
            return latent
        if self.normalization_type == "tanh":
            in_dtype = latent.dtype
            z = torch.clamp(latent.float() / self.tanh_output_scale, -self.tanh_clamp, self.tanh_clamp)
            return (torch.atanh(z) * self.tanh_input_scale).to(in_dtype)
        raise ValueError(f"Unsupported sound tokenizer normalization_type={self.normalization_type!r}")

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Sound latents ``[B, C, T]`` or ``[C, T]`` -> waveform in [-1, 1]
        (``[B, audio_channels, T * hop_size]``, squeezed for unbatched input)."""
        squeeze = latents.ndim == 2
        if squeeze:
            latents = latents.unsqueeze(0)
        dtype = self.decoder.conv2.weight_v.dtype
        audio = self.decoder(self._denormalize(latents).to(dtype)).clamp(-1.0, 1.0)
        return audio.squeeze(0) if squeeze else audio
