"""EarTTS RVQ audio codec (``tts_model.audio_codec.*``).

A DAC/Vocos-style ConvNeXt codec: a downsampling encoder and an upsampling
decoder, each a heterogeneous stack of strided resample convs and ConvNeXt
blocks, plus a projected-RVQ quantizer (``prvq``).

Channel schedule (from the checkpoint):
    encoder: 18 ->[384]x3 ->768 ->[768]x3 ->1536 ->[1536]x3 ->512(latent)
    decoder: 512 ->1536 ->[1536]x3 ->768 ->[768]x3 ->384 ->[384]x3 ->18

Built to exact param parity; the encode/decode forward is Phase 5 (needs the
EarTTS reference for stride/orientation of the resample convs). The three
top-level TTS buffers (``_control_codes``, ``codec_silence_tokens``,
``audio_prompt_latents``) are loaded here too.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.model.nemotron_duplex.components._util import RawWeight, param

# Per-index layer spec: ("conv", (shape...)) for a bare-weight resample conv,
# or ("block", ch) for a ConvNeXt block at channel width ch.
_ENCODER_LAYERS = [
    ("conv", (384, 18, 1)),
    ("block", 384), ("block", 384), ("block", 384),
    ("conv", (768, 384, 7)),
    ("block", 768), ("block", 768), ("block", 768),
    ("conv", (1536, 768, 7)),
    ("block", 1536), ("block", 1536), ("block", 1536),
    ("conv", (512, 1536, 9)),
]
_DECODER_LAYERS = [
    ("conv", (512, 1536, 9)),
    ("block", 1536), ("block", 1536), ("block", 1536),
    ("conv", (1536, 768, 7)),
    ("block", 768), ("block", 768), ("block", 768),
    ("conv", (768, 384, 7)),
    ("block", 384), ("block", 384), ("block", 384),
    ("conv", (18, 384, 1)),
]

NUM_QUANTIZERS = 31
CODEBOOK_SIZE = 1024
LATENT_DIM = 512


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: depthwise conv -> LayerNorm -> pointwise expand -> pointwise project."""

    def __init__(self, dim: int, expand: int = 4, kernel: int = 7):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel, groups=dim, padding=kernel // 2)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Conv1d(dim, expand * dim, kernel_size=1)
        self.pwconv2 = nn.Conv1d(expand * dim, dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.pwconv2(torch.nn.functional.gelu(self.pwconv1(x)))
        return residual + x


def _build_stack(spec) -> nn.ModuleList:
    layers: list[nn.Module] = []
    for kind, arg in spec:
        layers.append(RawWeight(*arg) if kind == "conv" else ConvNeXtBlock(arg))
    return nn.ModuleList(layers)


class ProjectedRVQ(nn.Module):
    """Projected residual vector quantizer: per-quantizer codebooks (``mus_list``)
    and running variances (``_variance_list``)."""

    def __init__(self, num_quantizers: int = NUM_QUANTIZERS, codebook: int = CODEBOOK_SIZE, dim: int = LATENT_DIM):
        super().__init__()
        self.mus_list = nn.ParameterList([param(codebook, dim) for _ in range(num_quantizers)])
        self._variance_list = nn.ModuleList([_Variance() for _ in range(num_quantizers)])


class _Variance(nn.Module):
    def __init__(self):
        super().__init__()
        self.variance = param()  # scalar


class AudioCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _CodecHalf(_ENCODER_LAYERS)
        self.decoder = _CodecHalf(_DECODER_LAYERS)
        self.prvq = ProjectedRVQ()

    @staticmethod
    def remap(name: str) -> str | None:
        prefix = "tts_model.audio_codec."
        return name[len(prefix):] if name.startswith(prefix) else None

    def load_weights(self, weights):
        """Load the ``tts_model.audio_codec.*`` subtree."""
        from mstar.model.loader import load_hf_weights

        return load_hf_weights(self, weights, stacked_params=[], name_remapper=self.remap)


class _CodecHalf(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.layers = _build_stack(spec)
