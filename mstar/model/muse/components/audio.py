"""Muse Spark's audio tower: raw 24 kHz mono waveform in, one LLM-width
embedding per 1920 samples (80 ms) out.

Ported from transformers' ``modeling_muse_spark.py`` (Apache-2.0). Four
stages; each folds ``patch_size`` steps into channels (a Linear and an
RMSNorm), then runs residual blocks of a causal depthwise conv (kernel 7,
left-padded), an RMSNorm and a GELU MLP. Then a norm, a projection to the
LLM's width, a norm, and the text embedding's weightless norm, which the
reference applies to audio features as it does to token embeddings.

Every op is causal or per token, so the ``audio_encoder`` node runs a clip in
chunks of whole tokens (`ChunkedCausalEncoder`) to its whole-clip output: each
conv carries its last ``kernel - 1`` input steps to the next chunk (the
reference's ``MuseSparkAudioConv1dPaddingCache``) in the node's recurrent pool
(``AUDIO_CONV``, one block per conv), which the convs bind.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.engine.resources.recurrent.config import RecurrentBlockConfig
from mstar.model.components import RMSNorm
from mstar.model.components.norm import rms_norm_no_weight
from mstar.model.muse.config import AUDIO_CONV, MuseSparkAudioConfig


def _state_block(stage: int, block: int) -> str:
    return f"s{stage}.{block}"


def conv_state_blocks(config: MuseSparkAudioConfig, dtype: torch.dtype) -> dict[str, RecurrentBlockConfig]:
    """The ``AUDIO_CONV`` pool's blocks: each conv's left context,
    ``(channels, kernel - 1)``."""
    return {
        _state_block(stage, block): RecurrentBlockConfig((hidden, config.conv_kernel_size - 1), dtype)
        for stage, (hidden, depth) in enumerate(zip(config.hidden_sizes, config.depths, strict=True))
        for block in range(depth)
    }


class _PatchEmbedding(nn.Module):
    def __init__(self, config: MuseSparkAudioConfig, stage: int):
        super().__init__()
        channels_in = 1 if stage == 0 else config.hidden_sizes[stage - 1]
        self.patch_size = config.patch_sizes[stage]
        self.proj = nn.Linear(channels_in * self.patch_size, config.hidden_sizes[stage], bias=True)
        self.norm = RMSNorm(config.hidden_sizes[stage], eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        return self.norm(self.proj(x.reshape(b, t // self.patch_size, self.patch_size * c)))


class _MLP(nn.Module):
    def __init__(self, hidden: int, ratio: float):
        super().__init__()
        self.fc1 = nn.Linear(hidden, int(hidden * ratio), bias=True)
        self.fc2 = nn.Linear(int(hidden * ratio), hidden, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _ConvBlock(nn.Module):
    def __init__(self, config: MuseSparkAudioConfig, stage: int, block: int):
        super().__init__()
        hidden = config.hidden_sizes[stage]
        self.conv = nn.Conv1d(hidden, hidden, config.conv_kernel_size, groups=hidden, bias=True)
        self.norm = RMSNorm(hidden, eps=config.rms_norm_eps)
        self.mlp = _MLP(hidden, config.mlp_ratio)
        self.left_pad = config.conv_kernel_size - 1
        # this conv's left context in the AUDIO_CONV pool
        self.state_block = _state_block(stage, block)

    def bind_resources(self, resources: dict) -> None:
        """See ``NodeSubmodule.bind_node_resources``."""
        self.pool = resources[AUDIO_CONV]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` is ``(rows, steps, channels)``, a row per row of the step."""
        steps = x.transpose(1, 2)
        # a row with no state (a clip's first chunk, a replay's padding row)
        # reads zeros whatever its slot holds
        held = self.pool.gather(self.state_block)
        padded = torch.cat([held.to(steps.dtype), steps], dim=-1)
        # the next chunk's left context
        tail = padded[..., padded.shape[-1] - self.left_pad:]
        self.pool.scatter_(self.state_block, tail.to(held.dtype))
        # contiguous: the norm's CUDA kernel needs rows of channels, not the conv's layout
        y = self.conv(padded).transpose(1, 2).contiguous()
        return x + self.mlp(self.norm(y))


class _Stage(nn.Module):
    def __init__(self, config: MuseSparkAudioConfig, stage: int):
        super().__init__()
        self.patch_embedder = _PatchEmbedding(config, stage)
        self.conv_blocks = nn.ModuleList(_ConvBlock(config, stage, i) for i in range(config.depths[stage]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embedder(x)
        for block in self.conv_blocks:
            x = block(x)
        return x


class MuseSparkAudioTower(nn.Module):
    def __init__(self, config: MuseSparkAudioConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(_Stage(config, i) for i in range(len(config.hidden_sizes)))
        self.norm = RMSNorm(config.hidden_sizes[-1], eps=config.rms_norm_eps)
        self.proj = nn.Linear(config.hidden_sizes[-1], config.out_hidden_size, bias=True)
        self.post_proj_norm = RMSNorm(config.out_hidden_size, eps=config.rms_norm_eps)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """``(rows, samples)``, a multiple of ``samples_per_token`` samples,
        to ``(rows, tokens, out_hidden_size)``."""
        x = values.unsqueeze(-1).to(self.proj.weight.dtype)
        for stage in self.layers:
            x = stage(x)
        x = self.post_proj_norm(self.proj(self.norm(x)))
        return rms_norm_no_weight(x, self.config.text_rms_norm_eps)
