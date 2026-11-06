"""Canary text decoder on the shared mstar attention components.

A pre-norm transformer: causal self-attention over the paged KV cache,
cross-attention over the encoder output, and a ReLU FFN. Positions are fixed
sinusoids added to the token embeddings, followed by a LayerNorm. Every
projection has a bias, cross-attention's K included.

Cross-attention K/V depend only on the encoder output, so prefill adds them
once to the context array (``add_context``) and later steps read it.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.canary.config import CROSS_KV_CACHE, KV_CACHE, CanaryModelConfig
from mstar.model.components.distributed.attention import ParallelAttention, ParallelCrossAttention


def sinusoid_positions(position_ids: torch.Tensor, channels: int) -> torch.Tensor:
    """NeMo's fixed decoder positions, ``[sin | cos] / sqrt(channels)``, built
    in fp32 on the fly: a buffer would be rounded by the engine's dtype cast."""
    increment = math.log(10000 ** ((channels - 2) / channels)) / (channels // 2 - 1)
    inv = torch.exp(-increment * torch.arange(channels // 2, device=position_ids.device, dtype=torch.float32))
    scaled = position_ids.to(torch.float32)[:, None] * inv[None, :]
    return torch.cat([scaled.sin(), scaled.cos()], dim=1) / math.sqrt(channels)


class _MLP(nn.Module):
    def __init__(self, config: CanaryModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class CanaryDecoderLayer(nn.Module):
    def __init__(self, config: CanaryModelConfig, layer: int):
        super().__init__()
        h = config.hidden_size
        self.input_layernorm = nn.LayerNorm(h)
        self.self_attn = ParallelAttention(
            hidden_size=h,
            num_heads=config.num_heads,
            num_kv_heads=config.num_heads,
            head_dim=config.head_dim,
            qkv_bias=True,
            o_bias=True,
            kv_key=KV_CACHE,
            pos_key=None,  # absolute sinusoids, added at embedding time
            layer=layer,
        )
        self.post_attention_layernorm = nn.LayerNorm(h)
        self.encoder_attn = ParallelCrossAttention(
            hidden_size=h,
            num_heads=config.num_heads,
            num_kv_heads=config.num_heads,
            head_dim=config.head_dim,
            k_bias=True,
            context_kv_key=CROSS_KV_CACHE,
            layer=layer,
        )
        self.final_layernorm = nn.LayerNorm(h)
        self.mlp = _MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.encoder_attn(self.post_attention_layernorm(x))
        return x + self.mlp(self.final_layernorm(x))


class CanaryDecoderModel(nn.Module):
    """Decoder stack; parameter paths mirror HF's ``model.decoder.*`` plus
    ``proj_out``."""

    def __init__(self, config: CanaryModelConfig):
        super().__init__()
        self.config = config
        h = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, h)
        self.embedding_layernorm = nn.LayerNorm(h)
        self.layers = nn.ModuleList(CanaryDecoderLayer(config, layer) for layer in range(config.num_layers))
        self.norm = nn.LayerNorm(h)
        self.proj_out = nn.Linear(h, config.vocab_size)

    def embed(self, input_ids: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Token embeddings plus sinusoids at the array's planned position ids
        (a static buffer under a captured graph), then the LayerNorm."""
        tokens = self.embed_tokens(input_ids)
        pos = sinusoid_positions(position_ids[:input_ids.shape[0]], self.config.hidden_size).to(tokens.dtype)
        return self.embedding_layernorm(tokens + pos)

    def add_context(self, encoder_states: torch.Tensor) -> None:
        """Project the encoder output to per-layer K/V and add it to the
        context array; for a batch, the requests' outputs concatenated in
        the step's segment order."""
        for layer in self.layers:
            layer.encoder_attn.add_context(encoder_states)

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.proj_out(hidden_states)

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = input_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)

    @staticmethod
    def remap(name: str) -> str | None:
        """HF checkpoint key -> parameter path (``None`` drops it)."""
        if name.startswith("proj_out."):
            return name
        if not name.startswith("model.decoder."):
            return None
        return name.removeprefix("model.decoder.").replace("encoder_attn.o_proj", "encoder_attn.out_proj")
