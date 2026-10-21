"""LTX-2.5's text connectors (``LTX2TextConnectors``), native.

Turn Gemma's stacked hidden states into the two text embeddings the DiT attends
over: a per-token RMS norm of every layer, a per-modality projection (video 4096,
audio 2048), then a small bidirectional transformer per modality over a fixed
1024-token sequence whose tail is learned registers rather than padding. Because
the registers fill the sequence, the DiT's text attention needs no mask.

The reference left-pads, then front-aligns the real tokens before appending the
registers; ``LTX2TextConnectors.forward`` here takes the real tokens alone (see
``Gemma4TextEncoder``) and builds the same front-aligned sequence directly.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.diffusion.attention import sdpa_attention
from mstar.model.ltx2_5.components.layers import (
    Attend,
    GatedAttention,
    GeluFeedForward,
    RotaryTable,
    split_rope_table,
)
from mstar.model.ltx2_5.config import ConnectorConfig


def per_token_rms_norm(hidden: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalizes each (token, layer) over the hidden dim, in the input dtype as the
    reference does (no fp32 upcast). ``hidden``: ``[..., hidden, layers]``."""
    variance = torch.mean(hidden ** 2, dim=-2, keepdim=True)
    return hidden * torch.rsqrt(variance + eps)


class ConnectorBlock(nn.Module):
    """``LTX2TransformerBlock1d``: pre-norm gated self-attention and feed-forward."""

    def __init__(self, dim: int, heads: int, head_dim: int, gated: bool, comm_group: CommGroup):
        super().__init__()
        self.dim = dim
        self.attn1 = GatedAttention(dim, heads, head_dim, comm_group, gated=gated)
        self.ff = GeluFeedForward(dim, comm_group)

    def forward(self, x: torch.Tensor, rope: RotaryTable, attend: Attend) -> torch.Tensor:
        x = x + self.attn1(F.rms_norm(x, (self.dim,), eps=1e-6), attend, q_rope=rope)
        return x + self.ff(F.rms_norm(x, (self.dim,), eps=1e-6))


class Connector1d(nn.Module):
    """``LTX2ConnectorTransformer1d`` for one modality."""

    def __init__(
        self, heads: int, head_dim: int, layers: int, registers: int, gated: bool,
        rope_base_seq_len: int, rope_theta: float, comm_group: CommGroup,
    ):
        super().__init__()
        self.dim = heads * head_dim
        self.heads = heads
        self.rope_base_seq_len = rope_base_seq_len
        self.rope_theta = rope_theta
        self.learnable_registers = nn.Parameter(torch.empty(registers, self.dim))
        self.transformer_blocks = nn.ModuleList(
            [ConnectorBlock(self.dim, heads, head_dim, gated, comm_group) for _ in range(layers)]
        )

    def rope(self, seq_len: int, device) -> RotaryTable:
        positions = torch.arange(seq_len, dtype=torch.float32, device=device).view(1, -1, 1)
        return split_rope_table(positions, self.dim, self.heads, self.rope_theta, (float(self.rope_base_seq_len),))

    def forward(self, tokens: torch.Tensor, valid: list[int], seq_len: int, attend: Attend) -> torch.Tensor:
        """``tokens``: ``[B, max(valid), dim]`` projected prompt tokens, front-aligned,
        row ``i`` real for its first ``valid[i]``. Returns ``[B, seq_len, dim]``."""
        registers = self.learnable_registers.repeat(seq_len // self.learnable_registers.shape[0], 1)
        rows = []
        for i, n in enumerate(valid):
            rows.append(torch.cat([tokens[i, :n], registers[n:].to(tokens.dtype)], dim=0))
        x = torch.stack(rows)
        rope = self.rope(seq_len, x.device)
        for block in self.transformer_blocks:
            x = block(x, rope, attend)
        return F.rms_norm(x, (self.dim,), eps=1e-6)


class LTX2TextConnectors(nn.Module):
    def __init__(self, cfg: ConnectorConfig, comm_group: CommGroup | None = None):
        super().__init__()
        comm_group = comm_group or CommGroup.trivial()
        self.cfg = cfg
        in_dim = cfg.caption_channels * cfg.text_proj_in_factor
        self.video_text_proj_in = nn.Linear(in_dim, cfg.video_hidden_dim, bias=cfg.proj_bias)
        self.audio_text_proj_in = nn.Linear(in_dim, cfg.audio_hidden_dim, bias=cfg.proj_bias)
        common = dict(rope_base_seq_len=cfg.connector_rope_base_seq_len, rope_theta=cfg.rope_theta,
                      comm_group=comm_group)
        self.video_connector = Connector1d(
            cfg.video_connector_num_attention_heads, cfg.video_connector_attention_head_dim,
            cfg.video_connector_num_layers, cfg.video_connector_num_learnable_registers, cfg.video_gated_attn,
            **common,
        )
        self.audio_connector = Connector1d(
            cfg.audio_connector_num_attention_heads, cfg.audio_connector_attention_head_dim,
            cfg.audio_connector_num_layers, cfg.audio_connector_num_learnable_registers, cfg.audio_gated_attn,
            **common,
        )

    def forward(
        self, hidden: torch.Tensor, valid: list[int], seq_len: int, attend: Attend = sdpa_attention,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``hidden``: ``[B, T, caption_channels, layers + 1]`` Gemma states of the real
        tokens, front-aligned (row ``i`` real for its first ``valid[i]``). Returns the
        video ``[B, seq_len, 4096]`` and audio ``[B, seq_len, 2048]`` text embeddings."""
        normed = per_token_rms_norm(hidden).flatten(2, 3)
        video = self.video_text_proj_in(normed * math.sqrt(self.cfg.video_hidden_dim / self.cfg.caption_channels))
        audio = self.audio_text_proj_in(normed * math.sqrt(self.cfg.audio_hidden_dim / self.cfg.caption_channels))
        return (
            self.video_connector(video, valid, seq_len, attend),
            self.audio_connector(audio, valid, seq_len, attend),
        )
