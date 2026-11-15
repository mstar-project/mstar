"""MiniCPM-o's audio tower: a Whisper-medium encoder, a two-layer projector into
the LLM's width, and a 5x average pool (~10 LLM tokens a second).

Ported from ``MiniCPMWhisperEncoder`` and ``MiniCPMO.get_audio_embedding`` in
the checkpoint's ``modeling_minicpmo.py`` (Apache-2.0).

The encoder attends block-causally in 1 s chunks (50 post-conv frames) even
for a whole clip: a frame sees its own chunk and every earlier one. That is
the ``RaggedBlockCausalAttentionSpec`` resource, with one segment per piece
(audio longer than 30 s arrives as 30 s pieces, each encoded from position 0).
The layers are Whisper's, which is what ``AuTEncoderLayer`` is.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.aut_encoder import AuTEncoderConfig, AuTEncoderLayer
from mstar.model.minicpm_o.config import AUDIO_ATTN, AudioConfig


class WhisperEncoder(nn.Module):
    """Parameter paths mirror the checkpoint's ``apm.*``."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        d = config.d_model
        self.conv1 = nn.Conv1d(config.num_mel_bins, d, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d, d, kernel_size=3, stride=2, padding=1)
        self.embed_positions = nn.Embedding(config.max_source_positions, d)
        layer_config = AuTEncoderConfig(
            d_model=d,
            encoder_layers=config.encoder_layers,
            encoder_attention_heads=config.encoder_attention_heads,
            encoder_ffn_dim=config.encoder_ffn_dim,
            activation_function=config.activation_function,
        )
        self.layers = nn.ModuleList(
            AuTEncoderLayer(layer_config, attn_key=AUDIO_ATTN) for _ in range(config.encoder_layers)
        )
        self.layer_norm = nn.LayerNorm(d)

    def frontend(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Per piece ``[num_mel_bins, frames]`` -> packed ``[total_frames, d]``
        with each piece's positions counted from 0."""
        dtype = self.conv1.weight.dtype
        out = []
        for mel in features:
            x = F.gelu(self.conv1(mel.to(dtype)[None]))
            x = F.gelu(self.conv2(x))[0].transpose(0, 1)
            out.append(x + self.embed_positions.weight[: x.shape[0]])
        return torch.cat(out)

    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.layers[0].self_attn.ragged is None:
            raise RuntimeError(
                f"MiniCPM-o's audio tower has no {AUDIO_ATTN!r} resource; declare a "
                "RaggedBlockCausalAttentionSpec for the audio_encoder node"
            )
        for layer in self.layers:
            # cu_seqlens is only read without a bound resource
            hidden_states = layer(hidden_states, cu_seqlens=None)
        return self.layer_norm(hidden_states)


class AudioProjector(nn.Module):
    """``audio_projection_layer.*``: linear, ReLU, linear."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.relu(self.linear1(x)))


class MiniCPMOAudio(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.apm = WhisperEncoder(config)
        self.audio_projection_layer = AudioProjector(config.projector_dim, config.output_dim)

    def zero_missing_biases(self) -> None:
        """Whisper's ``k_proj`` has no bias, so the checkpoint fills only the q
        and v slices of each fused ``qkv_proj`` bias. Zero the whole bias first."""
        for layer in self.apm.layers:
            layer.self_attn.qkv_proj.bias.data.zero_()

    def encode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The transformer stack and projector over packed frames: shape-only in
        the frame count, so a submodule can capture it."""
        return self.audio_projection_layer(self.apm.encode(hidden_states))

    def pool(self, hidden: torch.Tensor, frames_per_piece: list[int]) -> torch.Tensor:
        """5x average pool within each piece; ``AvgPool1d`` drops each tail."""
        pooled = []
        for piece in hidden.split(frames_per_piece):
            pooled.append(F.avg_pool1d(piece.transpose(0, 1)[None], self.config.pool_step)[0].transpose(0, 1))
        return torch.cat(pooled)

    def forward(self, features: list[torch.Tensor], frames_per_piece: list[int]) -> torch.Tensor:
        """Mel pieces -> ``[total_tokens, output_dim]``, pieces end to end.

        ``frames_per_piece`` is each piece's post-conv frame count, known on
        the host; the pooled token count per piece follows from it.
        """
        return self.pool(self.encode(self.apm.frontend(features)), frames_per_piece)
