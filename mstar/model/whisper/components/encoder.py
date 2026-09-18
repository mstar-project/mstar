"""Whisper audio encoder built from the shared mstar components.

Two GELU convolutions (the second strided by 2) turn a ``(num_mel_bins,
3000)`` log-mel window into 1500 frames, a fixed sinusoidal table is added,
and a pre-norm transformer stack (bidirectional self-attention + GELU FFN)
runs over them. Every window is exactly ``max_source_positions`` tokens, so
a batch is a dense ``[bs, 1500, d_model]`` tensor: attention is plain SDPA
with no mask, no cache and no resource, and the whole forward is a fixed
shape per batch size — which is what lets the submodule capture it as one
CUDA graph per batch size.

Parameter paths mirror HF's ``model.encoder.*`` so the checkpoint loads
through ``load_hf_weights`` with ``WHISPER_STACKED_PARAMS`` (``q/k/v_proj``
-> fused ``qkv_proj``). ``k_proj`` has no bias in the checkpoint while
``q/v_proj`` do, so the fused bias's K slice is zeroed after loading.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.linear import FusedColumnLinear
from mstar.model.whisper.config import WhisperModelConfig


class WhisperEncoderAttention(nn.Module):
    """Bidirectional multi-head self-attention over one encoder window."""

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads:
            raise ValueError(f"d_model {d_model} is not divisible by {num_heads} heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_proj = FusedColumnLinear(
            d_model, {"q": d_model, "k": d_model, "v": d_model}, bias=True,
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bs, seq_len, d_model = hidden_states.shape
        q, k, v = self.qkv_proj(hidden_states).split(d_model, dim=-1)
        q = q.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bs, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # SDPA's default scale is head_dim ** -0.5, Whisper's own.
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(bs, seq_len, d_model)
        return self.out_proj(out)


class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperModelConfig):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.self_attn = WhisperEncoderAttention(config.d_model, config.encoder_attention_heads)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, config.d_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = residual + self.fc2(F.gelu(self.fc1(hidden_states)))
        return hidden_states


class WhisperEncoderModel(nn.Module):
    """Encoder stack; parameter paths mirror HF's ``model.encoder.*``."""

    def __init__(self, config: WhisperModelConfig):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(config.num_mel_bins, config.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config.d_model, config.d_model, kernel_size=3, stride=2, padding=1)
        # The sinusoidal table ships in the checkpoint; kept as a parameter so
        # it loads like everything else (it is never trained here).
        self.embed_positions = nn.Embedding(config.max_source_positions, config.d_model)
        self.layers = nn.ModuleList(
            [WhisperEncoderLayer(config) for _ in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    @property
    def expected_num_frames(self) -> int:
        return self.config.max_source_positions * self.conv2.stride[0]

    def zero_missing_biases(self) -> None:
        """Zero the fused K-bias slice absent from the HF checkpoint."""
        with torch.no_grad():
            for layer in self.layers:
                attn = layer.self_attn
                d_model = attn.num_heads * attn.head_dim
                attn.qkv_proj.bias[d_model:2 * d_model].zero_()

    def load_weights(self, weights) -> set[str]:
        """Stream ``model.encoder.*``-relative ``(name, tensor)`` pairs in;
        raise if any parameter is left unloaded."""
        from mstar.model.loader import WHISPER_STACKED_PARAMS, load_hf_weights

        loaded = load_hf_weights(self, weights, stacked_params=WHISPER_STACKED_PARAMS)
        missing = sorted(set(dict(self.named_parameters())) - loaded)
        if missing:
            raise RuntimeError(
                f"Whisper encoder checkpoint left {len(missing)} parameter(s) "
                f"unloaded, e.g. {missing[:5]}"
            )
        self.zero_missing_biases()
        return loaded

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        """``(bs, num_mel_bins, 3000)`` log-mel -> ``(bs, 1500, d_model)``."""
        if input_features.shape[-1] != self.expected_num_frames:
            raise ValueError(
                f"Whisper expects {self.expected_num_frames} mel frames per window; "
                f"got {input_features.shape[-1]}. Pad or trim to the 30 s window first."
            )
        hidden_states = F.gelu(self.conv1(input_features))
        hidden_states = F.gelu(self.conv2(hidden_states))
        hidden_states = hidden_states.permute(0, 2, 1) + self.embed_positions.weight
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.layer_norm(hidden_states)
