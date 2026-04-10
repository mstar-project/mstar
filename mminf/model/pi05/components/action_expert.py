"""Action expert transformer with adaRMS timestep conditioning.

The action expert is a Gemma-style transformer that processes action suffix
tokens during the flow-matching loop. It shares KV-cache dimensions with the
PaliGemma expert (same num_kv_heads and head_dim) so it can attend to the
prefix KV cache that PaliGemma wrote during the prefill walk.

adaRMS conditioning: each layer takes per-iteration ``(scale, shift, gate)``
parameters derived from a sinusoidal timestep embedding + MLP. The norm output
is modulated as ``norm(x) * (1 + scale) + shift`` and the residual is gated by
``gate``. The same condition vector is shared across all action tokens within
one denoising step.
"""

import torch
from torch import nn

from mminf.engine.cache_manager import BatchedCacheManager
from mminf.model.pi05.components.paligemma import (
    Pi05GemmaMLP,
    Pi05GemmaRMSNorm,
    Pi05PaliGemmaAttention,
)
from mminf.model.pi05.config import Pi05Config
from mminf.utils.flashinfer_utils import run_rms_norm


class Pi05AdaLNMLP(nn.Module):
    """Maps a timestep embedding to per-layer adaRMS modulation parameters.

    Output is a tensor of shape ``(num_layers, 6, hidden_size)`` containing
    ``(scale_pre, shift_pre, gate_attn, scale_post, shift_post, gate_mlp)`` for
    every transformer layer in the action expert.
    """

    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.proj = nn.Linear(hidden_size, num_layers * 6 * hidden_size, bias=True)

    def forward(self, time_emb: torch.Tensor) -> torch.Tensor:
        # time_emb: [hidden_size] or [B, hidden_size]
        h = nn.functional.silu(time_emb)
        out = self.proj(h)
        return out.view(*out.shape[:-1], self.num_layers, 6, self.hidden_size)


def _modulate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Apply adaRMS modulation: ``x * (1 + scale) + shift`` (broadcasting on tokens)."""
    # x: [seq, hidden]; scale/shift: [hidden]
    return x * (1.0 + scale) + shift


class Pi05ActionExpertLayer(nn.Module):
    def __init__(self, config: Pi05Config):
        super().__init__()
        self.self_attn = Pi05PaliGemmaAttention(config)
        self.mlp = Pi05GemmaMLP(config.hidden_size, config.action_intermediate_size)
        self.input_layernorm = Pi05GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Pi05GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        query_sequence: torch.Tensor,
        cache_handle: BatchedCacheManager,
        adaln_params: torch.Tensor,
    ) -> torch.Tensor:
        # adaln_params: [6, hidden_size] for this layer
        scale_pre, shift_pre, gate_attn, scale_post, shift_post, gate_mlp = adaln_params.unbind(0)

        # Pre-attention norm + modulation.
        residual = query_sequence
        normed = run_rms_norm(
            query_sequence,
            self.input_layernorm.weight,
            eps=self.input_layernorm.variance_epsilon,
        )
        normed = _modulate(normed, scale_pre, shift_pre)
        attn_out = self.self_attn(query_sequence=normed, cache_handle=cache_handle)
        query_sequence = residual + gate_attn * attn_out

        # Post-attention norm + modulation.
        residual = query_sequence
        normed = run_rms_norm(
            query_sequence,
            self.post_attention_layernorm.weight,
            eps=self.post_attention_layernorm.variance_epsilon,
        )
        normed = _modulate(normed, scale_post, shift_post)
        mlp_out = self.mlp(normed)
        return residual + gate_mlp * mlp_out


class Pi05ActionExpert(nn.Module):
    """Action expert transformer with adaRMS conditioning.

    Forward signature takes the precomputed adaln parameters for the current
    Euler step. The cache_handle is expected to be in read-only mode
    (``write_cache=False``) so the action expert attends to the frozen PaliGemma
    prefix without mutating it.
    """

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Pi05ActionExpertLayer(config) for _ in range(config.num_layers)]
        )
        self.norm = Pi05GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        query_sequence: torch.Tensor,
        cache_handle: BatchedCacheManager,
        adaln_params: torch.Tensor,
    ) -> torch.Tensor:
        # adaln_params: [num_layers, 6, hidden_size]
        for layer_idx, layer in enumerate(self.layers):
            cache_handle.set_layer_idx(layer_idx)
            query_sequence = layer(
                query_sequence=query_sequence,
                cache_handle=cache_handle,
                adaln_params=adaln_params[layer_idx],
            )
        return run_rms_norm(
            query_sequence, self.norm.weight, eps=self.norm.variance_epsilon
        )
