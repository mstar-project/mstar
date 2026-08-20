"""Whisper text decoder built on the shared mstar components.

The decoder is a standard pre-norm transformer with three sublayers per
block: causal self-attention (paged KV cache via the engine resources),
cross-attention over the audio encoder's output, and a plain GELU FFN.

Whisper has no RoPE — positions are a learned ``embed_positions`` table,
looked up on the position ids the position resource plans for the step
— so the self-attention layers bind no position resource
(``pos_key=None``).

Cross-attention K/V depend only on the (static) encoder output, so they
are computed once per request at prefill (``write_cross_kv``) and written
into the context KV stream the cross-attention resource attends. Every
later step declares a zero-span segment on that label and runs the
planned wrapper — nothing is recomputed or rewritten.

HF checkpoint quirks handled here:
  * ``self_attn.out_proj`` → ``self_attn.o_proj`` (name_remapper in
    ``whisper_model.py``).
  * ``k_proj`` has no bias in the checkpoint while ``q/v_proj`` do; the
    shared ``Attention`` uses one ``qkv_bias`` flag, so ``k_proj.bias``
    is allocated and zeroed post-load (``zero_missing_biases``).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.attention import Attention, CrossAttention
from mstar.model.whisper.config import (
    ATTN,
    CONTEXT_LABEL,
    CROSS_ATTN,
    CROSS_KV_CACHE,
    KV_CACHE,
    WhisperModelConfig,
)


class WhisperDecoderLayer(nn.Module):
    def __init__(self, config: WhisperModelConfig):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.self_attn = Attention(
            hidden_size=config.d_model,
            num_heads=config.decoder_attention_heads,
            num_kv_heads=config.decoder_attention_heads,
            head_dim=config.head_dim,
            qkv_bias=True,
            o_bias=True,
            attn_key=ATTN,
            kv_key=KV_CACHE,
            # learned absolute positions, added at embedding time
            pos_key=None,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(config.d_model)
        # Whisper's bias layout is the shared default (q/v/o biased, k not).
        self.encoder_attn = CrossAttention(
            hidden_size=config.d_model,
            num_heads=config.decoder_attention_heads,
            head_dim=config.head_dim,
            cross_key=CROSS_ATTN,
            context_kv_key=CROSS_KV_CACHE,
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        label: str,
        layer_idx: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = residual + self.self_attn(
            hidden_states, label=label, layer_idx=layer_idx,
        )

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = residual + self.encoder_attn(
            hidden_states, label=label, layer_idx=layer_idx,
        )

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = residual + self.fc2(F.gelu(self.fc1(hidden_states)))
        return hidden_states


class WhisperDecoderModel(nn.Module):
    """Decoder stack; parameter paths mirror HF's ``model.decoder.*``."""

    def __init__(self, config: WhisperModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_positions = nn.Embedding(config.max_target_positions, config.d_model)
        self.layers = nn.ModuleList(
            [WhisperDecoderLayer(config) for _ in range(config.decoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    def zero_missing_biases(self) -> None:
        """Zero the self-attn ``k_proj`` biases absent from the HF checkpoint
        (allocated because the shared ``Attention`` has one qkv_bias flag)."""
        with torch.no_grad():
            for layer in self.layers:
                layer.self_attn.k_proj.bias.zero_()

    def embed(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Token + learned position embeddings.

        ``position_ids`` is the position resource's plan for this step — under
        a captured graph it is the slot's static buffer, so the lookup rides
        inside the capture instead of being staged as an embedding.
        """
        embeds = self.embed_tokens(input_ids)
        if self.config.scale_embedding:
            embeds = embeds * (self.config.d_model ** 0.5)
        return embeds + self.embed_positions(position_ids[:input_ids.shape[0]])

    def write_cross_kv(self, encoder_states: torch.Tensor) -> None:
        """Project the encoder output to per-layer K/V and write it into the
        context stream the cross-attention resource attends.

        Called once per request, from the prefill forward, under a step that
        declared a ``CONTEXT_LABEL`` segment spanning the encoder output. For
        a batch, ``encoder_states`` is the requests' outputs concatenated in
        the step's segment order.
        """
        for layer_idx, layer in enumerate(self.layers):
            cross_attn = layer.encoder_attn
            k, v = cross_attn.compute_kv(encoder_states)
            cross_attn.context_kv.set_layer_idx(layer_idx)
            cross_attn.context_kv.write_kv(k, v, label=CONTEXT_LABEL)

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # proj_out is tied to embed_tokens in the HF checkpoint.
        return F.linear(hidden_states, self.embed_tokens.weight)

    def forward(
        self,
        input_embeds: torch.Tensor,
        *,
        label: str,
    ) -> torch.Tensor:
        hidden_states = input_embeds
        for layer_idx, layer in enumerate(self.layers):
            # NOTE: Set layer_idx here and pass in layer_idx=None so that inductor doesn't
            # try to specialize on the layer_idx int. Both caches need it: the
            # self-attention's and the (separate) encoder-context one.
            layer.self_attn.kv.set_layer_idx(layer_idx)
            layer.encoder_attn.context_kv.set_layer_idx(layer_idx)
            hidden_states = layer(
                hidden_states, label=label, layer_idx=None,
            )
        # the advance is the runner's now, off the step declaration
        return self.layer_norm(hidden_states)
