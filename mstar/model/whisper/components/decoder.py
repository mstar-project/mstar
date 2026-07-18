"""Whisper text decoder built on the shared mstar components.

The decoder is a standard pre-norm transformer with three sublayers per
block: causal self-attention (paged KV cache via ``cache_handle``),
cross-attention over the audio encoder's output, and a plain GELU FFN.

Whisper has no RoPE — positions come from a learned ``embed_positions``
table added to the token embeddings by the submodule — so the
self-attention subclass makes ``_apply_rope`` a no-op.

Cross-attention K/V depend only on the (static) encoder output, so they
are computed once per request at prefill (``compute_cross_kv``) and
written into the engine's cross-attention context pool
(``cache_handle.add_cross_attn_kv``); every step then runs the
pre-planned ``cache_handle.run_cross_attn`` — see issue #160.

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

from mstar.engine.cache_manager import BatchedCacheManager
from mstar.model.components.attention import Attention, CrossAttention
from mstar.model.whisper.config import WhisperModelConfig

CrossKV = tuple[torch.Tensor, torch.Tensor]


class WhisperSelfAttention(Attention):
    def _apply_rope(
        self, q: torch.Tensor, k: torch.Tensor, cache_handle: BatchedCacheManager,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Whisper uses learned absolute positions added at embedding time.
        return q, k


class WhisperCrossAttention(CrossAttention):
    """Shared ``CrossAttention`` with Whisper's bias layout (the default:
    q/v/o biased, k unbiased), attending the single encoder context.

    Fast path (#160 recovery): when the caller supplies the request's
    contiguous per-layer encoder K/V (``cross_kv``), cross-attention runs as
    an inline ``scaled_dot_product_attention`` so it is *traced into the
    torch.compiled decoder graph* — avoiding the per-layer graph break and
    eager FlashInfer wrapper of ``cache_handle.run_cross_attn``. When
    ``cross_kv`` is ``None`` it falls back to the engine cross-attention pool
    (batched / concurrent serving still uses that path).

    The inline path relies on whisper decode NOT being CUDA-graphed, so the
    per-request K/V can ride in as compiled-forward args. Once the decode step
    is CUDA-graphed (the #160 batching follow-up), captured tensor addresses are
    fixed, so that path takes the ``cross_kv=None`` FlashInfer pool branch (or a
    static capture buffer) rather than these varying-address args."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
        cross_kv: CrossKV | None = None,
    ) -> torch.Tensor:
        if cross_kv is None:
            return super().forward(hidden_states, cache_handle)
        num_tokens = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(num_tokens, self.num_heads, self.head_dim)
        k, v = cross_kv  # each (enc_len, num_heads, head_dim), static per request
        # SDPA wants (batch, heads, seq, dim); single stream -> batch 1.
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        attn = F.scaled_dot_product_attention(q, k, v)  # non-causal, full context
        attn = attn.squeeze(0).transpose(0, 1).reshape(
            num_tokens, self.num_heads * self.head_dim
        )
        return self.out_proj(attn)


class WhisperDecoderLayer(nn.Module):
    def __init__(self, config: WhisperModelConfig):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.self_attn = WhisperSelfAttention(
            hidden_size=config.d_model,
            num_heads=config.decoder_attention_heads,
            num_kv_heads=config.decoder_attention_heads,
            head_dim=config.head_dim,
            qkv_bias=True,
            o_bias=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.encoder_attn = WhisperCrossAttention(
            hidden_size=config.d_model,
            num_heads=config.decoder_attention_heads,
            head_dim=config.head_dim,
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_handle: BatchedCacheManager,
        cross_kv: CrossKV | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states, cache_handle)

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = residual + self.encoder_attn(hidden_states, cache_handle, cross_kv)

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

    def embed(self, input_ids: torch.Tensor, start_pos: int) -> torch.Tensor:
        """Token + learned position embeddings for a contiguous span."""
        positions = torch.arange(
            start_pos, start_pos + input_ids.shape[0], device=input_ids.device,
        )
        embeds = self.embed_tokens(input_ids)
        if self.config.scale_embedding:
            embeds = embeds * (self.config.d_model ** 0.5)
        return embeds + self.embed_positions(positions)

    def compute_cross_kv(self, encoder_states: torch.Tensor) -> list[CrossKV]:
        return [layer.encoder_attn.compute_kv(encoder_states) for layer in self.layers]

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # proj_out is tied to embed_tokens in the HF checkpoint.
        return F.linear(hidden_states, self.embed_tokens.weight)

    def forward(
        self,
        input_embeds: torch.Tensor,
        cache_handle: BatchedCacheManager,
        cross_k: torch.Tensor | None = None,
        cross_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # cross_k/cross_v (when given): stacked contiguous encoder K/V,
        # (num_layers, enc_len, num_heads, head_dim). Supplying them routes
        # cross-attention through the inline SDPA fast path (compile-inline,
        # #160 recovery) instead of the engine cross-attention pool.
        hidden_states = input_embeds
        for layer_idx, layer in enumerate(self.layers):
            cache_handle.set_layer_idx(layer_idx)
            cross_kv = None if cross_k is None else (cross_k[layer_idx], cross_v[layer_idx])
            hidden_states = layer(hidden_states, cache_handle, cross_kv)
        cache_handle.advance_seq_lens()
        return self.layer_norm(hidden_states)
