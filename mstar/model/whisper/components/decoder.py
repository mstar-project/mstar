"""Whisper text decoder built on the shared mstar components.

The decoder is a standard pre-norm transformer with three sublayers per
block: causal self-attention over a paged KV array, cross-attention over
the audio encoder's output, and a plain GELU FFN.

Whisper has no RoPE — positions are a learned ``embed_positions`` table,
looked up on the position ids the array plans for the step — so the
self-attention layers rotate nothing (``pos_key=None``).

Cross-attention K/V depend only on the (static) encoder output, so they
are computed once per request at prefill and added to the context array
(``add_context``). Every later step declares a zero-span segment on that
array and reads it — nothing is recomputed or rewritten.

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

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed.attention import ParallelAttention, ParallelCrossAttention
from mstar.model.whisper.config import CROSS_KV_CACHE, KV_CACHE, WhisperModelConfig


class WhisperDecoderLayer(nn.Module):
    def __init__(self, config: WhisperModelConfig, layer: int, comm_group: CommGroup | None = None):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.self_attn = ParallelAttention(
            comm_group=comm_group,
            hidden_size=config.d_model,
            num_heads=config.decoder_attention_heads,
            num_kv_heads=config.decoder_attention_heads,
            head_dim=config.head_dim,
            qkv_bias=True,
            o_bias=True,
            kv_key=KV_CACHE,
            # learned absolute positions, added at embedding time
            pos_key=None,
            layer=layer,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(config.d_model)
        # Whisper's bias layout is the shared default (q/v/o biased, k not).
        self.encoder_attn = ParallelCrossAttention(
            comm_group=comm_group,
            hidden_size=config.d_model,
            num_heads=config.decoder_attention_heads,
            head_dim=config.head_dim,
            context_kv_key=CROSS_KV_CACHE,
            num_kv_heads=config.decoder_attention_heads,
            layer=layer,
        )
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, config.d_model)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = residual + self.self_attn(hidden_states)

        residual = hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        hidden_states = residual + self.encoder_attn(hidden_states)

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = residual + self.fc2(F.gelu(self.fc1(hidden_states)))
        return hidden_states


class WhisperDecoderModel(nn.Module):
    """Decoder stack; parameter paths mirror HF's ``model.decoder.*``."""

    def __init__(self, config: WhisperModelConfig, comm_group: CommGroup | None = None):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_positions = nn.Embedding(config.max_target_positions, config.d_model)
        self.layers = nn.ModuleList(
            [WhisperDecoderLayer(config, layer, comm_group=comm_group) for layer in range(config.decoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    def zero_missing_biases(self) -> None:
        """Zero the fused K-bias slice absent from the HF checkpoint."""
        with torch.no_grad():
            for layer in self.layers:
                attn = layer.self_attn
                q_size = attn.num_heads * attn.head_dim
                k_size = attn.num_kv_heads * attn.head_dim
                attn.qkv_proj.bias[q_size:q_size + k_size].zero_()

    def embed(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Token + learned position embeddings.

        ``position_ids`` is the array's plan for this step — under a captured
        graph it is the slot's static buffer, so the lookup rides inside the
        capture instead of being staged as an embedding.
        """
        embeds = self.embed_tokens(input_ids)
        if self.config.scale_embedding:
            embeds = embeds * (self.config.d_model ** 0.5)
        return embeds + self.embed_positions(position_ids[:input_ids.shape[0]])

    def add_context(self, encoder_states: torch.Tensor) -> None:
        """Project the encoder output to per-layer K/V and add it to the
        context array; for a batch, the requests' outputs concatenated in
        the step's segment order."""
        for layer in self.layers:
            layer.encoder_attn.add_context(encoder_states)

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # proj_out is tied to embed_tokens in the HF checkpoint.
        return F.linear(hidden_states, self.embed_tokens.weight)

    @torch.no_grad()
    def cross_attention_weights(
        self,
        tokens: torch.Tensor,
        encoder_states: torch.Tensor,
        heads: list[tuple[int, int]],
    ) -> torch.Tensor:
        """Teacher-forced pass over one token sequence, outside the paged
        resources, returning the cross-attention probabilities of the given
        ``(layer, head)`` pairs: ``(len(heads), len(tokens), enc_len)``.

        The served decoder's own projections, applied with plain causal
        self-attention and an explicit softmax over the encoder positions.
        That is the alignment signal word-level timestamps are read from
        (:mod:`.alignment`); one request at a time, so it stays eager.
        """
        wanted: dict[int, list[int]] = {}
        for layer_idx, head_idx in heads:
            wanted.setdefault(layer_idx, []).append(head_idx)
        num_tokens = tokens.shape[0]
        enc_len = encoder_states.shape[0]
        dtype = self.embed_tokens.weight.dtype
        encoder_states = encoder_states.to(dtype)
        positions = torch.arange(num_tokens, device=tokens.device)
        hidden = self.embed(tokens, positions)
        out: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            attn = layer.self_attn
            if attn.num_heads != attn.total_num_heads:
                raise NotImplementedError("word timestamps need the decoder unsharded (TP=1)")
            heads_n, head_dim = attn.num_heads, attn.head_dim
            h = layer.self_attn_layer_norm(hidden)
            qkv = attn.qkv_proj(h)
            kv_size = attn.num_kv_heads * head_dim
            q, k, v = qkv.split([heads_n * head_dim, kv_size, kv_size], dim=-1)
            q = q.view(num_tokens, heads_n, head_dim).transpose(0, 1)
            k = k.view(num_tokens, attn.num_kv_heads, head_dim).transpose(0, 1)
            v = v.view(num_tokens, attn.num_kv_heads, head_dim).transpose(0, 1)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            a = a.transpose(0, 1).reshape(num_tokens, heads_n * head_dim)
            hidden = hidden + attn.o_proj(a)

            cross = layer.encoder_attn
            h = layer.encoder_attn_layer_norm(hidden)
            q = cross.q_proj(h).view(num_tokens, heads_n, head_dim).transpose(0, 1)
            k, v = cross.compute_kv(encoder_states)  # (enc_len, heads, head_dim)
            k = k.transpose(0, 1)
            v = v.transpose(0, 1)
            scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * head_dim ** -0.5
            probs = scores.softmax(dim=-1)  # (heads, tokens, enc_len)
            for head_idx in wanted.get(layer_idx, []):
                out.append(probs[head_idx])
            a = torch.matmul(probs.to(v.dtype), v).transpose(0, 1).reshape(num_tokens, heads_n * head_dim)
            hidden = hidden + cross.out_proj(a)

            h = layer.final_layer_norm(hidden)
            hidden = hidden + layer.fc2(F.gelu(layer.fc1(h)))
        del enc_len
        return torch.stack(out) if out else torch.zeros(0, num_tokens, encoder_states.shape[0], device=tokens.device)

    def forward(self, input_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = input_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.layer_norm(hidden_states)
