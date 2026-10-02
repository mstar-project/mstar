"""PL-BERT: the ALBERT phoneme encoder that conditions Kokoro's prosody.

A 12-layer ALBERT shares one transformer layer across all depths. Attention is
bidirectional over the phoneme sequence with a key-padding mask, so the
encoder needs no engine resource: it runs as plain SDPA inside the batched
synthesis node.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components import MLP, FusedColumnLinear
from mstar.model.kokoro.components.masking import length_mask
from mstar.model.kokoro.config import KokoroBertConfig


class AlbertEmbeddings(nn.Module):
    def __init__(self, config: KokoroBertConfig, vocab_size: int):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, config.embedding_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        # Kokoro never uses segment ids; the table is a single learned offset.
        self.token_type_embeddings = nn.Embedding(2, config.embedding_size)
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        x = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(positions)[None]
            + self.token_type_embeddings.weight[0]
        )
        return self.LayerNorm(x)


class AlbertAttention(nn.Module):
    """Post-LayerNorm multi-head self-attention with a fused QKV projection."""

    def __init__(self, config: KokoroBertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        hidden = config.hidden_size
        self.qkv = FusedColumnLinear(hidden, {"q": hidden, "k": hidden, "v": hidden}, bias=True)
        # FusedColumnLinear allocates without initializing (it expects a
        # checkpoint); give it a sane init so a randomly built model runs.
        nn.init.normal_(self.qkv.weight, std=0.02)
        nn.init.zeros_(self.qkv.bias)
        self.dense = nn.Linear(hidden, hidden)
        self.LayerNorm = nn.LayerNorm(hidden, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        q, k, v = self.qkv(hidden_states).chunk(3, dim=-1)
        shape = (bsz, seq_len, self.num_heads, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        # True = may attend. Padded keys are excluded for every query; padded
        # queries still see the valid keys, so nothing turns NaN.
        context = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask[:, None, None, :])
        context = context.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.LayerNorm(hidden_states + self.dense(context))


class AlbertLayer(nn.Module):
    def __init__(self, config: KokoroBertConfig):
        super().__init__()
        self.attention = AlbertAttention(config)
        self.ffn = MLP(config.hidden_size, config.intermediate_size, activation="gelu_tanh")
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(hidden_states, key_mask)
        return self.full_layer_layer_norm(attention_output + self.ffn(attention_output))


class PLBert(nn.Module):
    """``[B, T] phoneme ids -> [B, T, hidden]`` contextual phoneme states."""

    def __init__(self, config: KokoroBertConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.embeddings = AlbertEmbeddings(config, vocab_size)
        self.mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        # One set of weights, applied ``num_hidden_layers`` times.
        self.layer = AlbertLayer(config)

    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        key_mask = length_mask(lengths, input_ids.shape[1])
        hidden = self.mapping_in(self.embeddings(input_ids))
        for _ in range(self.config.num_hidden_layers):
            hidden = self.layer(hidden, key_mask)
        return hidden
