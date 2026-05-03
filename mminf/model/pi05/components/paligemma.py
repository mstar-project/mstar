"""PaliGemma transformer expert for Pi0.5 (prefix processing).

A Gemma-style transformer that integrates with mminf's BatchedCacheManager
for paged KV cache. Used for the prefill graph walk where it processes the
prefix tokens (image + language + state) and writes the KV cache that the
action expert later reads during action generation.
"""

import torch
import torch.nn.functional as F
from torch import nn

from mminf.engine.cache_manager import BatchedCacheManager
from mminf.model.pi05.components import Pi05GemmaRMSNorm
from mminf.model.pi05.config import Pi05Config


class Pi05GemmaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class Pi05PaliGemmaAttention(nn.Module):
    """GQA self-attention with RoPE, integrated with the FlashInfer cache_handle.

    The input/output hidden size is parameterized so the same attention module
    can be reused for the action expert (which has a different ``hidden_size``
    than PaliGemma) while sharing num_kv_heads and head_dim so both experts
    can read/write the same KV cache layout.
    """

    def __init__(self, config: Pi05Config, input_hidden_size: int | None = None):
        super().__init__()
        self.config = config
        self.num_heads = config.num_qo_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = (
            input_hidden_size if input_hidden_size is not None else config.hidden_size
        )
        self.rope_theta = config.rope_theta

        qkv_out_dim = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.qkv_proj = nn.Linear(self.hidden_size, qkv_out_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        query_sequence: torch.Tensor,
        cache_handle: BatchedCacheManager,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(query_sequence)
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim
        q, k, v = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        q, k = cache_handle.apply_rope(q, k, rope_theta=self.rope_theta)
        attn_output = cache_handle.run_attention(q=q, k=k, v=v)
        # attn_output is [seq, num_heads, head_dim]; flatten the head dimension
        # to feed o_proj (in_features = num_heads * head_dim, which may differ
        # from hidden_size when paligemma and action expert use different
        # widths sharing the same num_kv_heads / head_dim).
        attn_output = attn_output.reshape(-1, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class Pi05PaliGemmaLayer(nn.Module):
    def __init__(self, config: Pi05Config):
        super().__init__()
        self.self_attn = Pi05PaliGemmaAttention(config)
        self.mlp = Pi05GemmaMLP(config.hidden_size, config.pali_intermediate_size)
        self.input_layernorm = Pi05GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Pi05GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        query_sequence: torch.Tensor,
        cache_handle: BatchedCacheManager,
    ) -> torch.Tensor:
        residual = query_sequence
        query_sequence = self.input_layernorm(query_sequence)
        query_sequence = self.self_attn(
            query_sequence=query_sequence, cache_handle=cache_handle
        )
        query_sequence = residual + query_sequence

        residual = query_sequence
        query_sequence = self.post_attention_layernorm(query_sequence)
        query_sequence = self.mlp(query_sequence)
        return residual + query_sequence


class Pi05PaliGemmaExpert(nn.Module):
    """Stack of PaliGemma transformer layers.

    The submodule's input embeddings (image tokens + language tokens + state
    tokens) are passed in directly. This module owns only the transformer
    blocks plus a final RMSNorm; the embedding table is held by the parent
    submodule and shared with the action expert.
    """

    def __init__(self, config: Pi05Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [Pi05PaliGemmaLayer(config) for _ in range(config.num_layers)]
        )
        self.norm = Pi05GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        query_sequence: torch.Tensor,
        cache_handle: BatchedCacheManager,
        write_cache: bool = True,
    ) -> torch.Tensor:
        for layer_idx, layer in enumerate(self.layers):
            cache_handle.set_layer_idx(layer_idx)
            query_sequence = layer(query_sequence=query_sequence, cache_handle=cache_handle)

        if write_cache:
            cache_handle.advance_seq_lens()

        return self.norm(query_sequence)
