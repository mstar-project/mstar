"""Orpheus language model: Llama 3.2 3B-style transformer.

Built from the shared transformer components in ``mminf.model.components``.
Orpheus uses standard Llama-style RMSNorm (Llama mode, not Gemma),
SwiGLU MLP, GQA self-attention with Llama-3 RoPE scaling, and no
QK-norm.

QKV and gate/up projections are fused from construction via the
parallel-linear classes (with a trivial single-rank comm group for the
non-TP case). Weight loading goes through ``load_weights`` with stacked
shard routing (no post-load consolidate step).
"""
from __future__ import annotations

import torch
from torch import nn

from mminf.engine.kv_cache_engine import BatchedCacheManager
from mminf.model.components import DecoderLayer, RMSNorm
from mminf.model.components.distributed import ParallelAttention, ParallelGatedMLP
from mminf.model.orpheus.config import OrpheusModelConfig


def _build_decoder_layer(config: OrpheusModelConfig) -> DecoderLayer:
    return DecoderLayer(
        self_attn=ParallelAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            rope_theta=config.rope_theta,
            rope_scale=config.rope_scaling["factor"],
            rope_low_freq_factor=config.rope_scaling["low_freq_factor"],
            rope_high_freq_factor=config.rope_scaling["high_freq_factor"],
            rope_old_context_len=config.rope_scaling["original_max_position_embeddings"],
        ),
        mlp=ParallelGatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation="silu",
        ),
        input_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
        post_attention_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
    )


class OrpheusLanguageModel(nn.Module):
    def __init__(self, config: OrpheusModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [_build_decoder_layer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        query_sequence: torch.Tensor,
        cache_handle: BatchedCacheManager,
    ) -> torch.Tensor:
        for layer_idx, decoder_layer in enumerate(self.layers):
            cache_handle.set_layer_idx(layer_idx)
            query_sequence = decoder_layer(
                hidden_states=query_sequence, cache_handle=cache_handle,
            )
        cache_handle.advance_seq_lens()
        return self.norm(query_sequence)


class OrpheusForCausalLM(nn.Module):
    def __init__(self, config: OrpheusModelConfig):
        super().__init__()
        self.model = OrpheusLanguageModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        query_sequence: torch.Tensor,
        cache_handle: BatchedCacheManager,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(query_sequence=query_sequence, cache_handle=cache_handle)

    def load_weights(self, weights):
        """Load HF Llama-style weights into the fused parameters."""
        from mminf.model.loader import LLAMA_STACKED_PARAMS, load_hf_weights

        return load_hf_weights(self, weights, stacked_params=LLAMA_STACKED_PARAMS)
