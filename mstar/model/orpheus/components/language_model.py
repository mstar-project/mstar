"""Orpheus language model: Llama 3.2 3B-style transformer.

Built from the shared transformer components in ``mstar.model.components``.
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

from mstar.distributed.communication import TPCommGroup
from mstar.engine.kv_cache_engine import BatchedCacheManager
from mstar.model.components import DecoderLayer, RMSNorm
from mstar.model.components.distributed import (
    ColumnParallelLinear,
    ParallelAttention,
    ParallelGatedMLP,
    VocabParallelEmbedding,
)
from mstar.model.orpheus.config import OrpheusModelConfig


def _build_decoder_layer(config: OrpheusModelConfig, comm_group: TPCommGroup | None = None) -> DecoderLayer:
    return DecoderLayer(
        self_attn=ParallelAttention(
            comm_group=comm_group,
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
            comm_group=comm_group,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            activation="silu",
        ),
        input_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
        post_attention_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
    )


class OrpheusLanguageModel(nn.Module):
    def __init__(self, config: OrpheusModelConfig, comm_group: TPCommGroup | None = None):
        super().__init__()
        # Vocab-parallel embedding: ``[V, H]`` is row-sharded so each rank
        # holds ``V/tp`` rows. Forward masks + ``all_reduce`` produces a
        # replicated ``[..., H]`` output, identical to a non-TP embedding.
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            comm_group=comm_group,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            [_build_decoder_layer(config, comm_group=comm_group) for _ in range(config.num_hidden_layers)]
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
    def __init__(self, config: OrpheusModelConfig, comm_group: TPCommGroup | None = None):
        super().__init__()
        self.model = OrpheusLanguageModel(config, comm_group=comm_group)
        # LM head is column-parallel over vocab: each rank computes
        # ``[B, V/tp]``; ``gather_output=True`` runs an all-gather on the
        # last dim so the caller (and the sampler) see the full ``[B, V]``
        # logits — no sampler-side coordination needed. For ``tp_size == 1``
        # the all-gather is a no-op and behaviour matches ``nn.Linear``.
        self.lm_head = ColumnParallelLinear(
            comm_group=comm_group,
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            gather_output=True,
        )

    def forward(
        self,
        query_sequence: torch.Tensor,
        cache_handle: BatchedCacheManager,
        **kwargs,
    ) -> torch.Tensor:
        return self.model(query_sequence=query_sequence, cache_handle=cache_handle)

    def load_weights(self, weights):
        """Load HF Llama-style weights into the fused parameters."""
        from mstar.model.loader import LLAMA_STACKED_PARAMS, load_hf_weights

        return load_hf_weights(self, weights, stacked_params=LLAMA_STACKED_PARAMS)
