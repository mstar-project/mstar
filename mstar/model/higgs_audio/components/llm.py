"""Higgs-Audio text LLM: dense Qwen3 (1.7B-class) transformer.

Built from the shared transformer components. Qwen3 = Llama-style
decoder with GQA, per-head QK-norm, SwiGLU MLP, and plain RoPE
(theta 1e6, no llama-3 scaling).

Checkpoint layout is flat (``embed_tokens.*``, ``layers.N.*``,
``norm.*``) with the text head at ``audio_decoder_proj.text_lm_head``;
the weight iterator in ``higgs_audio_model.py`` remaps that to
``lm_head``. QKV and gate/up projections are fused from construction via
the parallel-linear classes; loading routes the separate HF shards via
``LLAMA_STACKED_PARAMS``.
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.kv_cache_engine import BatchedCacheManager
from mstar.model.components import DecoderLayer, RMSNorm
from mstar.model.components.distributed import (
    ColumnParallelLinear,
    ParallelAttention,
    ParallelGatedMLP,
    VocabParallelEmbedding,
)
from mstar.model.higgs_audio.config import HiggsAudioModelConfig


def _build_decoder_layer(
    config: HiggsAudioModelConfig, comm_group: CommGroup | None = None,
) -> DecoderLayer:
    return DecoderLayer(
        self_attn=ParallelAttention(
            comm_group=comm_group,
            hidden_size=config.num_attention_heads * config.head_dim,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            qk_norm=True,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            input_hidden_size=config.hidden_size,
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


class HiggsAudioLLM(nn.Module):
    """Qwen3 causal LM; parameter paths mirror the (flat) HF checkpoint."""

    def __init__(self, config: HiggsAudioModelConfig, comm_group: CommGroup | None = None):
        super().__init__()
        self.config = config
        if comm_group is None:
            comm_group = CommGroup.trivial()
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            comm_group=comm_group,
        )
        self.layers = nn.ModuleList(
            [_build_decoder_layer(config, comm_group=comm_group)
             for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            comm_group=comm_group,
            input_size=config.hidden_size,
            output_size=config.vocab_size,
            bias=False,
            gather_output=True,
        )

    def forward(
        self,
        input_embeds: torch.Tensor,
        cache_handle: BatchedCacheManager,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = input_embeds
        for layer_idx, decoder_layer in enumerate(self.layers):
            cache_handle.set_layer_idx(layer_idx)
            hidden_states = decoder_layer(
                hidden_states=hidden_states, cache_handle=cache_handle,
            )
        cache_handle.advance_seq_lens()
        return self.norm(hidden_states)
