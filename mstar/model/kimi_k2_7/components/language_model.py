"""Kimi-K2.7 language-model builders over existing mstar primitives."""
from __future__ import annotations

from mstar.distributed.communication import CommGroup
from mstar.model.components import RMSNorm
from mstar.model.components.distributed import (
    ColumnParallelLinear,
    ParallelGatedMLP,
    VocabParallelEmbedding,
)
from mstar.model.kimi_k2_7.components.moe import KimiSparseMoeBlock
from mstar.model.kimi_k2_7.config import KimiK2Config


def build_embedding(
    config: KimiK2Config, comm_group: CommGroup | None = None
) -> VocabParallelEmbedding:
    return VocabParallelEmbedding(
        num_embeddings=config.vocab_size,
        embedding_dim=config.hidden_size,
        comm_group=comm_group,
        padding_idx=config.pad_token_id,
    )


def build_lm_head(
    config: KimiK2Config, comm_group: CommGroup | None = None
) -> ColumnParallelLinear:
    return ColumnParallelLinear(
        comm_group or CommGroup.trivial(),
        input_size=config.hidden_size,
        output_size=config.vocab_size,
        bias=False,
        gather_output=True,
    )


def build_rmsnorm(config: KimiK2Config) -> RMSNorm:
    return RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


def build_dense_mlp(
    config: KimiK2Config, comm_group: CommGroup | None = None
) -> ParallelGatedMLP:
    return ParallelGatedMLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        comm_group=comm_group,
        activation=config.hidden_act,
        bias=False,
    )


def is_moe_layer(config: KimiK2Config, layer_idx: int) -> bool:
    return (
        layer_idx >= config.first_k_dense_replace
        and layer_idx % config.moe_layer_freq == 0
    )


def build_moe_block(
    config: KimiK2Config, comm_group: CommGroup | None = None
) -> KimiSparseMoeBlock:
    return KimiSparseMoeBlock(config, comm_group=comm_group)


def build_mlp_for_layer(
    config: KimiK2Config, layer_idx: int, comm_group: CommGroup | None = None
):
    if is_moe_layer(config, layer_idx):
        return build_moe_block(config, comm_group=comm_group)
    return build_dense_mlp(config, comm_group=comm_group)
