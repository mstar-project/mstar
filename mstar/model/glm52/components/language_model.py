"""GLM-5.2 language-model builders over existing mstar primitives."""
from __future__ import annotations

from mstar.distributed.communication import CommGroup
from mstar.model.components import RMSNorm
from mstar.model.components.distributed import (
    ColumnParallelLinear,
    ParallelGatedMLP,
    VocabParallelEmbedding,
)
from mstar.model.glm52.components.moe import Glm52SparseMoeBlock
from mstar.model.glm52.config import Glm52ModelConfig


def build_embedding(
    config: Glm52ModelConfig, comm_group: CommGroup | None = None
) -> VocabParallelEmbedding:
    return VocabParallelEmbedding(
        num_embeddings=config.vocab_size,
        embedding_dim=config.hidden_size,
        comm_group=comm_group,
        padding_idx=config.pad_token_id,
    )


def build_lm_head(
    config: Glm52ModelConfig, comm_group: CommGroup | None = None
) -> ColumnParallelLinear:
    return ColumnParallelLinear(
        comm_group or CommGroup.trivial(),
        input_size=config.hidden_size,
        output_size=config.vocab_size,
        bias=False,
        gather_output=True,
    )


def build_rmsnorm(config: Glm52ModelConfig) -> RMSNorm:
    return RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


def build_dense_mlp(
    config: Glm52ModelConfig, comm_group: CommGroup | None = None
) -> ParallelGatedMLP:
    return ParallelGatedMLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        comm_group=comm_group,
        activation=config.hidden_act,
        bias=False,
    )


def is_moe_layer(config: Glm52ModelConfig, layer_idx: int) -> bool:
    """Layers 0..first_k_dense_replace-1 are dense; every layer after is MoE."""
    return layer_idx >= config.first_k_dense_replace


def build_mlp_for_layer(
    config: Glm52ModelConfig, layer_idx: int, comm_group: CommGroup | None = None
):
    if is_moe_layer(config, layer_idx):
        return Glm52SparseMoeBlock(config, comm_group=comm_group)
    return build_dense_mlp(config, comm_group=comm_group)
