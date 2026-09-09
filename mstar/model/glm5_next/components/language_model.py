"""GLM-5.3-Flash language-model builders over existing mstar primitives.

Same shape as ``glm52/components/language_model.py``; the two deltas are
that ``build_mlp_for_layer`` reads the config's ``mlp_layer_types``
schedule (not a ``first_k_dense_replace`` formula — though the config
validates they agree) and that the dense MLP carries the SwiGLU clamp.
"""
from __future__ import annotations

from mstar.distributed.communication import CommGroup
from mstar.model.components import RMSNorm
from mstar.model.components.distributed import (
    ColumnParallelLinear,
    VocabParallelEmbedding,
)
from mstar.model.glm5_next.components.moe import (
    Glm5NextGatedMLP,
    Glm5NextSparseMoeBlock,
)
from mstar.model.glm5_next.config import Glm5NextModelConfig


def build_embedding(
    config: Glm5NextModelConfig, comm_group: CommGroup | None = None
) -> VocabParallelEmbedding:
    return VocabParallelEmbedding(
        num_embeddings=config.vocab_size,
        embedding_dim=config.hidden_size,
        comm_group=comm_group,
        padding_idx=config.pad_token_id,
    )


def build_lm_head(
    config: Glm5NextModelConfig, comm_group: CommGroup | None = None
) -> ColumnParallelLinear:
    return ColumnParallelLinear(
        comm_group or CommGroup.trivial(),
        input_size=config.hidden_size,
        output_size=config.vocab_size,
        bias=False,
        gather_output=True,
    )


def build_rmsnorm(config: Glm5NextModelConfig) -> RMSNorm:
    return RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


def build_dense_mlp(
    config: Glm5NextModelConfig, comm_group: CommGroup | None = None
) -> Glm5NextGatedMLP:
    return Glm5NextGatedMLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        comm_group=comm_group,
        activation=config.hidden_act,
        bias=False,
        swiglu_limit=config.swiglu_limit,
    )


def build_mlp_for_layer(
    config: Glm5NextModelConfig, layer_idx: int, comm_group: CommGroup | None = None
):
    """Dense on the ``mlp_layer_types`` "dense" entries (layers 0..2), MoE
    everywhere else — the MTP layer (idx == num_hidden_layers) is MoE."""
    if config.is_dense_mlp_layer(layer_idx):
        return build_dense_mlp(config, comm_group=comm_group)
    return Glm5NextSparseMoeBlock(config, comm_group=comm_group)
