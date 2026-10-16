"""GLM-5.2 decoder layer with MLA position ids threaded through attention."""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.glm52.components.attention import Glm52MLAAttention
from mstar.model.glm52.components.language_model import (
    build_mlp_for_layer,
    build_rmsnorm,
)
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.glm52.dsa import Glm52DsaForwardContext


class Glm52DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Glm52ModelConfig,
        layer_idx: int,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.comm_group = comm_group or CommGroup.trivial()
        # Each all-reduce is fused into the next add + RMSNorm: o_proj and the FFN return
        # partials, and the next norm reduces them.
        self.fused_add_rmsnorm = config.fused_add_rmsnorm
        self.self_attn = Glm52MLAAttention(
            config, comm_group=comm_group, layer_idx=layer_idx)
        if self.fused_add_rmsnorm:
            self.self_attn.o_proj.reduce_results = False
        self.mlp = build_mlp_for_layer(
            config, layer_idx, comm_group=comm_group,
            reduce_results=not self.fused_add_rmsnorm)
        self.input_layernorm = build_rmsnorm(config)
        self.post_attention_layernorm = build_rmsnorm(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        dsa_ctx: Glm52DsaForwardContext | None = None,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
        rows: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``rows``: run the FFN half on these rows only (and return only them).

        With ``fused_add_rmsnorm`` the layer returns ``(FFN partial, residual)``, and takes
        the previous layer's pair (``residual`` None: ``hidden_states`` is the embedding).
        """
        if self.fused_add_rmsnorm:
            return self._forward_fused(
                hidden_states, residual, position_ids, dsa_ctx, rope_cos_sin, rows)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, position_ids, dsa_ctx=dsa_ctx, rope_cos_sin=rope_cos_sin)
        hidden_states = residual + hidden_states
        if rows is not None:
            hidden_states = hidden_states.index_select(0, rows)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def forward_block_row(
        self, hidden_states: torch.Tensor, position_ids: torch.Tensor, row: int,
        block: int, rows: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """One query per request at ``row`` of the step's block
        (``Glm52MLAAttention.forward_block_row``); the MTP layer's own norms reduce nothing."""
        assert not self.fused_add_rmsnorm, "the block-row pass is the MTP layer's, built unfused"
        residual = hidden_states
        hidden_states = self.self_attn.forward_block_row(
            self.input_layernorm(hidden_states), position_ids, row, block, rows)
        hidden_states = residual + hidden_states
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

    def _forward_fused(self, hidden_states, residual, position_ids, dsa_ctx, rope_cos_sin, rows):
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm.forward_residual(
                hidden_states, residual, self.comm_group)
        hidden_states = self.self_attn(
            hidden_states, position_ids, dsa_ctx=dsa_ctx, rope_cos_sin=rope_cos_sin)
        if rows is None:
            hidden_states, residual = self.post_attention_layernorm.forward_residual(
                hidden_states, residual, self.comm_group)
        else:
            residual = (self.comm_group.all_reduce(hidden_states) + residual).index_select(0, rows)
            hidden_states = self.post_attention_layernorm(residual)
        return self.mlp(hidden_states), residual
