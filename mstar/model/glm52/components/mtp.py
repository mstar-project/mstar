"""GLM-5.2 MTP (multi-token-prediction) draft module."""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.glm52.components.decoder_layer import Glm52DecoderLayer
from mstar.model.glm52.components.language_model import build_rmsnorm
from mstar.model.glm52.config import Glm52ModelConfig


class Glm52SharedHead(nn.Module):
    """Named to mirror the checkpoint's ``shared_head.norm`` key: the MTP
    module's final norm. The actual head weight is the target's ``lm_head``
    (not duplicated in the checkpoint), applied by the caller."""

    def __init__(self, config: Glm52ModelConfig) -> None:
        super().__init__()
        self.norm = build_rmsnorm(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)


class Glm52MTPModule(nn.Module):
    """One draft iteration: fuse (token embedding, previous hidden) and run
    the layer-78 decoder layer.
    """

    def __init__(
        self, config: Glm52ModelConfig, comm_group: CommGroup | None = None
    ) -> None:
        super().__init__()
        # the checkpoint ships an indexer at layer 78: a config that puts the MTP layer on a
        # SHARED slot would build it without one and miss those weights
        from mstar.model.glm52.components.indexer import is_full_indexer_layer

        if not is_full_indexer_layer(config, config.num_hidden_layers):
            raise ValueError(
                f"MTP layer_idx={config.num_hidden_layers} is SHARED under "
                f"the IndexShare formula (offset="
                f"{config.index_skip_topk_offset}, freq={config.index_topk_freq})"
                " — the MTP module requires its own FULL indexer"
            )
        self.enorm = build_rmsnorm(config)
        self.hnorm = build_rmsnorm(config)
        self.eh_proj = nn.Linear(
            2 * config.hidden_size, config.hidden_size, bias=False
        )
        # the draft layer has no next norm to take its partials: it reduces its own
        self.transformer_layer = Glm52DecoderLayer(
            config, layer_idx=config.num_hidden_layers, comm_group=comm_group,
            fused_add_rmsnorm=False,
        )
        self.shared_head = Glm52SharedHead(config)

    def fuse(
        self, token_embeds: torch.Tensor, prev_hidden: torch.Tensor
    ) -> torch.Tensor:
        return self.eh_proj(
            torch.cat([self.enorm(token_embeds), self.hnorm(prev_hidden)], dim=-1)
        )

    def forward(
        self,
        token_embeds: torch.Tensor,
        prev_hidden: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(head_input, raw_hidden)``: the shared_head-normed state, which the caller's
        ``lm_head`` reads and the next draft chains, and the layer output under that norm.
        ``prev_hidden`` is the trunk's post-final-norm state or the previous draft's head input,
        as the reference pairs and chains them (``hnorm`` normalizes it again)."""
        hidden_states = self.fuse(token_embeds, prev_hidden)
        hidden_states = self.transformer_layer(hidden_states, position_ids)
        return self.shared_head(hidden_states), hidden_states

    def forward_block_row(
        self,
        token_embeds: torch.Tensor,
        prev_hidden: torch.Tensor,
        position_ids: torch.Tensor,
        row: int,
        block: int,
        rows: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``forward`` for one draft per request whose query sits at ``row`` of the step's
        block (``Glm52DecoderLayer.forward_block_row``); ``rows`` carries the block's cache
        rows from one draft to the next."""
        hidden_states = self.transformer_layer.forward_block_row(
            self.fuse(token_embeds, prev_hidden), position_ids, row, block, rows)
        return self.shared_head(hidden_states), hidden_states


# Checkpoint sub-keys under ``model.layers.78.`` that map 1:1 onto Glm52MTPModule's glue;
# everything else under the prefix routes into ``transformer_layer.``.
MTP_GLUE_PREFIXES = ("enorm", "hnorm", "eh_proj", "shared_head")


def remap_mtp_key(sub_key: str) -> str:
    """``model.layers.78.<sub_key>`` → Glm52MTPModule state-dict key."""
    if sub_key.startswith(MTP_GLUE_PREFIXES):
        return sub_key
    return f"transformer_layer.{sub_key}"
