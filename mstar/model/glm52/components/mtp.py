"""GLM-5.2 MTP (multi-token-prediction) draft module."""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.glm52.components.decoder_layer import Glm52DecoderLayer
from mstar.model.glm52.components.language_model import build_rmsnorm
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.glm52.dsa import Glm52DsaForwardContext


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
        # The MTP layer must be FULL under the IndexShare formula: the
        # checkpoint ships indexer weights at layer 78 (GLM placed 78 =
        # offset-1 + 19·freq deliberately), and index_share_for_mtp_iteration
        # reuses THIS layer's selection across draft iterations. A config
        # whose MTP position lands SHARED would construct indexer-less and
        # desync from the checkpoint — fail loudly instead.
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
        self.transformer_layer = Glm52DecoderLayer(
            config, layer_idx=config.num_hidden_layers, comm_group=comm_group
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
        dsa_ctx: Glm52DsaForwardContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(head_input, raw_hidden)``: the shared_head-normed
        state for ``lm_head`` (applied by the caller that owns the head),
        and the raw layer output for chaining the next draft iteration.
        ``hnorm``/``fuse`` expect the UN-normalized stream, exactly as the
        trunk pairing does; handing them a normed hidden state double-norms
        the fusion input and drops acceptance to nothing.
        """
        hidden_states = self.fuse(token_embeds, prev_hidden)
        hidden_states = self.transformer_layer(
            hidden_states, position_ids, dsa_ctx=dsa_ctx
        )
        return self.shared_head(hidden_states), hidden_states


# Checkpoint sub-keys under ``model.layers.78.`` that belong to the MTP glue
# and map 1:1 onto Glm52MTPModule attributes; everything else under the
# prefix routes into ``transformer_layer.``. The loader remap and its test
# both import this so the contract lives in exactly one place.
MTP_GLUE_PREFIXES = ("enorm", "hnorm", "eh_proj", "shared_head")


def remap_mtp_key(sub_key: str) -> str:
    """``model.layers.78.<sub_key>`` → Glm52MTPModule state-dict key."""
    if sub_key.startswith(MTP_GLUE_PREFIXES):
        return sub_key
    return f"transformer_layer.{sub_key}"


def mtp_greedy_verify(
    draft_tokens: torch.Tensor, target_argmax: torch.Tensor
) -> tuple[int, torch.Tensor]:
    """Greedy (temp-0) acceptance: the drafts' longest prefix matching argmax."""
    k = draft_tokens.shape[0]
    if target_argmax.shape[0] != k + 1:
        raise ValueError(
            f"target_argmax must have k+1={k + 1} entries, got "
            f"{target_argmax.shape[0]}"
        )
    mismatch = draft_tokens != target_argmax[:k]
    num_accepted = int(mismatch.nonzero()[0, 0]) if bool(mismatch.any()) else k
    return num_accepted, target_argmax[num_accepted]


def mtp_greedy_verify_host(
    draft_tokens: list[int], target_argmax: list[int]
) -> int:
    """The same rule on host lists — no device round trip."""
    k = len(draft_tokens)
    if len(target_argmax) != k + 1:
        raise ValueError(
            f"target_argmax must have k+1={k + 1} entries, got "
            f"{len(target_argmax)}"
        )
    for j in range(k):
        if draft_tokens[j] != target_argmax[j]:
            return j
    return k
