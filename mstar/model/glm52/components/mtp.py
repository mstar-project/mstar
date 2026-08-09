"""GLM-5.2 MTP (multi-token-prediction) module — the M3 draft model.

Checkpoint anatomy [code-verified against model.safetensors.index.json]:
``model.layers.78.*`` is one more full GLM decoder layer (own MLA stack,
own FULL DSA indexer — layer 78 satisfies the IndexShare formula — and its
own 256+1-expert fp8 MoE) plus the DeepSeek-V3 MTP glue:

- ``enorm`` / ``hnorm``: RMSNorms over the draft token's embedding and the
  previous step's final hidden state,
- ``eh_proj``: (2·hidden → hidden) fusion of the two,
- ``shared_head.norm``: final norm before logits.

The module owns NO embedding and NO head — the checkpoint ships neither
under layer 78; drafts reuse the target's ``embed_tokens`` and ``lm_head``
(callers pass embeddings in and apply the head to what comes back).

``num_nextn_predict_layers=1``: this one module is *iterated* k times per
step for k draft tokens, threading its own hidden state; with
``index_share_for_mtp_iteration=True`` the DSA selection computed on the
first iteration is reused by later ones instead of re-running the indexer
(the same pattern as SHARED layers consuming a FULL layer's selection).
"""
from __future__ import annotations

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.cache_manager import BatchedCacheManager
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

    The KV written by this layer lives at its own layer slot
    (``config.num_hidden_layers``); the engine half owns allocating that
    slot and rewinding it together with the trunk KV on draft rejection.
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
        cache_handle: BatchedCacheManager,
        position_ids: torch.Tensor,
        dsa_ctx: Glm52DsaForwardContext | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(head_input, raw_hidden)``: the shared_head-normed
        state for ``lm_head`` (applied by the caller that owns the head),
        and the raw layer output for chaining the next draft iteration —
        ``hnorm``/``fuse`` expect the UN-normalized stream, exactly as the
        trunk pairing does (feeding normed hidden double-norms the fusion
        input; measured to zero out acceptance, 2026-08-09)."""
        hidden_states = self.fuse(token_embeds, prev_hidden)
        hidden_states = self.transformer_layer(
            hidden_states, cache_handle, position_ids, dsa_ctx=dsa_ctx
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
    """Greedy (temp-0) acceptance: the M3-v1 rule.

    ``draft_tokens``: (k,) tokens the MTP module proposed.
    ``target_argmax``: (k+1,) the target model's argmax at each verify
    position — position i is what the target emits given the context plus
    drafts[:i]; position k is the bonus position after all k drafts.

    Returns ``(num_accepted, next_token)``: the longest prefix where draft
    and target agree, and the token to emit after it — the target's
    correction on first mismatch, or its bonus token when all k accepted.
    Every step emits ``num_accepted + 1`` tokens, so k=0 degenerates to
    plain decode (0 accepted, the argmax emitted). Greedy acceptance keeps
    the output stream bit-identical to non-speculative decoding by
    construction — the M3 acceptance test asserts exactly that.
    """
    k = draft_tokens.shape[0]
    if target_argmax.shape[0] != k + 1:
        raise ValueError(
            f"target_argmax must have k+1={k + 1} entries, got "
            f"{target_argmax.shape[0]}"
        )
    mismatch = draft_tokens != target_argmax[:k]
    num_accepted = int(mismatch.nonzero()[0, 0]) if bool(mismatch.any()) else k
    return num_accepted, target_argmax[num_accepted]
