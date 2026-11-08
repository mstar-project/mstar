"""Qwen3.5's multi-token-prediction head as a drafter on the engine's
interface (`mstar.engine.resources.draft.drafter`).

The checkpoints ship one MTP layer (``mtp.*``). Row ``p`` of the head takes the
target's hidden state at ``p`` (after the final norm) and the embedding of the
token at ``p + 1``, and predicts the token at ``p + 2``:
``norm(layer(fc([norm_e(embed) | norm_h(hidden)])))`` into the target's LM
head. Its one full-attention layer keeps its K/V as one more layer of the
target's KV pool, at the row's own slot and position, so it shares the
target's pages, lengths, take-back, eviction and prefix pages.

Inside a verify step at slots ``L .. L + k`` (the block the target runs):

    propose   the first draft from the carry, the head's output at row
              ``L - 1``; then ``k - 1`` chained passes over the block's own
              slots, pass ``m`` running row ``L + m - 1`` on the last pass's
              output and the draft it made (``keep``: only that row past
              attention); the earlier rows rerun on the same inputs, and
              the rows past it write slots the next pass or ``observe`` overwrite
    target    the target over ``[input, d1 .. dk]``
    observe   the head over the whole block on the target's hidden states and
              each row's following token (the verified run, then the bonus
              token): its K/V for every kept row as the prompt would have
              written them, and the new carry, its output at the last kept
              row, where the next step's first draft comes from

A prompt step runs the head over every prompt row the same way, the last
row's following token being the one the step samples (a chunk's last row
takes the next chunk's first token, ``mtp_next_ids``), and stores its output
at the last row as the request's carry.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.draft.drafter import (
    UNBOUND_KV_LAYER,
    CarrySpec,
    ChunkBoundary,
    DrafterNeeds,
    DraftInputs,
    ModelGeometry,
    VerifyResult,
)
from mstar.model.components.chained_draft import at_last_rows, chain_drafts, kept_rows
from mstar.model.qwen3_5.config import ATTN, KV_CACHE, Qwen3_5Config

# The per-token input a prompt step's head reads: each row's following token
# id, the prompt's own, -1 where it is the token the step samples.
MTP_NEXT_IDS = "mtp_next_ids"
# the carry's dtype: a lossless store of any compute dtype's hidden state
CARRY_DTYPE = torch.float32
# the layer set the head's one layer joins the target's KV pool with
LAYER_SET = "mtp"


@dataclass(frozen=True)
class MTP:
    """The MTP drafter as its node declares it, from config alone."""

    config: Qwen3_5Config
    num_draft_tokens: int
    # the checkpoint's weight dtype, as config.json names it
    dtype: str

    def __post_init__(self):
        if self.config.mtp_num_hidden_layers != 1:
            raise ValueError(
                f"the Qwen3.5 MTP drafter runs one MTP layer; the checkpoint has "
                f"{self.config.mtp_num_hidden_layers}"
            )
        if self.config.mtp_use_dedicated_embeddings:
            # it would draft with the target's table: exact output, nothing accepted
            raise ValueError("the checkpoint's MTP head has its own embedding table "
                             "(mtp_use_dedicated_embeddings), which this drafter doesn't load")

    @property
    def needs(self) -> DrafterNeeds:
        cfg = self.config
        geometry = ModelGeometry(
            num_layers=1, num_kv_heads=cfg.num_key_value_heads, head_dim=cfg.head_dim,
            hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size, dtype=self.dtype,
            # it attends through the target's attention resource
            num_qo_heads=cfg.num_attention_heads,
        )
        return DrafterNeeds(
            # its one layer joins the target's KV pool, which its attention is built against
            kv=geometry.layer_set(KV_CACHE, LAYER_SET),
            carry=CarrySpec(cfg.hidden_size, CARRY_DTYPE),
            target_hidden=True,
            # a media row's following input is an embedding, which the prompt step has whole; video
            # isn't tested on a GPU yet, so a drafting node refuses it per request
            input_kinds=frozenset({"text", "image"}),
            chunk_boundary=ChunkBoundary.READS_NEXT_TOKEN,
            geometry=geometry,
        )

    def node_resources(self, nodes: set[str]) -> list:
        """None of its own: its layer joins the target's KV pool, it attends and
        rotates through the target's resources, and the sampler keeps its carry."""
        return []


class Qwen3_5MTPHead(nn.Module):
    """The checkpoint's one MTP layer (``mtp.*``): ``fc``, the two pre-fc norms,
    a gated full-attention decoder layer writing KV layer ``kv_layer_idx``, and
    a final norm whose output feeds the target's LM head and, chained, the next
    row. ``kv_layer_idx`` is bound from the pool's layer set (`bind_resources`)."""

    def __init__(self, config: Qwen3_5Config, comm_group: CommGroup, kv_layer_idx: int = UNBOUND_KV_LAYER):
        super().__init__()
        from mstar.model.qwen3_5.components.language_model import RopeCache, _build_full_attn_layer, _norm

        self.config = config
        self.kv_layer_idx = kv_layer_idx
        self.comm_group = comm_group
        hidden = config.hidden_size
        # replicated: its input is the replicated hidden state
        self.fc = nn.Linear(2 * hidden, hidden, bias=False)
        self.pre_fc_norm_embedding = _norm(config)
        self.pre_fc_norm_hidden = _norm(config)
        self.rope_cache = RopeCache()
        self.layers = nn.ModuleList([_build_full_attn_layer(config, self.rope_cache, comm_group)])
        self.norm = _norm(config)
        layer = self.layers[0]
        # the residual adds fuse into the next norm, which also reduces the TP partial sums
        layer.self_attn.o_proj.reduce_results = False
        layer.mlp.down_proj.reduce_results = False

    def bind_resources(self, resources: dict) -> None:
        self.kv_layer_idx = resources[KV_CACHE].layer_set(LAYER_SET).first

    @property
    def kv_layers_unbound(self) -> bool:
        return self.kv_layer_idx == UNBOUND_KV_LAYER

    def forward(
        self, hidden: torch.Tensor, next_embeds: torch.Tensor, *,
        cos_sin: tuple[torch.Tensor, torch.Tensor], keep: torch.Tensor | None = None, label: str = "main",
    ) -> torch.Tensor:
        """The head over every row, its K/V written for each; its output at the
        rows ``keep`` indexes (all by default), past attention only those."""
        cg = self.comm_group if self.comm_group.world_size > 1 else None
        x = self.fc(torch.cat(
            (self.pre_fc_norm_embedding(next_embeds), self.pre_fc_norm_hidden(hidden)), dim=-1,
        ))
        self.rope_cache.set(*cos_sin)
        layer = self.layers[0]
        attend = layer.self_attn.attend
        attend.bind_step(label)
        _set_layer(attend, self.kv_layer_idx)
        h = layer.self_attn(layer.input_layernorm(x))
        if keep is not None:
            h, x = h.index_select(0, keep), x.index_select(0, keep)
        h, residual = layer.post_attention_layernorm.forward_residual(h, x, cg)
        h = layer.mlp(h)
        normed, _ = self.norm.forward_residual(h, residual, cg)
        return normed


@torch.compiler.disable
def _set_layer(attend, idx: int) -> None:
    # eager, as the stack's cursors are, so no compiled frame specializes on the index
    attend.set_layer_idx(idx)


class Qwen3_5MTPDrafter:
    """The drafter a verifying Qwen3.5 node calls: the target's MTP head, its
    embedding and LM head, all the target's own modules (no copy)."""

    def __init__(self, decl: MTP, lm: nn.Module):
        if getattr(lm, "mtp", None) is None:
            raise ValueError("the MTP drafter needs a target built with its MTP head")
        self.decl = decl
        self.needs = decl.needs
        self.lm = lm
        self.k = decl.num_draft_tokens
        # The verify block's rotary tables, the target's for the same step (`bind_block`): the
        # head runs at the block's positions, in `propose` and `observe` of one forward.
        self._block_cos_sin = None

    @property
    def kv_layers_unbound(self) -> bool:
        return self.lm.mtp.kv_layers_unbound

    # ── prompt step ──

    def prefill(
        self, resources, hidden: torch.Tensor, cos_sin, next_ids: torch.Tensor | None = None,
        next_embeds: torch.Tensor | None = None, sampled: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The head over a prompt step's rows; returns each prompt's carry, its
        output at the prompt's last row. Each row's following token is
        ``next_ids`` (-1 where the step samples it: ``sampled``), or for a
        step over embeddings the next row's, the last row's from ``sampled``."""
        embed = self.lm.model.embed_tokens
        tokens = hidden.shape[0]
        last, at = resources[ATTN].prompt_last_rows(tokens, hidden.device)
        sampled = sampled.reshape(-1)
        if next_ids is not None:
            fill = at_last_rows(sampled.to(next_ids.dtype), at, tokens)
            following = embed(torch.where(next_ids >= 0, next_ids, fill))
        else:
            rolled = torch.cat([next_embeds.roll(-1, 0), next_embeds[:1]])
            following = rolled.index_copy(0, at, embed(sampled).to(next_embeds.dtype))[:tokens]
        out = self.lm.mtp(hidden, following.to(hidden.dtype), cos_sin=cos_sin, keep=last)
        return out.to(CARRY_DTYPE)

    # ── verify step ──

    def bind_block(self, cos_sin) -> None:
        """The step's verify block's rotary tables, which the target runs at too."""
        self._block_cos_sin = cos_sin

    def propose(self, inputs: DraftInputs) -> torch.Tensor:
        lm = self.lm
        embed = lm.model.embed_tokens
        cos_sin = self._block_cos_sin
        assert cos_sin is not None, "the verify step binds the block's rotary tables before proposing"
        return chain_drafts(
            inputs.carry.to(embed.weight.dtype), self.k,
            lambda hidden, ids, keep: lm.mtp(hidden, embed(ids), cos_sin=cos_sin, keep=keep),
            lambda out: lm.lm_head(out).argmax(-1),
        )

    def observe(self, block: torch.Tensor, result: VerifyResult, hidden: torch.Tensor) -> torch.Tensor:
        """The head's K/V over the block from the target's hidden states, and the
        new carry: its output at each row's last kept position."""
        out = self.lm.mtp(
            hidden, self.lm.model.embed_tokens(result.tokens.reshape(-1)).to(hidden.dtype),
            cos_sin=self._block_cos_sin, keep=kept_rows(block, result.accepted),
        )
        return out.to(CARRY_DTYPE)
