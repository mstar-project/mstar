"""DeepSeek-V4.1's draft head (DSpark, ``mtp.*`` in the checkpoint): a block of draft tokens
for each request in one pass.

A draft block is ``dspark_block_size`` rows after a request's last token: the token to
continue from, then a noise token per further row. The draft layers, window-only blocks with
their own experts, run over those rows. A row attends every row of its block, the later ones
too, and a window of the backbone's hidden states: the mean of the streams entering the target
layers, projected (``main_proj``) and kept as a latent per position in each draft layer's own
window (``WINDOW`` in ``STATE``, at the draft layer's index after the backbone's). The
backbone's head gives each row's logits, a low-rank Markov head corrects them row by row from
the token drafted before, and a confidence head scores each row.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.model.components.distributed import ColumnParallelLinear, VocabParallelEmbedding
from mstar.model.deepseek_v41.components.attention import DeepseekV41Attention
from mstar.model.deepseek_v41.components.layers import DeepseekV41Block, collapse, vocab_logits
from mstar.model.deepseek_v41.components.linear import RMSNorm, RoundedLinear
from mstar.model.deepseek_v41.components.rope import Rope
from mstar.model.deepseek_v41.config import CSA, LABEL, WINDOW, DeepseekV41Config

if TYPE_CHECKING:
    from mstar.engine.resources.csa.manager import CSARows


class Draft(NamedTuple):
    """A step's draft blocks, a row per request."""
    ids: torch.Tensor         # (requests, block): the drafted tokens, each row's argmax
    logits: torch.Tensor      # (requests, block, vocab) fp32, Markov correction included
    confidence: torch.Tensor  # (requests, block) fp32: the confidence head's score of each row


@dataclass
class DraftRows:
    """What a draft pass hands its layers. ``main (T, hidden)``: the step's projected hidden
    states, a row per step row (``rows``). Per draft row: ``pos`` its position, and its
    entries for the CSA resource (``idx``, ``columns``): its request's window in the order of
    the reference's ring, then the request's draft rows."""

    main: torch.Tensor
    rows: CSARows
    pos: torch.Tensor
    idx: torch.Tensor
    columns: torch.Tensor


class DraftAttention(DeepseekV41Attention):
    """A draft layer's attention. Its window holds the projected hidden states of the
    request's last tokens, not the layer's own rows."""

    def _keep(self, main: torch.Tensor, rows: CSARows, rope: Rope) -> torch.Tensor:
        """The requests' windows ``(requests, window, D)`` with the step's rows in: each
        position's projected hidden state as this layer's latent, at ring slot ``p % window``."""
        ring = self._state.gather(WINDOW, self.layer, label=LABEL)
        latent = self._latent(main, rope.angles(rows.pos, False))
        tail = rows.pos >= rows.end - self.window
        ring[rows.row_req[tail], rows.pos[tail] % self.window] = latent[tail].to(ring.dtype)
        self._state.scatter_(WINDOW, ring, self.layer, label=LABEL)
        return ring

    def forward(self, x: torch.Tensor, rope: Rope, draft: DraftRows) -> torch.Tensor:
        """``x (requests * block, hidden)``, the draft rows request by request."""
        ring = self._keep(draft.main, draft.rows, rope)
        angles = rope.angles(draft.pos, False)
        _, q = self._query(x, angles)
        source = torch.cat([ring.flatten(0, 1), self._latent(x, angles)], dim=0)
        return self._output(self._csa.attend(q, self.attn_sink, source, draft.idx, draft.columns), angles)


class MarkovHead(nn.Module):
    """A rank-``dspark_markov_rank`` bias on a row's logits from the token before it: that
    token's embedding through a head over the vocabulary, sharded as the backbone's are."""

    def __init__(self, config: DeepseekV41Config, comm_group: CommGroup) -> None:
        super().__init__()
        self.comm_group = comm_group
        self.embed = VocabParallelEmbedding(
            num_embeddings=config.vocab_size, embedding_dim=config.dspark_markov_rank, comm_group=comm_group)
        self.head = ColumnParallelLinear(
            comm_group, input_size=config.dspark_markov_rank, output_size=config.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``tokens (requests,)`` -> their fp32 logit bias ``(requests, vocab)`` and embedding."""
        embedding = self.embed(tokens)
        return vocab_logits(embedding, self.head.weight, self.comm_group), embedding


class ConfidenceHead(nn.Module):
    """A score per draft row from its final hidden state and its Markov embedding, in fp32."""

    def __init__(self, config: DeepseekV41Config) -> None:
        super().__init__()
        self.proj = RoundedLinear(config.hidden_size + config.dspark_markov_rank, 1, rounded=False)

    def forward(self, hidden: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        return F.linear(torch.cat([hidden, embedding], dim=-1).float(), self.proj.weight.float()).squeeze(-1)


class DraftBlock(DeepseekV41Block):
    """One draft layer: the backbone's block around ``DraftAttention``, with the draft
    layers' expert counts. The first also projects the backbone's hidden states, and the last
    holds the final norm and the two heads."""

    attention_cls = DraftAttention

    def __init__(self, config: DeepseekV41Config, stage: int, comm_group: CommGroup | None = None,
                 moe_parallel=None) -> None:
        super().__init__(config.draft_block_config(), config.num_hidden_layers + stage, comm_group, moe_parallel)
        hidden = config.hidden_size
        if stage == 0:
            self.main_proj = RoundedLinear(hidden * len(config.dspark_target_layer_ids), hidden)
            self.main_norm = RMSNorm(hidden, eps=config.rms_norm_eps)
        if stage == config.num_nextn_predict_layers - 1:
            self.norm = RMSNorm(hidden, eps=config.rms_norm_eps)
            self.markov_head = MarkovHead(config, comm_group or CommGroup.trivial())
            self.confidence_head = ConfidenceHead(config)


class DraftHead(nn.ModuleList):
    """The draft layers, named as the checkpoint names them (``mtp.N``)."""

    def __init__(self, config: DeepseekV41Config, comm_group: CommGroup | None = None,
                 moe_parallel=None) -> None:
        super().__init__(
            DraftBlock(config, stage, comm_group, moe_parallel) for stage in range(config.num_nextn_predict_layers))
        self.block, self.window, self.hc_mult = config.dspark_block_size, config.sliding_window, config.hc_mult
        self.noise_token_id = config.dspark_noise_token_id
        self._csa = None

    def bind_resources(self, resources: dict) -> None:
        self._csa = resources[CSA]

    def _rows(self, rows: CSARows, main: torch.Tensor) -> DraftRows:
        """The draft rows after each request's last row of the step. A row's entries are laid
        out as the checkpoint's reference lays them, since the softmax accumulates over them in
        blocks: the window's ring slots in order (slot ``j`` holds the latest position that is
        ``j`` modulo the window), as many as the request has tokens, then the block's rows."""
        size, win, device = self.block, self.window, main.device
        requests = rows.req_first.shape[0]
        last = rows.pos[rows.req_first + rows.req_rows - 1]
        req = torch.arange(requests, device=device).repeat_interleave(size)
        pos = (last[:, None] + 1 + torch.arange(size, device=device)).flatten()
        # as wide as the longest row, which the host knows from the step's spans
        width = min(win, max(start + n for _, n, start in rows.spans)) + size
        col = torch.arange(width, device=device)[None, :]
        held = (last[req, None] + 1).clamp_max(win)
        in_ring = req[:, None] * win + col
        in_block = requests * win + req[:, None] * size + col - held
        idx = torch.where(col < held, in_ring, torch.where(col < held + size, in_block, -1))
        return DraftRows(main, rows, pos, idx, (held + size).squeeze(1))

    def forward(self, tokens: torch.Tensor, taps: torch.Tensor, embed: nn.Module, logits, rope: Rope) -> Draft:
        """A draft block per request, continuing from ``tokens (requests,)``, the token each
        request's last row of the step sampled. ``taps (T, targets * hidden)``: the mean of the
        streams entering each target layer, for every row of the step, which the draft layers'
        windows take in first. ``embed`` and ``logits`` are the backbone's embedding and head."""
        first, last = self[0], self[-1]
        draft = self._rows(self._csa.current(LABEL), first.main_norm(first.main_proj(taps)))
        requests, size = tokens.shape[0], self.block
        ids = torch.full((requests, size), self.noise_token_id, dtype=tokens.dtype, device=tokens.device)
        ids[:, 0] = tokens
        hidden = embed(ids.flatten())
        streams = hidden.unsqueeze(1).repeat(1, self.hc_mult, 1)
        pre = torch.zeros(hidden.shape[0], self.hc_mult, dtype=torch.float32, device=hidden.device)
        pre[:, 0] = 1.0
        for block in self:
            streams, pre = block(streams, pre, rope, draft)
        hidden = collapse(streams, pre).view(requests, size, -1)
        out = logits(last.norm(hidden))
        # each row's logits corrected from the token drafted before it, left to right
        drafted, embeddings, prev = [], [], tokens
        for row in range(size):
            bias, embedding = last.markov_head(prev)
            out[:, row] += bias
            prev = out[:, row].argmax(-1)
            drafted.append(prev)
            embeddings.append(embedding)
        confidence = last.confidence_head(hidden, torch.stack(embeddings, dim=1))
        return Draft(torch.stack(drafted, dim=1), out, confidence)
