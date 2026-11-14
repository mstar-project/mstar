"""DeepSeek-V4.1 attention: one latent per token that is key and value, attended over a
sliding window of the layer's own rows plus an indexer's pick of a compressed cache.

The engine holds the state and runs the attention: the compressed latents and their index keys
on paged caches (a row per pooled group), each layer's window and the compressors' open groups in
the recurrent pool (``STATE``), and the caches' writes, scoring, selection and the softmax in the
CSA resource. The layers here project, pool, rotate and round as the checkpoint's reference does.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.model.deepseek_v41.components.linear import COLUMN, ROW, RMSNorm, RoundedLinear
from mstar.model.deepseek_v41.components.rope import Rope, rotate_tail
from mstar.model.deepseek_v41.config import (
    CSA,
    LABEL,
    POOL_KV,
    POOL_SCORE,
    STATE,
    WINDOW,
    DeepseekV41Config,
)
from mstar.utils.mx_round import round_fp4, round_fp8


@dataclass
class SharedSelection:
    """What a forward's layers hand down the stack: the last indexer's pick, which the layers
    after it attend, and the candidate source's mask, which later indexers pick inside."""

    slots: torch.Tensor | None = None
    candidates: torch.Tensor | None = None


class Compressor(nn.Module):
    """A compressed source's latents: every ``ratio`` tokens pooled into one by a softmax gate
    per dim (``ratio`` 1: a plain projection). A group still open at the end of a step waits
    in the request's ``POOL_KV`` / ``POOL_SCORE`` rows of this layer: rings by position, which
    hold the request's last tokens, more of them than a group so that tokens taken back
    (``take_back_tokens``) overwrite none of an open group."""

    def __init__(self, config: DeepseekV41Config, layer: int) -> None:
        super().__init__()
        self.ratio = config.compress_ratios[layer]
        # the pooling runs in fp32, so its projections do
        dtype = torch.float32 if self.ratio > 1 else None
        self.wkv = RoundedLinear(config.hidden_size, config.head_dim, rounded=False, dtype=dtype)
        self.norm = RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        if self.ratio > 1:
            self.wgate = RoundedLinear(config.hidden_size, config.head_dim, rounded=False, dtype=dtype)
        self.layer = layer
        self._state = None

    def bind_resources(self, resources: dict) -> None:
        self._state = resources[STATE]

    def forward(self, x: torch.Tensor, rows) -> torch.Tensor:
        """``x (T, hidden)`` -> ``(T, head_dim)``: for a row that ends a group, the group's
        unrotated latent. The other rows' are of no group, and the cache drops them."""
        ratio = self.ratio
        if ratio == 1:
            return self.norm(self.wkv(x))
        kv, score = self.wkv(x.float()), self.wgate(x.float())
        held_kv = self._state.gather(POOL_KV, self.layer, label=LABEL)  # (requests, slots, D)
        held_score = self._state.gather(POOL_SCORE, self.layer, label=LABEL)
        slots = held_kv.shape[1]
        row = torch.arange(x.shape[0], device=x.device)
        in_span = row - rows.first
        members_kv, members_score = [], []
        for back in range(ratio - 1, -1, -1):  # the group's tokens, oldest first
            here = (in_span >= back)[:, None]
            src = (row - back).clamp_min(0)
            slot = (rows.pos - back) % slots
            members_kv.append(torch.where(here, kv[src], held_kv[rows.row_req, slot]))
            members_score.append(torch.where(here, score[src], held_score[rows.row_req, slot]))
        members_kv, members_score = torch.stack(members_kv, 1), torch.stack(members_score, 1)
        pooled = (members_kv * members_score.softmax(dim=1)).sum(dim=1)
        # the step's last tokens, among them the next step's open group
        tail = rows.tail(slots)
        req, slot = rows.row_req[tail], rows.pos[tail] % slots
        held_kv[req, slot] = kv[tail]
        held_score[req, slot] = score[tail]
        self._state.scatter_(POOL_KV, held_kv, self.layer, label=LABEL)
        self._state.scatter_(POOL_SCORE, held_score, self.layer, label=LABEL)
        return self.norm(pooled.to(x.dtype))


class Indexer(nn.Module):
    """One index source's projections: the key each compressed latent stores (on the layers
    that write a cache), and each query row's index heads and head weights. The checkpoint was
    trained with keys and queries rounded to FP4: the cache rounds the keys, the queries are
    rounded here to the same grid."""

    def __init__(self, config: DeepseekV41Config, layer: int, comm_group: CommGroup) -> None:
        super().__init__()
        self.comm_group = comm_group
        self.owns_keys = layer in config.kv_source_layer_ids
        self.n_heads = divide(config.index_n_heads, comm_group.world_size)
        self.head_dim = config.index_head_dim
        self.key_rows = config.key_rows
        self.wq_b = RoundedLinear(
            config.q_lora_rank, config.index_n_heads * config.index_head_dim, comm_group, COLUMN)
        self.weights_proj = RoundedLinear(
            config.hidden_size, config.index_n_heads, comm_group, COLUMN, rounded=False)
        if self.owns_keys:
            self.wk = RoundedLinear(config.head_dim, config.index_head_dim, rounded=False)
            self.k_norm = RMSNorm(config.index_head_dim, eps=config.rms_norm_eps)
        # the score's softmax scale and its mean over heads, folded into the head weights
        self.weight_scale = config.index_head_dim ** -0.5 * config.index_n_heads ** -0.5

    def keys(self, latent: torch.Tensor, angles) -> torch.Tensor:
        """The keys of the groups ``latent``'s rows complete, which their cache rounds to FP4."""
        return rotate_tail(self.k_norm(self.wk(latent)), angles)

    def query(self, x: torch.Tensor, q_latent: torch.Tensor, angles) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.wq_b(q_latent).unflatten(-1, (self.n_heads, self.head_dim))
        q = round_fp4(rotate_tail(q, angles), self.key_rows.block, self.key_rows.e4m3_scale)
        return q, self.weights_proj(x) * self.weight_scale


class DeepseekV41Attention(nn.Module):
    def __init__(self, config: DeepseekV41Config, layer: int, comm_group: CommGroup | None = None) -> None:
        super().__init__()
        comm_group = comm_group or CommGroup.trivial()
        self.comm_group = comm_group
        tp = comm_group.world_size
        self.layer = layer
        self.head_dim = config.head_dim
        self.num_heads = divide(config.num_attention_heads, tp)
        self.num_groups = divide(config.o_groups, tp)
        self.o_lora_rank = config.o_lora_rank
        self.window = config.sliding_window
        self.block = config.fp8_block
        self.ratio = config.compress_ratios[layer]
        self.plane = config.plane_of(layer) if self.ratio else None
        heads, width = config.num_attention_heads, config.head_dim

        self.attn_sink = nn.Parameter(torch.empty(self.num_heads, dtype=torch.float32))
        self.wq_a = RoundedLinear(config.hidden_size, config.q_lora_rank)
        self.q_norm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = RoundedLinear(config.q_lora_rank, heads * width, comm_group, COLUMN)
        self.wkv = RoundedLinear(config.hidden_size, width)
        self.kv_norm = RMSNorm(width, eps=config.rms_norm_eps)
        # block-diagonal over the head groups: each group projects only its own heads
        self.wo_a = RoundedLinear(
            heads * width // config.o_groups, config.o_groups * config.o_lora_rank, comm_group,
            COLUMN, rounded=False)
        self.wo_b = RoundedLinear(config.o_groups * config.o_lora_rank, config.hidden_size, comm_group, ROW)

        self.compressor = Compressor(config, layer) if layer in config.kv_source_layer_ids else None
        self.indexer = (Indexer(config, layer, comm_group)
                        if layer in config.index_source_layer_ids else None)
        self.is_candidate_source = layer == config.candidate_source_layer_id
        self.uses_candidates = config.uses_candidates(layer)
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self._csa = self._state = None

    def bind_resources(self, resources: dict) -> None:
        self._csa, self._state = resources[CSA], resources[STATE]

    def _query(self, x: torch.Tensor, angles) -> tuple[torch.Tensor, torch.Tensor]:
        """A row's low-rank query latent, and its heads' rotated queries."""
        q_latent = self.q_norm(self.wq_a(x))
        return q_latent, rotate_tail(self.wq_b(q_latent).unflatten(-1, (self.num_heads, self.head_dim)), angles)

    def _latent(self, x: torch.Tensor, angles) -> torch.Tensor:
        """A row's window latent: rotated, then rounded to FP8 as the checkpoint keeps it."""
        return round_fp8(rotate_tail(self.kv_norm(self.wkv(x)), angles), self.block)

    def _output(self, out: torch.Tensor, angles) -> torch.Tensor:
        """The attended latents ``(rows, heads, D)`` projected back to the hidden size."""
        # the cache holds rotated latents, so the output carries the query's rotation
        out = rotate_tail(out, angles, inverse=True)
        dtype = out.dtype
        out = round_fp8(out.reshape(out.shape[0], self.num_groups, -1), self.block)
        wo_a = self.wo_a.weight.view(self.num_groups, self.o_lora_rank, -1)
        out = torch.einsum("sgd,grd->sgr", out.float(), wo_a.float()).to(dtype)
        return self.wo_b(out.flatten(1))

    def _window(self, kv: torch.Tensor, rows) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The rows each query's window holds, as ``(rows, idx, columns)`` for the CSA resource:
        the step's own ``kv`` and, for positions before the step, the request's ring in ``STATE``
        (a token at position p sits in ring slot ``p % slots``; the ring has ``take_back_tokens``
        slots more than the window, so tokens taken back overwrite none the next row attends).
        Then the ring takes the step's last rows.

        The columns are laid out as the checkpoint's reference lays them, since the softmax
        accumulates over a row's entries in blocks: a prompt's row has as many columns as the
        prompt has tokens, up to the window, from its oldest row on; a decode row has the whole
        window, oldest first, the positions before the request's first token leading."""
        win = self.window
        ring = self._state.gather(WINDOW, self.layer, label=LABEL)  # (requests, slots, D)
        requests, slots = ring.shape[:2]
        row = torch.arange(kv.shape[0], device=kv.device)
        col = torch.arange(win, device=kv.device)[None, :]
        start = rows.pos - (row - rows.first)
        decode = start > 0
        if not rows.decode:  # a prompt's later step: its one-token steps alone are decode rows
            decode = decode & (rows.req_rows[rows.row_req] == 1)
        columns = torch.where(decode, win, rows.end.clamp_max(win))
        pos = torch.where(decode, rows.pos - win + 1, (rows.pos - win + 1).clamp_min(0))[:, None] + col
        held = (pos >= 0) & (pos <= rows.pos[:, None]) & (col < columns[:, None])
        in_step = requests * slots + rows.first[:, None] + pos - start[:, None]
        in_ring = rows.row_req[:, None] * slots + pos % slots
        idx = torch.where(held, torch.where(pos >= start[:, None], in_step, in_ring), -1)
        source = torch.cat([ring.flatten(0, 1), kv], dim=0)
        tail = rows.tail(slots)
        ring[rows.row_req[tail], rows.pos[tail] % slots] = kv[tail].to(ring.dtype)
        self._state.scatter_(WINDOW, ring, self.layer, label=LABEL)
        return source, idx, columns

    def _select(self, x, q_latent, latent, rows, rope: Rope, angles, shared: SharedSelection):
        """This layer's pick of the compressed cache: its own indexer's, or the last one's."""
        if self.indexer is None:
            return shared.slots
        ratio = self.ratio
        if self.indexer.owns_keys:
            # a latent stands at its group's first token
            keys = self.indexer.keys(latent, rope.angles(rows.pos + 1 - ratio, True))
            self._csa.write(ratio, self.plane, keys=keys, label=LABEL)
        q, w = self.indexer.query(x, q_latent, angles)
        scores = self._csa.scores(ratio, self.plane, q, w, label=LABEL)
        if self.comm_group.world_size > 1:
            # the ranks' sum, rounded as one rank's scores are
            scores = self.comm_group.all_reduce(scores).to(x.dtype).float()
        keep = None
        if self.is_candidate_source:
            shared.candidates = self._csa.candidates(
                ratio, scores, self.candidate_topk_blocks, self.candidate_block_size, label=LABEL)
        elif self.uses_candidates:
            keep = shared.candidates
        shared.slots = self._csa.select(ratio, scores, keep, label=LABEL)
        return shared.slots

    def forward(self, x: torch.Tensor, rope: Rope, shared: SharedSelection) -> torch.Tensor:
        rows = self._csa.current(LABEL)
        compressed = self.ratio > 0
        angles = rope.angles(rows.pos, compressed)

        q_latent, q = self._query(x, angles)
        window, window_idx, window_cols = self._window(self._latent(x, angles), rows)

        slots = None
        if compressed:
            latent = self.compressor(x, rows) if self.compressor is not None else None
            slots = self._select(x, q_latent, latent, rows, rope, angles, shared)
            if latent is not None:
                latent = rotate_tail(latent, rope.angles(rows.pos + 1 - self.ratio, True))
                self._csa.write(self.ratio, self.plane, latent=latent, label=LABEL)
        out = self._csa.attend(q, self.attn_sink, window, window_idx, window_cols, self.ratio, self.plane, slots)
        return self._output(out, angles)
