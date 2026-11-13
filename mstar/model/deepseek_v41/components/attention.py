"""DeepSeek-V4.1 attention: one latent per token that is key and value, attended over a
sliding window of the layer's own rows plus an indexer's pick of a compressed cache.

The engine holds the state and runs the attention: the compressed latents and their index keys
on the paged cache (``KV_CACHE``), each layer's window and the compressors' open groups in the
recurrent pool (``STATE``), and scoring, selection and the softmax in the CSA resource. The
layers here project, pool, rotate and round as the checkpoint's reference does.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.model.components import RMSNorm
from mstar.model.deepseek_v41.components.linear import COLUMN, ROW, RoundedLinear
from mstar.model.deepseek_v41.components.rope import Rope, rotate_tail
from mstar.model.deepseek_v41.config import (
    CSA,
    KV_CACHE,
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
    in the request's ``POOL_KV`` / ``POOL_SCORE`` rows of this layer."""

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

    def forward(self, x: torch.Tensor, rows) -> tuple[torch.Tensor, torch.Tensor]:
        """``x (T, hidden)`` -> the unrotated latent of the group each row completes ``(T,
        head_dim)``, zeros where it completes none, and that mask ``(T,)``."""
        ratio = self.ratio
        if ratio == 1:
            return self.norm(self.wkv(x)), torch.ones_like(rows.pos, dtype=torch.bool)
        kv, score = self.wkv(x.float()), self.wgate(x.float())
        held_kv = self._state.gather(POOL_KV, self.layer, label=LABEL)  # (requests, ratio, D)
        held_score = self._state.gather(POOL_SCORE, self.layer, label=LABEL)
        row = torch.arange(x.shape[0], device=x.device)
        in_span = row - rows.first
        members_kv, members_score = [], []
        for back in range(ratio - 1, -1, -1):  # the group's tokens, oldest first
            here = (in_span >= back)[:, None]
            src = (row - back).clamp_min(0)
            slot = (rows.pos - back) % ratio
            members_kv.append(torch.where(here, kv[src], held_kv[rows.row_req, slot]))
            members_score.append(torch.where(here, score[src], held_score[rows.row_req, slot]))
        members_kv, members_score = torch.stack(members_kv, 1), torch.stack(members_score, 1)
        pooled = (members_kv * members_score.softmax(dim=1)).sum(dim=1)
        done = (rows.pos + 1) % ratio == 0
        latent = self.norm(pooled.to(x.dtype)) * done[:, None]
        # the step's last tokens are the next step's open group
        tail = rows.pos >= rows.end - ratio
        req, slot = rows.row_req[tail], rows.pos[tail] % ratio
        held_kv[req, slot] = kv[tail]
        held_score[req, slot] = score[tail]
        self._state.scatter_(POOL_KV, held_kv, self.layer, label=LABEL)
        self._state.scatter_(POOL_SCORE, held_score, self.layer, label=LABEL)
        return latent, done


class Indexer(nn.Module):
    """One index source's projections: the key each compressed latent stores (on the layers
    that write a cache), and each query row's index heads and head weights. Keys and queries
    are rounded to FP4, as the checkpoint was trained."""

    def __init__(self, config: DeepseekV41Config, layer: int, comm_group: CommGroup) -> None:
        super().__init__()
        self.comm_group = comm_group
        self.owns_keys = layer in config.kv_source_layer_ids
        self.n_heads = divide(config.index_n_heads, comm_group.world_size)
        self.head_dim = config.index_head_dim
        self.block = config.fp8_block
        self.wq_b = RoundedLinear(
            config.q_lora_rank, config.index_n_heads * config.index_head_dim, comm_group, COLUMN)
        self.weights_proj = RoundedLinear(
            config.hidden_size, config.index_n_heads, comm_group, COLUMN, rounded=False)
        if self.owns_keys:
            self.wk = RoundedLinear(config.head_dim, config.index_head_dim, rounded=False)
            self.k_norm = RMSNorm(config.index_head_dim, eps=config.rms_norm_eps)
        # the score's softmax scale and its mean over heads, folded into the head weights
        self.weight_scale = config.index_head_dim ** -0.5 * config.index_n_heads ** -0.5

    def keys(self, latent: torch.Tensor, angles, done: torch.Tensor) -> torch.Tensor:
        k = rotate_tail(self.k_norm(self.wk(latent)), angles)
        return round_fp4(k, self.block) * done[:, None]

    def query(self, x: torch.Tensor, q_latent: torch.Tensor, angles) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.wq_b(q_latent).unflatten(-1, (self.n_heads, self.head_dim))
        q = round_fp4(rotate_tail(q, angles), self.block)
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
        self._csa = self._kv = self._state = None

    def bind_resources(self, resources: dict) -> None:
        self._csa, self._kv, self._state = resources[CSA], resources[KV_CACHE], resources[STATE]

    def _window(self, kv: torch.Tensor, rows) -> tuple[torch.Tensor, torch.Tensor]:
        """The rows each query's window holds, as ``(rows, idx)`` for the CSA resource: the
        step's own ``kv`` and, for positions before the step, the request's ring in ``STATE``
        (a token at position p sits in ring slot ``p % window``). Then the ring takes the
        step's last ``window`` rows."""
        win = self.window
        ring = self._state.gather(WINDOW, self.layer, label=LABEL)  # (requests, window, D)
        requests = ring.shape[0]
        row = torch.arange(kv.shape[0], device=kv.device)
        back = torch.arange(win, device=kv.device)
        pos = rows.pos[:, None] - back[None, :]
        in_step = (row - rows.first)[:, None] >= back[None, :]
        from_ring = rows.row_req[:, None] * win + pos % win
        idx = torch.where(in_step, requests * win + row[:, None] - back[None, :], from_ring)
        idx = torch.where(pos >= 0, idx, -1)
        source = torch.cat([ring.flatten(0, 1), kv], dim=0)
        tail = rows.pos >= rows.end - win
        ring[rows.row_req[tail], rows.pos[tail] % win] = kv[tail].to(ring.dtype)
        self._state.scatter_(WINDOW, ring, self.layer, label=LABEL)
        return source, idx

    def _select(self, x, q_latent, latent, done, rows, rope: Rope, angles, shared: SharedSelection):
        """This layer's pick of the compressed cache: its own indexer's, or the last one's."""
        if self.indexer is None:
            return shared.slots
        ratio = self.ratio
        if self.indexer.owns_keys:
            # a latent stands at its group's first token
            self._csa.write_index(
                self.plane, self.indexer.keys(latent, rope.angles(rows.pos + 1 - ratio, True), done),
                label=LABEL)
        q, w = self.indexer.query(x, q_latent, angles)
        scores = self._csa.scores(self.plane, q, w, ratio, label=LABEL)
        if self.comm_group.world_size > 1:
            scores = self.comm_group.all_reduce(scores)
        keep = None
        if self.is_candidate_source:
            shared.candidates = self._csa.candidates(
                scores, ratio, self.candidate_topk_blocks, self.candidate_block_size, label=LABEL)
        elif self.uses_candidates:
            keep = shared.candidates
        shared.slots = self._csa.select(scores, ratio, keep, label=LABEL)
        return shared.slots

    def forward(self, x: torch.Tensor, rope: Rope, shared: SharedSelection) -> torch.Tensor:
        rows = self._csa.current(LABEL)
        compressed = self.ratio > 0
        angles = rope.angles(rows.pos, compressed)

        q_latent = self.q_norm(self.wq_a(x))
        q = rotate_tail(self.wq_b(q_latent).unflatten(-1, (self.num_heads, self.head_dim)), angles)
        kv = round_fp8(rotate_tail(self.kv_norm(self.wkv(x)), angles), self.block)
        window, window_idx = self._window(kv, rows)

        slots = None
        if compressed:
            latent = done = None
            if self.compressor is not None:
                latent, done = self.compressor(x, rows)
            slots = self._select(x, q_latent, latent, done, rows, rope, angles, shared)
            if latent is not None:
                latent = rotate_tail(latent, rope.angles(rows.pos + 1 - self.ratio, True))
                # the compressed cache's FP4 has an E4M3 scale per 16 values
                latent = round_fp4(latent, 16, e4m3_scale=True) * done[:, None]
                self._kv.write_kv(latent, None, layer_idx=self.plane, label=LABEL)
        out = self._csa.attend(q, self.attn_sink, window, window_idx, self.plane, slots)
        # the cache holds rotated latents, so the output carries the query's rotation
        out = rotate_tail(out, angles, inverse=True)

        out = round_fp8(out.reshape(x.shape[0], self.num_groups, -1), self.block)
        wo_a = self.wo_a.weight.view(self.num_groups, self.o_lora_rank, -1)
        out = torch.einsum("sgd,grd->sgr", out.float(), wo_a.float()).to(x.dtype)
        return self.wo_b(out.flatten(1))
