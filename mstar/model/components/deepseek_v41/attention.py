"""DeepSeek-V4.1 attention, compressed sparse attention (CSA2) indexer, and SWA + sink softmax."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from mstar.distributed.communication import CommGroup
from mstar.distributed.utils import divide
from mstar.model.components.deepseek_v41.compressor import DeepSeekV41Compressor
from mstar.model.components.deepseek_v41.config import DeepSeekV41Config
from mstar.model.components.deepseek_v41.quant import (
    COLUMN,
    ROW,
    ReferenceRMSNorm,
    RoundedLinear,
    round_fp4,
    round_fp8,
)
from mstar.model.components.deepseek_v41.rope import Angles, DeepSeekV41Rope, rotate_tail


@dataclass
class SharedSelection:
    """Cross-layer CSA2 state passed down the layer stack within a single forward step."""

    slots: torch.Tensor | None = None
    candidates: torch.Tensor | None = None


def top_scores(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic top-k (earlier column wins on exact ties)."""
    order = scores.sort(dim=-1, descending=True, stable=True)
    return order.values[:, :k], order.indices[:, :k]


def select_candidate_blocks(
    scores: torch.Tensor,
    groups: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Level-1 candidate block selection keeping ``topk_blocks`` blocks plus the newest partial block."""
    width = scores.shape[-1]
    if width == 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    pad = -width % block_size
    padded = F.pad(scores, (0, pad), value=-torch.inf) if pad else scores
    block = padded.unflatten(-1, (-1, block_size)).amax(-1)
    last = (groups - 1) // block_size
    ids = torch.arange(block.shape[-1], device=scores.device)
    pin = (groups > 0)[:, None] & (ids[None, :] == last[:, None])
    block = block.masked_fill(pin, torch.inf)
    values, indices = top_scores(block, min(topk_blocks, block.shape[-1]))
    keep = torch.zeros_like(block, dtype=torch.bool).scatter_(-1, indices, values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class DeepSeekV41Indexer(nn.Module):
    """CSA2 lightning indexer with FP4-rounded keys and queries."""

    def __init__(
        self,
        config: DeepSeekV41Config,
        layer_id: int,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.comm_group = comm_group or CommGroup.trivial()
        self.layer_id = layer_id
        self.owns_keys = layer_id in config.kv_source_layer_ids
        self.n_heads = divide(config.index_n_heads, self.comm_group.world_size)
        self.head_dim = config.index_head_dim
        self.block = config.fp8_block
        self.wq_b = RoundedLinear(
            config.q_lora_rank,
            config.index_n_heads * config.index_head_dim,
            self.comm_group,
            COLUMN,
            block=config.fp8_block,
        )
        self.weights_proj = RoundedLinear(
            config.hidden_size,
            config.index_n_heads,
            self.comm_group,
            COLUMN,
            rounded=False,
            block=config.fp8_block,
        )
        if self.owns_keys:
            self.wk = RoundedLinear(
                config.head_dim,
                config.index_head_dim,
                rounded=False,
                block=config.fp8_block,
            )
            self.k_norm = ReferenceRMSNorm(config.index_head_dim, eps=config.rms_norm_eps)
        self.weight_scale = config.index_head_dim**-0.5 * config.index_n_heads**-0.5

    def keys(self, latent: torch.Tensor, angles: Angles) -> torch.Tensor:
        """Compute FP4-rounded rotated index keys from unrotated compressed latents."""
        k = rotate_tail(self.k_norm(self.wk(latent)), angles)
        return round_fp4(k, self.block, e4m3_scale=False)

    def query(self, x: torch.Tensor, q_latent: torch.Tensor, angles: Angles) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute FP4-rounded rotated index queries and per-head weights."""
        q = self.wq_b(q_latent).unflatten(-1, (self.n_heads, self.head_dim))
        q = round_fp4(rotate_tail(q, angles), self.block, e4m3_scale=False)
        w = self.weights_proj(x) * self.weight_scale
        return q, w


class DeepSeekV41Attention(nn.Module):
    """DeepSeek-V4.1 self-tied 512-dim latent attention with SWA, CSA2, sink, and inverse RoPE."""

    def __init__(
        self,
        config: DeepSeekV41Config,
        layer_id: int,
        comm_group: CommGroup | None = None,
    ) -> None:
        super().__init__()
        self.comm_group = comm_group or CommGroup.trivial()
        tp = self.comm_group.world_size
        self.layer_id = layer_id
        self.head_dim = config.head_dim
        self.num_heads = divide(config.num_attention_heads, tp)
        self.num_groups = divide(config.o_groups, tp)
        self.o_lora_rank = config.o_lora_rank
        self.window = config.sliding_window
        self.block = config.fp8_block
        self.sm_scale = config.softmax_scale
        self.ratio = config.compress_ratios[layer_id]
        self.kv_source_layer = config.kv_source_of(layer_id)
        self.index_topk = config.index_topk
        self.is_candidate_source = layer_id == config.candidate_source_layer_id
        self.uses_candidates = config.uses_candidates(layer_id)
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size

        heads, width = config.num_attention_heads, config.head_dim
        self.attn_sink = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.wq_a = RoundedLinear(config.hidden_size, config.q_lora_rank, block=config.fp8_block)
        self.q_norm = ReferenceRMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = RoundedLinear(
            config.q_lora_rank,
            heads * width,
            self.comm_group,
            COLUMN,
            block=config.fp8_block,
        )
        self.wkv = RoundedLinear(config.hidden_size, width, block=config.fp8_block)
        self.kv_norm = ReferenceRMSNorm(width, eps=config.rms_norm_eps)
        self.wo_a = RoundedLinear(
            heads * width // config.o_groups,
            config.o_groups * config.o_lora_rank,
            self.comm_group,
            COLUMN,
            rounded=False,
            block=config.fp8_block,
        )
        self.wo_b = RoundedLinear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            self.comm_group,
            ROW,
            block=config.fp8_block,
        )

        self.compressor = (
            DeepSeekV41Compressor(
                config.hidden_size,
                width,
                self.ratio,
                norm_eps=config.rms_norm_eps,
            )
            if layer_id in config.kv_source_layer_ids
            else None
        )
        self.indexer = (
            DeepSeekV41Indexer(config, layer_id, self.comm_group)
            if layer_id in config.index_source_layer_ids
            else None
        )

    def _apply(self, fn, recurse=True):
        sink_orig = self.attn_sink.detach()
        result = super()._apply(fn, recurse=recurse)
        self.attn_sink.data = (
            self.attn_sink.data.float()
            if sink_orig.is_meta
            else sink_orig.to(device=self.attn_sink.device, dtype=torch.float32)
        )
        return result

    def _query(self, x: torch.Tensor, angles: Angles) -> tuple[torch.Tensor, torch.Tensor]:
        q_latent = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_latent).unflatten(-1, (self.num_heads, self.head_dim))
        return q_latent, rotate_tail(q, angles)

    def _window_latent(self, x: torch.Tensor, angles: Angles) -> torch.Tensor:
        return round_fp8(rotate_tail(self.kv_norm(self.wkv(x)), angles), self.block)

    def _output(self, out: torch.Tensor, angles: Angles) -> torch.Tensor:
        out = rotate_tail(out, angles, inverse=True)
        dtype = out.dtype
        out = round_fp8(out.reshape(out.shape[0], self.num_groups, -1), self.block)
        wo_a = self.wo_a.weight.view(self.num_groups, self.o_lora_rank, -1)
        out = torch.einsum("sgd,grd->sgr", out.float(), wo_a.float()).to(dtype)
        return self.wo_b(out.flatten(1))

    def _build_window(
        self,
        kv: torch.Tensor,
        held_window: torch.Tensor,
        pos: torch.Tensor,
        row_req: torch.Tensor,
        req_first: torch.Tensor,
        req_end_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather SWA entries for each token and produce the updated ring window."""
        win = self.window
        num_reqs = held_window.shape[0]
        total_tokens = kv.shape[0]
        row = torch.arange(total_tokens, device=kv.device)
        col = torch.arange(win, device=kv.device)[None, :]
        req_start_pos = pos - (row - req_first)
        req_rows = req_end_pos[row_req] - req_start_pos
        is_decode = (req_start_pos > 0) & (req_rows == 1)
        columns = torch.where(is_decode, win, req_end_pos[row_req].clamp_max(win))
        k_pos = torch.where(is_decode, pos - win + 1, (pos - win + 1).clamp_min(0))[:, None] + col
        valid = (k_pos >= 0) & (k_pos <= pos[:, None]) & (col < columns[:, None])
        in_step = num_reqs * win + req_first[:, None] + (k_pos - req_start_pos[:, None])
        in_ring = row_req[:, None] * win + (k_pos % win)
        idx = torch.where(valid, torch.where(k_pos >= req_start_pos[:, None], in_step, in_ring), 0)
        source = torch.cat([held_window.flatten(0, 1).to(kv.dtype), kv], dim=0)
        entries = source[idx]
        next_window = held_window.clone()
        tail = pos >= (req_end_pos[row_req] - win)
        if tail.any():
            next_window[row_req[tail], pos[tail] % win] = kv[tail].to(next_window.dtype)
        return entries, valid, next_window

    def _score_and_select(
        self,
        x: torch.Tensor,
        q_latent: torch.Tensor,
        angles: Angles,
        pos: torch.Tensor,
        row_req: torch.Tensor,
        compress_k: torch.Tensor,
        shared: SharedSelection,
    ) -> torch.Tensor:
        """Run indexer scoring, optional level-1 candidate pruning, and top-k selection."""
        if self.indexer is None:
            assert shared.slots is not None
            return shared.slots
        seen = (pos + 1) // self.ratio
        max_seen = int(seen.max().item()) if seen.numel() else 0
        width = max(1, max_seen)
        q, w = self.indexer.query(x, q_latent, angles)
        group_ids = torch.arange(width, device=x.device)
        keys = compress_k[row_req[:, None], group_ids[None, :]].to(q.dtype)  # [T, width, index_head_dim]
        dots = torch.einsum("thd,tgd->thg", q.float(), keys.float()).relu_()
        scores = (dots * w.float().unsqueeze(-1)).sum(dim=1)
        if self.comm_group.world_size > 1:
            scores = self.comm_group.all_reduce(scores).to(x.dtype).float()
        scores = scores.masked_fill(group_ids[None, :] >= seen[:, None], -torch.inf)

        keep = None
        if self.is_candidate_source:
            shared.candidates = select_candidate_blocks(
                scores, seen, self.candidate_topk_blocks, self.candidate_block_size
            )
        elif self.uses_candidates:
            keep = shared.candidates
        if keep is not None:
            scores = scores.masked_fill(~keep[..., :width], -torch.inf)
        values, indices = top_scores(scores, min(self.index_topk, width))
        slots = torch.where(values > -torch.inf, indices, -1)
        shared.slots = slots
        return slots

    def forward_packed(
        self,
        x: torch.Tensor,
        rope: DeepSeekV41Rope,
        pos: torch.Tensor,
        row_req: torch.Tensor,
        req_first: torch.Tensor,
        req_end_pos: torch.Tensor,
        held_window: torch.Tensor,
        compress_kv: torch.Tensor | None = None,
        compress_k: torch.Tensor | None = None,
        partial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
        shared: SharedSelection | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        """Packed multi-request forward returning updated per-request window/compressed/partial state."""
        if shared is None:
            shared = SharedSelection()
        compressed = self.ratio > 0
        angles = rope.angles(pos, compressed)
        q_latent, q = self._query(x, angles)
        kv_win = self._window_latent(x, angles)
        win_entries, win_valid, next_window = self._build_window(
            kv_win, held_window, pos, row_req, req_first, req_end_pos
        )

        next_compress_kv = compress_kv
        next_compress_k = compress_k
        next_partial = partial_state
        comp_entries = None
        comp_valid = None

        if compressed:
            assert compress_kv is not None and compress_k is not None
            if self.compressor is not None:
                held_kv, held_score = partial_state if partial_state is not None else (None, None)
                latent, next_pkv, next_pscore = self.compressor.forward_packed(
                    x, pos, row_req, req_first, req_end_pos, held_kv, held_score
                )
                if next_pkv is not None and next_pscore is not None:
                    next_partial = (next_pkv, next_pscore)
                complete = (pos + 1) % self.ratio == 0
                if complete.any():
                    req_c = row_req[complete]
                    grp_c = pos[complete] // self.ratio
                    grp_angles = rope.angles(pos[complete] + 1 - self.ratio, True)
                    lat_c = latent[complete]
                    next_compress_kv = compress_kv.clone()
                    rot_lat = round_fp4(rotate_tail(lat_c, grp_angles), block=16, e4m3_scale=True)
                    next_compress_kv[req_c, grp_c] = rot_lat.to(next_compress_kv.dtype)
                    if self.indexer is not None and self.indexer.owns_keys:
                        next_compress_k = compress_k.clone()
                        next_compress_k[req_c, grp_c] = self.indexer.keys(lat_c, grp_angles).to(
                            next_compress_k.dtype
                        )

            slots = self._score_and_select(x, q_latent, angles, pos, row_req, next_compress_k, shared)
            comp_valid = slots >= 0
            comp_entries = next_compress_kv[row_req[:, None], slots.clamp_min(0)]

        if comp_entries is not None and comp_valid is not None:
            entries = torch.cat([win_entries, comp_entries.to(win_entries.dtype)], dim=1)
            valid = torch.cat([win_valid, comp_valid], dim=1)
        else:
            entries = win_entries
            valid = win_valid

        scores = torch.einsum("thd,tkd->thk", q.float(), entries.float()) * self.sm_scale
        scores = scores.masked_fill(~valid[:, None, :], -torch.inf)
        sink = self.attn_sink.float()[None, :, None].expand(q.shape[0], self.num_heads, 1)
        probs = torch.cat([scores, sink], dim=-1).softmax(dim=-1)[..., :-1]
        attended = torch.einsum("thk,tkd->thd", probs.to(q.dtype).float(), entries.float()).to(q.dtype)
        return self._output(attended, angles), next_window, next_compress_kv, next_compress_k, next_partial

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        rope: DeepSeekV41Rope,
        cache,
        shared: SharedSelection,
    ) -> torch.Tensor:
        """``[B, S, D]`` forward compatible with ``DeepSeekV41Layer``."""
        bsz, seq, _ = x.shape
        device = x.device
        pos = torch.arange(start_pos, start_pos + seq, dtype=torch.long, device=device).repeat(bsz)
        row_req = torch.arange(bsz, dtype=torch.long, device=device).repeat_interleave(seq)
        req_first = (torch.arange(bsz, dtype=torch.long, device=device) * seq).repeat_interleave(seq)
        req_end_pos = torch.full((bsz,), start_pos + seq, dtype=torch.long, device=device)

        src_layer = self.kv_source_layer
        has_partial = self.compressor is not None and self.ratio > 1
        held_window, ckv, ck, partial = cache.read_layer(None, src_layer, has_partial)

        out, next_win, next_ckv, next_ck, next_partial = self.forward_packed(
            x.reshape(-1, x.shape[-1]),
            rope,
            pos,
            row_req,
            req_first,
            req_end_pos,
            held_window,
            ckv,
            ck,
            partial,
            shared,
        )
        cache.write_layer(
            None,
            src_layer,
            self.compressor is not None,
            next_win,
            next_ckv,
            next_ck,
            next_partial,
        )
        return out.view(bsz, seq, -1)
