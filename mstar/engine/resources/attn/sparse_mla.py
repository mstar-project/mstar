"""Sparse MLA: absorbed attention of each query row over its own list of cache slots.

A DSA indexer picks, per query row, which tokens of the paged MLA cache the row attends.
FlashInfer's MLA kernel runs over them as one-token pages, the slot list being the page table:
under a CUDA graph a ``SparseGraphPlan`` re-planned outside the graph every step, otherwise an
``EagerSparsePlan`` built once per forward. ``reference`` is the same math in torch.
"""
from __future__ import annotations

from collections.abc import Callable

import torch

# FlashInfer's kernel over one-token pages faults past ~16k rows, so an eager plan attends in
# chunks of this many rows.
MAX_ROWS = 8192
WORKSPACE_BYTES = 128 << 20
# A prefill's selection shards across TP ranks from this many rows per rank.
SHARD_MIN_ROWS = 64


def uses_kernel(q_nope: torch.Tensor, latent: torch.Tensor) -> bool:
    return q_nope.is_cuda and latent.dtype == torch.bfloat16


def _plan(wrapper, indices: torch.Tensor, lens: list[int], heads: int,
          kv_lora_rank: int, kpe: int, sm_scale: float) -> None:
    # packed: row r's slots at indices[sum(lens[:r]):], what FlashInfer >= 0.7 checks the
    # page counts against (one-token pages); _pack lays the slots out to match. lens are
    # already held to the row width (_lens)
    qo = torch.arange(len(lens) + 1, dtype=torch.int32)
    kv = torch.zeros(len(lens) + 1, dtype=torch.int32)
    kv[1:] = torch.tensor(lens, dtype=torch.int32).cumsum(0)
    wrapper.plan(qo, kv, indices, torch.tensor(lens, dtype=torch.int32), heads,
                 kv_lora_rank, kpe, 1, False, sm_scale, torch.bfloat16, torch.bfloat16)


def _lens(lens: list[int], width: int) -> list[int]:
    """A row attends at most its ``width`` slots, as ``reference`` does: packed, a longer
    length would read entries no slot was written to."""
    return [min(n, width) for n in lens]


def _pack(dst: torch.Tensor, slots: torch.Tensor, indptr: torch.Tensor,
          lens: torch.Tensor) -> None:
    """``slots [rows, width]`` into ``dst`` as the plan's CSR: row r's first ``lens[r]`` at
    ``dst[indptr[r]:]``, the rest to ``dst``'s spare last entry. Fixed shapes, so it captures."""
    rows, width = slots.shape
    j = torch.arange(width, dtype=torch.int32, device=slots.device)
    pos = torch.where(j < lens[:rows, None], indptr[:rows, None] + j, dst.numel() - 1)
    dst.scatter_(0, pos.reshape(-1).long(), slots.reshape(-1).to(dst.dtype))


def _split(q_nope: torch.Tensor, q_pe: torch.Tensor, latent: torch.Tensor):
    rank = q_nope.shape[-1]
    flat = latent.view(-1, 1, latent.shape[-1])
    return (q_nope.to(torch.bfloat16).contiguous(), q_pe.to(torch.bfloat16).contiguous(),
            flat[..., :rank], flat[..., rank:])


class SparseGraphPlan:
    """A CUDA-graph wrapper for ``rows`` query rows of up to ``width`` slots: owned length
    buffers and a view of the capture slot's index buffer (``rows * width + 1`` entries, the
    last a spare for ``_pack``), planned outside the graph every step and read by the replay."""

    def __init__(self, rows: int, width: int, workspace: torch.Tensor,
                 indices: torch.Tensor | None = None):
        import flashinfer

        device = workspace.device
        self.rows, self.width = rows, width
        size = rows * width + 1
        assert indices is None or indices.numel() >= size, (indices.numel(), size)
        self.indices = (torch.zeros(size, dtype=torch.int32, device=device)
                        if indices is None else indices[:size])
        self._qo = torch.zeros(rows + 1, dtype=torch.int32, device=device)
        self._kv = torch.zeros(rows + 1, dtype=torch.int32, device=device)
        self._lens = torch.zeros(rows, dtype=torch.int32, device=device)
        self.wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            workspace, use_cuda_graph=True, qo_indptr=self._qo, kv_indptr=self._kv,
            kv_indices=self.indices, kv_len_arr=self._lens, backend="auto")
        self._planned: torch.cuda.Event | None = None

    def plan(self, lens: list[int], heads: int, kv_lora_rank: int, kpe: int,
             sm_scale: float) -> None:
        assert len(lens) == self.rows
        if self._planned is not None:
            # the last plan's copy out of the wrapper's pinned buffer may still be queued
            self._planned.synchronize()
        _plan(self.wrapper, self.indices, _lens(lens, self.width), heads, kv_lora_rank, kpe,
              sm_scale)
        self._planned = torch.cuda.Event()
        self._planned.record()

    def attend(self, q_nope, q_pe, latent, slots) -> torch.Tensor:
        # the plan's offsets and lengths are in the wrapper's device buffers by now
        _pack(self.indices, slots, self._kv, self._lens)
        return self.wrapper.run(*_split(q_nope, q_pe, latent))


class EagerSparsePlan:
    """An eager forward's plan: a wrapper per ``MAX_ROWS`` rows, sharing one workspace since
    the chunks run in turn. Every layer attends with the same lengths, so it is built once."""

    def __init__(self, lens: list[int], width: int, heads: int, kv_lora_rank: int, kpe: int,
                 sm_scale: float, device: torch.device):
        import flashinfer

        # plan() copies a pinned host buffer to the device with its own async memcpy, which
        # torch does not track: a buffer freed with an earlier plan can come back here and be
        # rewritten before that copy has run
        torch.cuda.current_stream(device).synchronize()
        workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
        lens = _lens(lens, width)
        self.chunks = []
        for r0 in range(0, len(lens), MAX_ROWS):
            r1 = min(r0 + MAX_ROWS, len(lens))
            wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="auto")
            indices = torch.empty((r1 - r0) * width + 1, dtype=torch.int32, device=device)
            chunk = torch.tensor(lens[r0:r1], dtype=torch.int32)
            indptr = torch.cat([chunk.new_zeros(1), chunk.cumsum(0, dtype=torch.int32)])
            _plan(wrapper, indices, lens[r0:r1], heads, kv_lora_rank, kpe, sm_scale)
            self.chunks.append((r0, r1, wrapper, indices,
                                indptr.to(device, non_blocking=True), chunk.to(device, non_blocking=True)))

    def attend(self, q_nope, q_pe, latent, slots) -> torch.Tensor:
        q_nope, q_pe, ckv, kpe = _split(q_nope, q_pe, latent)
        out = q_nope.new_empty(q_nope.shape)
        for r0, r1, wrapper, indices, indptr, lens in self.chunks:
            _pack(indices, slots[r0:r1], indptr, lens)
            out[r0:r1] = wrapper.run(q_nope[r0:r1], q_pe[r0:r1], ckv, kpe)
        return out


def reference(q_nope: torch.Tensor, q_pe: torch.Tensor, latent: torch.Tensor,
              slots: torch.Tensor, lens: list[int], sm_scale: float) -> torch.Tensor:
    """``[rows, heads, kv_lora_rank]``: row r attends ``slots[r, :lens[r]]``, slots of
    ``latent`` (``[pages, page_size, kv_lora_rank + kpe]``) as ``page * page_size + offset``."""
    rank = q_nope.shape[-1]
    flat = latent.view(-1, latent.shape[-1])
    query = torch.cat([q_nope, q_pe], dim=-1).float()
    out = torch.empty_like(q_nope)
    with torch.autocast(q_nope.device.type, enabled=False):  # fp32 under the engine's autocast too
        for r, n in enumerate(lens):
            picked = flat[slots[r, :n].long()].float()
            attn = (torch.einsum("hd,kd->hk", query[r], picked[:, :query.shape[-1]])
                    * sm_scale).softmax(-1)
            out[r] = torch.einsum("hk,kd->hd", attn, picked[:, :rank]).to(out.dtype)
    return out


def select_rows(select: Callable[[int, int], torch.Tensor], r0: int, r1: int, width: int,
                device: torch.device, group=None) -> torch.Tensor:
    """``select(a, b)`` (``[b - a, width]`` int32) over rows ``[r0, r1)`` of one request's
    prefill. With a TP ``group`` each rank selects a contiguous block and the blocks are
    all-gathered: every rank would compute the same selection."""
    n = r1 - r0
    world = group.world_size if group is not None else 1
    if world == 1 or n < world * SHARD_MIN_ROWS:
        return select(r0, r1)
    per = -(-n // world)
    a = min(r0 + group.rank * per, r1)
    b = min(a + per, r1)
    block = torch.full((per, width), -1, dtype=torch.int32, device=device)
    if b > a:
        block[:b - a] = select(a, b)
    return group.all_gather(block, dim=0)[:n]
