"""Sparse MLA: absorbed attention of each query row over its own list of cache slots.

A DSA indexer picks, per query row, which tokens of the paged MLA cache the row attends.
FlashInfer's MLA kernel runs over them as one-token pages, the slot list being the page table:
under a CUDA graph a ``SparseGraphPlan`` re-planned outside the graph every step, otherwise an
``EagerSparsePlan`` built once per forward. ``reference`` is the same math in torch.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch

from mstar.engine.resources.attn.workspace import Workspace, WorkspacePool

# FlashInfer's kernel over one-token pages faults past ~16k rows, so an eager plan attends in
# chunks of this many rows.
MAX_ROWS = 8192
# A prefill's selection shards across TP ranks from this many rows per rank.
SHARD_MIN_ROWS = 64


def uses_kernel(q_nope: torch.Tensor, latent: torch.Tensor) -> bool:
    from mstar.engine.resources.attn.flashinfer_mla import flashinfer_mla_supports

    # the kernel reads out of bounds outside its latent shape (a reduced test model's)
    rank = q_nope.shape[-1]
    return (q_nope.is_cuda and latent.dtype == torch.bfloat16
            and flashinfer_mla_supports(rank, latent.shape[-1] - rank))


def _plan(workspace: Workspace, wrapper, indices: torch.Tensor, lens: list[int], heads: int,
          kv_lora_rank: int, kpe: int, sm_scale: float) -> None:
    # packed: row r's slots at indices[sum(lens[:r]):], which FlashInfer >= 0.7 checks each
    # row's page count against; _pack lays them out to match, lens already held to the width
    qo = torch.arange(len(lens) + 1, dtype=torch.int32)
    kv = torch.zeros(len(lens) + 1, dtype=torch.int32)
    kv[1:] = torch.tensor(lens, dtype=torch.int32).cumsum(0)
    # FlashInfer stages the plan through a host copy that ATen splits over every intra-op
    # thread; with a worker per GPU, each with an OpenMP pool as wide as the host, that copy
    # waits on oversubscribed cores. Under OpenMP the count is this thread's alone.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        workspace.plan(wrapper.plan, qo, kv, indices, torch.tensor(lens, dtype=torch.int32),
                       heads, kv_lora_rank, kpe, 1, False, sm_scale, torch.bfloat16,
                       torch.bfloat16)
    finally:
        torch.set_num_threads(threads)


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

    def __init__(self, rows: int, width: int, workspace: Workspace,
                 indices: torch.Tensor | None = None):
        import flashinfer

        device = workspace.tensor.device
        self._workspace = workspace
        self.rows, self.width = rows, width
        size = rows * width + 1
        assert indices is None or indices.numel() >= size, (indices.numel(), size)
        self.indices = (torch.zeros(size, dtype=torch.int32, device=device)
                        if indices is None else indices[:size])
        self._qo = torch.zeros(rows + 1, dtype=torch.int32, device=device)
        self._kv = torch.zeros(rows + 1, dtype=torch.int32, device=device)
        self._lens = torch.zeros(rows, dtype=torch.int32, device=device)
        self.wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
            workspace.tensor, use_cuda_graph=True, qo_indptr=self._qo, kv_indptr=self._kv,
            kv_indices=self.indices, kv_len_arr=self._lens, backend="auto")
        self._planned: torch.cuda.Event | None = None

    def plan(self, lens: list[int], heads: int, kv_lora_rank: int, kpe: int,
             sm_scale: float) -> None:
        assert len(lens) == self.rows
        if self._planned is not None:
            # the last plan's copy out of the wrapper's pinned buffer may still be queued
            self._planned.synchronize()
        _plan(self._workspace, self.wrapper, self.indices, _lens(lens, self.width), heads,
              kv_lora_rank, kpe, sm_scale)
        self._planned = torch.cuda.Event()
        self._planned.record()

    def attend(self, q_nope, q_pe, latent, slots, valid: torch.Tensor | None = None,
               pad_slot: int | None = None) -> torch.Tensor:
        """``valid`` (device, per row): a plan made for longer rows than the step has; each
        row's slots past its own length go to ``pad_slot``, a row the caller masks."""
        # the plan's offsets and lengths are in the wrapper's device buffers by now
        if valid is not None:
            j = torch.arange(slots.shape[1], dtype=valid.dtype, device=slots.device)
            slots = torch.where(j < valid[:, None], slots, pad_slot)
        _pack(self.indices, slots, self._kv, self._lens)
        return self.wrapper.run(*_split(q_nope, q_pe, latent))


class SparseChunk(NamedTuple):
    """Rows ``[r0, r1)`` of an eager plan: their wrapper, its index buffer, and the plan's
    offsets and lengths on the device, which ``_pack`` lays the slots out by."""

    r0: int
    r1: int
    wrapper: object
    indices: torch.Tensor
    indptr: torch.Tensor
    lens: torch.Tensor


class EagerSparsePlan:
    """An eager forward's plan (``EagerSparsePlans.plan``): a wrapper per ``MAX_ROWS`` rows,
    which run in turn. Every layer attends with the same lengths, so it is planned once."""

    def __init__(self, chunks: list[SparseChunk]):
        self.chunks = chunks

    def attend(self, q_nope, q_pe, latent, slots) -> torch.Tensor:
        q_nope, q_pe, ckv, kpe = _split(q_nope, q_pe, latent)
        out = q_nope.new_empty(q_nope.shape)
        for chunk in self.chunks:
            rows = slice(chunk.r0, chunk.r1)
            _pack(chunk.indices, slots[rows], chunk.indptr, chunk.lens)
            out[rows] = chunk.wrapper.run(q_nope[rows], q_pe[rows], ckv, kpe)
        return out


class EagerSparsePlans:
    """Eager plans a resource reuses across forwards and a long prefill's windows: one
    workspace, and two sets of chunk wrappers taken in turn. FlashInfer copies a plan to the
    device out of a pinned buffer its wrapper keeps, so re-planning a set waits for that set's
    last copy (an event recorded after it), not for the stream; after the first plans nothing
    is allocated or freed under a copy in flight."""

    def __init__(self, workspaces: WorkspacePool):
        self._workspaces = workspaces
        # per set: (wrapper, index buffer) per chunk of MAX_ROWS rows
        self._sets: tuple[list, list] = ([], [])
        self._planned: list[torch.cuda.Event | None] = [None, None]
        self._turn = 0

    def plan(self, lens: list[int], width: int, heads: int, kv_lora_rank: int, kpe: int,
             sm_scale: float) -> EagerSparsePlan:
        import flashinfer

        turn, self._turn = self._turn, 1 - self._turn
        if self._planned[turn] is not None:
            self._planned[turn].synchronize()
        workspace = self._workspaces.get("eager")  # allocated at the first eager plan
        device = workspace.tensor.device
        wrappers = self._sets[turn]
        lens = _lens(lens, width)
        chunks = []
        for i, r0 in enumerate(range(0, len(lens), MAX_ROWS)):
            r1 = min(r0 + MAX_ROWS, len(lens))
            need = (r1 - r0) * width + 1  # + the spare entry _pack writes the unused slots to
            if i == len(wrappers):
                wrappers.append((
                    flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace.tensor, backend="auto"),
                    torch.empty(MAX_ROWS * width + 1, dtype=torch.int32, device=device)))
            wrapper, indices = wrappers[i]
            if indices.numel() < need:  # a wider plan than the buffer's first
                indices = torch.empty(MAX_ROWS * width + 1, dtype=torch.int32, device=device)
                wrappers[i] = (wrapper, indices)
            chunk = torch.tensor(lens[r0:r1], dtype=torch.int32)
            indptr = torch.cat([chunk.new_zeros(1), chunk.cumsum(0, dtype=torch.int32)])
            _plan(workspace, wrapper, indices[:need], lens[r0:r1], heads, kv_lora_rank, kpe,
                  sm_scale)
            chunks.append(SparseChunk(r0, r1, wrapper, indices[:need],
                                      indptr.to(device, non_blocking=True),
                                      chunk.to(device, non_blocking=True)))
        self._planned[turn] = torch.cuda.Event()
        self._planned[turn].record()
        return EagerSparsePlan(chunks)


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


class RequestPages(NamedTuple):
    """One request's cache pages, in position order, on this rank. A selection is this rank's
    cache slots and the ranks' page tables need not agree, so it crosses ranks as the
    request's token positions."""

    pages: torch.Tensor
    page_size: int

    def to_positions(self, slots: torch.Tensor) -> torch.Tensor:
        """The token positions of ``slots`` (-1: none), all in this request's pages."""
        sorted_pages, order = self.pages.sort()
        slot = slots.clamp_min(0).long()
        page = (slot // self.page_size).to(sorted_pages.dtype)
        at = torch.searchsorted(sorted_pages, page).clamp_max(order.numel() - 1)
        pos = order[at] * self.page_size + slot % self.page_size
        return torch.where(slots >= 0, pos.to(slots.dtype), -1)

    def to_slots(self, positions: torch.Tensor) -> torch.Tensor:
        """This rank's cache slots of ``positions`` (-1: none)."""
        pos = positions.clamp_min(0).long()
        index = (pos // self.page_size).clamp_max(self.pages.numel() - 1)
        slot = self.pages[index].long() * self.page_size + pos % self.page_size
        return torch.where(positions >= 0, slot.to(positions.dtype), -1)


def select_rows(select: Callable[[int, int], torch.Tensor], r0: int, r1: int, width: int,
                device: torch.device, group=None,
                request: RequestPages | None = None) -> torch.Tensor:
    """``select(a, b)`` (``[b - a, width]`` int32 cache slots) over rows ``[r0, r1)`` of one
    request's prefill. With a TP ``group`` each rank selects a contiguous block and the blocks
    are all-gathered as token positions in ``request`` (``RequestPages``)."""
    n = r1 - r0
    world = group.world_size if group is not None else 1
    if world == 1 or n < world * SHARD_MIN_ROWS:
        return select(r0, r1)
    if request is None:
        raise ValueError("a selection sharded across ranks needs the request's pages")
    per = -(-n // world)
    a = min(r0 + group.rank * per, r1)
    b = min(a + per, r1)
    block = torch.full((per, width), -1, dtype=torch.int32, device=device)
    if b > a:
        block[:b - a] = request.to_positions(select(a, b))
    return request.to_slots(group.all_gather(block, dim=0)[:n])
