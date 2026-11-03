"""The KV pool's resident tables: each request's block table row and stored length."""
from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from mstar.engine.resources.resident import (
    CapturedResidentStep,
    ResidentPlan,
    ResidentPlanner,
    ResidentWrites,
    advance_column,
    gather_column,
)


@triton.jit
def _gather_write_slots_kernel(
    slots_ptr, kv_len_ptr, store_ptr, page_ptr, offset_ptr,
    STRIDE: tl.constexpr, BLOCK: tl.constexpr, PAGE: tl.constexpr,
):
    i = tl.program_id(0)
    n = tl.load(kv_len_ptr + i)
    tl.store(page_ptr + i, tl.load(store_ptr + tl.load(slots_ptr + i) * STRIDE + BLOCK + n // PAGE))
    tl.store(offset_ptr + i, n % PAGE)


class KVResidentTables:
    """``block_table[slot, col]`` is a stream's col-th page, ``kv_len[slot]`` its
    stored length: what a step appending one token per row reads to find where
    it writes and what it attends to.

    A step that only reads the stream (declared at span 0, as a context is)
    leaves it where it is: the row holds the stream's pages and length, and
    has no write addressing, no advance and no crossings.

    The stream stays the truth. ``host_len`` advances by the rule the graph
    does, so nothing comes back from the device, and ``host_cols`` counts the
    columns a row holds, which is how the host sees a crossing.
    """

    def __init__(self, planner: ResidentPlanner, key: str, page_size: int, max_pages: int):
        self.planner = planner
        self.key = key
        self.page_size = page_size
        self.block_table = planner.table(max_pages)
        self.kv_len = planner.length_table(key)
        self.host_len = np.zeros(planner.max_slots, dtype=np.int64)
        self.host_cols = np.zeros(planner.max_slots, dtype=np.int64)

    def row(self, stored_len: int, pages: list[int], slot: int, row: list[int], writes: ResidentWrites) -> None:
        """The cells of a stream about to append one token, or to be read. Its
        host mirrors move here, not at the flush: a plan that never runs leaves
        the row stale."""
        if len(pages) > self.block_table.width:
            raise ValueError(f"{len(pages)} pages of a stream don't fit a block table row of {self.block_table.width}")
        page = self.page_size
        # a stream that appends holds the page its next token goes to, too
        cols = -(-stored_len // page) if writes.streams[self.key].reads else stored_len // page + 1
        row[self.kv_len.col] = stored_len
        writes.kv_len[self.key] = stored_len
        if cols:
            writes.run(slot, self.block_table.col, pages[:cols])
        self.host_len[slot] = stored_len
        self.host_cols[slot] = cols

    def refresh_len(self, stored_len: int, slot: int, writes: ResidentWrites) -> None:
        """A row a host-planned step advanced; its columns stay until `stage_step`."""
        writes.put(self.kv_len.cell(slot), stored_len)
        self.host_len[slot] = stored_len

    def stage_step(self, plan: ResidentPlan, stream_len, stream_pages, rewrite) -> None:
        """Bring each row's columns up to this step. The usual case, one new
        column, rides the graph's scatter, ahead of its gather: only columns
        past the row's length change, so a step still replaying reads none of
        them. A row further behind (a host-planned step took its pages) is
        rewritten whole.

        A stream the step only reads moves only under a host-planned step, so
        just the rows one moved are compared, and one that changed is
        rewritten whole."""
        if plan.writes.streams[self.key].reads:
            slot_of = self.planner.slot_of
            for rid in plan.moved:
                slot = slot_of(rid)
                if self.host_len[slot] != stream_len(rid):
                    plan.taken.add(rid)
                    rewrite(rid, slot, plan.writes)
            return
        slots = plan.slots
        cols = self.host_cols[slots]
        need = self.host_len[slots] // self.page_size + 1
        rows = np.flatnonzero(need > cols)
        if not rows.size:
            return
        for i in rows.tolist():
            rid, slot, last = plan.rids[i], int(slots[i]), int(need[i]) - 1
            if last >= self.block_table.width:
                raise ValueError(f"page {last} of a stream doesn't fit a block table row of {self.block_table.width}")
            if last > cols[i]:
                plan.taken.add(rid)
                rewrite(rid, slot, plan.writes)
            else:
                plan.writes.put(self.block_table.cell(slot, last), stream_pages(rid)[last])
        plan.crossed[self.key] = (rows, need[rows])

    def advance(self, plan: ResidentPlan) -> None:
        if plan.writes.streams[self.key].reads:
            return
        self.host_len[plan.slots] += 1
        crossed = plan.crossed.get(self.key)
        if crossed is not None:
            rows, need = crossed
            self.host_cols[plan.slots[rows]] = need

    def gather(
        self, step: CapturedResidentStep, token_to_page: torch.Tensor | None, token_to_cache: torch.Tensor | None,
    ) -> None:
        """The rows' stored lengths, and where each writes unless the step only
        reads. The planner has scattered the step's single cells already."""
        store, table = self.planner.store, self.block_table
        stride, page_size = table.stride, self.page_size
        step.block_tables[self.key] = table
        kv_len = step.buffers.kv_len[self.key]
        gather_column(store, step.slots, stride, self.kv_len.col, kv_len)
        if step.streams[self.key].reads:
            return
        if store.is_cuda:
            _gather_write_slots_kernel[(step.bucket.bs,)](
                step.slots, kv_len, store, token_to_page, token_to_cache,
                STRIDE=stride, BLOCK=table.col, PAGE=page_size, num_warps=1,
            )
            return
        col = torch.div(kv_len, page_size, rounding_mode="floor")
        bs = step.bucket.bs
        torch.index_select(store, 0, step.slots * stride + col + table.col, out=token_to_page[:bs])
        token_to_cache[:bs].copy_(kv_len - col * page_size)

    def commit(self, step: CapturedResidentStep) -> None:
        if not step.streams[self.key].reads:
            advance_column(self.planner.store, step.slots, self.kv_len.stride, self.kv_len.col)
