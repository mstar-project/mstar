"""Unit tests for the FlashAttention paged backend's index construction.

The backend's whole job on the host is turning the KV plan's `SequenceView`s
into the three tensors the kernel reads — `page_table`, `cache_seqlens`,
`cu_seqlens_q`. That is pure CPU work over a plan object, so it is checked
here without a kernel; the kernel itself is covered on the GPU.

The invariant worth the most is buffer-address stability: a captured graph
bakes these addresses into its replay, so a plan must write *into* them and
never reallocate.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources import (
    AttentionStep,
    BucketKey,
    PagedKVConfig,
    SlotLease,
    StepContext,
)
from mstar.engine.resources.attn.flash_attn import FlashAttnManager, check_fa3_head_dim
from mstar.engine.resources.kv.plan import SINK_PAGE, KVPlanOutput, SequenceView

PAGE_SIZE = 16
NUM_KV_HEADS = 2
HEAD_DIM = 64
MAX_PAGES = 64
MAX_SEQ_LEN = 128  # -> 8 blocks per request


def _kv_config() -> PagedKVConfig:
    return PagedKVConfig(
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_seq_len=MAX_SEQ_LEN,
        max_num_pages=MAX_PAGES,
        page_size=PAGE_SIZE,
    )


def _manager() -> FlashAttnManager:
    return FlashAttnManager(
        kv_cache="kv",
        device=torch.device("cpu"),
        dtype=torch.float32,
        kv_config=_kv_config(),
    )


def _view(rid: str, pages: list[int], length: int, to_compute: int):
    return SequenceView(
        request_id=rid, label="main", page_idxs=pages,
        length=length, to_compute=to_compute,
    )


def _ctx(views: list[SequenceView], **kwargs) -> StepContext:
    return StepContext(
        request_ids=tuple(dict.fromkeys(v.request_id for v in views)),
        graph_walk="decode",
        slot=0,
        capture=False,
        plan_results={
            "kv": {"main": KVPlanOutput(cpu_indptrs=None, cuda_indptrs=None, views=views)}
        },
        **kwargs,
    )


def _planned(manager, views, causal=True, lease=None):
    ctx = _ctx(views)
    if lease is not None:
        ctx.slot_lease = lease
    manager.plan(AttentionStep(causal=causal), ctx)
    return manager._current_plans["main"]


class TestIndexConstruction:
    def test_ragged_decode_lays_out_one_row_per_request(self):
        manager = _manager()
        plan = _planned(manager, [
            _view("r0", [3, 4], length=20, to_compute=1),
            _view("r1", [7], length=5, to_compute=1),
        ])
        b = plan
        assert b.cache_seqlens[:2].tolist() == [20, 5]
        # one token each -> a strictly unit-stepping query indptr
        assert b.cu_seqlens_q[:3].tolist() == [0, 1, 2]
        assert plan.max_q == 1
        # the request's own pages, then SINK_PAGE to the block ceiling
        assert b.page_table[0].tolist() == [3, 4] + [SINK_PAGE] * 6
        assert b.page_table[1].tolist() == [7] + [SINK_PAGE] * 7

    def test_packed_prefill_accumulates_query_lengths(self):
        manager = _manager()
        plan = _planned(manager, [
            _view("r0", [0, 1], length=20, to_compute=20),
            _view("r1", [2], length=6, to_compute=6),
        ])
        assert plan.cu_seqlens_q[:3].tolist() == [0, 20, 26]
        assert plan.max_q == 20

    def test_causal_rides_on_the_step(self):
        manager = _manager()
        views = [_view("r0", [0], length=4, to_compute=1)]
        assert _planned(manager, views, causal=True).causal is True
        assert _planned(manager, views, causal=False).causal is False

    def test_block_ceiling_is_per_request_not_the_pool(self):
        """A page-table row covers one sequence, so it is sized by
        max_seq_len/page_size — not by the pool's max_num_pages, which would
        make the table 8x wider here for nothing."""
        manager = _manager()
        assert manager._max_blocks == MAX_SEQ_LEN // PAGE_SIZE
        plan = _planned(manager, [_view("r0", [0], length=4, to_compute=1)])
        assert plan.page_table.shape[1] == MAX_SEQ_LEN // PAGE_SIZE


class TestCapturePadding:
    """A replay always runs the bucket's full width, so the rows past the live
    batch have to be well-formed rather than merely zeroed."""

    def _lease(self, bs: int) -> SlotLease:
        return SlotLease(
            slot=0, bucket=BucketKey(graph_walk="decode", bs=bs, num_tokens=bs),
        )

    def test_padded_rows_are_well_formed_sink_reads(self):
        manager = _manager()
        plan = _planned(
            manager,
            [_view("r0", [5], length=9, to_compute=1)],
            lease=self._lease(4),
        )
        b = plan
        assert b.cache_seqlens.shape[0] == 4
        # length 1, not 0: a zero-length row has nothing in bounds to read
        assert b.cache_seqlens.tolist() == [9, 1, 1, 1]
        assert all(p == SINK_PAGE for p in b.page_table[1].tolist())
        # the query indptr still spans every row, so q keeps its padded width
        assert b.cu_seqlens_q.tolist() == [0, 1, 2, 3, 4]

    def test_buffers_are_the_same_tensors_across_steps(self):
        """The capture invariant: same (bucket, slot, label) -> same storage,
        written in place. A reallocation here is a wrong-address replay."""
        manager = _manager()
        lease = self._lease(4)
        first = _planned(
            manager, [_view("r0", [5], length=9, to_compute=1)], lease=lease,
        )
        second = _planned(
            manager,
            [_view("r0", [5], length=10, to_compute=1),
             _view("r1", [6], length=3, to_compute=1)],
            lease=lease,
        )

        assert second.page_table.data_ptr() == first.page_table.data_ptr()
        assert second.cache_seqlens.data_ptr() == first.cache_seqlens.data_ptr()
        assert second.cu_seqlens_q.data_ptr() == first.cu_seqlens_q.data_ptr()
        # and the values did move
        assert second.cache_seqlens.tolist() == [10, 3, 1, 1]

    def test_a_different_bucket_gets_its_own_buffers(self):
        manager = _manager()
        views = [_view("r0", [5], length=9, to_compute=1)]
        small = _planned(manager, views, lease=self._lease(2))
        large = _planned(manager, views, lease=self._lease(8))
        assert small.cache_seqlens.data_ptr() != large.cache_seqlens.data_ptr()
        assert small.cache_seqlens.shape[0] == 2
        assert large.cache_seqlens.shape[0] == 8

    def test_more_views_than_the_bucket_is_an_error_not_an_overrun(self):
        manager = _manager()
        with pytest.raises(RuntimeError, match="rows but its bucket holds"):
            _planned(
                manager,
                [_view(f"r{i}", [i], length=4, to_compute=1) for i in range(4)],
                lease=self._lease(2),
            )


class TestEagerWidth:
    """Regression: an eager step must describe exactly its own query rows.

    The eager buffers are reused and sized to the widest batch seen, so
    padding out to the buffer — which is right for a captured replay — made
    `cu_seqlens_q` claim rows that `q` did not have. The kernel indexes `q`
    through that, so it read off the end and reported an illegal access from
    whatever ran next, nowhere near here.
    """

    def test_a_narrower_step_does_not_inherit_the_wider_buffer_width(self):
        manager = _manager()
        wide = _planned(
            manager,
            [_view(f"r{i}", [i], length=4, to_compute=1) for i in range(6)],
        )
        assert wide.total_q == 6
        assert wide.cu_seqlens_q.tolist() == [0, 1, 2, 3, 4, 5, 6]

        narrow = _planned(
            manager, [_view("r0", [0], length=4, to_compute=1)],
        )
        # the buffer still holds 6 rows, but this step is one row wide
        assert narrow.cache_seqlens.shape[0] == 1
        assert narrow.cu_seqlens_q.tolist() == [0, 1]
        assert narrow.total_q == 1

    def test_total_q_is_the_packed_query_count(self):
        manager = _manager()
        plan = _planned(manager, [
            _view("r0", [0], length=9, to_compute=5),
            _view("r1", [1], length=4, to_compute=2),
        ])
        assert plan.total_q == 7

    def test_a_captured_step_keeps_the_padded_width(self):
        """The other side of the same coin: a replay runs the bucket, so its
        total_q counts the padding rows."""
        manager = _manager()
        plan = _planned(
            manager, [_view("r0", [0], length=9, to_compute=1)],
            lease=SlotLease(
                slot=0, bucket=BucketKey(graph_walk="decode", bs=4, num_tokens=4),
            ),
        )
        assert plan.total_q == 4


class TestPageTableCeiling:
    def test_a_request_naming_more_pages_than_a_row_holds_is_an_error(self):
        """Truncating would leave cache_seqlens describing tokens whose pages
        are not in the table: silently wrong attention, or a row overrun."""
        manager = _manager()
        too_many = list(range(MAX_SEQ_LEN // PAGE_SIZE + 1))
        with pytest.raises(RuntimeError, match="names .* pages but a page table row holds"):
            _planned(
                manager,
                [_view("r0", too_many, length=MAX_SEQ_LEN, to_compute=1)],
            )


class TestPreplanStaging:
    """Building the indices is host work, so it stages a step ahead and the
    next plan promotes it — there is no kernel-side schedule, but the list
    building and the host->device copy are still worth overlapping."""

    def _lease(self, bs: int, slot: int = 0) -> SlotLease:
        return SlotLease(
            slot=slot,
            bucket=BucketKey(graph_walk="decode", bs=bs, num_tokens=bs),
        )

    def _preplan(self, manager, views, lease):
        ctx = _ctx(views)
        ctx.slot_lease = lease
        ctx.is_preplan = True
        manager.plan(AttentionStep(causal=True), ctx)

    def test_it_stages_and_the_next_plan_promotes(self):
        manager = _manager()
        assert manager.supports_preplan is True
        views = [_view("r0", [5], length=9, to_compute=1)]
        # staged on the plan thread against a slot the live replay is not on
        self._preplan(manager, views, self._lease(4, slot=1))
        assert manager._current_plans == {}
        assert "main" in manager._preplan_plans
        staged = manager._preplan_plans["main"]

        # the live plan for the same step hands back exactly what was staged
        ctx = _ctx(views)
        ctx.slot_lease = self._lease(4, slot=1)
        manager.plan(AttentionStep(causal=True), ctx)
        assert manager._current_plans["main"] is staged
        assert manager._preplan_plans == {}

    def test_staging_twice_without_promoting_is_refused(self):
        manager = _manager()
        views = [_view("r0", [5], length=9, to_compute=1)]
        self._preplan(manager, views, self._lease(4, slot=1))
        with pytest.raises(AssertionError, match="preplan is already pending"):
            self._preplan(manager, views, self._lease(4, slot=1))

    def test_clear_preplan_drops_the_stage(self):
        manager = _manager()
        self._preplan(
            manager, [_view("r0", [5], length=9, to_compute=1)],
            self._lease(4, slot=1),
        )
        manager.clear_preplan()
        assert manager._preplan_plans == {}
        # and the next plan builds afresh rather than promoting a stale stage
        plan = _planned(
            manager, [_view("r0", [5], length=12, to_compute=1)],
            lease=self._lease(4, slot=0),
        )
        assert plan.cache_seqlens.tolist()[0] == 12

    def test_preplan_without_a_lease_is_refused(self):
        """An eager step shares its slot with the captured one, so there is no
        disjoint buffer to stage into."""
        manager = _manager()
        ctx = _ctx([_view("r0", [0], length=4, to_compute=1)])
        ctx.is_preplan = True
        with pytest.raises(AssertionError, match="preplan requires a cuda graph step"):
            manager.plan(AttentionStep(causal=True), ctx)

    def test_a_staged_slot_does_not_share_buffers_with_the_live_one(self):
        """The whole point of the second slot: plan(N+1) must not write the
        device buffers replay(N) is reading."""
        manager = _manager()
        views = [_view("r0", [5], length=9, to_compute=1)]
        live = _planned(manager, views, lease=self._lease(4, slot=0))
        self._preplan(manager, views, self._lease(4, slot=1))
        staged = manager._preplan_plans["main"]
        assert staged.page_table.data_ptr() != live.page_table.data_ptr()
        assert staged.cache_seqlens.data_ptr() != live.cache_seqlens.data_ptr()


class TestDoubleBuffering:
    def test_double_buffering_is_required(self):
        """Not just for pre-plan: `plan` writes device buffers a replay reads,
        so two consecutive steps on one slot would race."""
        assert _manager().force_double_buffer is True


class TestSubmoduleSurface:
    def test_qo_indptr_buf_is_the_query_indptr(self):
        manager = _manager()
        plan = _planned(manager, [
            _view("r0", [0], length=8, to_compute=3),
            _view("r1", [1], length=4, to_compute=2),
        ])
        assert torch.equal(manager.qo_indptr_buf("main"), plan.cu_seqlens_q)

    def test_qo_indptr_buf_is_none_for_an_unplanned_label(self):
        assert _manager().qo_indptr_buf("nope") is None

    def test_select_last_hidden_picks_each_requests_final_row(self):
        manager = _manager()
        _planned(manager, [
            _view("r0", [0], length=8, to_compute=3),
            _view("r1", [1], length=4, to_compute=2),
        ])
        hidden = torch.arange(5, dtype=torch.float32)[:, None]
        # rows 0,1,2 are r0 and 3,4 are r1 -> last of each is 2 and 4
        assert manager.select_last_hidden(hidden).flatten().tolist() == [2.0, 4.0]


class TestHeadDimGuard:
    def test_an_unbuilt_head_dim_is_refused_with_the_fix(self):
        with pytest.raises(ValueError, match="FLASH_ATTENTION_DISABLE_HDIM"):
            check_fa3_head_dim(72, torch.device("cuda"))

    def test_cpu_is_not_checked(self):
        # CPU tests drive the manager with toy head dims and no kernel
        check_fa3_head_dim(72, torch.device("cpu"))

    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_the_built_head_dims_pass(self, head_dim):
        check_fa3_head_dim(head_dim, torch.device("cuda"))
