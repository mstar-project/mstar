"""Unit tests for the KV cache pool and its boundary types.

``KVCachePool`` is the segment lifecycle over per-request cache storage:
``admit`` reserves pages for a segment, ``view`` describes the stream it
extends, ``commit`` advances stored length and position counter. The values
crossing the boundary (``Segment``, ``Reservation``, ``SequenceView``,
``PositionPlan``) are immutable and valid for one step.
"""

from __future__ import annotations

import dataclasses
import sys
import threading

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.kv_store import (
    AllocationFailedError,
    KVCacheConfig,
    PageAllocator,
    PagedAllocationManager,
    StoreWritePolicy,
)
from mstar.engine.resources import (
    KVCachePool,
    PageArena,
    PositionPlan,
    Reservation,
    Segment,
    SequenceView,
)


def _make_pool(max_num_pages: int = 16, page_size: int = 8) -> tuple[KVCachePool, PagedAllocationManager]:
    manager = PagedAllocationManager.__new__(PagedAllocationManager)
    manager.config = KVCacheConfig(
        num_layers=1,
        num_kv_heads=1,
        head_dim=1,
        max_seq_len=max_num_pages * page_size,
        max_num_pages=max_num_pages,
        page_size=page_size,
    )
    manager.page_allocator = PageAllocator(max_num_pages)
    manager.request_states = {}
    manager.kv_cache = None
    manager.write_policy = StoreWritePolicy.ALWAYS
    manager._kv_transfer_engine = None
    manager._offload_stream = None
    manager.pending_reads = {}
    manager._lock = threading.RLock()
    return KVCachePool(manager), manager


class TestAdmit:
    def test_admit_reserves_whole_pages(self):
        pool, manager = _make_pool(page_size=8)
        manager.add_request("r", ["main"])

        reservation = pool.admit(Segment("r", "main", 20))  # 3 pages of 8
        assert reservation == Reservation(resident=0, to_compute=20, pending=False)
        assert len(pool.view(Segment("r", "main", 0)).page_indices) == 3
        assert pool.num_free_pages == 13

    def test_admit_grows_from_resident_length(self):
        pool, manager = _make_pool(page_size=8)
        manager.add_request("r", ["main"])

        first = Segment("r", "main", 8)  # exactly one page
        pool.admit(first)
        pool.commit(first)

        # 5 more tokens cross into a second page.
        reservation = pool.admit(Segment("r", "main", 5))
        assert reservation.resident == 8
        assert reservation.to_compute == 5
        assert len(pool.view(Segment("r", "main", 0)).page_indices) == 2

    def test_admit_within_last_page_reserves_nothing(self):
        pool, manager = _make_pool(page_size=8)
        manager.add_request("r", ["main"])

        first = Segment("r", "main", 5)
        pool.admit(first)
        pool.commit(first)
        free_before = pool.num_free_pages

        pool.admit(Segment("r", "main", 3))  # fills page to exactly 8
        assert pool.num_free_pages == free_before

    def test_zero_span_admit_reserves_nothing(self):
        pool, manager = _make_pool()
        manager.add_request("r", ["main"])
        free_before = pool.num_free_pages

        reservation = pool.admit(Segment("r", "main", 0))
        assert reservation == Reservation(resident=0, to_compute=0, pending=False)
        assert pool.num_free_pages == free_before

    def test_admit_failure_carries_diagnostics(self):
        pool, manager = _make_pool(max_num_pages=2, page_size=8)
        manager.add_request("r", ["main"])

        with pytest.raises(AllocationFailedError) as excinfo:
            pool.admit(Segment("r", "main", 100))  # needs 13 pages, has 2
        assert excinfo.value.pages_short == 11
        assert excinfo.value.request_id == "r"
        assert excinfo.value.label == "main"
        # A failed admit must not leak partial reservations.
        assert pool.num_free_pages == 2

    def test_admit_reports_pending_residency(self):
        pool, manager = _make_pool()
        manager.add_request("r", ["main"])
        manager.get_state("r", "main").read_in_progress = True

        assert pool.admit(Segment("r", "main", 0)).pending is True


class TestViewAndCommit:
    def test_view_extent_covers_resident_plus_span(self):
        pool, manager = _make_pool(page_size=8)
        manager.add_request("r", ["main"])

        first = Segment("r", "main", 10)
        pool.admit(first)
        view = pool.view(first)
        assert view.start == 0
        assert view.length == 10
        assert view.pool is pool

        pool.commit(first)
        # Before the next step's admit, a zero-span view sees the committed
        # stream.
        assert pool.view(Segment("r", "main", 0)).length == 10

    def test_view_pages_are_an_immutable_copy(self):
        pool, manager = _make_pool(page_size=8)
        manager.add_request("r", ["main"])

        segment = Segment("r", "main", 20)
        pool.admit(segment)
        view = pool.view(segment)
        assert isinstance(view.page_indices, tuple)
        # Mutating the live state afterwards must not change the view.
        pages_at_plan_time = view.page_indices
        manager.reset_label("r", "main")
        assert view.page_indices == pages_at_plan_time

    def test_commit_advances_length_and_positions_together(self):
        pool, manager = _make_pool()
        manager.add_request("r", ["main"])

        segment = Segment("r", "main", 7)
        pool.admit(segment)
        pool.commit(segment)
        assert pool.view(Segment("r", "main", 0)).length == 7
        assert pool.positions("r", "main") == 7

    def test_commit_with_position_override(self):
        """Steps whose position span differs from their token count (vision
        grids, single-position image blocks) advance positions by an explicit
        amount."""
        pool, manager = _make_pool()
        manager.add_request("r", ["main"])

        segment = Segment("r", "main", 6)
        pool.admit(segment)
        pool.commit(segment, pos_advance=1)
        assert pool.view(Segment("r", "main", 0)).length == 6
        assert pool.positions("r", "main") == 1

    def test_streams_advance_independently(self):
        pool, manager = _make_pool()
        manager.add_request("r", ["main", "cfg"])

        main_seg = Segment("r", "main", 4)
        pool.admit(main_seg)
        pool.commit(main_seg)
        assert pool.view(Segment("r", "cfg", 0)).length == 0
        assert pool.positions("r", "cfg") == 0

    def test_write_once_context_flow(self):
        """The cross-attention shape: one admitted write, position counter
        pinned, then zero-span reads."""
        pool, manager = _make_pool(page_size=8)
        manager.add_request("r", ["ctx"])

        segment = Segment("r", "ctx", 12)
        pool.admit(segment)
        pool.commit(segment, pos_advance=0)

        view = pool.view(Segment("r", "ctx", 0))
        assert view.length == 12
        assert len(view.page_indices) == 2
        assert pool.positions("r", "ctx") == 0


class TestPageArena:
    def test_allocate_free_roundtrip(self):
        arena = PageArena(tensor=None, allocator=PageAllocator(4), page_size=8)
        assert arena.total_pages == 4
        pages = arena.allocate(3)
        assert arena.num_free == 1
        arena.free(pages)
        assert arena.num_free == 4

    def test_try_allocate_shortfall_returns_none(self):
        arena = PageArena(tensor=None, allocator=PageAllocator(2), page_size=8)
        assert arena.try_allocate(3) is None
        assert arena.num_free == 2


class TestBoundaryValuesAreImmutable:
    def test_segment_frozen(self):
        segment = Segment("r", "main", 4)
        with pytest.raises(dataclasses.FrozenInstanceError):
            segment.span = 5

    def test_reservation_frozen(self):
        reservation = Reservation(resident=1, to_compute=2)
        with pytest.raises(dataclasses.FrozenInstanceError):
            reservation.resident = 3

    def test_sequence_view_frozen(self):
        view = SequenceView(pool=None, page_indices=(1, 2), start=0, length=9)
        with pytest.raises(dataclasses.FrozenInstanceError):
            view.length = 10

    def test_position_plan_frozen(self):
        plan = PositionPlan(pos_ids=torch.arange(3), advance=(3,))
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.advance = (4,)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
