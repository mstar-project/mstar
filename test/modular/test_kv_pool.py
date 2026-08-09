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

from mstar.conductor.request_info import SequenceInfo
from mstar.engine.cpu_page_pool import OffloadedState
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


class _StubTransferEngine:
    """Transfer engine that never actually moves bytes: reads complete
    immediately and the transfer descriptor is a sentinel."""

    def __init__(self):
        self.transfer_info = object()

    def read_batched_async(self, remote_kv_info, read_info):
        return None

    def get_kv_transfer_info(self):
        return self.transfer_info


class _FakeCpuPool:
    """CPU tier double: tracks offloaded page bookkeeping without pinned
    memory or GPU copies."""

    def __init__(self, max_pages: int = 16):
        self.page_allocator = PageAllocator(max_pages)
        self.offloaded: dict[str, dict[str, OffloadedState]] = {}

    def is_offloaded(self, request_id: str) -> bool:
        return bool(self.offloaded.get(request_id))

    def offload_pages(
        self, request_id, label, gpu_kv_cache, gpu_page_indices,
        seq_len, position_id_start,
    ) -> None:
        cpu_pages = self.page_allocator.try_allocate(len(gpu_page_indices))
        if cpu_pages is None:
            return
        self.offloaded.setdefault(request_id, {})[label] = OffloadedState(
            cpu_page_indices=cpu_pages,
            seq_len=seq_len,
            position_id_start=position_id_start,
        )

    def reload_pages(self, request_id, label, gpu_kv_cache, gpu_page_indices):
        state = self.offloaded[request_id][label]
        self.page_allocator.free(state.cpu_page_indices)
        del self.offloaded[request_id][label]
        if not self.offloaded[request_id]:
            del self.offloaded[request_id]
        return state.seq_len, state.position_id_start

    def sync(self) -> None:
        return

    def remove_request(self, request_id: str) -> None:
        for state in self.offloaded.pop(request_id, {}).values():
            self.page_allocator.free(state.cpu_page_indices)


def _make_pool(
    max_num_pages: int = 16,
    page_size: int = 8,
    with_tensor: bool = False,
    cpu_pool: _FakeCpuPool | None = None,
) -> tuple[KVCachePool, PagedAllocationManager]:
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
    manager.kv_cache = (
        torch.zeros(1, max_num_pages, 2, page_size, 1, 1) if with_tensor else None
    )
    manager.write_policy = StoreWritePolicy.ALWAYS
    manager._kv_transfer_engine = _StubTransferEngine()
    manager._offload_stream = None
    manager.pending_reads = {}
    manager._lock = threading.RLock()
    return KVCachePool(manager, cpu_pool=cpu_pool), manager


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


class TestFork:
    def _fill_stream(self, pool, manager, rid, label, tokens):
        segment = Segment(rid, label, tokens)
        pool.admit(segment)
        pool.commit(segment)
        # Stamp each page with its own index so copies are checkable.
        for page in manager.get_state(rid, label).page_indices:
            manager.kv_cache[:, page] = float(page + 1)

    def test_fork_mirrors_accounting_and_copies_pages(self):
        pool, manager = _make_pool(page_size=8, with_tensor=True)
        manager.add_request("r", ["main", "snap"])
        self._fill_stream(pool, manager, "r", "main", 12)
        pool.commit(Segment("r", "main", 0), pos_advance=30)  # counter != length

        pool.fork("r", "main", "snap")

        main_state = manager.get_state("r", "main")
        snap_state = manager.get_state("r", "snap")
        assert snap_state.seq_len == 12
        assert snap_state.position_id_start == 42
        assert snap_state.page_indices != main_state.page_indices
        for src, dst in zip(
            main_state.page_indices, snap_state.page_indices, strict=True
        ):
            assert torch.equal(manager.kv_cache[:, dst], manager.kv_cache[:, src])

    def test_fork_tops_up_only_past_the_destination_length(self):
        pool, manager = _make_pool(page_size=8, with_tensor=True)
        manager.add_request("r", ["main", "snap"])
        self._fill_stream(pool, manager, "r", "main", 16)  # 2 full pages

        # The destination already holds one committed page of its own data.
        snap_seg = Segment("r", "snap", 8)
        pool.admit(snap_seg)
        pool.commit(snap_seg)
        first_snap_page = manager.get_state("r", "snap").page_indices[0]
        manager.kv_cache[:, first_snap_page] = -1.0

        pool.fork("r", "main", "snap")

        snap_state = manager.get_state("r", "snap")
        assert snap_state.seq_len == 16
        # Page 0 was already resident on the destination and is not re-copied.
        assert torch.all(manager.kv_cache[:, snap_state.page_indices[0]] == -1.0)
        src_page_1 = manager.get_state("r", "main").page_indices[1]
        assert torch.equal(
            manager.kv_cache[:, snap_state.page_indices[1]],
            manager.kv_cache[:, src_page_1],
        )

    def test_fork_with_realloc_replaces_the_destination(self):
        pool, manager = _make_pool(page_size=8, with_tensor=True)
        manager.add_request("r", ["main", "snap"])
        self._fill_stream(pool, manager, "r", "main", 8)
        self._fill_stream(pool, manager, "r", "snap", 16)
        free_before = pool.num_free_pages

        pool.fork("r", "main", "snap", realloc=True)

        snap_state = manager.get_state("r", "snap")
        assert snap_state.seq_len == 8
        assert len(snap_state.page_indices) == 1
        # The two old destination pages went back, one new page came out.
        assert pool.num_free_pages == free_before + 1
        src_page = manager.get_state("r", "main").page_indices[0]
        assert torch.equal(
            manager.kv_cache[:, snap_state.page_indices[0]],
            manager.kv_cache[:, src_page],
        )

    def test_fork_can_fail_like_any_admit(self):
        pool, manager = _make_pool(max_num_pages=2, page_size=8, with_tensor=True)
        manager.add_request("r", ["main", "snap"])
        self._fill_stream(pool, manager, "r", "main", 16)  # both pages

        with pytest.raises(AllocationFailedError):
            pool.fork("r", "main", "snap")


class TestRetrieveAndPublish:
    def test_retrieve_installs_published_state(self):
        pool, manager = _make_pool(page_size=8)
        manager.add_request("r", ["main"])
        seq_info = SequenceInfo(
            seq_len=12,
            pos_id=34,
            latest_kv_transfer_info=object(),
            page_indices=[5, 6],
        )

        pool.retrieve("r", "main", seq_info)

        state = manager.get_state("r", "main")
        assert state.seq_len == 12
        assert state.position_id_start == 34
        assert len(state.page_indices) == 2
        assert state.read_in_progress is False

    def test_publish_describes_every_stream(self):
        pool, manager = _make_pool(page_size=8)
        manager.add_request("r", ["main", "cfg"])
        for label, span, pos in (("main", 12, 12), ("cfg", 3, 0)):
            segment = Segment("r", label, span)
            pool.admit(segment)
            pool.commit(segment, pos_advance=pos)

        published = pool.publish("r")

        assert set(published) == {"main", "cfg"}
        assert published["main"].seq_len == 12
        assert published["main"].pos_id == 12
        assert published["cfg"].seq_len == 3
        assert published["cfg"].pos_id == 0
        transfer_info = manager._kv_transfer_engine.transfer_info
        for label, info in published.items():
            assert info.latest_kv_transfer_info is transfer_info
            assert info.page_indices == manager.get_state("r", label).page_indices

    def test_publish_roundtrips_through_retrieve(self):
        """What one pool publishes, another can retrieve: the consumer ends
        up with the producer's stored length and position counter."""
        producer, producer_manager = _make_pool(page_size=8)
        producer_manager.add_request("r", ["main"])
        segment = Segment("r", "main", 20)
        producer.admit(segment)
        producer.commit(segment)

        consumer, consumer_manager = _make_pool(page_size=8)
        consumer_manager.add_request("r", ["main"])
        consumer.retrieve("r", "main", producer.publish("r")["main"])

        assert consumer.view(Segment("r", "main", 0)).length == 20
        assert consumer.positions("r", "main") == 20

    def test_publish_unknown_request_is_empty(self):
        pool, _ = _make_pool()
        assert pool.publish("ghost") == {}


class TestOffloadTier:
    def _fill(self, pool, tokens=20):
        segment = Segment("r", "main", tokens)
        pool.admit(segment)
        pool.commit(segment)

    def test_without_tier_the_pool_declines(self):
        pool, manager = _make_pool()
        manager.add_request("r", ["main"])
        self._fill(pool)

        assert pool.supports_offload is False
        assert pool.offload_candidates() == []
        assert pool.offload("r") == 0
        assert pool.is_offloaded("r") is False
        assert pool.reload("r") is False

    def test_candidates_list_page_holding_requests(self):
        pool, manager = _make_pool(cpu_pool=_FakeCpuPool())
        manager.add_request("r", ["main"])
        manager.add_request("empty", ["main"])
        self._fill(pool, tokens=20)  # 3 pages for "r", none for "empty"

        assert dict(pool.offload_candidates()) == {"r": 3}

    def test_offload_frees_gpu_pages_and_reload_restores(self):
        pool, manager = _make_pool(with_tensor=True, cpu_pool=_FakeCpuPool())
        manager.add_request("r", ["main"])
        self._fill(pool, tokens=20)  # 3 pages
        free_before = pool.num_free_pages

        freed = pool.offload("r")
        assert freed == 3
        assert pool.num_free_pages == free_before + 3
        assert pool.is_offloaded("r") is True
        assert manager.get_state("r", "main").page_indices == []

        assert pool.reload("r") is True
        assert pool.is_offloaded("r") is False
        state = manager.get_state("r", "main")
        assert state.seq_len == 20
        assert len(state.page_indices) == 3
        assert pool.num_free_pages == free_before

    def test_reload_declines_when_gpu_pages_are_short(self):
        pool, manager = _make_pool(
            max_num_pages=3, with_tensor=True, cpu_pool=_FakeCpuPool(),
        )
        manager.add_request("r", ["main"])
        self._fill(pool, tokens=20)  # all 3 pages
        assert pool.offload("r") == 3

        # Another request takes the freed pages; the reload cannot fit.
        manager.add_request("greedy", ["main"])
        greedy = Segment("greedy", "main", 24)
        pool.admit(greedy)
        pool.commit(greedy)

        assert pool.reload("r") is False
        assert pool.is_offloaded("r") is True

    def test_remove_request_clears_the_cpu_tier(self):
        cpu_pool = _FakeCpuPool(max_pages=4)
        pool, manager = _make_pool(with_tensor=True, cpu_pool=cpu_pool)
        manager.add_request("r", ["main"])
        self._fill(pool, tokens=20)
        pool.offload("r")
        assert cpu_pool.page_allocator.num_free == 1

        pool.remove_request("r")
        assert cpu_pool.page_allocator.num_free == 4
        assert "r" not in manager.request_states

    def test_write_policy_reaches_the_manager(self):
        pool, manager = _make_pool()
        pool.set_write_policy(StoreWritePolicy.NEVER)
        assert manager.write_policy is StoreWritePolicy.NEVER

    def test_add_request_opens_streams(self):
        pool, manager = _make_pool()
        pool.add_request("r", ["main", "cfg"])
        assert pool.labels("r") == ["main", "cfg"]


class _FakeFuture:
    def __init__(self):
        self._done = False

    def done(self) -> bool:
        return self._done

    def result(self):
        return None


class TestAdmitRetrieve:
    def _seq_info(self, seq_len=12, pos_id=34):
        return SequenceInfo(
            seq_len=seq_len,
            pos_id=pos_id,
            latest_kv_transfer_info=object(),
            page_indices=[5, 6],
        )

    def test_instant_transfer_is_not_pending(self):
        pool, manager = _make_pool(page_size=8)
        manager.add_request("r", ["main"])

        outcome = pool.admit_retrieve("r", "main", self._seq_info())

        assert outcome.pending is False
        assert outcome.resident == 12
        assert manager.get_state("r", "main").position_id_start == 34

    def test_pending_until_the_transfer_lands(self):
        pool, manager = _make_pool(page_size=8)
        manager.add_request("r", ["main"])
        future = _FakeFuture()
        manager._kv_transfer_engine.read_batched_async = (
            lambda remote_kv_info, read_info: future
        )

        first = pool.admit_retrieve("r", "main", self._seq_info())
        assert first.pending is True

        # Polling again neither restarts the transfer nor clears it early.
        second = pool.admit_retrieve("r", "main", self._seq_info())
        assert second.pending is True

        future._done = True
        third = pool.admit_retrieve("r", "main", self._seq_info())
        assert third.pending is False

    def test_admit_retrieve_can_fail_like_any_admit(self):
        pool, manager = _make_pool(max_num_pages=1, page_size=8)
        manager.add_request("r", ["main"])

        with pytest.raises(AllocationFailedError):
            pool.admit_retrieve("r", "main", self._seq_info(seq_len=100))


class TestPosInfoAndLabels:
    def test_pos_info_reads_one_stream(self):
        pool, manager = _make_pool()
        manager.add_request("r", ["main"])
        segment = Segment("r", "main", 9)
        pool.admit(segment)
        pool.commit(segment, pos_advance=4)

        info = pool.pos_info("r", "main")
        assert info.full_seq_len == 9
        assert info.position_id_start == 4

    def test_labels_lists_the_request_streams(self):
        pool, manager = _make_pool()
        manager.add_request("r", ["main", "cfg"])
        assert pool.labels("r") == ["main", "cfg"]


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
