"""An OOM eviction moves on to the next victim when one won't move.

``Worker._try_offload_cold_request`` picks the least recently used request and
offloads it. A victim can refuse for reasons of its own: its streams don't fit
what is left of the host pool (``CPUPagePool`` allocates per stream), or a step
is admitted against its pages (``step_in_flight``). Nothing about it changes
before the retry, so if eviction stops at that victim every retry re-picks it
and the blamed request is held for as long as the victim stays LRU, a big idle
duplex or realtime session say, while a smaller request that would move is
never tried.

These drive the real ``_try_offload_cold_request`` over a real ``KVManager``.
The host pool is the real ``CPUPagePool`` accounting (its ``PageAllocator``
and per-stream refusal) without the pinned buffer and the CUDA copies.
"""

from __future__ import annotations

import logging
import sys
import threading
from concurrent.futures import Future
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.cache import PageAllocator
from mstar.engine.resources.kv.config import KVStep, PagedKVConfig
from mstar.engine.resources.kv.cpu_page_pool import CPUPagePool, OffloadedStream
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import Segment, StepContext
from mstar.worker.worker import EvictionPolicy, Worker

PAGE = 16
NODE = "LLM"


class _StubTransferManager:
    def __init__(self, *args, **kwargs):
        pass

    def get_kv_transfer_info(self, **kwargs):
        return None

    def owns_transfer_info(self, *args, **kwargs):
        return False

    def remove_request(self, request_id):
        pass

    def cleanup(self):
        pass


class _HostPool(CPUPagePool):
    """``CPUPagePool`` with no pinned buffer and no copies: the same page
    accounting and the same refusal of a stream that doesn't fit."""

    def __init__(self, capacity: int):
        self.page_allocator = PageAllocator(capacity)
        self.offloaded = {}
        self._stream = None

    def offload_stream(
        self, rid, label, gpu_kv_cache, gpu_page_indices, stored_len, position,
        released=0, protected_prefix=0,
    ) -> bool:
        if not gpu_page_indices:
            return False
        cpu_pages = self.page_allocator.try_allocate(len(gpu_page_indices))
        if cpu_pages is None:
            return False
        self.offloaded.setdefault(rid, {})[label] = OffloadedStream(
            cpu_page_indices=cpu_pages, stored_len=stored_len, position=position,
            released=released, protected_prefix=protected_prefix,
        )
        return True


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransferManager)


def _kv(host_pages: int) -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=64, page_size=PAGE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv._cpu_pool = _HostPool(host_pages)
    return kv


def _step(rid: int, tokens: int) -> tuple[KVStep, StepContext]:
    step = KVStep(segments=(Segment(rid, "main", tokens),))
    ctx = StepContext(request_ids=(rid,), graph_walk="decode", slot=0, capture=False)
    return step, ctx


def _fill(kv: KVManager, rid: int, pages: int) -> None:
    kv.ingest_request(rid)
    step, ctx = _step(rid, pages * PAGE)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)


class _OneResourceEngine:
    """``Engine``'s eviction surface for a node whose one evictable resource
    is *kv*."""

    def __init__(self, kv: KVManager):
        self.kv = kv
        self.offload_calls: list[int] = []

    def evictable(self, node_name):
        return True

    def is_offloaded(self, node_name, request_id):
        return self.kv.is_offloaded(request_id)

    def reclaimable(self, node_name, request_id, affected_resources=None):
        return self.kv.reclaimable(request_id)

    def offload_request(self, node_name, request_id):
        self.offload_calls.append(request_id)
        return self.kv.offload(request_id)


class _Evictor(Worker):
    """A ``Worker`` holding only what eviction reads."""

    def __init__(self, engine: _OneResourceEngine, last_active: dict[int, float]):
        self.engine = engine
        self.engine_manager = SimpleNamespace(get_engine=lambda node_name: engine)
        self._last_active = {(rid, NODE): t for rid, t in last_active.items()}
        self.eviction_policy = EvictionPolicy.LRU

    def evict(self, batch_ids: set[int]) -> int | None:
        return self._try_offload_cold_request(NODE, batch_ids)


BIG, SMALL, OTHER, BLAMED = 1, 2, 3, 4


def _big_idle_and_small(host_pages: int = 3) -> tuple[KVManager, _Evictor]:
    """An 8-page request that has been idle longest and a 2-page one; the host
    pool has room for the small one only."""
    kv = _kv(host_pages)
    _fill(kv, BIG, 8)
    _fill(kv, SMALL, 2)
    kv.ingest_request(BLAMED)
    return kv, _Evictor(_OneResourceEngine(kv), {BIG: 1.0, SMALL: 5.0, BLAMED: 9.0})


def test_control_the_small_request_alone_would_move():
    kv, _ = _big_idle_and_small()
    assert kv.offload(BIG) == 0
    assert kv.offload(SMALL) == 2


def test_a_victim_too_big_for_the_host_pool_falls_through_to_one_that_fits():
    kv, worker = _big_idle_and_small()
    victim = worker.evict({BLAMED})
    assert victim == SMALL, (
        f"eviction returned {victim!r}: it tried only the LRU victim {BIG}, "
        f"which doesn't fit the host pool, although {SMALL} would have moved"
    )
    assert worker.engine.offload_calls == [BIG, SMALL]
    assert kv.is_offloaded(SMALL) and not kv.is_offloaded(BIG)


def test_retries_do_not_re_pick_the_same_immovable_victim():
    kv, worker = _big_idle_and_small(host_pages=5)
    _fill(kv, OTHER, 2)
    worker._last_active[(OTHER, NODE)] = 6.0
    victims = [worker.evict({BLAMED}) for _ in range(3)]
    assert victims == [SMALL, OTHER, None], (
        f"three OOM retries evicted {victims}: each should take the next "
        f"request that moves and skip {BIG}, which never fits"
    )


def test_a_victim_with_a_step_in_flight_falls_through_to_the_next():
    kv = _kv(host_pages=16)
    _fill(kv, BIG, 2)
    _fill(kv, SMALL, 2)
    # a step admitted against BIG's pages and not yet committed
    step, ctx = _step(BIG, PAGE)
    assert kv.admit(step, ctx).ok
    assert kv.offload(BIG) == 0
    worker = _Evictor(_OneResourceEngine(kv), {BIG: 1.0, SMALL: 5.0})
    assert worker.evict({BLAMED}) == SMALL


def test_a_victim_that_cannot_fit_does_not_wait_on_its_reads():
    """Trying the next victim must not block the worker on a read of one that
    would free nothing anyway."""
    kv, worker = _big_idle_and_small(host_pages=1)
    pending = Future()
    kv._streams[SMALL]["main"].read_future = pending
    kv._streams[SMALL]["main"].read_pending = True
    done = threading.Event()
    result = []

    def evict():
        result.append(worker.evict({BLAMED}))
        done.set()

    threading.Thread(target=evict, daemon=True).start()
    try:
        assert done.wait(2), "eviction waited on the read of a victim that does not fit"
    finally:
        pending.set_result(None)
    assert result == [None]
    assert not kv.is_offloaded(SMALL)


def test_requests_outside_the_batch_are_still_tried_first():
    kv = _kv(host_pages=16)
    _fill(kv, BIG, 2)       # in the batch, and least recently used
    _fill(kv, SMALL, 2)
    worker = _Evictor(_OneResourceEngine(kv), {BIG: 1.0, SMALL: 5.0})
    assert worker.evict({BIG}) == SMALL
    assert worker.engine.offload_calls == [SMALL]


def test_a_batch_member_is_tried_when_no_outside_request_moves():
    kv = _kv(host_pages=3)
    _fill(kv, BIG, 8)       # outside the batch, too big for the host pool
    _fill(kv, SMALL, 2)     # in the batch
    worker = _Evictor(_OneResourceEngine(kv), {BIG: 1.0, SMALL: 5.0})
    assert worker.evict({SMALL}) == SMALL
    assert worker.engine.offload_calls == [BIG, SMALL]


def test_nothing_moves_reports_a_full_host_pool_once(caplog):
    kv = _kv(host_pages=1)
    _fill(kv, BIG, 8)
    _fill(kv, SMALL, 2)
    worker = _Evictor(_OneResourceEngine(kv), {BIG: 1.0, SMALL: 5.0})
    with caplog.at_level(logging.WARNING, logger="mstar.worker.worker"):
        assert worker.evict({BLAMED}) is None
    assert worker.engine.offload_calls == [BIG, SMALL]
    full = [r for r in caplog.records if "host pool is most likely full" in r.getMessage()]
    assert len(full) == 1
    assert not kv.is_offloaded(BIG) and not kv.is_offloaded(SMALL)
    assert kv._cpu_pool.num_free_pages == 1
