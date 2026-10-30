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
The host pool is the real ``CPUPagePool`` accounting (its ``PageArena`` and
per-stream refusal in ``place``) without the pinned buffer and the CUDA copies.
"""

from __future__ import annotations

import logging
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.engine import Engine
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.cache import PageAllocator
from mstar.engine.resources.kv.config import KVStep, PagedKVConfig
from mstar.engine.resources.kv.cpu_page_pool import CPUPagePool
from mstar.engine.resources.kv.manager import KVManager, PageArena
from mstar.engine.resources.kv.prefix_index import PrefixIndex
from mstar.engine.resources.step import Segment, StepContext
from mstar.worker import worker as worker_mod
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
        self.arena = PageArena(kv_cache=None, allocator=PageAllocator(capacity))
        self.index = PrefixIndex(self.arena)
        self.offloaded = {}
        self._refused_logged_at = None
        self._refused_unlogged = 0
        self.dropped = 0
        self._dropped_logged_at = None

    def copy_out(self, gpu_kv_cache, copies):
        pass

    def copy_in(self, gpu_kv_cache, copies):
        pass

    def sync(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransferManager)


def _kv(host_pages: int, name: str = "kv") -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=64, page_size=PAGE,
        ),
        name=name, joint_comm_group=None, transfer_engine_info=None,
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
    assert _grow(kv, rid, pages * PAGE)


def _grow(kv: KVManager, rid: int, tokens: int) -> bool:
    """Run one step of *tokens* for *rid*; False when its admit refuses."""
    step, ctx = _step(rid, tokens)
    if not kv.admit(step, ctx).ok:
        return False
    kv.plan(step, ctx)
    kv.commit(step, ctx)
    return True


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

    def offload_request(self, node_name, request_id, affected_resources=None):
        self.offload_calls.append(request_id)
        return self.kv.offload(request_id)


class _TwoResourceEngine:
    """The real ``Engine`` eviction methods over a node with two evictable KV
    resources, the way BAGEL and Omni nodes have several."""

    evictable = Engine.evictable
    is_offloaded = Engine.is_offloaded
    reclaimable = Engine.reclaimable

    def __init__(self, *kvs: KVManager):
        self._resources = {kv.name: kv for kv in kvs}
        self._submodules = {NODE: SimpleNamespace(resources=dict(self._resources))}
        self._resident_delta = {}
        self._tp_follower_nodes = set()
        self.offload_calls: list[int] = []

    def offload_request(self, node_name, request_id, affected_resources=None):
        self.offload_calls.append(request_id)
        return Engine.offload_request(self, node_name, request_id, affected_resources)


class _Evictor(Worker):
    """A ``Worker`` holding only what eviction reads."""

    def __init__(self, engine, last_active: dict[int, float]):
        self.engine = engine
        self.engine_manager = SimpleNamespace(get_engine=lambda node_name: engine)
        self._last_active = {(rid, NODE): t for rid, t in last_active.items()}
        self.eviction_policy = EvictionPolicy.LRU
        self._no_victim_logged = {}

    def evict(
        self, batch_ids: set[int], affected_resources: set[str] | None = None,
    ) -> int | None:
        return self._try_offload_cold_request(NODE, batch_ids, affected_resources)


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



# A node with several KV resources (BAGEL, Omni) offloads a victim from all of
# them. Pages freed in a resource that didn't run out leave the retry as short
# as before, so they must not count as the victim having moved.


def _two_resources():
    """``BIG`` is least recently used and holds pages in both resources; its
    ``kv_a`` stream is too big for that host pool, so offloading it frees only
    ``kv_b``. ``SMALL`` holds pages in ``kv_a`` that fit."""
    kv_a, kv_b = _kv(host_pages=3, name="kv_a"), _kv(host_pages=16, name="kv_b")
    _fill(kv_a, BIG, 8)
    _fill(kv_b, BIG, 2)
    _fill(kv_a, SMALL, 2)
    engine = _TwoResourceEngine(kv_a, kv_b)
    return kv_a, kv_b, _Evictor(engine, {BIG: 1.0, SMALL: 5.0, BLAMED: 9.0})


def test_control_the_lru_victim_frees_pages_only_in_the_other_resource():
    kv_a, kv_b, worker = _two_resources()
    assert worker.engine.reclaimable(NODE, BIG, {"kv_a"}) > 0
    assert kv_a.offload(BIG) == 0
    assert kv_b.offload(BIG) == 2


def test_a_victim_that_frees_only_another_resource_falls_through():
    kv_a, kv_b, worker = _two_resources()
    victim = worker.evict({BLAMED}, affected_resources={"kv_a"})
    assert victim == SMALL, (
        f"eviction returned {victim!r}: {BIG} freed pages only in kv_b, but "
        f"kv_a ran out, so the retry would refuse again"
    )
    assert worker.engine.offload_calls == [BIG, SMALL]
    assert kv_a.is_offloaded(SMALL)
    # BIG's kv_b stream did move, and check_ready brings it back like any victim
    assert kv_b.is_offloaded(BIG) and not kv_a.is_offloaded(BIG)


def test_unscoped_eviction_still_counts_every_resource():
    _, kv_b, worker = _two_resources()
    assert worker.evict({BLAMED}) == BIG
    assert worker.engine.offload_calls == [BIG]
    assert kv_b.is_offloaded(BIG)


class _RefusingEngine:
    """An evictable node whose candidates hold pages and never move, until
    ``moves`` is set."""

    def __init__(self):
        self.moves = False

    def evictable(self, node_name):
        return True

    def is_offloaded(self, node_name, request_id):
        return False

    def reclaimable(self, node_name, request_id, affected_resources=None):
        return 1

    def offload_request(self, node_name, request_id, affected_resources=None):
        return 1 if self.moves else 0


def _no_victim_warnings(caplog):
    return [
        r.getMessage() for r in caplog.records
        if "host pool is most likely full" in r.getMessage()
    ]


def test_a_stalled_batch_logs_no_victim_once_per_interval_with_a_count(
    monkeypatch, caplog,
):
    """One stalled run logged this line on every retry, about 12.6k times."""
    clock = [100.0]
    monkeypatch.setattr(worker_mod._time, "monotonic", lambda: clock[0])
    worker = _Evictor(_RefusingEngine(), {BIG: 1.0, SMALL: 5.0})
    with caplog.at_level(logging.WARNING, logger="mstar.worker.worker"):
        for _ in range(50):
            assert worker.evict({BLAMED}, {"kv"}) is None
        assert len(_no_victim_warnings(caplog)) == 1
        assert "(0 earlier refusals not logged)" in _no_victim_warnings(caplog)[0]

        clock[0] += worker_mod._HOLD_LOG_INTERVAL + 0.1
        assert worker.evict({BLAMED}, {"kv"}) is None
    lines = _no_victim_warnings(caplog)
    assert len(lines) == 2
    assert "(49 earlier refusals not logged)" in lines[1]
    assert "freed pages of kv;" in lines[1]


def test_an_eviction_that_moves_starts_a_new_stall_episode(monkeypatch, caplog):
    monkeypatch.setattr(worker_mod._time, "monotonic", lambda: 100.0)
    engine = _RefusingEngine()
    worker = _Evictor(engine, {BIG: 1.0, SMALL: 5.0})
    with caplog.at_level(logging.WARNING, logger="mstar.worker.worker"):
        assert worker.evict({BLAMED}) is None
        assert worker.evict({BLAMED}) is None
        engine.moves = True
        assert worker.evict({BLAMED}) == BIG
        assert NODE not in worker._no_victim_logged
        engine.moves = False
        # same instant: a new stall logs at once instead of waiting out the
        # previous one's interval
        assert worker.evict({BLAMED}) is None
    lines = _no_victim_warnings(caplog)
    assert len(lines) == 2
    assert "(0 earlier refusals not logged)" in lines[1]
