"""An OOM eviction moves on to the next victim when one won't move.

``Worker._try_offload_cold_request`` picks the request a step of the refused
key would reach last (``MicroScheduler.last_in_line``) and offloads it. A
victim can refuse for reasons of its own: its streams don't fit what is left of
the host pool (``CPUPagePool`` allocates per stream), or a step is admitted
against its pages (``step_in_flight``). Nothing about it changes before the
retry, so if eviction stops at that victim every retry re-picks it and the
blamed request is held for as long as the victim stays last in line, a big idle
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

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.arena import new_page_arena
from mstar.engine.resources.kv.config import KVStep, PagedKVConfig
from mstar.engine.resources.kv.cpu_page_pool import CPUPagePool
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.step import Segment, StepContext
from mstar.model.submodule_base import StepOrder
from mstar.worker.micro_scheduler import MicroScheduler
from mstar.worker.worker import EvictionPolicy, Worker

PAGE = 16
NODE = "LLM"
WALK = "decode"


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
    """The real ``CPUPagePool`` with an unpinned host buffer and no copies: the
    same arena and prefix index, and the same per-stream refusal in ``place``."""

    def __init__(self, **kwargs):
        empty = torch.empty

        def unpinned(*args, pin_memory=False, **kw):
            return empty(*args, **kw)

        torch.empty = unpinned
        try:
            super().__init__(**kwargs)
        finally:
            torch.empty = empty

    def copy_out(self, gpu_kv_cache, copies):
        pass

    def copy_in(self, gpu_kv_cache, copies):
        pass

    def sync(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransferManager)


def _kv(host_pages: int) -> KVManager:
    cfg = PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
        max_num_pages=64, page_size=PAGE,
    )
    kv = KVManager(
        cfg=cfg, name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv._cpu_pool = _HostPool(
        config=cfg, kv_cache=kv.kv_cache,
        arena=new_page_arena(cfg.units(host_pages)),
    )
    return kv


def _step(rid: int, tokens: int) -> tuple[KVStep, StepContext]:
    step = KVStep(segments=(Segment(rid, "main", tokens),))
    ctx = StepContext(request_ids=(rid,), graph_walk=WALK, slot=0, capture=False)
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

    def evictable(self, node_name, resources=None):
        return True

    def get_step_order(self, node_name, graph_walk):
        return StepOrder.ARRIVAL

    def get_max_batch_tokens(self, node_name, graph_walk):
        return None

    def is_offloaded(self, node_name, request_id):
        return self.kv.is_offloaded(request_id)

    def reclaimable(self, node_name, request_id, affected_resources=None):
        return self.kv.reclaimable(request_id)

    def offload_request(self, node_name, request_id):
        self.offload_calls.append(request_id)
        return self.kv.offload(request_id)


def _line(engine, *rids: int) -> MicroScheduler:
    """The real ``MicroScheduler`` line order over a node that has run no step
    yet: plain arrival order, so *rids* are the victims in the order eviction
    tries them (``last_in_line`` takes the latest arrival first)."""
    sched = MicroScheduler.__new__(MicroScheduler)
    sched.engine_manager = SimpleNamespace(get_engine=lambda node_name: engine)
    sched.arrival = {rid: n for n, rid in enumerate(reversed(rids), start=1)}
    sched._arrivals = len(rids)
    sched.last_ran = {}
    sched._latest_step = {}
    sched.chunk_started = {}
    sched.combined_walk_of = {}
    sched._walks_of_key = {}
    sched._orders = {}
    sched._budgeted = {}
    return sched


class _Evictor(Worker):
    """A ``Worker`` holding only what eviction reads."""

    def __init__(self, engine: _OneResourceEngine, *order: int):
        self.engine = engine
        self.engine_manager = SimpleNamespace(get_engine=lambda node_name: engine)
        self.scheduler = _line(engine, *order)
        self.eviction_policy = EvictionPolicy.LAST_IN_LINE

    def evict(self, batch_ids: set[int]) -> int | None:
        return self._try_offload_cold_request(NODE, WALK, batch_ids)


BIG, SMALL, OTHER, BLAMED = 1, 2, 3, 4


def _big_idle_and_small(host_pages: int = 3) -> tuple[KVManager, _Evictor]:
    """An 8-page request that is last in line and a 2-page one behind it; the
    host pool has room for the small one only."""
    kv = _kv(host_pages)
    _fill(kv, BIG, 8)
    _fill(kv, SMALL, 2)
    kv.ingest_request(BLAMED)
    return kv, _Evictor(_OneResourceEngine(kv), BIG, SMALL, OTHER, BLAMED)


def test_control_the_small_request_alone_would_move():
    kv, _ = _big_idle_and_small()
    assert kv.offload(BIG) == 0
    assert kv.offload(SMALL) == 2


def test_a_victim_too_big_for_the_host_pool_falls_through_to_one_that_fits():
    kv, worker = _big_idle_and_small()
    victim = worker.evict({BLAMED})
    assert victim == SMALL, (
        f"eviction returned {victim!r}: it tried only the first victim {BIG}, "
        f"which doesn't fit the host pool, although {SMALL} would have moved"
    )
    assert worker.engine.offload_calls == [BIG, SMALL]
    assert kv.is_offloaded(SMALL) and not kv.is_offloaded(BIG)


def test_retries_do_not_re_pick_the_same_immovable_victim():
    kv, worker = _big_idle_and_small(host_pages=5)
    _fill(kv, OTHER, 2)
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
    worker = _Evictor(_OneResourceEngine(kv), BIG, SMALL)
    assert worker.evict({BLAMED}) == SMALL


def test_requests_outside_the_batch_are_still_tried_first():
    kv = _kv(host_pages=16)
    _fill(kv, BIG, 2)       # in the batch, and first in the victim order
    _fill(kv, SMALL, 2)
    worker = _Evictor(_OneResourceEngine(kv), BIG, SMALL)
    assert worker.evict({BIG}) == SMALL
    assert worker.engine.offload_calls == [SMALL]


def test_a_batch_member_is_tried_when_no_outside_request_moves():
    kv = _kv(host_pages=3)
    _fill(kv, BIG, 8)       # outside the batch, too big for the host pool
    _fill(kv, SMALL, 2)     # in the batch
    worker = _Evictor(_OneResourceEngine(kv), BIG, SMALL)
    assert worker.evict({SMALL}) == SMALL
    assert worker.engine.offload_calls == [BIG, SMALL]


def test_nothing_moves_reports_a_full_host_pool_once(caplog):
    kv = _kv(host_pages=1)
    _fill(kv, BIG, 8)
    _fill(kv, SMALL, 2)
    worker = _Evictor(_OneResourceEngine(kv), BIG, SMALL)
    with caplog.at_level(logging.WARNING, logger="mstar.worker.worker"):
        assert worker.evict({BLAMED}) is None
    assert worker.engine.offload_calls == [BIG, SMALL]
    full = [r for r in caplog.records if "host pool is most likely full" in r.getMessage()]
    assert len(full) == 1
    assert not kv.is_offloaded(BIG) and not kv.is_offloaded(SMALL)
    assert kv._cpu_pool.num_free_pages == 1

