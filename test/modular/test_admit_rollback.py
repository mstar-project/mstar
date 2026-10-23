"""A refused admit gives back exactly what the step reserved, row by row.

Three ways that promise broke: a KV stream stayed marked ``step_in_flight``
after its pages were unwound (so it could not be offloaded until it next
committed), the unbatched fallback returned on a refusal without unwinding the
requests it had already admitted, and the recurrent pool kept every slot a
refused step took.

CPU-only: real ``KVManager``, ``RecurrentStatePool`` and ``StepRunner``; the KV
transfer engine and the pinned host pool are stubbed.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.engine import Engine, ExecutingBatch, SubmoduleManagement
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import RecurrentStatePool
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import (
    AllocationFailed,
    Segment,
    StepContext,
    SubmoduleStep,
)

PAGE = 16


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def get_kv_transfer_info(self, **kwargs):
        return None

    def owns_transfer_info(self, *args, **kwargs):
        return False

    def remove_request(self, request_id):
        pass

    def start_async_retrieve(self, **kwargs):
        pass

    def cleanup(self):
        pass


class _HostPoolStub:
    """Stands in for the pinned host pool (pinning needs a GPU). Every copy
    succeeds, so ``offload`` is gated only by the manager's own claim."""

    def __init__(self):
        self.offloaded: dict = {}

    def is_offloaded(self, rid):
        return rid in self.offloaded

    def offload_stream(self, rid, label, **kwargs):
        self.offloaded.setdefault(rid, {})[label] = kwargs
        return True

    def sync(self):
        pass

    def discard(self, rid, label):
        self.offloaded.get(rid, {}).pop(label, None)

    def remove_request(self, rid):
        self.offloaded.pop(rid, None)


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _kv(name: str, pages: int) -> KVManager:
    mgr = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=pages, page_size=PAGE, cpu_offload_pages=0,
        ),
        name=name, joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    mgr._cpu_pool = _HostPoolStub()
    return mgr


def _pool(max_slots: int) -> RecurrentStatePool:
    spec = RecurrentStateSpec(
        "gdn_state", {"llm"},
        RecurrentStateConfig(
            num_layers=1,
            blocks={"state": RecurrentBlockConfig(shape=(2, 4), dtype=torch.float32)},
            max_slots=max_slots,
        ),
    )
    return RecurrentStatePool.build(spec, EngineResourceInfo(device=torch.device("cpu")))


def _ctx(*rids) -> StepContext:
    return StepContext(request_ids=tuple(rids), graph_walk="w", slot=0, capture=False)


def _kv_step(rid, span: int) -> KVStep:
    return KVStep(segments=(Segment(rid, "main", span),))


# --- the step_in_flight mark goes back with the pages ------------------------
#
# Whisper's decoder shape: two KV resources, and ``topo_sort`` orders
# independent keys alphabetically, so ``cross_kv_cache`` admits (and marks its
# stream) before ``kv_cache`` refuses.


def _whisper_step(cross_span: int, self_span: int, rid: str = "a") -> SubmoduleStep:
    return SubmoduleStep(steps={
        "cross_kv_cache": _kv_step(rid, cross_span),
        "kv_cache": _kv_step(rid, self_span),
    })


def _drive(runner: StepRunner, step: SubmoduleStep, ctx: StepContext):
    step.set_ctx(ctx)
    outcome = runner.admit(step)
    if outcome.ok:
        runner.plan(step)
        runner.commit(step)
    return outcome


def _whisper():
    cross = _kv("cross_kv_cache", pages=32)
    selfkv = _kv("kv_cache", pages=4)  # 3 usable: page 0 is the sink
    runner = StepRunner({"cross_kv_cache": cross, "kv_cache": selfkv})
    assert runner.order == ("cross_kv_cache", "kv_cache")
    for mgr in (cross, selfkv):
        mgr.ingest_request("a", None)
    assert _drive(runner, _whisper_step(2 * PAGE, PAGE), _ctx("a")).ok
    # a decode step needing more self-attention pages than exist
    outcome = _drive(runner, _whisper_step(0, 5 * PAGE), _ctx("a"))
    assert not outcome.ok and isinstance(outcome.reason, AllocationFailed)
    assert outcome.failed_resource == "kv_cache"
    return cross, selfkv


def test_a_refused_admit_clears_the_mark_an_earlier_resource_set():
    cross, _ = _whisper()
    stream = cross._streams["a"]["main"]
    assert len(stream.page_indices) == 2
    assert not stream.step_in_flight, "cross_kv_cache left marked by a step that never ran"


def test_a_refused_request_can_still_be_offloaded():
    cross, _ = _whisper()
    # the eviction that follows a KV refusal may pick this very request
    assert cross.offload("a") == 2


def test_a_refused_request_can_still_release_oldest():
    cross, _ = _whisper()
    cross.release_oldest("a", PAGE)


# --- one row's admit can be unwound after later rows admitted -----------------


def test_an_earlier_rows_admit_unwinds_after_a_later_row_admitted():
    """The unbatched fallback admits every request before any runs, so the
    unwind record has to hold each row's reservation, not only the last."""
    kv = _kv("kv", pages=16)
    for rid in (1, 2):
        kv.ingest_request(rid, None)
    free = kv._arena.num_free
    first, second = _kv_step(1, 2 * PAGE), _kv_step(2, 2 * PAGE)
    assert kv.admit(first, _ctx(1)).ok
    assert kv.admit(second, _ctx(2)).ok

    kv.rollback_admit(first, _ctx(1))

    s1, s2 = kv._streams[1]["main"], kv._streams[2]["main"]
    assert s1.page_indices == [] and not s1.step_in_flight
    assert len(s2.page_indices) == 2 and s2.step_in_flight, (
        "unwinding row 1 touched row 2's reservation"
    )
    kv.rollback_admit(second, _ctx(2))
    assert kv._arena.num_free == free


def test_a_commit_keeps_another_rows_unwind_record():
    kv = _kv("kv", pages=16)
    for rid in (1, 2):
        kv.ingest_request(rid, None)
    free = kv._arena.num_free
    first, second = _kv_step(1, 2 * PAGE), _kv_step(2, 2 * PAGE)
    assert kv.admit(first, _ctx(1)).ok
    assert kv.admit(second, _ctx(2)).ok
    kv.plan(first, _ctx(1))
    kv.commit(first, _ctx(1))

    kv.rollback_admit(second, _ctx(2))

    assert kv._streams[2]["main"].page_indices == []
    assert kv._arena.num_free == free - 2, "only row 1's committed pages stay held"


# --- the unbatched fallback admits all or nothing -----------------------------


class _Sub:
    """One segment per row of ``input_seq_len`` tokens, on every resource."""

    def __init__(self, keys):
        self._keys = keys

    def declare_step(self, graph_walk, request_ids, inputs, **kwargs):
        del graph_walk, kwargs
        segments = tuple(
            Segment(rid, "main", inp.input_seq_len)
            for rid, inp in zip(request_ids, inputs, strict=True)
        )
        return SubmoduleStep(steps={
            key: (KVStep if key == "kv" else RecurrentStep)(segments=segments)
            for key in self._keys
        })


def _engine(resources: dict) -> Engine:
    eng = Engine.__new__(Engine)
    eng._runner = StepRunner(resources)
    eng._enable_nvtx = False
    eng._enable_profile = False
    eng._device = torch.device("cpu")
    eng._tp_follower_nodes = set()
    eng._submodules = {"n": SubmoduleManagement(
        submodule=_Sub(list(resources)), forward=None, forward_batched=None,
        joint_comm_group=None, resources=resources,
    )}
    return eng


def _per_request_batch(rids, span: int) -> ExecutingBatch:
    batch = ExecutingBatch(
        node_name="n",
        per_request_info={rid: SimpleNamespace() for rid in rids},
        step_context=StepContext(request_ids=tuple(rids), graph_walk="w", slot=0, capture=False),
    )
    batch.inputs = [SimpleNamespace(input_seq_len=span)] * len(rids)
    return batch


def test_a_per_request_refusal_unwinds_the_rows_already_admitted():
    kv = _kv("kv", pages=6)  # 5 usable: two rows fit, the third does not
    eng = _engine({"kv": kv})
    for rid in (1, 2, 3):
        kv.ingest_request(rid, None)
    free = kv._arena.num_free

    batch = _per_request_batch([1, 2, 3], 2 * PAGE)
    eng._exec_per_request(batch)

    assert isinstance(batch.admit_error, AllocationFailed)
    assert kv._arena.num_free == free, (
        f"{free - kv._arena.num_free} pages kept for a step that never ran"
    )
    for rid in (1, 2, 3):
        stream = kv._streams[rid]["main"]
        assert stream.page_indices == [] and not stream.step_in_flight, rid


def test_a_per_request_refusal_unwinds_every_resource_of_the_earlier_rows():
    kv = _kv("kv", pages=4)  # 3 usable: the first row fits, the second does not
    pool = _pool(max_slots=8)
    eng = _engine({"gdn_state": pool, "kv": kv})
    for rid in (1, 2):
        kv.ingest_request(rid, None)
        pool.ingest_request(rid)
    slots = pool.num_free_slots

    batch = _per_request_batch([1, 2], 2 * PAGE)
    eng._exec_per_request(batch)

    assert batch.admit_error is not None
    assert pool.num_free_slots == slots, "row 1 kept its recurrent slot"
    assert pool._slots[1] == {} and pool._slots[2] == {}


# --- the recurrent pool gives back the slots a refused step took --------------


def _forking_step(rids) -> RecurrentStep:
    return RecurrentStep(
        segments=tuple(Segment(r, "main", 4) for r in rids),
        pre_forks=(("main", "fork"),),
    )


def test_a_refused_recurrent_admit_returns_the_slots_it_took():
    pool = _pool(max_slots=3)  # the sink plus 2 usable
    for r in ("a", "b"):
        pool.ingest_request(r)
    free = pool.num_free_slots
    step, ctx = _forking_step(["a", "b"]), _ctx("a", "b")

    assert not pool.admit(step, ctx).ok
    pool.rollback_admit(step, ctx)

    assert pool.num_free_slots == free


def test_each_forking_request_fits_alone_after_a_refused_pair():
    """Otherwise each holds the main slot the other's fork needs, every
    smaller retry refuses too, and the pool has no eviction to break it."""
    pool = _pool(max_slots=3)
    for r in ("a", "b"):
        pool.ingest_request(r)
    pair, ctx = _forking_step(["a", "b"]), _ctx("a", "b")
    pool.admit(pair, ctx)
    pool.rollback_admit(pair, ctx)

    assert pool.admit(_forking_step(["a"]), _ctx("a")).ok


def test_a_recurrent_rollback_keeps_the_slots_a_request_already_held():
    pool = _pool(max_slots=3)  # 2 usable: main fits, main plus fork for two rows do not
    for r in ("a", "b"):
        pool.ingest_request(r)
    prefill = RecurrentStep(segments=(Segment("a", "main", 4),))
    assert pool.admit(prefill, _ctx("a")).ok
    pool.plan(prefill, _ctx("a"))
    pool.commit(prefill, _ctx("a"))
    held = pool._slots["a"]["main"]

    pair, ctx = _forking_step(["a", "b"]), _ctx("a", "b")
    assert not pool.admit(pair, ctx).ok
    pool.rollback_admit(pair, ctx)

    assert pool._slots["a"] == {"main": held} and held.has_state
    assert pool._slots["b"] == {}
    assert pool.num_free_slots == 1
