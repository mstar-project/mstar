"""Lazy publication (MSTAR_LAZY_PUBLISH): finalize_batch snapshots a few scalars
per request and the worker finishes the publication only for the requests whose
frame or completion carries it. The finished publication has to equal what the
eager publish exported at snapshot time, even when the next step committed in
between (the next step is already running when the worker gets to it)."""
from __future__ import annotations

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine import engine as engine_mod
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import StepRunner
from mstar.engine.resources.base import Resource
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVStep, PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.position.manager import (
    PublishedPositionInfo,
    RopeManager,
)
from mstar.engine.resources.step import ADMIT_OK, Segment, StepContext

OWN_HANDLE = ("own", 0)


class _StubTransfer:
    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        # what the real engines stamp: the lengths the caller handed over
        return (OWN_HANDLE, kwargs.get("seq_len"), tuple(kwargs.get("page_indices") or ()))

    def start_async_retrieve(self, **kwargs):
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del kwargs
        return transfer_info[0] == OWN_HANDLE

    def cleanup(self):
        pass

    def remove_request(self, request_id):
        del request_id


@pytest.fixture(autouse=True)
def _stub(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _manager(max_num_pages=64, page_size=16) -> KVManager:
    return KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
            max_num_pages=max_num_pages, page_size=page_size,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )


def _ctx(*rids):
    return StepContext(request_ids=tuple(rids), graph_walk="w", slot=0, capture=False)


def _grow(kv, rid, span, label="main"):
    step = KVStep(segments=(Segment(rid, label, span),))
    ctx = _ctx(rid)
    assert kv.admit(step, ctx).ok
    kv.commit(step, ctx)


def _seq(published, label="main"):
    return published.get(0)[label]


# -- KV ----------------------------------------------------------------------


def test_kv_snapshot_finishes_to_the_eager_publication():
    kv = _manager()
    kv.ingest_request("r0", None)
    _grow(kv, "r0", 40)
    eager = kv.publish("r0", "node", "w")
    snap = kv.publish_snapshot_for_step("r0", "node", "w")
    lazy = kv.publish_from_snapshot("r0", snap, "node", "w")
    assert _seq(lazy).seq_len == _seq(eager).seq_len == 40
    assert _seq(lazy).page_indices == _seq(eager).page_indices
    assert _seq(lazy).reset_generation == _seq(eager).reset_generation
    assert _seq(lazy).latest_kv_transfer_info == _seq(eager).latest_kv_transfer_info
    assert lazy.world_size == eager.world_size


def test_kv_snapshot_keeps_this_steps_lengths_when_the_next_step_committed():
    kv = _manager(page_size=16)
    kv.ingest_request("r0", None)
    _grow(kv, "r0", 31)
    expected = kv.publish("r0", "node", "w")
    snap = kv.publish_snapshot_for_step("r0", "node", "w")
    # the next step lands (crosses a page boundary: a third page appears)
    _grow(kv, "r0", 1)
    _grow(kv, "r0", 1)
    later = kv.publish("r0", "node", "w")
    assert _seq(later).seq_len == 33 and len(_seq(later).page_indices) == 3
    lazy = kv.publish_from_snapshot("r0", snap, "node", "w")
    assert _seq(lazy).seq_len == _seq(expected).seq_len == 31
    assert _seq(lazy).page_indices == _seq(expected).page_indices
    assert len(_seq(lazy).page_indices) == 2
    # the transfer descriptor is stamped with the snapshot's lengths too
    assert _seq(lazy).latest_kv_transfer_info == _seq(expected).latest_kv_transfer_info


def test_kv_snapshot_is_a_copy_not_a_view_of_the_page_list():
    kv = _manager()
    kv.ingest_request("r0", None)
    _grow(kv, "r0", 20)
    snap = kv.publish_snapshot_for_step("r0", "node", "w")
    lazy = kv.publish_from_snapshot("r0", snap, "node", "w")
    pages = list(_seq(lazy).page_indices)
    _grow(kv, "r0", 40)
    assert _seq(lazy).page_indices == pages


def test_kv_snapshot_is_none_for_an_unknown_request_and_mirrors_an_empty_stream():
    kv = _manager()
    assert kv.publish_snapshot_for_step("nobody", "node", "w") is None
    assert kv.publish_from_snapshot("r0", None, "node", "w") is None
    kv.ingest_request("r0", None)
    # ingest opens the stream at length 0; eager publish exports that, and so does the snapshot
    eager = kv.publish("r0", "node", "w")
    snap = kv.publish_snapshot_for_step("r0", "node", "w")
    lazy = kv.publish_from_snapshot("r0", snap, "node", "w")
    assert _seq(lazy).seq_len == _seq(eager).seq_len == 0
    assert _seq(lazy).page_indices == _seq(eager).page_indices == []


# -- positions -----------------------------------------------------------------


def test_position_snapshot_matches_publish_and_does_not_track_later_advances():
    pm = RopeManager.__new__(RopeManager)
    pm._counters = {"r0": {"main": 17}}
    eager = pm.publish("r0")
    snap = pm.publish_snapshot_for_step("r0", "node", "w")
    pm._counters["r0"]["main"] = 18
    lazy = pm.publish_from_snapshot("r0", snap, "node", "w")
    assert isinstance(lazy, PublishedPositionInfo)
    assert lazy.counters == eager.counters == {"main": 17}
    assert pm.publish_snapshot_for_step("r1", "node", "w") is None
    assert pm.publish_from_snapshot("r1", None, "node", "w") is None


# -- runner + engine -----------------------------------------------------------


class _Pub(Resource):
    """A resource with only the default hooks: its snapshot is the eager value."""

    def __init__(self, published):
        self._published = published
        self.publish_calls = 0

    @classmethod
    def build(cls, spec, info):
        raise NotImplementedError

    def admit(self, step, ctx):
        return ADMIT_OK

    def plan(self, step, ctx):
        return None

    def publish(self, request_id):
        self.publish_calls += 1
        return None if self._published is None else f"{self._published}:{request_id}"


def test_runner_snapshot_and_finish_equal_publish_and_skip_the_silent():
    kv, sampler = _Pub("kv-info"), _Pub(None)
    runner = StepRunner({"kv": kv, "sampler": sampler})
    eager = runner.publish(["r1", "r2"])
    snaps = runner.publish_snapshot(["r1", "r2"])
    assert set(snaps) == {"r1", "r2"} and set(snaps["r1"]) == {"kv"}
    assert runner.publish_from_snapshots(snaps, ["r2"]) == {"r2": eager["r2"]}
    assert runner.publish_from_snapshots(snaps, ["r9"]) == {}


def _engine_with(runner):
    eng = Engine.__new__(Engine)
    eng._runner = runner
    eng._enable_nvtx = False
    return eng


def _batch(rids):
    return ExecutingBatch(
        node_name="LLM",
        per_request_info={
            rid: CurrentForwardPassInfo(
                request_id=str(rid), graph_walk="decode", fwd_index=0,
                random_seed=0, max_tokens=16,
            )
            for rid in rids
        },
        step_context=StepContext(
            request_ids=tuple(rids), graph_walk="decode", slot=0, capture=False,
        ),
    )


def test_lazy_finalize_publishes_nothing_until_materialized(monkeypatch):
    monkeypatch.setattr(engine_mod, "_LAZY_PUBLISH", True)
    kv = _Pub("kv-info")
    eng = _engine_with(StepRunner({"kv": kv}))
    batch = _batch([1, 2, 3])
    out = eng.finalize_batch(batch)
    assert out == {} and batch.resource_publish_info == {}
    assert set(batch.publish_snapshots) == {1, 2, 3}
    assert all(not info.resource_publish_info for info in batch.per_request_info.values())

    eng.materialize_publish(batch, [2])
    assert batch.resource_publish_info == {2: {"kv": "kv-info:2"}}
    assert batch.per_request_info[2].resource_publish_info == {"kv": "kv-info:2"}
    assert not batch.per_request_info[1].resource_publish_info
    # idempotent, and a rid the batch never ran is ignored
    calls = kv.publish_calls
    eng.materialize_publish(batch, [2, 7])
    assert kv.publish_calls == calls
    assert batch.resource_publish_info == {2: {"kv": "kv-info:2"}}


def test_eager_finalize_still_publishes_every_request(monkeypatch):
    monkeypatch.setattr(engine_mod, "_LAZY_PUBLISH", False)
    kv = _Pub("kv-info")
    eng = _engine_with(StepRunner({"kv": kv}))
    batch = _batch([1, 2])
    out = eng.finalize_batch(batch)
    assert out == {1: {"kv": "kv-info:1"}, 2: {"kv": "kv-info:2"}}
    assert batch.per_request_info[1].resource_publish_info == {"kv": "kv-info:1"}
    assert batch.publish_snapshots is None
    eng.materialize_publish(batch, [1])  # nothing to finish, no error
