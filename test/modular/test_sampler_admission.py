"""The sampler's in-graph rows are admitted by the runner, as a recurrent state slot is.

With the in-graph scatter the offset and last-token masters are allocated once
(`sampler_slots` rows), since captured graphs keep their addresses. A request
takes its row when the runner admits it, waits while none is free, and takes a
removed request's row afterwards; a deployment needs no cap for them.

CPU-only: the buffers' gather/scatter run on CPU views.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch

from mstar.engine.resources import StepRunner
from mstar.engine.resources.sampler.config import SamplerStep
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.step import ASKS, SlotLease, StepContext

CPU = torch.device("cpu")
NODE, WALK = "llm", "decode"


def _resource(monkeypatch, slots: int, max_bs: int, comm_group=None, ingraph: bool = True) -> SamplerResource:
    monkeypatch.setenv("MSTAR_SAMPLER_INGRAPH_SCATTER", "1" if ingraph else "0")
    monkeypatch.setenv("MSTAR_SAMPLER_SLOTS", str(slots))
    monkeypatch.setenv("MSTAR_DEVICE_LOOPBACK", "1")
    res = SamplerResource(vocab_size=None, enable_repetion_penalty=False, device=CPU, comm_group=comm_group)
    res.enable_device_loopback_reader()
    res.build_accelerator_graph_buffers([SimpleNamespace(slot=s) for s in (0, 1)], max_bs, 1)
    return res


def _waiting(rid, node_name):
    return None


def test_a_request_waits_for_a_row_and_takes_the_one_a_removed_request_frees(monkeypatch):
    res = _resource(monkeypatch, slots=2, max_bs=1)
    bufs = res._cg_buffers
    offset, last_token = bufs.offset.master, bufs.last_token.master
    runner = StepRunner({"sampler": res}, node_resources={NODE: ["sampler"]})
    for rid in ("a", "b", "c"):
        res.ingest_request(rid)
    assert not bufs.has_slot("a"), "ingest took a row before admission"

    assert runner.admit_retrieve("a", NODE, WALK).ready
    assert runner.admit_retrieve("b", NODE, WALK).ready, "the second row, past the grown masters, wasn't free"
    assert runner.admit_retrieve("c", NODE, WALK).queued
    assert runner.admit_waiting(_waiting) == [] and not bufs.has_slot("c")

    freed = bufs._rid_to_slot["a"]
    runner.remove_request("a")
    assert runner.admit_waiting(_waiting) == [(NODE, "c")]
    assert bufs._rid_to_slot["c"] == freed and bufs.free_rows == 0
    assert bufs.offset.master is offset and bufs.last_token.master is last_token


def test_a_row_taken_for_a_request_another_resource_holds_back_is_given_back(monkeypatch):
    res = _resource(monkeypatch, slots=2, max_bs=2)
    res.ingest_request("a")
    assert res.reserve("a", NODE, WALK, None) is None and res._cg_buffers.has_slot("a")
    res.unreserve("a")
    assert not res._cg_buffers.has_slot("a") and res._cg_buffers.free_rows == 2
    assert res.ask("a", NODE, WALK) is ASKS


def test_without_fixed_rows_the_sampler_admits_every_request(monkeypatch):
    # knob off, and above one rank, where no resource decides admission
    for kwargs in (dict(ingraph=False), dict(comm_group=SimpleNamespace(world_size=2))):
        res = _resource(monkeypatch, slots=2, max_bs=1, **kwargs)
        bufs = res._cg_buffers
        assert not bufs.ingraph_scatter
        for rid in ("a", "b", "c"):
            res.ingest_request(rid)
            assert res.ask(rid, NODE, WALK) is None and bufs.has_slot(rid), kwargs
        assert bufs._master_capacity >= 3, "the growing masters didn't grow"


def test_a_dropped_rows_offset_stays_uncommitted_through_plan_and_commit(monkeypatch):
    """``SamplerStep.kept_rids`` on a captured step with the in-graph scatter: plan stages
    the dropped row's scatter index at the trash row, so the replay leaves its offset."""
    res = _resource(monkeypatch, slots=8, max_bs=4)
    runner = StepRunner({"sampler": res}, node_resources={NODE: ["sampler"]})
    for rid in (1, 2):
        res.ingest_request(rid)
        assert runner.admit_retrieve(rid, NODE, WALK).ready
    bufs = res._cg_buffers
    slots = [bufs._rid_to_slot[rid] for rid in (1, 2)]

    def step(kept):
        sampler_step = SamplerStep(kept_rids=kept)
        ctx = StepContext(
            request_ids=(1, 2), graph_walk=WALK, slot=0, capture=False,
            slot_lease=SlotLease(slot=1, bucket=SimpleNamespace(bs=4)),
        )
        res.plan(sampler_step, ctx)
        # what the captured sample does around its draw
        s = res._cg_sampler
        s.gather_in_graph()
        s.offset_buf += 1
        s.scatter_in_graph(torch.tensor([5, 6, 0, 0]))
        res.commit(sampler_step, ctx)

    step(frozenset({1}))
    assert bufs.offset.master[slots].tolist() == [1, 0], "the dropped row's offset was committed"
    step(None)
    assert bufs.offset.master[slots].tolist() == [2, 1]
