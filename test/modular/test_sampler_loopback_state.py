"""Declared loop-back state on the sampler (``SamplerSpec.loopback_state``):
kept by slot beside the last token, written per step by the forward,
scattered by commit, gathered by the next step; grown with the masters and
reset when a slot is reused. Also ``overwrite_rows``, the routed-row override
the loop-back readers share."""
from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.sampler.config import LoopbackState, SamplerStep
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.engine.resources.sampler.utils import SamplerBuffers
from mstar.engine.resources.step import BucketKey, SlotLease

STATE = LoopbackState(width=3, dtype=torch.long, default=(7, 0, -1))
DEFAULT = list(STATE.default)


def _buffers(max_bs=2, cg_slots=2):
    return SamplerBuffers.allocate(
        max_batch_size=max_bs, device=torch.device("cpu"), cg_slots=cg_slots,
        loopback_state={"rules": STATE},
    )


def test_a_default_that_does_not_fit_the_width_is_refused():
    with pytest.raises(ValueError, match="width"):
        LoopbackState(width=2, dtype=torch.long, default=(1,))


def test_eager_rows_round_trip_by_request_and_start_at_the_default():
    bufs = _buffers()
    for rid in ("a", "b"):
        bufs.register_request(rid)
    assert bufs.loopback_for("rules", ["a", "b"]).tolist() == [DEFAULT, DEFAULT]
    bufs.write_loopback("rules", ["b", "a"], torch.tensor([[1, 2, 3], [4, 5, 6]]))
    assert bufs.loopback_for("rules", ["a", "b"]).tolist() == [[4, 5, 6], [1, 2, 3]]


def test_the_per_step_rows_scatter_to_their_slots_and_gather_back():
    bufs = _buffers(max_bs=4, cg_slots=2)
    for rid in ("a", "b", "c"):
        bufs.register_request(rid)
    bufs.write_loopback("rules", ["c"], torch.tensor([[9, 9, 9]]))
    # a step of [b, a] padded to 4 on cg slot 1
    bufs.gather_static(["b", "a"], 4, 1)
    bufs.gather_dynamic(["b", "a"], 4, 1)
    assert bufs.gather_loopback("rules", 1, 4).tolist()[:2] == [DEFAULT, DEFAULT]
    # what the captured forward does: write the step's rows into the slot row
    bufs.loopback["rules"].slot_view(1, 4).copy_(torch.tensor([[2] * 3, [1] * 3, [5] * 3, [5] * 3]))
    bufs.scatter_loopback("rules", 1)
    # real rows reached their slots; another request's slot is untouched
    assert bufs.loopback_for("rules", ["a", "b", "c"]).tolist() == [[1] * 3, [2] * 3, [9] * 3]
    assert bufs.gather_loopback("rules", 1, 4).tolist()[:2] == [[2] * 3, [1] * 3]


def test_a_dropped_row_keeps_its_slot():
    bufs = _buffers(max_bs=2, cg_slots=1)
    for rid in ("a", "b"):
        bufs.register_request(rid)
    bufs.gather_static(["a", "b"], 2, 0)
    bufs.gather_dynamic(["a", "b"], 2, 0)
    bufs.loopback["rules"].slot_view(0, 2).copy_(torch.tensor([[1] * 3, [2] * 3]))
    bufs.scatter_loopback("rules", 0, keep=torch.tensor([True, False]))
    assert bufs.loopback_for("rules", ["a", "b"]).tolist() == [[1] * 3, DEFAULT]


def test_the_state_grows_with_the_masters():
    bufs = _buffers(max_bs=2, cg_slots=1)
    rids = [f"r{i}" for i in range(5)]  # past the capacity of 2, twice
    for rid in rids:
        bufs.register_request(rid)
    values = torch.arange(15).view(5, 3)
    bufs.write_loopback("rules", rids, values)
    master = bufs.loopback["rules"].master
    assert master.shape[0] == bufs._master_capacity + 1 == bufs.last_token.master.shape[0]
    assert bufs.loopback_for("rules", rids).tolist() == values.tolist()


def test_a_reused_slot_starts_again_at_the_default():
    bufs = _buffers(max_bs=2, cg_slots=1)
    bufs.register_request("a")
    bufs.write_loopback("rules", ["a"], torch.tensor([[1, 2, 3]]))
    slot = bufs._rid_to_slot["a"]
    bufs.unregister_request("a")
    bufs.register_request("b")
    assert bufs._rid_to_slot["b"] == slot
    assert bufs.loopback_for("rules", ["b"]).tolist() == [DEFAULT]
    # under a lease too
    bufs.write_loopback("rules", ["b"], torch.tensor([[4, 5, 6]]))
    bufs.unregister_request("b")
    bufs.register_request("c")
    bufs.gather_static(["c"], 1, 0)
    bufs.gather_dynamic(["c"], 1, 0)
    assert bufs.gather_loopback("rules", 0, 1).tolist() == [DEFAULT]


def _resource(monkeypatch):
    monkeypatch.setenv("MSTAR_DEVICE_LOOPBACK", "1")
    res = SamplerResource(
        vocab_size=None, enable_repetion_penalty=False, device=torch.device("cpu"),
        loopback_state={"rules": STATE},
    )
    res.enable_device_loopback_reader()
    res.build_accelerator_graph_buffers([SimpleNamespace(slot=0), SimpleNamespace(slot=1)], 4, 0)
    return res


def _ctx(rids, padded_bs, slot):
    return SimpleNamespace(
        request_ids=list(rids), padded_request_ids=list(rids) + ["pad"] * (padded_bs - len(rids)),
        slot_lease=SlotLease(slot=slot, bucket=BucketKey(graph_walk="decode", bs=padded_bs, num_tokens=padded_bs)),
        is_preplan=False, plan_phase="res.plan", slot=slot, is_padding_row=lambda rid: rid == "pad",
    )


def test_resource_stores_by_step_and_commit_persists_what_the_step_declares(monkeypatch):
    res = _resource(monkeypatch)
    for rid in ("a", "b"):
        res.ingest_request(rid)
    # an eager step reads and writes by request
    assert res.loopback_state("rules", ["a", "b"]).tolist() == [DEFAULT, DEFAULT]
    res.store_loopback_state("rules", ["a", "b"], torch.tensor([[1] * 3, [2] * 3]))
    assert res.loopback_state("rules", ["b"]).tolist() == [[2] * 3]

    # a leased step: the forward's rows go to the per-step row, commit scatters
    ctx = _ctx(["b", "a"], 4, 1)
    res.plan(SamplerStep(), ctx)
    assert res.loopback_state("rules", ctx.request_ids).tolist()[:2] == [[2] * 3, [1] * 3]
    res.store_loopback_state("rules", ctx.request_ids, torch.tensor([[5] * 3, [6] * 3, [0] * 3, [0] * 3]))
    res.commit(SamplerStep(), ctx)  # not declared: nothing persisted
    assert res._cg_buffers.loopback_for("rules", ["a", "b"]).tolist() == [[1] * 3, [2] * 3]
    res.plan(SamplerStep(), ctx)
    res.store_loopback_state("rules", ctx.request_ids, torch.tensor([[5] * 3, [6] * 3, [0] * 3, [0] * 3]))
    res.commit(SamplerStep(loopback_writes=frozenset({"rules"})), ctx)
    assert res._cg_buffers.loopback_for("rules", ["a", "b"]).tolist() == [[6] * 3, [5] * 3]


def test_overwrite_rows_puts_routed_values_over_the_read_ones(monkeypatch):
    res = _resource(monkeypatch)
    ids = torch.tensor([1, 2, 3, 4])
    rules = torch.zeros(4, 3, dtype=torch.long)
    res.overwrite_rows([1, 3], (ids, torch.tensor([20, 40], dtype=torch.int32)), (rules, torch.ones(2, 3)))
    assert ids.tolist() == [1, 20, 3, 40]
    assert rules.tolist() == [[0] * 3, [1] * 3, [0] * 3, [1] * 3]
    res.overwrite_rows([], (ids, torch.tensor([])))  # nothing routed: untouched
    assert ids.tolist() == [1, 20, 3, 40]
