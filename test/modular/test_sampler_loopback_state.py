"""Declared loop-back state on the sampler (``SamplerSpec.loopback_state``):
kept by slot beside the last token, written per step by the forward,
scattered by commit, gathered by the next step; grown with the masters and
reset when a slot is reused; a walk's graph reads staged off the masters
into fixed rows before a leased step. Also ``overwrite_rows``, the routed-row
override the loop-back readers share."""
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
        vocab_size=None, enable_repetition_penalty=False, device=torch.device("cpu"),
        loopback_state={"rules": STATE},
    )
    res.enable_device_loopback_reader()
    res.build_accelerator_graph_buffers([SimpleNamespace(slot=0), SimpleNamespace(slot=1)], 4, 0)
    return res


def _ctx(rids, padded_bs, slot):
    return SimpleNamespace(
        request_ids=list(rids), padded_request_ids=list(rids) + ["pad"] * (padded_bs - len(rids)),
        slot_lease=SlotLease(slot=slot, bucket=BucketKey(graph_walk="decode", bs=padded_bs, num_tokens=padded_bs)),
        graph_walk="decode", is_preplan=False, plan_phase="res.plan", slot=slot,
        is_padding_row=lambda rid: rid == "pad", chunk_tokens_after={}, rebuild_rids=frozenset(), lean=frozenset(),
    )


def test_resource_stores_by_step_and_commit_persists_what_the_step_declares(monkeypatch):
    res = _resource(monkeypatch)
    for rid in ("a", "b"):
        res.ingest_request(rid)
        if res.ask(rid, "decoder", "decode") is not None:
            # with the in-graph scatter the row is taken at admission
            assert res.reserve(rid, "decoder", "decode", None) is None
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


def _growing(monkeypatch):
    # the masters grow as above one rank (no in-graph scatter, so no row is taken at admission)
    monkeypatch.setenv("MSTAR_SAMPLER_INGRAPH_SCATTER", "0")
    res = _resource(monkeypatch)
    res.bind_graph_reads("decode", frozenset({"rules"}))
    assert not res._cg_buffers.ingraph_scatter
    return res


def test_a_leased_step_reads_its_graph_reads_from_fixed_rows_while_the_masters_grow(monkeypatch):
    """What a capture of the walk reads is the slot's staged rows, at one address: staged off the masters on
    each step, so a master that grew into a new tensor between steps still reaches the graph."""
    res = _growing(monkeypatch)
    for rid in ("a", "b"):
        res.ingest_request(rid)
    res.store_loopback_state("rules", ["a", "b"], torch.tensor([[1] * 3, [2] * 3]))
    ctx = _ctx(["b", "a"], 4, 1)
    res.plan(SamplerStep(), ctx)
    first = res.loopback_state("rules", ctx.request_ids)
    staged = res._cg_buffers.loopback["rules"].read_view(1, 4)
    assert first.data_ptr() == staged.data_ptr()
    assert first.tolist()[:2] == [[2] * 3, [1] * 3]
    master = res._cg_buffers.loopback["rules"].master
    for i in range(12):  # past the capacity of 4, twice
        res.ingest_request(f"r{i}")
    # an eager step's write, by request
    res._cg_buffers.write_loopback("rules", ["a", "b"], torch.tensor([[3] * 3, [4] * 3]))
    assert res._cg_buffers.loopback["rules"].master is not master
    res.plan(SamplerStep(), ctx)
    again = res.loopback_state("rules", ctx.request_ids)
    assert again.data_ptr() == staged.data_ptr()
    assert again.tolist()[:2] == [[4] * 3, [3] * 3]
    # another walk's leased step stages nothing and reads the masters as before
    other = SimpleNamespace(**{**vars(ctx), "graph_walk": "prefill"})
    res.plan(SamplerStep(), other)
    assert res.loopback_state("rules", ctx.request_ids).data_ptr() != staged.data_ptr()


def test_a_routed_row_reads_its_state_from_its_slot(monkeypatch):
    """A row's first step after its prompt brings its token routed, which overrides the token read; its graph
    reads come staged from its slot, where the prompt step stored them, untouched by that override."""
    res = _growing(monkeypatch)
    for rid in ("a", "b"):
        res.ingest_request(rid)
    # the prompt step's store (eager), then a leased step whose rows are all routed
    res.store_loopback_state("rules", ["a", "b"], torch.tensor([[1] * 3, [2] * 3]))
    ctx = _ctx(["a", "b"], 4, 0)
    res.plan(SamplerStep(), ctx)
    tokens = res.loopback_tokens(ctx.request_ids)
    res.overwrite_rows([0, 1], (tokens, torch.tensor([11, 12])))
    assert tokens.tolist()[:2] == [11, 12]
    assert res.loopback_state("rules", ctx.request_ids).tolist()[:2] == [[1] * 3, [2] * 3]



def test_the_debug_carry_digest_counts_stored_and_staged_reads_that_differ_from_rank_0(monkeypatch, caplog):
    """Under ``MSTAR_KV_DEBUG_ASSERTS`` above one rank, a step that stores a drafter's carry compares, in one
    broadcast, the stored carry's digest and (on a leased step that staged it) the staged read's with rank 0's,
    counting the steps that differ on the device; a step storing none is skipped. The tally is logged whole once
    the sampler holds no request."""
    import logging

    from mstar.engine.resources.base import tensor_digest
    from mstar.engine.resources.draft.drafter import DRAFT_CARRY
    from mstar.engine.resources.kv import manager as kv_manager

    monkeypatch.setattr(kv_manager, "_DEBUG_ASSERTS", True)
    monkeypatch.setenv("MSTAR_DEVICE_LOOPBACK", "1")
    monkeypatch.setenv("MSTAR_SAMPLER_INGRAPH_SCATTER", "0")
    lead: dict[str, torch.Tensor] = {}
    group = SimpleNamespace(world_size=2, rank=1, broadcast=lambda t, src=0: lead.get("digest", t))
    res = SamplerResource(
        vocab_size=None, enable_repetition_penalty=False, device=torch.device("cpu"), comm_group=group,
        loopback_state={DRAFT_CARRY: STATE},
    )
    res.enable_device_loopback_reader()
    res.build_accelerator_graph_buffers([SimpleNamespace(slot=0), SimpleNamespace(slot=1)], 4, 0)
    res.bind_graph_reads("decode", frozenset({DRAFT_CARRY}))
    res.ingest_request("a")
    res.store_loopback_state(DRAFT_CARRY, ["a"], torch.tensor([[1, 2, 3]]))
    stores = SamplerStep(loopback_writes=frozenset({DRAFT_CARRY}))
    eager = SimpleNamespace(request_ids=["a"])
    res.commit(stores, eager)
    assert res._carry_tally.tolist() == [1, 0, 0, 0]
    lead["digest"] = torch.zeros(4, dtype=torch.long)
    res.commit(stores, eager)
    assert res._carry_tally.tolist() == [2, 1, 0, 0]
    res.commit(SamplerStep(), eager)
    assert res._carry_tally.tolist() == [2, 1, 0, 0]
    # a leased verify step: its staged read is compared too, against rank 0's
    ctx = _ctx(["a"], 4, 1)
    res.plan(stores, ctx)
    staged = res._cg_buffers.loopback[DRAFT_CARRY].read_view(1, 4)[:1]
    res.store_loopback_state(DRAFT_CARRY, ctx.request_ids, torch.tensor([[4, 5, 6]] * 4))
    stored = res._cg_buffers.loopback[DRAFT_CARRY].slot_view(1, 4)[:1]
    lead["digest"] = torch.cat([tensor_digest(stored), tensor_digest(staged)])
    res.commit(stores, ctx)
    assert res._carry_tally.tolist() == [3, 1, 1, 0]
    res.plan(stores, ctx)
    res.store_loopback_state(DRAFT_CARRY, ctx.request_ids, torch.tensor([[4, 5, 6]] * 4))
    lead["digest"] = torch.cat([tensor_digest(stored), torch.zeros(2, dtype=torch.long)])
    res.commit(stores, ctx)
    assert res._carry_tally.tolist() == [4, 1, 2, 1]
    with caplog.at_level(logging.INFO, logger="mstar.engine.resources.sampler.resource"):
        res.remove_request("a")
    assert "stored carry differs on 1 of 4 verify and prompt steps, staged read on 1 of 2" in caplog.text
