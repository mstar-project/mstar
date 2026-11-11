"""A host field changes what a model sends and is kept between steps, so the engine counts each write exactly once.

A pre-plan declares a step that exec may declare again, and a step that fails admission declares a step that never runs.
Writes apply where a step commits, so neither of those counts. The store is keyed by the worker's recycled handles, so a
field a removal left behind would attach to whichever request gets that handle next.
"""

from __future__ import annotations

import re
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.engine import Engine
from mstar.engine.resources.step import Segment, StepContext, SubmoduleStep
from mstar.engine.state import Array, ArrayStep, Attend, Causal, Entry, Fork, KVEntry, Read, Value, host
from mstar.engine.state.adapters import (
    HOST_VALUE,
    StateSetup,
    _resolve,
    build_handle,
    build_specs,
    kv_layouts,
    node_handles,
    state_steps,
    storage_of,
)
from mstar.engine.state.analysis import host_field_holders
from mstar.engine.state.calls import ArrayHandle
from mstar.engine.state.host_fields import HostFields, is_host_value
from mstar.testing_utils.engine import make_engine, make_mgmt
from mstar.testing_utils.model import make_model

RID = 3


def _flags(**host_types) -> Array:
    return Value("flags", nodes={"talker"}, layers={0, 1}, entry=Entry({}, host=host_types or {"sent": bool}))


def _handle(array=None, *requests) -> ArrayHandle:
    array = array or _flags()
    fields = HostFields()
    model = SimpleNamespace(declare_state=lambda: [array])
    handle = node_handles(model, "talker", {}, setup=StateSetup(host_fields=fields))[array.name]
    for request in requests or (RID,):
        fields.open(request)
    return handle


def _step(handle, ctx_rows=(RID,), replays=False, **declared) -> SubmoduleStep:
    segments = (Segment(RID, "main", 1),)
    step = SubmoduleStep(segments=segments, state={handle.name: ArrayStep(**declared)})
    expanded = state_steps(step, {handle.name: handle})
    ctx = StepContext(request_ids=list(ctx_rows), graph_walk="decode", slot=0, capture=False)
    if replays:
        ctx.slot_lease = SimpleNamespace()
    expanded.set_ctx(ctx)
    return expanded


def _run(step) -> None:
    step.begin_calls()
    step.commit_calls()
    step.end_calls()


def test_a_value_of_host_fields_builds_no_spec_and_no_step():
    assert build_specs([_flags()]) == []
    step = _step(_handle(), host={(RID, "main", 0): {"sent": True}})
    assert step.steps == {}, "no resource holds a host field"


def test_a_value_of_host_fields_only_has_its_own_storage_whose_handle_refuses_a_read_by_name():
    assert storage_of(_flags()) is HOST_VALUE, "a Value with no tensor has no pool to hold"
    step = _step(_handle())
    assert step.steps == {} and len(step.state_handles) == 1, "the step runs, to commit the writes, with no resource"
    with pytest.raises(ValueError, match="Array 'flags': no layout yet: read on a HostValueHandle"):
        step.state_handles[0].read(0, torch.zeros(1))


def test_a_value_of_host_fields_has_no_layout():
    assert kv_layouts([_flags()], {"flags": {}}, {}) == {}


@pytest.mark.parametrize("kind, zero", [
    (bool, False), (int, 0), (float, 0.0), (str, ""), (bytes, b""), (tuple[int, ...], ()),
])
def test_each_field_starts_at_its_types_zero(kind, zero):
    h = _handle(_flags(field=kind))
    assert host(h, RID, "main", 0)["field"] == zero and type(host(h, RID, "main", 0)["field"]) is type(zero)


def test_a_write_applies_at_commit_and_not_before():
    h = _handle()
    step = _step(h, host={(RID, "main", 0): {"sent": True}})
    assert not host(h, RID, "main", 0)["sent"], "declaring must not write, since a pre-plan declares too"
    step.begin_calls()
    assert not host(h, RID, "main", 0)["sent"], "opening must not write either"
    step.commit_calls()
    step.end_calls()
    assert host(h, RID, "main", 0)["sent"]
    assert not host(h, RID, "main", 1)["sent"], "a field is per layer"


def test_a_step_declared_twice_applies_only_the_declaration_that_runs():
    h = _handle(_flags(count=int))
    _step(h, host={(RID, "main", 0): {"count": 1}})
    second = _step(h, host={(RID, "main", 0): {"count": 2}})
    assert host(h, RID, "main", 0)["count"] == 0, "declaring writes nothing"
    _run(second)
    assert host(h, RID, "main", 0)["count"] == 2


def _engine_run(h, step, forward):
    """``Engine._forward_and_commit`` over ``step``, with its forward and its resources stubbed."""
    engine = Engine.__new__(Engine)
    engine._enable_nvtx, engine._enable_profile, engine._foreign_planners = False, False, {}
    order = []
    engine._runner = SimpleNamespace(commit=lambda _: order.append(host(h, RID, "main", 0)["sent"]))
    engine._forward = forward
    mgmt = SimpleNamespace(
        accelerator_graph_runner=None, resources={}, piecewise_runners={}, fp32_precision=None, resident=None,
        submodule=SimpleNamespace(preprocess=lambda *a, **k: {}),
    )
    batch = SimpleNamespace(
        per_request_input_metadata=None, launch_started_event=None, committed=None, preplan_event=None,
        exec_timings=SimpleNamespace(fwd_start=0.0), node_name="Talker",
    )
    engine._forward_and_commit(batch, mgmt, [RID], [], {}, step.ctx, None, False, step, False)
    return order


def test_the_engine_applies_a_steps_writes_after_its_resources_commit():
    h = _handle()
    step = _step(h, host={(RID, "main", 0): {"sent": True}})
    assert _engine_run(h, step, lambda *a, **k: {}) == [False], "the resources commit first"
    assert host(h, RID, "main", 0)["sent"]


def test_the_engine_applies_nothing_when_a_forward_raises():
    def forward(*a, **k):
        raise RuntimeError("the forward failed")

    h = _handle()
    with pytest.raises(RuntimeError, match="the forward failed"):
        _engine_run(h, _step(h, host={(RID, "main", 0): {"sent": True}}), forward)
    assert not host(h, RID, "main", 0)["sent"]


def test_a_step_that_never_commits_applies_nothing():
    h = _handle()
    step = _step(h, host={(RID, "main", 0): {"sent": True}})
    step.begin_calls()
    step.end_calls()
    assert not host(h, RID, "main", 0)["sent"]


@pytest.mark.parametrize("walk, combined", [("decode", {}), ("steps", {"talker": {"steps": {"prefill", "decode"}}})],
                         ids=["walk", "combined-walk"])
def test_a_node_holding_host_fields_that_reports_step_is_input_free_is_refused_at_load(walk, combined):
    """For an input-free walk, exec runs the step a pre-plan declared over the bucket's template rows, so its writes
    would come from the template and not from the requests."""
    graph = SimpleNamespace(get_nodes=lambda: {"talker": None})
    model = make_model(
        declare_state=lambda: [_flags()], get_graph_walk_graphs=lambda: {"prefill": graph, "decode": graph},
        get_combined_graph_walks=lambda: combined,
    )
    talker = SimpleNamespace(step_is_input_free=lambda graph_walk: graph_walk == walk)
    engine = make_engine(_submodules={"talker": make_mgmt(submodule=talker)})
    with pytest.raises(ValueError, match=re.escape(
        f"'talker' holds the host fields of ['flags'] and reports step_is_input_free for [{walk!r}]",
    )):
        engine._check_host_fields(model)


def test_the_holders_of_host_fields_are_the_arrays_whose_entry_declares_them():
    talker = Value("talker_step", nodes={"Talker"}, layers=frozenset({0}), entry=Entry({}, host={"n": int}))
    kv = Array("kv", nodes={"Talker"}, layers=frozenset({0}), entry=KVEntry(heads=1, head_dim=2), max_len=8,
               reads={"attn": Read(mask=Causal(), op=Attend(q_heads=1))})
    assert host_field_holders([kv, talker]) == [talker], "an array without host fields would be checked as a holder"


def test_a_replay_applies_its_writes():
    h = _handle()
    _run(_step(h, replays=True, host={(RID, "main", 0): {"sent": True}}))
    assert host(h, RID, "main", 0)["sent"], "a captured step's rows still declare and commit"


def test_a_replay_still_refuses_a_bad_write():
    h = _handle()
    with pytest.raises(ValueError, match="field 'nope' isn't declared"):
        _step(h, replays=True, host={(RID, "main", 0): {"nope": True}}).begin_calls()


def test_a_layer_goes_through_the_normalizer_when_read():
    h = _handle()
    assert host(h, RID, "main", torch.tensor(1)) == {"sent": False}
    with pytest.raises(ValueError, match="a layer tensor must be 0-d int64 on the CPU"):
        host(h, RID, "main", torch.tensor([1]))


def test_a_read_of_an_entry_that_doesnt_exist_is_refused_by_name():
    h = _handle()
    with pytest.raises(ValueError, match="Array 'flags': no host fields for request 3, label 'other', layer 0"):
        host(h, RID, "other", 0)


def test_a_field_read_cant_be_written_through():
    with pytest.raises(TypeError):
        host(_handle(), RID, "main", 0)["sent"] = True


@pytest.mark.parametrize("declared, cause", [
    ({"keep": False, "host": {(RID, "main", 0): {"sent": True}}}, "a step that keeps nothing"),
    ({"host": {(RID, "main", 0): {"other": True}}}, "field 'other' isn't declared"),
    ({"host": {(RID, "main", 0): {"sent": 1}}}, "field 'sent' takes"),
    ({"host": {(RID, "cfg", 0): {"sent": True}}}, "the array declares labels"),
    ({"host": {(RID, "main", 7): {"sent": True}}}, "the array declares labels"),
    ({"host": {(99, "main", 0): {"sent": True}}}, "isn't one of the step's requests"),
])
def test_a_bad_write_is_refused_naming_the_array_and_the_cause(declared, cause):
    h = _handle()
    with pytest.raises(ValueError, match=f"Array 'flags': .*{cause}"):
        _step(h, ctx_rows=(RID,), **declared).begin_calls()


def test_a_write_on_a_padding_row_is_refused():
    h = _handle()
    step = _step(h, host={(-1, "main", 0): {"sent": True}})
    step.ctx.set_padded_rids([RID, -1])
    with pytest.raises(ValueError, match="row -1, .*isn't one of the step's requests"):
        step.begin_calls()


def test_a_write_for_a_request_with_no_open_fields_is_refused():
    h = _handle()
    h.host_fields.remove_request(RID)
    with pytest.raises(ValueError, match="that request has no open fields"):
        _step(h, host={(RID, "main", 0): {"sent": True}}).begin_calls()


def test_a_write_on_an_array_that_holds_no_host_fields_is_refused():
    array = Array("kv", nodes={"talker"}, layers={0}, entry=KVEntry(heads=2, head_dim=4), max_len=8,
                  reads={"attn": Read(mask=Causal(), op=Attend(q_heads=2))})
    resources = {"kv": SimpleNamespace(layer_view=lambda i: None), "attn": SimpleNamespace()}
    h = build_handle(array, resources, _resolve([array], StateSetup()))
    declared = ArrayStep(host={(RID, "main", 0): {"sent": True}})
    step = SubmoduleStep(segments=(Segment(RID, "main", 1),), state={"kv": declared})
    step.set_ctx(StepContext(request_ids=[RID], graph_walk="decode", slot=0, capture=False))
    with pytest.raises(ValueError, match="Array 'kv': a host write, and the array declares no host fields"):
        h.begin_step(step)


def _forked(at, **written) -> ArrayHandle:
    array = Array("flags", nodes={"talker"}, layers={0}, entry=Entry({}, host={"n": int}), max_len=1, rewritable=True,
                  labels={"main", "cfg"})
    h = _handle(array)
    _run(_step(h, host={(RID, "main", 0): {"n": 1}}))
    _run(_step(h, forks=(Fork(src="main", dst="cfg", at=at),), **written))
    return h


def test_a_fork_copies_the_fields():
    assert host(_forked("end"), RID, "cfg", 0)["n"] == 1


def test_a_fork_at_the_start_comes_before_the_steps_writes_and_one_at_the_end_after():
    written = {"host": {(RID, "main", 0): {"n": 2}}}
    assert host(_forked("start", **written), RID, "cfg", 0)["n"] == 1
    assert host(_forked("end", **written), RID, "cfg", 0)["n"] == 2


def test_the_store_is_empty_after_a_removal():
    fields = HostFields()
    fields.declare(_flags())
    fields.open(RID)
    assert fields.is_open(RID)
    fields.remove_request(RID)
    assert not fields._fields, "a leaked handle would attach to the next request that gets it"


def test_a_removed_handle_reopens_at_zero():
    h = _handle()
    _run(_step(h, host={(RID, "main", 0): {"sent": True}}))
    h.host_fields.remove_request(RID)
    h.host_fields.open(RID)
    assert not host(h, RID, "main", 0)["sent"], "the next request on this handle starts clean"


def test_an_engine_with_no_host_fields_keeps_nothing_per_request():
    fields = HostFields()
    fields.open(RID)
    assert not fields._fields


@pytest.mark.parametrize("value, kind, matches", [
    (True, bool, True), (1, bool, False), (True, int, False), (1.0, float, True), (1, float, False),
    ((1, 2), tuple[int, ...], True), ((1, "a"), tuple[int, ...], False), ([1], tuple[int, ...], False),
    ((1, "a"), tuple[int, str], True), ((1,), tuple[int, str], False),
])
def test_a_field_takes_exactly_its_declared_type(value, kind, matches):
    assert is_host_value(value, kind) is matches
