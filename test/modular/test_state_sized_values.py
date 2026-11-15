"""A Value whose fields take sizes each request sets.

LingBot World's VAE keeps a causal-conv cache per stream whose slots follow the request's latent size. A Value of
fixed shapes, sized for the largest request, would hold 18 GB from boot and copy all of it every chunk, and on a
loop-back edge the caches cost 3.8 GB of copies to the host a chunk. A sized Value reserves each request's own bytes
at admission and holds the tensors its steps write as they are. Its read hands them over, so a step that loses them
is refused, and nothing else may keep them alive past the request.
"""

from __future__ import annotations

import sys
import weakref
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.communication import wire
from mstar.communication import wire_types as _wire_types  # noqa: F401
from mstar.engine.accelerator_graph_runner import blocked_capture_resources
from mstar.engine.engine import Engine
from mstar.engine.resources.recurrent.config import RecurrentStateSpec
from mstar.engine.resources.sized import SizedValueReqConfig, SizedValueSpec, SizedValueStep
from mstar.engine.resources.sized.store import SizedValueStore
from mstar.engine.resources.step import ASKS, Segment, StepContext, SubmoduleStep
from mstar.engine.state import Array, ArrayStep, Entry, Fork, KVEntry, Layout, Size, TensorType, Value
from mstar.engine.state.adapters import SIZED_VALUE, build_specs, node_handles, state_sizes, state_steps, storage_of

# a slot at twice the latent and one at the latent, as a VAE's cache holds them
SLOT, OUT = "slot", "out"
ENTRY = Entry({
    SLOT: TensorType((1, 4, 2, Size("lat_h", scale=2), Size("lat_w", scale=2)), dtype=torch.float32),
    OUT: TensorType((1, 8, 1, Size("lat_h"), Size("lat_w")), dtype=torch.bfloat16),
})
# bytes a latent pixel: 4 * 2 * 4 fp32 and 8 bf16
PIXEL = 4 * 2 * 4 * 4 + 8 * 2
LARGEST = {"lat_h": 1, "lat_w": 12}
SMALL = {"lat_h": 2, "lat_w": 3}


def _caches(**fields) -> Array:
    return Value("caches", **{"nodes": {"dit"}, "layers": {0}, "entry": ENTRY, "largest": LARGEST, **fields})


def _store(*rids, sizes=SMALL, max_bytes=None):
    """The store and the handle of `_caches`, with each of ``rids`` opened at ``sizes`` as intake opens it and
    admitted."""
    array = _caches()
    (spec,) = build_specs([array], kernel_settings={"caches": {"max_bytes": max_bytes}} if max_bytes else {})
    store = SizedValueStore(spec.config, torch.device("cpu"), name="caches")
    handle = node_handles(SimpleNamespace(declare_state=lambda: [array]), "dit", {"caches": store})["caches"]
    for rid in rids:
        _open(store, spec, rid, sizes)
        assert store.reserve(rid, "dit", "rollout", None) is None
    return handle, store, spec


def _open(store, spec, rid, sizes) -> None:
    layout = {"caches": {"main": Layout(sizes=sizes)}}
    store.ingest_request(rid, spec.request_config({"state_sizes": state_sizes([_caches()], layout)}))


def _step(handle, *rids, **declared) -> SubmoduleStep:
    step = state_steps(SubmoduleStep(segments=[Segment(rid, "main", 1) for rid in rids],
                                     state={"caches": ArrayStep(**declared)}), {"caches": handle}, graph_walk="rollout")
    step.set_ctx(StepContext(request_ids=list(rids), graph_walk="rollout", slot=0, capture=False))
    return step


def _slot(lat_h=2, lat_w=3) -> torch.Tensor:
    return torch.zeros(1, 4, 2, 2 * lat_h, 2 * lat_w)


def _write(handle, rid="a", **fields) -> None:
    """A whole step of ``rid`` that reads the row and writes ``fields`` and what it was handed back."""
    step = _step(handle, rid)
    step.begin_calls()
    held = {name: tensor for name, tensor in handle.read(0).fields.items() if tensor is not None}
    handle.write(0, **{**held, **fields})
    step.check_calls()
    step.end_calls()


# The declaration


def test_a_value_with_sizes_takes_its_own_storage_and_builds_no_pool():
    assert storage_of(_caches()) is SIZED_VALUE
    (spec,) = build_specs([_caches()])
    assert isinstance(spec, SizedValueSpec) and not isinstance(spec, RecurrentStateSpec)
    assert spec.config.budget == PIXEL * 12, "the store should keep one request at the largest sizes by default"


@pytest.mark.parametrize(("make", "message"), [
    (lambda: _caches(largest={"lat_h": 1}), r"largest gives \['lat_h'\], and its fields' Size axes name"),
    (lambda: _caches(largest={}), "only a Value that declares largest takes a Size axis"),
    (lambda: _caches(largest={**LARGEST, "lat_f": 4}), r"largest gives \['lat_f', 'lat_h', 'lat_w'\]"),
    (lambda: _caches(largest={"lat_h": 0, "lat_w": 12}), "largest must give positive ints"),
    (lambda: Array("caches", nodes={"dit"}, layers={0}, entry=ENTRY, max_len=64, largest=LARGEST),
     "only a Value .* is sized by its requests"),
    (lambda: _caches(entry=Entry({SLOT: TensorType((1, Size("lat_h"), Size("lat_w")), dtype=None)})),
     "dtype is None, and a sized field's bytes are counted before"),
    (lambda: _caches(entry=Entry({SLOT: TensorType((Size("lat_h"), Size("lat_w")), dtype=torch.float32, shard=0)})),
     "shard is a Size axis"),
    (lambda: _caches(entry=Entry(ENTRY.tensors, host={"count": int})), "a sized Value holds tensors only"),
    (lambda: Size("lat_h", scale=0), "Size.scale must be a positive int, got 0"),
], ids=["size_unbounded", "no_largest", "bound_unnamed", "bound_zero", "not_a_value", "dtype", "shard", "host",
        "scale"])
def test_a_sized_value_refuses_what_it_cannot_count(make, message):
    with pytest.raises(ValueError, match=message):
        make()


# Intake


def test_a_request_s_sizes_may_pass_largest_one_by_one():
    """The bound is the bytes at ``largest``: a request picks any aspect up to an area."""
    wide = {"caches": {"main": Layout(sizes={"lat_h": 3, "lat_w": 4})}}
    assert state_sizes([_caches()], wide) == {"caches": {"main": {"lat_h": 3, "lat_w": 4}}}


@pytest.mark.parametrize(("layout", "message"), [
    ({"caches": {"main": Layout(sizes={"lat_h": 3, "lat_w": 5})}}, "the request's fields take 2160 bytes"),
    ({"caches": {"main": Layout(sizes={"lat_h": 3})}}, r"must give a positive int for each of \['lat_h', 'lat_w'\]"),
    ({"caches": {"main": Layout(sizes={"lat_h": 3, "lat_w": 2.0})}}, "must give a positive int for each"),
    ({"kv": {"main": Layout(sizes=SMALL)}}, "give sizes, and 'kv' is no sized Value"),
    ({}, r"state_layout gives no sizes for \['caches'\]"),
    ({"caches": {}}, r"state_layout gives no sizes for \['caches'\]"),
], ids=["past_largest", "size_missing", "size_float", "not_sized", "left_out", "no_label"])
def test_sizes_the_model_got_wrong_fail_the_request_at_intake(layout, message):
    kv = Array("kv", nodes={"dit"}, layers={0}, entry=KVEntry(heads=1, head_dim=8), max_len=0)
    with pytest.raises(RuntimeError, match=message):
        state_sizes([_caches(), kv], layout)


def test_a_request_past_the_budget_is_refused_at_intake():
    """A deployment that lowers the budget serves smaller requests, and a larger one is a 400 and never waits."""
    _, store, spec = _store(max_bytes=PIXEL * 6)
    spec.request_config({"state_sizes": {"caches": {"main": SMALL}}}).validate(spec)
    large = spec.request_config({"state_sizes": {"caches": {"main": {"lat_h": 2, "lat_w": 4}}}})
    with pytest.raises(ValueError, match="resources: caches: max_bytes"):
        large.validate(spec)
    store.ingest_request("a", large)
    asked = store.ask("a", "dit", "rollout")
    assert not asked.ok and "resources: caches: max_bytes" in asked.reason.message, "the request would wait for ever"


def test_a_request_s_sizes_survive_the_wire():
    request = SizedValueReqConfig(sizes={"main": {"lat_h": 58, "lat_w": 104}})
    assert wire.decode(wire.encode(request)) == request, "the conductor's config never reached a worker"


# Admission


def test_a_request_reserves_its_own_bytes_not_the_largest():
    _, store, spec = _store()
    _open(store, spec, "a", SMALL)
    assert store.ask("a", "dit", "rollout") is ASKS
    assert store.reserve("a", "dit", "rollout", None) is None
    assert store._used == PIXEL * 6 and store.ask("a", "dit", "rollout") is None
    store.unreserve("a")
    assert store._used == 0 and store.ask("a", "dit", "rollout") is ASKS, "an unreserved request should ask again"


def test_a_request_that_doesn_t_fit_waits_for_another_s_bytes():
    _, store, spec = _store("a")
    _open(store, spec, "b", {"lat_h": 1, "lat_w": 7})
    mark = store.reserve("b", "dit", "rollout", None)
    assert mark == PIXEL * 6 and not store.grew_since(mark)
    store.remove_request("a")
    assert store.grew_since(mark) and store.reserve("b", "dit", "rollout", None) is None


def test_a_request_that_cleared_every_field_doesn_t_reserve_again():
    handle, store, _ = _store("a")
    _step(handle, "a").begin_calls()
    handle.clear(0, SLOT, OUT)
    assert store._used == 0
    assert store.ask("a", "dit", "rollout") is None, "the request's next node would reserve the cleared bytes again"
    assert store.reserve("a", "dit", "rollout", None) is None and store._used == 0


def test_a_request_admitted_again_reserves_only_the_fields_it_still_holds():
    handle, store, _ = _store("a")
    _step(handle, "a").begin_calls()
    handle.clear(0, SLOT)
    store.unreserve("a")
    assert store.ask("a", "dit", "rollout") is ASKS and store.reserve("a", "dit", "rollout", None) is None
    assert store._used == 8 * 2 * 6, "the cleared slot's bytes were reserved again, and nothing gives them back"


def test_cleared_fields_give_their_bytes_back():
    handle, store, _ = _store("a")
    step = _step(handle, "a")
    step.begin_calls()
    handle.read(0)
    handle.clear(0, SLOT)
    assert store._used == 8 * 2 * 6, "the cleared slot's bytes should be free for a waiting request"
    with pytest.raises(ValueError, match="Array 'caches': a second write or a write after the clear of 'slot'"):
        handle.write(0, slot=_slot())


def test_a_cleared_field_stays_unset_for_the_request():
    handle, store, _ = _store("a")
    step = _step(handle, "a")
    step.begin_calls()
    handle.clear(0, SLOT)
    step.check_calls()
    step.end_calls()
    step = _step(handle, "a")
    step.begin_calls()
    with pytest.raises(ValueError, match="field 'slot' was cleared for request a"):
        handle.write(0, slot=_slot())


def test_the_store_holds_nothing_at_load_and_keeps_its_budget_from_the_capture_plan(monkeypatch):
    """The capture plan gives graphs the device's free memory, and the store's tensors arrive with the first
    request, so the plan leaves the store's budget out."""
    _, store, _ = _store()
    assert store.deferred_bytes == PIXEL * 12 and not store._requests
    engine = object.__new__(Engine)
    engine._device, engine._gpu_memory_fraction = torch.device("cuda"), None
    engine._submodules, engine._resources = {}, {"caches": store}
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (10_000, 20_000))
    plans = []

    def plan_captures(pools, free, reusable):
        plans.append(free)
        return SimpleNamespace(floor=0)

    monkeypatch.setattr("mstar.engine.engine.plan_captures", plan_captures)
    engine._plan_captures({}, {})
    assert plans[0] == 10_000 - PIXEL * 12, "graphs would take the memory the first request's tensors need"


def _engine(monkeypatch, room: int, **resources) -> Engine:
    """An engine on a CUDA device with ``room`` bytes left after load."""
    engine = object.__new__(Engine)
    engine._device, engine._resources = torch.device("cuda"), resources
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (room, 2 * room))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
    return engine


def test_two_stores_that_fit_the_device_alone_and_not_together_are_refused_at_load(monkeypatch):
    """Each store's budget is within what load left free. A request in each at once is not, and the forward that
    held both would run out of memory."""
    for name in ("caches", "frames"):
        _engine(monkeypatch, PIXEL * 20, **{name: _store()[1]})._check_deferred_room()
    engine = _engine(monkeypatch, PIXEL * 20, caches=_store()[1], frames=_store()[1])
    with pytest.raises(RuntimeError, match=r"hold none of at load \(caches 0 MiB, frames 0 MiB\)"):
        engine._check_deferred_room()


def test_a_step_that_lists_a_sized_value_isn_t_captured():
    handle, store, _ = _store("a")
    assert not store.can_capture()
    assert isinstance(_step(handle, "a").steps["caches"], SizedValueStep)
    assert blocked_capture_resources(_step(handle, "a"), {"caches": store}) == ["caches"]


def test_a_sized_value_keeps_the_prefix_caches_on_its_nodes_shut():
    (spec,) = build_specs([_caches()])
    assert spec.restores_prefix, "a prefix hit would skip the steps that write the request's tensors"


# The calls


def test_an_unwritten_field_reads_none():
    handle, _, _ = _store("a")
    _step(handle, "a").begin_calls()
    assert handle.read(0).fields == {SLOT: None, OUT: None}


def test_a_read_hands_over_the_tensor_written_without_a_copy():
    handle, store, _ = _store("a")
    slot = _slot()
    _write(handle, slot=slot)
    assert store._requests["a"].tensors == {("main", 0, SLOT): slot}
    _step(handle, "a").begin_calls()
    assert handle.read(0).fields[SLOT] is slot, "the compiled decode guards on the strides it wrote"
    assert not store._requests["a"].tensors, "the store kept a reference, so a replaced slot would outlive its chunk"


def test_a_write_within_the_row_s_shape_is_kept_as_written():
    handle, store, _ = _store("a")
    short = torch.zeros(1, 4, 1, 4, 6)
    _write(handle, slot=short)
    assert store._requests["a"].tensors["main", 0, SLOT] is short


@pytest.mark.parametrize(("fields", "message"), [
    ({SLOT: _slot(2, 4)}, r"'slot' is written at shape \(1, 4, 2, 4, 8\), which doesn't fit within \(1, 4, 2, 4, 6\)"),
    ({SLOT: _slot().double()}, "'slot' is written as torch.float64, and it declares torch.float32"),
    ({SLOT: _slot(2, 6)[..., :6]}, "'slot' is written as a view of a larger buffer"),
    ({"other": _slot()}, r"write names fields \['other'\], of \['out', 'slot'\]"),
], ids=["past_the_shape", "dtype", "view", "undeclared"])
def test_a_write_the_store_cannot_count_is_refused(fields, message):
    handle, _, _ = _store("a")
    _step(handle, "a").begin_calls()
    with pytest.raises(ValueError, match=f"Array 'caches': {message}"):
        handle.write(0, **fields)


def test_a_tensor_on_another_device_is_refused():
    """The budget is memory on the store's device, which a tensor on the host takes none of."""
    _, _, spec = _store()
    store = SizedValueStore(spec.config, torch.device("cuda"), name="caches")
    _open(store, spec, "a", SMALL)
    with pytest.raises(ValueError, match="field 'slot' is written on cpu, and the store counts its bytes on cuda"):
        store.put("a", "main", 0, {SLOT: _slot()})


def test_a_step_that_reads_a_field_writes_it_back_or_clears_it():
    handle, _, _ = _store("a")
    _write(handle, slot=_slot())
    step = _step(handle, "a")
    step.begin_calls()
    handle.read(0)
    with pytest.raises(ValueError, match=r"read layer 0 and neither wrote nor cleared \['slot'\]"):
        step.check_calls()


def test_a_row_a_failed_step_took_is_refused_at_the_next_read():
    """A forward that raises after its read leaves the store empty for the request, which is what chunk 0 reads.
    The next chunk must not start the stream again."""
    handle, _, _ = _store("a")
    _write(handle, slot=_slot())
    failed = _step(handle, "a")
    failed.begin_calls()
    handle.read(0)
    failed.end_calls()
    _step(handle, "a").begin_calls()
    with pytest.raises(RuntimeError, match="caches: request a's tensors on layer 0 were read by a step that never"):
        handle.read(0)


@pytest.mark.parametrize(("rids", "declared", "message"), [
    (("a",), {"keep": False}, "keep=False on a sized Value"),
    (("a",), {"forks": (Fork(src="main", dst="main", at="end"),)}, "a step forks a sized Value"),
    (("a", "b"), {}, "a step of 2 rows on a sized Value"),
    (("c",), {}, "request c gave no sizes for label 'main'"),
], ids=["scratch", "forks", "two_rows", "no_sizes"])
def test_a_sized_step_holds_one_admitted_row(rids, declared, message):
    handle, _, _ = _store("a", "b", sizes={"lat_h": 1, "lat_w": 2})
    with pytest.raises(ValueError, match=f"Array 'caches': .*{message}"):
        _step(handle, *rids, **declared).begin_calls()


def test_removing_a_request_frees_its_tensors():
    handle, store, _ = _store("a")
    slot = _slot()
    _write(handle, slot=slot)
    alive = weakref.ref(slot)
    del slot
    assert alive() is not None
    store.remove_request("a")
    assert alive() is None and not store._used, "the request's tensors would outlive it"
