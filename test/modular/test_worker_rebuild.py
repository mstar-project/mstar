"""A dropped request comes back through a rebuild row, which the graph runtime never sees.

The runtime keeps the request in its decode walk with its loop-back input pending while the row prefills its kept
ids. Every worker path that reads a step's rows has to know the row is there, or it routes a discarded sample,
fails the wrong requests on a refused admit, or speculates a step the rebuild was meant to precede.
"""

from __future__ import annotations

from array import array
from collections import defaultdict, deque
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from test_kv_preemption import _pool
from test_kv_prefix_lease import _StubTransfer

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import AllocationFailed, SamplerStep, StepContext, SubmoduleStep
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, PrefixSpan, StreamLayout
from mstar.engine.resources.step import AdmitRuntimeError
from mstar.model.orpheus.submodules import OrpheusLLMSubmodule
from mstar.model.submodule_base import ARNodeInputs, BatchedModelOutput, HostRows
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.worker import KeptIds, PendingBatch, PrefillCopies, RebuildRow, Speculation, Worker, _SpecFairness
from mstar.worker.worker_metrics import PreemptionCounts

NODE, PREFILL, DECODE = "LLM", "prefill", "decode"
R, A = 7, 8


def _cleanup(node, rids, wgs):
    assert not rids, "the runtime was asked about a row it never scheduled"
    return [], []


def _worker(fresh: ScheduledBatch | None = None):
    eng = SimpleNamespace(dropped=[], landed=[], is_offloaded=lambda node, rid: False,
                          lay_out_rebuild=lambda node, rid, ids: (PrefixSpan(len(ids) - 1, len(ids) - 1, PREFILL),),
                          get_max_batch_size=lambda node, walk: 1)
    eng.is_resident, eng.drop_request = lambda _, rid: rid not in eng.dropped, lambda _, rid: eng.dropped.append(rid)
    eng.land_rebuild = lambda node, rid: eng.landed.append(rid)
    w = Worker.__new__(Worker)
    w.worker_id, w._phase_period, w.enable_nvtx, w.enable_prof, w.device = "w", 0, False, False, torch.device("cpu")
    w._in_flight_rids, w._last_active, w._stream_buffer_rids = set(), {}, set()
    w._kv_keys, w._kept_ids, w._rebuild_rows = {NODE: ["kv"]}, {NODE: {}}, {}
    w._copied_inputs = {}
    w._dropped, w._victims, w._preempted, w._preemptions = {}, {}, {}, defaultdict(PreemptionCounts)
    w._walk_inputs = {(PREFILL, NODE): {"text_inputs"}, (DECODE, NODE): {"text_inputs"}, ("embeds", NODE): {"embeds"},
                      ("pair", NODE): {"a", "b"}, ("pair_decode", NODE): {"a", "b"}}
    w._growth_loops = {(DECODE, NODE): "loop", ("embeds", NODE): "embeds_loop", ("pair_decode", NODE): "pair_loop"}
    w._pending_removes, w._pending_drains, w._draining_rids = {}, set(), set()
    # holds no row, so any routing, stop or loop-count call fails the test
    w._graph_runtime = SimpleNamespace(cleanup_consumed_inputs=_cleanup, clear_pending_loop_stops=lambda: None,
                                       get_rid_string=str, push_back_node=lambda node, rids, wgs: w.pushed.extend(rids))
    w.engine_manager = SimpleNamespace(get_engine=lambda node: eng)
    w.asked, w.held, w.pushed, w.failed = [], [], [], {}
    w.scheduler = SimpleNamespace(held_until={}, backlog={}, failed_rids=set(), hold_requests=w.held.extend,
                                  get_next_batch=lambda state, **kw: w.asked.append(kw) or fresh)
    w.infos = {rid: CurrentForwardPassInfo(str(rid), DECODE, 0, 0, 8) for rid in (R, A)}
    w.request_state = SimpleNamespace(get_partition_for_node=lambda node: "p", get_fwd_info=lambda r, p: w.infos[r])
    w.tensor_manager = SimpleNamespace(get_tensor=None, cleanup_collectable=lambda *a: None)
    w._fail_requests = w.failed.update
    return w, eng


def _rebuilding(fresh: ScheduledBatch | None = None):
    """A worker with R dropped and its rebuild row queued, and the step the worker builds for it."""
    w, eng = _worker(fresh)
    w._kept_ids[NODE][R] = KeptIds(array("i", [1, 2, 3]).tobytes(), array("i", [4]))
    eng.dropped.append(R)
    w._rebuild(R, NODE)
    batch, rows = w._schedule_rebuild()
    return w, eng, batch, w._build_executing_batch(batch, rows)


def _config(growth=(("loop", 1),), labels=("main",), walk=PREFILL) -> dict[str, KVReqConfig]:
    layout = StreamLayout(spans=(PrefixSpan(2, 2, walk),), growth=growth)
    ids = array("i", [1, 2]).tobytes()
    return {"kv": KVReqConfig(
        prompt_ids=dict.fromkeys(labels[:1], ids), kv_layout=dict.fromkeys(labels, layout),
        loop_caps={loop: 1 for loop, _ in growth},
    )}


@pytest.mark.parametrize("sampled", [[4, 5, 7], [[4], [5], [7]]])  # the sampler's [B] output, and a [B, 1] one
def test_a_request_keeps_its_ids_once_and_only_its_kept_samples(sampled):
    w, _ = _worker()
    for rid in (R, A, 11):
        w._keep_ids(rid, _config())
    w._kept_ids[NODE][R].sampled.append(3)
    # R's second call is its other partition's NewRequest.
    # 9 lays out a second label, which has no ids.
    # 10 has no growth loop.
    # 12's decode reads embeddings, as Fish S2 Pro's does.
    # 13 reads two inputs, whose ids one row can't carry.
    w._keep_ids(R, _config())
    w._keep_ids(9, _config(labels=("main", "cfg_img")))
    w._keep_ids(10, _config(growth=()))
    w._keep_ids(12, _config(growth=(("embeds_loop", 1),)))
    w._keep_ids(13, _config(growth=(("pair_loop", 1),), walk="pair"))
    sampler = SamplerStep(kept_rids=frozenset({R, 11}))
    batch = _step((R, A, 11), rebuild=11, step=SimpleNamespace(steps={"sampler": sampler}))

    w._keep_samples(batch, {}, HostRows((R, A, 11), {"new_token": torch.tensor(sampled)}))
    w._keep_samples(batch, {R: {"new_token": [torch.tensor([6])]}}, None)

    assert list(w._kept_ids[NODE][R].sampled) == [3, 4, 6], "a rebuild would prefill other ids than the request's"
    assert not (w._kept_ids[NODE][A].sampled or w._kept_ids[NODE][11].sampled), "it kept a discarded sample"
    assert w._kept_ids[NODE].keys() == {R, A, 11}, "one row can't rebuild two labels, or KV not grown from ids"


@pytest.mark.parametrize("shared", [False, True], ids=["own pool", "pool shared with a node that keeps no ids"])
def test_a_request_its_kept_ids_cant_rebuild_is_guaranteed_in_its_pools(monkeypatch, shared):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    w, _ = _worker()
    if shared:
        w._kv_keys["talker"] = ["kv"]
    kv, w._my_consumer_connections, w._unprocessed_messages = _pool(), [], {}
    w._graph_runtime.add_request, w._graph_runtime.get_sharding_config = lambda request_id, **kw: int(request_id), id
    w.engine_manager.evictable_nodes = list
    w.engine_manager.add_request = lambda rid, configs: kv.ingest_request(rid, configs["kv"])
    w.request_state.add_request, w.request_state.per_request_info = lambda rid, info: None, defaultdict(dict)
    w.tensor_manager.register_request, w.tensor_manager.start_read_tensors = lambda *a: None, lambda *a, **k: []
    w.wakeup_event = SimpleNamespace(register_futures=lambda futures: None)
    # 12's decode reads embeddings, as Fish S2 Pro's does, so it keeps no ids.
    # 10 lays out nothing, so it isn't guaranteed.
    configs = {R: _config(), 12: _config(growth=(("embeds_loop", 1),)), 10: {"kv": KVReqConfig(kv_layout={})}}
    for rid, config in configs.items():
        info = SimpleNamespace(partition_name="p", graph_walk=PREFILL, resource_configs=config, stream_lead_items={})
        w._add_new_request(SimpleNamespace(request_id=str(rid), partition_worker_graph_ids=None, loop_caps={},
                                           worker_graph_to_workers={}, initial_inputs=[], request_info=info))
    guaranteed = {rid for rid in configs if kv._overrides[rid].guaranteed}
    assert guaranteed == ({R, 12} if shared else {12}), "a request nothing can drop would reserve only its prompt"


def test_postprocess_appends_kept_samples_and_lands_a_rebuild_row_beside_them():
    from test_dropped_row_stream_stop import _engine, _ReachedRoutingError, _Runtime, _Sub, _worker

    eng, landed = _engine(_Sub()), []
    eng.land_rebuild = lambda node, rid: landed.append(rid)
    w, node, walk = _worker(eng, _Runtime()), "vocoder", "w"
    w._kept_ids, w._rebuild_rows = {node: {1: KeptIds(b"", array("i"))}}, {R: deque([RebuildRow(node, walk, {}, None)])}
    batch = _step((R, 1, 2), walk, node)
    eng.prepare_inputs(batch)
    pending = PendingBatch(ScheduledBatch(node, walk, {1: 0, 2: 0}), batch, node, "p", walk, Future())
    outputs = BatchedModelOutput(per_rid_outputs={rid: {"new_token": [torch.tensor([rid + 4])]} for rid in (1, 2)})
    with pytest.raises(_ReachedRoutingError):  # the harness stops at output routing, after the append and the landing
        w._postprocess_batch(pending, outputs)
    assert list(w._kept_ids[node][1].sampled) == [5], "a drop would find no samples and refuse"
    assert landed == [R], "the rebuilt request would never be ready to decode again"


@pytest.mark.parametrize("build", ["executing", "speculative"])
def test_a_prefill_row_keeps_a_host_copy_of_what_it_read_besides_its_ids(build):
    from test_dropped_row_stream_stop import _engine, _ReachedRoutingError, _Runtime, _Sub, _worker

    eng = _engine(_Sub())
    eng.land_rebuild = lambda node, rid: None
    w, node, walk = _worker(eng, _Runtime()), "vocoder", "w"
    w._kept_ids = {node: {rid: KeptIds(b"", array("i"), []) for rid in (R, 1)}}
    w._copied_inputs, w._rebuild_rows = {(walk, node): {"embeds"}}, {R: deque([RebuildRow(node, walk, {}, None)])}
    embeds = {rid: torch.full((2,), rid) for rid in (R, 1, 2)}
    batch = _step((R, 1, 2), walk, node, per_request_input_tensors={
        rid: {"embeds": [] if build == "speculative" else [t], "ids": [t]} for rid, t in embeds.items()
    })
    if build == "speculative":
        # a speculated step gets its embeds only once the encoder has run
        spec = Speculation(ScheduledBatch(node, walk, {1: 0, 2: 0}), batch, {("embeds", node)}, {R, 1, 2}, "p",
                           is_new_iter=False, is_same_node=False)
        w._thread_outputs_to_speculative(spec, BatchedModelOutput({rid: {"embeds": [t]} for rid, t in embeds.items()}))
    for rid, info in batch.per_request_info.items():
        info.step_metadata = {"prefill_order": (rid,)}
    eng.prepare_inputs(batch)
    pending = PendingBatch(ScheduledBatch(node, walk, {1: 0, 2: 0}), batch, node, "p", walk, Future())
    outputs = BatchedModelOutput(per_rid_outputs={rid: {"new_token": [torch.tensor([rid + 4])]} for rid in (1, 2)})
    with pytest.raises(_ReachedRoutingError):
        w._postprocess_batch(pending, outputs)

    (copies,) = w._kept_ids[node][1].copies
    assert copies.inputs == {"embeds": [embeds[1]]}, "a rebuild would need the encoder to run again"
    assert copies.step_metadata == {"prefill_order": (1,)}, "the rebuild row would run under decode's metadata"
    assert copies.step_metadata is not batch.per_request_info[1].step_metadata, "a later step could rewrite the copy"
    assert not w._kept_ids[node][R].copies, "a rebuild row's inputs would stand in for the request's copy"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_a_host_copy_lands_whole_before_its_request_can_be_dropped():
    w, _ = _worker()
    w.device, w.device_capability, w._d2h_stream = torch.device("cuda"), SimpleNamespace(is_accelerator=True), None
    w._kept_ids[NODE][R] = KeptIds(b"", array("i", [5]), [])
    rows = 1 << 22
    embeds = torch.arange(1, 1 + rows, dtype=torch.float32, device="cuda")
    expected = embeds.cpu()
    with w._side_stream():
        # a cached block for the copy back, since allocating a block waits for the device
        torch.full((rows,), -1.0, device="cuda")
    batch = _step((R,), "vision", rebuild=A, per_request_input_tensors={R: {"embeds": [embeds]}})
    batch.per_request_info[R].step_metadata = {}
    # a busy side stream keeps the copy queued past the point where the worker lets go of the step's tensors
    with w._side_stream():
        torch.cuda._sleep(1 << 30)
    w._copy_inputs(batch, {"embeds"})

    assert not w._rebuildable(R, NODE), "the request could be dropped before its copy has landed"
    del batch, embeds
    torch.full((rows,), -1.0, device="cuda")
    torch.cuda.synchronize()
    (copies,) = w._kept_ids[NODE][R].copies
    assert w._rebuildable(R, NODE)
    assert torch.equal(copies.inputs["embeds"][0], expected), "the next step's tensor took the memory before the copy"

    with w._side_stream():
        torch.cuda._sleep(1 << 30)
    on_device = w._to_device(copies)
    assert torch.equal(on_device["embeds"][0].cpu(), expected), "the rebuild row would read its inputs before they land"


def test_a_rebuild_row_runs_under_its_copied_metadata_even_when_empty():
    w, _ = _worker()
    w.infos[R].step_metadata = {"prefill_order": ()}  # decode's metadata
    w._rebuild_rows[R] = deque([RebuildRow(NODE, PREFILL, {"text_inputs": [torch.tensor([1, 2])]}, {})])

    node_batch = w._build_executing_batch(*w._schedule_rebuild())

    assert node_batch.per_request_info[R].step_metadata == {}, "the rebuild row would run under decode's metadata"


@pytest.mark.parametrize("kept", [None, frozenset({R, A})])
def test_a_rebuild_rows_sample_is_never_kept_even_under_its_prefills_metadata(kept):
    engine = SimpleNamespace(_runner=SimpleNamespace(admit=lambda step: SimpleNamespace(ok=True)))
    batch = _step((R, A))
    step = SubmoduleStep(steps={"sampler": SamplerStep(kept_rids=kept)})

    _, step = Engine._declare_and_admit(
        engine, batch, rids=[R, A], inputs=[], submodule=None, nvtx=False, ctx=batch.step_context, step=step,
    )

    assert step.steps["sampler"].kept_rids == {A}, "the redrawn sample would replace the token decode resumes from"


def test_a_node_sharing_a_cache_whose_growth_walk_it_never_runs_copies_nothing():
    w, _ = _worker()
    # as Cosmos3's dit beside its reasoner, or Bagel's CFG nodes beside the LLM
    w._kv_keys["dit"], w._kept_ids["dit"], w._growth_loops[("reasoner_decode", "dit")] = ["kv"], {}, "loop"
    w._walk_inputs[("vision", NODE)], w._walk_inputs[("denoise", "dit")] = {"text_inputs", "image"}, {"latents"}

    w._set_copied_inputs()

    assert w._copied_inputs == {("vision", NODE): {"image"}}, "the worker would fail to boot, or copy for nothing"


@pytest.mark.parametrize("request_, guaranteed", [
    ("image chat on one walk", False), ("image chat on two walks", True), ("video", True), ("above one rank", True),
    ("image last", True), ("keyed", True), ("ids short of their spans", True), ("text alone on the image walk", True),
])
def test_only_an_image_prompt_on_one_walk_at_one_rank_is_preemptible(request_, guaranteed):
    w, _ = _worker()
    w._walk_inputs[("vision", NODE)], w._copied_inputs = {"text_inputs", "embeds"}, {("vision", NODE): {"embeds"}}
    w.parallel_nodes = {NODE} if request_ == "above one rank" else set()
    image_walk = "vit" if request_ == "image chat on two walks" else "vision"
    image = "rows" if request_ == "video" else "digest"
    # as Qwen3.5 lays out an image prompt, and Bagel an image chat whose image has a walk of its own
    spans = (PrefixSpan(2, 2, "vision"), PrefixSpan(6, 4, image_walk, kind=image), PrefixSpan(2, 2, "vision"))
    # the samples of a prompt that ends on its image would have no text part to extend
    spans = spans[:2] if request_ == "image last" else spans
    # a rebuild row of ids alone would leave out the other inputs the walk reads
    spans = (PrefixSpan(4, 4, "vision"),) if request_ == "text alone on the image walk" else spans
    count = sum(span.length for span in spans if span.kind == "ids") - (request_ == "ids short of their spans")
    ids = None if image == "rows" else {"main": array("i", range(count)).tobytes()}
    configs = {"kv": KVReqConfig(prompt_ids=ids, kv_layout={"main": StreamLayout(spans=spans, growth=(("loop", 1),))},
                                 loop_caps={"loop": 1}, prefix_keys={"main": [b"k"]} if request_ == "keyed" else None)}

    w._keep_ids(R, configs)
    w._guarantee(R, configs)

    assert configs["kv"].guaranteed == guaranteed, "a request no copy can rebuild would be preempted, or the reverse"
    assert guaranteed or w._kept_ids[NODE][R].copies is not None, "the request could be dropped before its copy exists"


@pytest.mark.parametrize("why", [
    "no ids", "in flight", "backlog", "rebuild queued", "prefill walk", "samples short", "copy not taken",
    "copy not landed",
])
def test_a_drop_is_refused_when_its_rebuild_could_not_follow(why):
    w, eng = _worker()
    copies = {"copy not taken": [], "copy not landed": [PrefillCopies({}, {}, SimpleNamespace(query=bool))]}
    w._kept_ids[NODE].update({} if why == "no ids" else {R: KeptIds(b"", array("i", [5]), copies.get(why))})
    w._in_flight_rids.update({R} if why == "in flight" else ())
    w.scheduler.backlog.update({(NODE, DECODE): ScheduledBatch(NODE, DECODE, {R: 0})} if why == "backlog" else {})
    w._rebuild_rows.update({R: deque([RebuildRow(NODE, PREFILL, {}, None)])} if why == "rebuild queued" else {})
    w.infos[R].graph_walk = PREFILL if why == "prefill walk" else DECODE
    w.infos[R].dynamic_loop_iter_counts["loop"] = 1 if why == "samples short" else 0

    with pytest.raises(RuntimeError):
        w._drop(R, NODE)
    assert not eng.dropped, "the request would lose pages a step still reads, or come back as nothing or wrongly keyed"


def test_an_image_request_is_dropped_once_its_copy_has_landed():
    w, eng = _worker()
    w._kept_ids[NODE][R] = KeptIds(b"", array("i", [5]), [PrefillCopies({}, {}, SimpleNamespace(query=lambda: True))])

    w._drop(R, NODE)

    assert eng.dropped == [R], "no image request could ever be preempted"


@pytest.mark.parametrize("state", ["resident", "queued", "in its step"])
def test_a_request_is_rebuilt_only_once_per_drop(state):
    w, eng = _worker()
    w._kept_ids[NODE][R] = KeptIds(b"", array("i", [5]))
    eng.dropped += [] if state == "resident" else [R]
    w._rebuild_rows.update(queued := {R: deque([RebuildRow(NODE, PREFILL, {}, None)])} if state == "queued" else {})
    w._in_flight_rids.update({R} if state == "in its step" else ())
    with pytest.raises(RuntimeError):
        w._rebuild(R, NODE)
    assert w._rebuild_rows == queued, "a rebuild would lay out a stream that is live or already being rebuilt"


def test_a_step_of_rebuild_rows_alone_routes_nothing_and_makes_them_resident():
    w, eng, batch, node_batch = _rebuilding()
    assert node_batch.rebuild_rids == {R} and node_batch.per_request_info[R].graph_walk == PREFILL
    assert node_batch.per_request_input_tensors[R]["text_inputs"][0].tolist() == [1, 2, 3]

    w._postprocess_batch(PendingBatch(batch, node_batch, NODE, "p", PREFILL, Future()), BatchedModelOutput())

    assert eng.landed == [R], "the request would never be ready to decode again"
    assert w.infos[R].graph_walk == DECODE, "the runtime's view of the request left its decode walk"
    assert (R, NODE) in w._last_active, "the rebuilt request would be the coldest eviction victim"


def test_a_rebuild_row_counts_against_the_prompt_rows_cap_and_stays_out_of_the_runtime():
    w, eng, batch, node_batch = _rebuilding(fresh=ScheduledBatch(NODE, PREFILL, {A: 0}))

    merge = {"target": (NODE, PREFILL), "pre_existing_batch_size": 1, "capture_group_of": R}
    assert w.asked == [merge], "the rebuild row would escape the prompt-row cap"
    assert list(node_batch.request_ids) == [R, A], "the step's rows would not lead with the rebuild row"
    assert list(batch.request_to_worker_graph) == [A], "the runtime would route the rebuild row"
    assert w._publishable_request_ids(node_batch) == [A], "the rebuild row would publish with MSTAR_LAZY_PUBLISH=0"


def test_the_run_loop_runs_a_queued_rebuild_row_once_and_a_remove_waits_for_its_step(monkeypatch):
    from test_worker_metrics import _looping_worker, _StopLoopError

    w, _, postprocessed, _ = _looping_worker(monkeypatch, schedule=[None, None], speculate={}, refuse=set())
    w._rebuild_rows, w._pending_removes = {R: deque([RebuildRow(NODE, "rebuild", {}, None)])}, {}
    w._rid = lambda wire_rid: wire_rid
    w._graph_runtime = SimpleNamespace(get_sharding_config=lambda rid: pytest.fail("removed under its rebuild step"))
    calls, build, step = [], w._build_executing_batch, (ScheduledBatch(NODE, "rebuild"), {R: w._rebuild_rows[R][0]})
    # the loop asks only while a row is queued, and this answers twice at most so the loop ends
    w._schedule_rebuild = lambda: step if calls.append(None) or len(calls) <= 2 else None
    w._handle_main_loop_error = lambda *args: None  # the loop keeps running, as with the real handler

    def build_after_one_raise(batch, rebuild):
        if len(calls) == 1:
            raise RuntimeError("the step's inputs failed")  # before `_set_pending`, so the row must stay queued
        return replace(build(batch), rebuild_rids=frozenset(rebuild or ()))

    w._build_executing_batch = build_after_one_raise
    # Remove R once, on the pass after its step went out, while that step is still pending.
    w._process_messages = lambda: R in w._in_flight_rids and not w._pending_removes and w._remove_request(
        SimpleNamespace(request_id=R))
    with pytest.raises(_StopLoopError):
        w.run()
    assert postprocessed == ["rebuild"], "a raise before `_set_pending` would lose the rebuild row"
    assert w._pending_removes == {R: None}, "the REMOVE would tear its request down mid-step"


@pytest.mark.parametrize("reason, outcome", [
    (AllocationFailed("short", 1, "main", R), "held"),
    (AdmitRuntimeError("unservable", request_ids=frozenset({R})), "failed"),
    (AllocationFailed("short", 1, "main", A), "requeued"),
])
def test_a_refused_admit_puts_the_rebuild_row_back_or_fails_only_its_request(reason, outcome):
    w, _, batch, node_batch = _rebuilding(fresh=ScheduledBatch(NODE, PREFILL, {A: 0}))
    node_batch.admit_error = reason
    node_batch.rebuild_rids |= {11}  # a second rebuild row, which left the step when its prepare_inputs raised
    w._handle_allocation_failure = lambda *a: w.held.append("base")

    w._handle_admit_failure(batch, node_batch)

    assert (R in w._rebuild_rows) == (outcome != "failed"), "the rebuild row left its queue for good"
    assert 11 not in w._rebuild_rows, "the failed request's row would run again"
    assert w.failed.keys() == ({R} if outcome == "failed" else set()), "the refusal failed the wrong requests"
    assert w.held == {"held": [R], "failed": [], "requeued": ["base"]}[outcome], "the other rows lost the base path"
    assert w.pushed == ([] if outcome == "requeued" else [A]), "the step's prompt row would never be scheduled again"


@pytest.mark.parametrize("why", ["failed", "removed", "mixed merge", "cap"])
def test_a_rebuild_step_holds_nothing_the_scheduler_wouldnt_run(why):
    w, _ = _worker(fresh=ScheduledBatch(NODE, "mixed", {A: 0}, request_walks={A: DECODE}))
    for rid in (R, 11) if why == "cap" else (R,):
        w._rebuild_rows[rid] = deque([RebuildRow(NODE, PREFILL, {}, None)])
    w.scheduler.failed_rids.update({R} if why == "failed" else ())
    w._pending_removes.update({R: None} if why == "removed" else {})

    scheduled = w._schedule_rebuild()

    if why in ("mixed merge", "cap"):
        assert not scheduled[0].request_to_worker_graph and w.pushed == [A], "a decode row would run as a prefill row"
        assert list(scheduled[1]) == [R], "the step would outgrow the walk's largest capture bucket"
    else:
        assert scheduled is None, "a torn-down request would prefill its whole history"


@pytest.mark.parametrize("held", [False, True])
def test_nothing_is_speculated_while_a_rebuild_row_could_run(held):
    w, _ = _worker()
    w._rebuild_rows[R] = deque([RebuildRow(NODE, PREFILL, {}, None)])
    w.scheduler.held_until[R] = float("inf") if held else 0.0
    w._can_speculate, w._tp_lead_needs_marker = lambda batch: True, lambda pending, speculation: False
    w._choose_spec_target = lambda batch: SimpleNamespace(info=SimpleNamespace(node_name=NODE, graph_walk=DECODE))
    w._try_speculate_next = lambda pending, target: pytest.fail("speculated past a queued rebuild row")
    # unless the row is held, nothing else is ready, so only the queued row can stop speculation
    w.scheduler.has_ready_excluding = lambda state, key: held
    pending = SimpleNamespace(batch=ScheduledBatch(NODE, DECODE, {A: 0}), node_name=NODE, graph_walk=DECODE,
                              node_batch=SimpleNamespace(admit_error=None))

    w._build_speculation(pending, 1, fair := _SpecFairness(peek=True, max_consecutive=8))
    assert bool(w.asked) == held, "a yield-away step would run ahead of the rebuild, or wait out its hold"
    w._choose_spec_target = lambda batch: pytest.fail("chose a target from a step of rebuild rows alone")
    w._build_speculation(SimpleNamespace(**{**vars(pending), "batch": ScheduledBatch(NODE, PREFILL)}), 1, fair)


def test_orpheus_keeps_the_prompts_sample_and_not_a_rebuild_rows():
    rows = [ARNodeInputs(input_seq_len=2, input_ids=torch.tensor([1, 2])) for _ in (R, A)]
    info = {rid: SimpleNamespace(step_metadata={"is_prefill": rid == A}) for rid in (R, A)}
    step = OrpheusLLMSubmodule.declare_step(None, PREFILL, [R, A], rows, per_request_info=info)
    assert step.steps["sampler"].kept_rids == {A}, "the penalty would see the rebuild's sample, or miss the prompt's"


def _step(rids=(R, A), walk=PREFILL, node=NODE, rebuild=R, **fields) -> ExecutingBatch:
    """A step of ``rids``, where the row ``rebuild`` only recomputes KV."""
    return ExecutingBatch(node_name=node, per_request_info={rid: SimpleNamespace(rid=rid) for rid in rids},
                          rebuild_rids=frozenset({rebuild}), **fields,
                          step_context=StepContext(request_ids=rids, graph_walk=walk, slot=0, capture=False))


@pytest.mark.parametrize("batched", [True, False])
def test_a_rebuild_rows_discarded_sample_never_stops_its_loop(batched):
    ends = {"decode_loop"}
    sub = SimpleNamespace(check_stop=lambda rid, info, outputs: ends,  # as Orpheus's, which doesn't test the walk
                          check_stop_batched=lambda rids, *_, step_walk: dict.fromkeys(rids, ends) if batched else None)
    engine = SimpleNamespace(_submodules={NODE: SimpleNamespace(submodule=sub)})

    stops = Engine.check_stop_for_batch(engine, _step(), dict.fromkeys((R, A), {"new_token": [0]}), host_rows=object())

    assert stops == {A: ends}, "the rebuild's sample would end the decode loop it rebuilds"


@pytest.mark.parametrize("branch", ["rows_only", "mixed", "tensor"])
def test_a_rebuild_row_leaves_its_requests_loopback_entry(branch):
    engine = SimpleNamespace(_loopback_steps={R: (NODE, DECODE), A: (NODE, DECODE)})
    batch = _step(rows_only=branch == "rows_only", loopback_rids=frozenset({A} if branch == "mixed" else ()))
    if branch == "mixed":
        batch.step_context.request_walks = {R: PREFILL, A: DECODE}

    Engine._record_loopback(engine, batch)

    assert engine._loopback_steps[R] == (NODE, DECODE), "the resumed decode would not read the last kept id"

