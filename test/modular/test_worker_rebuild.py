"""A dropped request comes back through a rebuild row, which the graph runtime never sees.

The runtime keeps the request in its decode walk with its loop-back input pending while the row prefills its kept
ids. Every worker path that reads a step's rows has to know the row is there, or it routes a discarded sample,
fails the wrong requests on a refused admit, or speculates a step the rebuild was meant to precede.
"""

from __future__ import annotations

from array import array
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import AllocationFailed, StepContext
from mstar.engine.resources.step import AdmitRuntimeError
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import BatchedModelOutput
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.rebuild import KeptIds, RebuildRow, Rebuilds
from mstar.worker.worker import PendingBatch, Worker, _SpecFairness

NODE, PREFILL, DECODE = "LLM", "prefill", "decode"
R, A = 7, 8
POLICY = RebuildPolicy(walk=PREFILL, ids_input="text_inputs", sample_output="new_token", loop="loop")


def _graphs() -> dict:
    """``NODE`` reads ``text_inputs`` and outputs ``new_token``, and loop ``loop`` runs it in ``DECODE``."""
    node = SimpleNamespace(input_names={"text_inputs"}, outputs=[SimpleNamespace(name="new_token")])
    loop = SimpleNamespace(get_nodes=lambda: {NODE: node})
    return {
        PREFILL: SimpleNamespace(get_nodes=lambda: {NODE: node}, get_loops=dict),
        DECODE: SimpleNamespace(get_nodes=lambda: {NODE: node}, get_loops=lambda: {"loop": loop}),
    }


def _kept(*sampled: int, prompt=(1, 2, 3)) -> KeptIds:
    return KeptIds(torch.tensor(prompt), array("i", sampled))


def _cleanup(node, rids, wgs):
    assert not rids, "the runtime was asked about a row it never scheduled"
    return [], []


def _worker(fresh: ScheduledBatch | None = None):
    eng = SimpleNamespace(dropped=[], landed=[], prepared=[], is_offloaded=lambda node, rid: False,
                          get_max_batch_size=lambda node, walk: 1)
    eng.prepare_rebuild = lambda node, rid, rebuild: eng.prepared.append((rid, rebuild))
    eng.is_resident, eng.drop_request = lambda _, rid: rid not in eng.dropped, lambda _, rid: eng.dropped.append(rid)
    eng.land_rebuild = lambda node, rid: eng.landed.append(rid)
    w = Worker.__new__(Worker)
    w.worker_id, w._phase_period, w.enable_nvtx, w.enable_prof, w.device = "w", 0, False, False, torch.device("cpu")
    w._in_flight_rids, w._last_active, w._stream_buffer_rids = set(), {}, set()
    w._rebuilds, w._rebuild_rows = Rebuilds({NODE: POLICY}, _graphs()), {}
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
    # A's prompt row carries no inputs here, so its ids count as kept already
    w._rebuilds._kept[NODE].update({R: _kept(4), A: _kept()})
    eng.dropped.append(R)
    w._rebuild(R, NODE)
    batch, rows = w._schedule_rebuild()
    return w, eng, batch, w._build_executing_batch(batch, rows)


def test_postprocess_appends_kept_samples_and_lands_a_rebuild_row_beside_them():
    from test_dropped_row_stream_stop import _engine, _ReachedRoutingError, _Runtime, _Sub, _worker

    eng, landed = _engine(_Sub()), []
    eng.land_rebuild = lambda node, rid: landed.append(rid)
    w, node, walk = _worker(eng, _Runtime()), "vocoder", "w"
    w._rebuilds._policies, w._rebuilds._kept = {node: POLICY}, {node: {1: _kept()}}
    batch = _step((R, 1, 2), walk, node)
    eng.prepare_inputs(batch)
    pending = PendingBatch(ScheduledBatch(node, walk, {1: 0, 2: 0}), batch, node, "p", walk, Future())
    outputs = BatchedModelOutput(per_rid_outputs={rid: {"new_token": [torch.tensor([rid + 4])]} for rid in (1, 2)})
    with pytest.raises(_ReachedRoutingError):  # the harness stops at output routing, after the append and the landing
        w._postprocess_batch(pending, outputs)
    assert list(w._rebuilds._kept[node][1].sampled) == [5], "a drop would find no samples and refuse"
    assert landed == [R], "the rebuilt request would never be ready to decode again"


@pytest.mark.parametrize("why", ["no ids", "in flight", "backlog", "rebuild queued", "prefill walk", "samples short"])
def test_a_drop_is_refused_when_its_rebuild_could_not_follow(why):
    w, eng = _worker()
    w._rebuilds._kept[NODE].update({} if why == "no ids" else {R: _kept(5)})
    w._in_flight_rids.update({R} if why == "in flight" else ())
    w.scheduler.backlog.update({(NODE, DECODE): ScheduledBatch(NODE, DECODE, {R: 0})} if why == "backlog" else {})
    w._rebuild_rows.update({R: RebuildRow(NODE, PREFILL, {})} if why == "rebuild queued" else {})
    w.infos[R].graph_walk = PREFILL if why == "prefill walk" else DECODE
    w.infos[R].dynamic_loop_iter_counts["loop"] = 1 if why == "samples short" else 0

    with pytest.raises(RuntimeError):
        w._drop(R, NODE)
    assert not eng.dropped, "the request would lose pages a step still reads, or come back as nothing or wrongly keyed"


@pytest.mark.parametrize("state", ["resident", "queued", "in its step"])
def test_a_request_is_rebuilt_only_once_per_drop(state):
    w, eng = _worker()
    w._rebuilds._kept[NODE][R] = _kept(5)
    eng.dropped += [] if state == "resident" else [R]
    w._rebuild_rows.update(queued := {R: RebuildRow(NODE, PREFILL, {})} if state == "queued" else {})
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
    w._rebuild_rows, w._pending_removes, w._rid = {R: RebuildRow(NODE, "rebuild", {})}, {}, lambda wire_rid: wire_rid
    w._graph_runtime = SimpleNamespace(get_sharding_config=lambda rid: pytest.fail("removed under its rebuild step"))
    calls, build, step = [], w._build_executing_batch, (ScheduledBatch(NODE, "rebuild"), dict(w._rebuild_rows))
    # the loop asks only while a row is queued, and this answers twice at most so the loop ends
    w._schedule_rebuild = lambda: step if calls.append(None) or len(calls) <= 2 else None
    w._handle_main_loop_error = lambda *args: None  # the loop keeps running, as with the real handler

    def build_after_one_raise(batch, rebuild):
        if len(calls) == 1:
            raise RuntimeError("the step's inputs failed")  # before `_set_pending`, so the row must stay queued
        return replace(build(batch), rebuild_rids=frozenset(rebuild or ()))

    w._build_executing_batch = build_after_one_raise
    # Remove R once, on the pass after its step went out, while that step is still pending.
    w._process_messages = lambda: len(calls) > 1 and not (w._rebuild_rows or w._pending_removes) and w._remove_request(
        SimpleNamespace(request_id=R))
    with pytest.raises(_StopLoopError):
        w.run()
    assert postprocessed == ["rebuild"], "a raise before `_set_pending` would lose the rebuild row"
    assert w._pending_removes == {R: None}, "the row stayed queued, or the REMOVE tore its request down mid-step"


@pytest.mark.parametrize("reason, outcome", [
    (AllocationFailed("short", 1, "main", R), "held"),
    (AdmitRuntimeError("unservable", request_ids=frozenset({R})), "failed"),
    (AllocationFailed("short", 1, "main", A), "requeued"),
])
def test_a_refused_admit_puts_the_rebuild_row_back_or_fails_only_its_request(reason, outcome):
    w, _, batch, node_batch = _rebuilding(fresh=ScheduledBatch(NODE, PREFILL, {A: 0}))
    del w._rebuild_rows[R]  # as the main loop does once the step is pending
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
    w._rebuild_rows.update(dict.fromkeys((R, 11) if why == "cap" else (R,), RebuildRow(NODE, PREFILL, {})))
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
    w._rebuild_rows[R] = RebuildRow(NODE, PREFILL, {})
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

