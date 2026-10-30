"""A dropped request comes back through a rebuild row, which the graph runtime never sees.

The runtime keeps the request in its decode walk with its loop-back input pending while the row prefills its kept
ids. The scheduler hands the row out with a step of its walk, and every worker path that reads a step's rows has to
know the row is there, or it routes a discarded sample, fails the wrong requests on a refused admit, or loses the
row when its step never goes out.
"""

from __future__ import annotations

from array import array
from collections import defaultdict
from concurrent.futures import Future
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from test_kv_preemption import _pool
from test_kv_prefix_lease import _StubTransfer

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import AllocationFailed, StepContext
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, PrefixSpan, StreamLayout
from mstar.engine.resources.step import AdmitRuntimeError
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import BatchedModelOutput
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.rebuild import KeptIds, RebuildRow, Rebuilds
from mstar.worker.stream_buffers import StreamBufferManager
from mstar.worker.worker import PendingBatch, Speculation, Worker
from mstar.worker.worker_metrics import PreemptionCounts

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


class _Scheduler:
    """The micro-scheduler's rebuild queue and holds, as the worker uses them."""

    def __init__(self, held: list):
        self.rows, self.taken, self.held_until, self.backlog, self.failed_rids = {}, [], {}, {}, set()
        self.hold_requests = held.extend

    def queue_rebuild(self, rid, row):
        self.rows[rid] = row

    def rebuild_queued(self, rid):
        return rid in self.rows

    def take_rebuilds(self, rids):
        self.taken.extend(rids)
        for rid in rids:
            self.rows.pop(rid, None)


def _worker():
    eng = SimpleNamespace(dropped=[], landed=[], prepared=[], is_offloaded=lambda node, rid: False)
    eng.prepare_rebuild = lambda node, rid, rebuild: eng.prepared.append((rid, rebuild))
    eng.is_resident = lambda _, rid: rid not in eng.dropped
    eng.drop_request = lambda _, rid, kept: eng.dropped.append(rid)
    eng.land_rebuild = lambda node, rid: eng.landed.append(rid)
    w = Worker.__new__(Worker)
    w.worker_id, w._phase_period, w.enable_nvtx, w.enable_prof, w.device = "w", 0, False, False, torch.device("cpu")
    w._in_flight_rids, w._last_active = set(), {}
    w.streams = StreamBufferManager(SimpleNamespace(per_request_info={}), None, [])  # no streams
    w._rebuilds = Rebuilds({NODE: POLICY}, _graphs())
    w._kv_keys, w._rebuilding = {NODE: ["kv"]}, frozenset()
    w._dropped, w._victims, w._preempted, w._preemptions = {}, {}, {}, defaultdict(PreemptionCounts)
    w._logged_pools = []
    w._pending_removes, w._pending_drains, w._draining_rids = {}, set(), set()
    w.parallel_nodes = w.parallel_leader_nodes = set()
    # holds no row, so any routing, stop or loop-count call fails the test
    w._graph_runtime = SimpleNamespace(cleanup_consumed_inputs=_cleanup, clear_pending_loop_stops=lambda: None,
                                       get_rid_string=str, push_back_node=lambda node, rids, wgs: w.pushed.extend(rids))
    w.engine_manager = SimpleNamespace(get_engine=lambda node: eng)
    w.held, w.pushed, w.failed = [], [], {}
    w.scheduler = _Scheduler(w.held)
    w.infos = {rid: CurrentForwardPassInfo(str(rid), DECODE, 0, 0, 8) for rid in (R, A)}
    w.request_state = SimpleNamespace(get_partition_for_node=lambda node: "p", get_fwd_info=lambda r, p: w.infos[r])
    w.tensor_manager = SimpleNamespace(get_tensor=None, cleanup_collectable=lambda *a: None)
    w._fail_requests = w.failed.update
    return w, eng


def _rebuilding(fresh: dict[int, int] | None = None):
    """A worker with R dropped and its rebuild row queued, and the step the worker builds when the scheduler
    hands that row out with ``fresh`` prompt rows of its walk."""
    w, eng = _worker()
    # A's prompt row carries no inputs here, so its ids count as kept already
    w._rebuilds._kept[NODE].update({R: _kept(4), A: _kept()})
    eng.dropped.append(R)
    w._rebuild(R, NODE)
    batch = ScheduledBatch(NODE, PREFILL, dict(fresh or {}), rebuild_rows=dict(w.scheduler.rows))
    return w, eng, batch, w._build_executing_batch(batch)


def _config(growth=(("loop", 1),), labels=("main",), walk=PREFILL) -> dict[str, KVReqConfig]:
    layout = StreamLayout(spans=(PrefixSpan(2, 2, walk),), growth=growth)
    return {"kv": KVReqConfig(kv_layout=dict.fromkeys(labels, layout), loop_caps={loop: 1 for loop, _ in growth})}


def test_a_rebuild_tells_the_node_what_its_row_writes_and_queues_the_row():
    w, eng, _, _ = _rebuilding()

    [(rid, rebuild)] = eng.prepared
    assert (rid, rebuild.walk, list(rebuild.ids)) == (R, PREFILL, [1, 2, 3, 4]), "the KV would lay out other ids"
    assert w.scheduler.rebuild_queued(R), "the row would never reach a step"


@pytest.mark.parametrize("shared", [False, True], ids=["own pool", "pool shared with a node with no rebuild policy"])
def test_a_request_no_rebuild_can_write_again_is_guaranteed_in_its_pools(monkeypatch, shared):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    w, _ = _worker()
    if shared:
        w._kv_keys["talker"] = ["kv"]
    kv, w._unprocessed_messages = _pool(), {}
    # neither the publish skip nor the streams bear on the class
    w._skip_unshared_publish, w.streams = lambda rid, workers: None, SimpleNamespace(add_request=lambda rid, info: None)
    w._graph_runtime.add_request, w._graph_runtime.get_sharding_config = lambda request_id, **kw: int(request_id), id
    w.engine_manager.evictable_nodes = list
    w.engine_manager.add_request = lambda rid, configs, info=None: kv.ingest_request(rid, configs["kv"])
    w.request_state.add_request, w.request_state.per_request_info = lambda rid, info: None, defaultdict(dict)
    w.tensor_manager.register_request, w.tensor_manager.start_read_tensors = lambda *a: None, lambda *a, **k: []
    w.wakeup_event = SimpleNamespace(register_futures=lambda futures: None)
    # 12's KV grows in a loop the node's policy doesn't name, so no rebuild can write it again.
    # 13's prompt was written in a walk the node's policy doesn't prefill, so it keeps no ids.
    # 10 lays out nothing, so it isn't guaranteed.
    configs = {R: _config(), 12: _config(growth=(("embeds_loop", 1),)), 13: _config(walk="audio"),
               10: {"kv": KVReqConfig(kv_layout={})}}
    for rid, config in configs.items():
        info = SimpleNamespace(partition_name="p", graph_walk=PREFILL, resource_configs=config, stream_lead_items={})
        w._add_new_request(SimpleNamespace(request_id=str(rid), partition_worker_graph_ids=None, loop_caps={},
                                           worker_graph_to_workers={}, initial_inputs=[], request_info=info))
    guaranteed = {rid for rid in configs if kv._overrides[rid].guaranteed}
    assert guaranteed == ({R, 12, 13} if shared else {12, 13}), (
        "a request nothing can drop would reserve only its prompt"
    )


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
    w.scheduler.rows.update({R: RebuildRow(NODE, PREFILL, {})} if why == "rebuild queued" else {})
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
    w.scheduler.rows.update(queued := {R: RebuildRow(NODE, PREFILL, {})} if state == "queued" else {})
    if state == "in its step":
        w._in_flight_rids, w._rebuilding = {R}, frozenset({R})
    with pytest.raises(RuntimeError):
        w._rebuild(R, NODE)
    assert w.scheduler.rows == queued and not eng.prepared, (
        "a rebuild would lay out a stream that is live or already being rebuilt"
    )


def test_a_dropped_request_is_rebuilt_while_a_step_of_another_node_holds_it():
    """Orpheus's SNAC decoder can run on the LLM's worker, so a SNAC step can be in flight when the request's LLM
    KV is rebuilt."""
    w, eng = _worker()
    w._rebuilds._kept[NODE][R] = _kept(5)
    eng.dropped.append(R)
    w._in_flight_rids.add(R)

    w._rebuild(R, NODE)

    assert w.scheduler.rebuild_queued(R), "the request never comes back, or its worker's main loop raises"


def test_a_step_of_rebuild_rows_alone_routes_nothing_and_makes_them_resident():
    w, eng, batch, node_batch = _rebuilding()
    assert node_batch.rebuild_rids == {R} and node_batch.per_request_info[R].graph_walk == PREFILL
    assert node_batch.per_request_input_tensors[R]["text_inputs"][0].tolist() == [1, 2, 3]

    w._postprocess_batch(PendingBatch(batch, node_batch, NODE, "p", PREFILL, Future()), BatchedModelOutput())

    assert eng.landed == [R], "the request would never be ready to decode again"
    assert w.infos[R].graph_walk == DECODE, "the runtime's view of the request left its decode walk"
    assert (R, NODE) in w._last_active, "the rebuilt request would be the coldest eviction victim"


def test_a_rebuild_row_leads_its_step_and_stays_out_of_the_runtime():
    w, eng, batch, node_batch = _rebuilding(fresh={A: 0})

    assert list(node_batch.request_ids) == [R, A], "the step's rows would not lead with the rebuild row"
    assert list(batch.request_to_worker_graph) == [A], "the runtime would route the rebuild row"
    assert w._publishable_request_ids(node_batch) == [A], "the rebuild row would publish with MSTAR_LAZY_PUBLISH=0"


def test_the_run_loop_takes_a_rebuild_row_once_its_step_goes_out_and_a_remove_waits_for_it(monkeypatch):
    from test_worker_metrics import _looping_worker, _StopLoopError

    w, _, postprocessed, _ = _looping_worker(monkeypatch, schedule=[None, None], speculate={}, refuse=set())
    w._pending_removes, w._rid = {}, lambda wire_rid: wire_rid
    w._graph_runtime = SimpleNamespace(get_sharding_config=lambda rid: pytest.fail("removed under its rebuild step"))
    queue, taken, calls, build = {R: RebuildRow(NODE, "rebuild", {})}, [], [], w._build_executing_batch
    w.scheduler.take_rebuilds = lambda rids: taken.extend(rids) or [queue.pop(rid) for rid in rids]
    after = w.scheduler.get_next_batch

    def next_batch(state, **kw):
        """The queued row while it is queued, twice at most, then the harness's steps, which end the loop."""
        if queue and len(calls) < 2:
            calls.append(None)
            return ScheduledBatch(NODE, "rebuild", rebuild_rows=dict(queue))
        return after(state, **kw)

    w.scheduler.get_next_batch = next_batch
    w._handle_main_loop_error = lambda *args: None  # the loop keeps running, as with the real handler

    def build_after_one_raise(batch):
        if len(calls) == 1:
            raise RuntimeError("the step's inputs failed")  # before `_set_pending`, so the row must stay queued
        node_batch = build(batch)
        node_batch.rebuild_rids = frozenset(batch.rebuild_rows)
        return node_batch

    w._build_executing_batch = build_after_one_raise
    # Remove R once, on the pass after its step went out, while that step is still pending.
    w._process_messages = lambda: taken and not w._pending_removes and w._remove_request(SimpleNamespace(request_id=R))
    with pytest.raises(_StopLoopError):
        w.run()
    assert taken == [R] and postprocessed == ["rebuild"], "a raise before `_set_pending` would lose the rebuild row"
    assert w._pending_removes == {R: None}, "the REMOVE tore its request down mid-step"


def test_a_yield_away_to_rebuild_rows_alone_runs_them(monkeypatch):
    from test_worker_metrics import _looping_worker, _StopLoopError

    w, _, postprocessed, _ = _looping_worker(monkeypatch, schedule=[("decode", 1), None, None], speculate={},
                                             refuse=set())
    taken, built = [], []
    w.scheduler.take_rebuilds = taken.extend

    def yield_to_rebuild(pending, consecutive, fair):
        """A chain that yields to a queued rebuild row, once."""
        if built or pending is None:
            return None, None
        batch = ScheduledBatch(NODE, "rebuild", rebuild_rows={R: RebuildRow(NODE, "rebuild", {})})
        node_batch = w._build_executing_batch(batch)
        node_batch.rebuild_rids = frozenset(batch.rebuild_rows)
        built.append(batch)
        return Speculation(scheduled_batch=batch, node_batch=node_batch, consumed_edges=set(), continuing_rids=set(),
                           partition="p", is_new_iter=False, is_same_node=False, is_yield_away=True), None

    w._build_speculation = yield_to_rebuild
    with pytest.raises(_StopLoopError):
        w.run()
    assert postprocessed == ["decode", "rebuild"] and taken == [R], "the yield-away step was built and thrown away"


@pytest.mark.parametrize("where", ["being built", "in flight"])
def test_a_step_that_raises_fails_its_rebuild_rows_with_its_other_rows(where):
    w, _ = _worker()
    w._clear_in_flight_flag = lambda batch: None
    batch = ScheduledBatch(NODE, PREFILL, {A: 0}, rebuild_rows={R: RebuildRow(NODE, PREFILL, {})})
    stale = PendingBatch(batch, _step(), NODE, "p", PREFILL, None) if where == "in flight" else None

    w._handle_main_loop_error(RuntimeError("boom"), (stale,), None if stale else batch)

    assert w.failed.keys() == {R, A}, "the dropped request would never be rebuilt or failed"


@pytest.mark.parametrize("reason, outcome", [
    (AllocationFailed("short", 1, "main", R), "held"),
    (AdmitRuntimeError("unservable", request_ids=frozenset({R})), "failed"),
    (AllocationFailed("short", 1, "main", A), "requeued"),
])
def test_a_refused_admit_puts_the_rebuild_row_back_or_fails_only_its_request(reason, outcome):
    w, _, batch, node_batch = _rebuilding(fresh={A: 0})
    node_batch.admit_error = reason
    node_batch.rebuild_rids |= {11}  # a second rebuild row, which left the step when its prepare_inputs raised
    w._handle_allocation_failure = lambda *a: w.held.append("base")
    w.scheduler.take_rebuilds(node_batch.rebuild_rids)  # as the step's launch does, before its admit

    w._handle_admit_failure(batch, node_batch)

    assert w.scheduler.rebuild_queued(R) == (outcome != "failed"), "the rebuild row left its queue for good"
    assert not w.scheduler.rebuild_queued(11), "a row that left the step would run again"
    assert w.failed.keys() == ({R} if outcome == "failed" else set()), "the refusal failed the wrong requests"
    assert w.held == {"held": [R], "failed": [], "requeued": ["base"]}[outcome], "the other rows lost the base path"
    assert w.pushed == ([] if outcome == "requeued" else [A]), "the step's prompt row would never be scheduled again"


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


def test_a_follower_leaves_a_refused_rebuild_row_to_rank_0s_next_step():
    """Rank 0 puts the row back and sends it again in a later step, and the follower builds the row from that step."""
    w, _, batch, node_batch = _rebuilding(fresh={A: 0})
    w.scheduler.take_rebuilds(node_batch.rebuild_rids)  # as the step's launch does
    w.parallel_nodes = {NODE}
    node_batch.admit_error = AllocationFailed("short", 1, "main", R)
    w._handle_allocation_failure = lambda *a: w.held.append("base")

    w._handle_admit_failure(batch, node_batch)

    assert not w.scheduler.rows and w.held == ["base"], "the follower would schedule the rebuild row itself"


def test_a_followers_yield_away_onto_rank_0s_rebuild_step_builds_its_rebuild_rows():
    """A follower that also runs a node of its own can yield away onto rank 0's next step. That step's rebuild rows
    aren't taken from a ready queue, so the follower builds them from its own kept ids, as the serial path does, or
    the follower runs fewer rows than rank 0."""
    rank_0s = ScheduledBatch(node_name=NODE, graph_walk=PREFILL, request_to_worker_graph={}, rebuild_rids=(7,))
    built = []
    w = SimpleNamespace(
        _can_speculate=lambda batch: True, _choose_spec_target=lambda batch: None,
        scheduler=SimpleNamespace(get_next_batch=lambda request_state, exclude_target=None: rank_0s),
        request_state=SimpleNamespace(get_partition_for_node=lambda node: "p"),
        _tp_lead_needs_marker=lambda pending, spec: False, _span=lambda name: nullcontext(),
        _build_executing_batch=lambda batch: built.append(dict(batch.rebuild_rows)) or SimpleNamespace(request_ids=[]),
        maybe_send_zmq_to_tp_followers=lambda node_batch: -1, enable_nvtx=False, _phase_period=0, device=None,
        _rebuilds=SimpleNamespace(rebuild=lambda rid, node, device: (None, (node, rid))),
    )
    w._follower_rebuild = lambda batch: Worker._follower_rebuild(w, batch)
    own = SimpleNamespace(
        node_name="own", graph_walk=DECODE, tp_seq=-1, node_batch=SimpleNamespace(admit_error=None),
        batch=SimpleNamespace(request_to_worker_graph={1: 0}),
    )

    speculation, _ = Worker._build_speculation(w, own, 5, SimpleNamespace(peek=False, max_consecutive=1))

    assert speculation.is_yield_away and built == [{7: (NODE, 7)}], "rank 0's rebuild row was left out"
