"""A dropped request comes back through rebuild rows, which the graph runtime never holds.

The runtime keeps the request in its decode walk with its loop-back input pending while rows of the policy's walk
prefill its kept ids. Those rows are rows of the step like any other, so the batch builder, admission, refusals
and failures treat them as rows, and only where the worker hands a step's rows to the graph runtime does it leave
them out. Postprocess takes them off the step first and lands their rebuild, or the runtime would route a
discarded sample or count a row it never scheduled.
"""

from __future__ import annotations

from array import array
from collections import defaultdict
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch
from test_kv_preemption import _pool
from test_kv_prefix_lease import _StubTransfer

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.chunking import ChunkRange
from mstar.engine.engine import Engine, ExecutingBatch
from mstar.engine.resources import Residency, StepContext
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, PrefixSpan, StreamLayout
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import BatchedModelOutput
from mstar.worker.host_pool import HostCopy, HostPool, HostSlice
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.preemption import Preemption
from mstar.worker.rebuild import KeptIds, NodeWrites, RebuildCursor, Rebuilds
from mstar.worker.stream_buffers import StreamBufferManager
from mstar.worker.worker import PendingBatch, Speculation, Worker

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
    view = torch.tensor(prompt, dtype=torch.int32)
    return KeptIds(HostCopy(HostSlice(0, view.nbytes), view, None), torch.long, array("i", sampled))


def _cleanup(node, rids, wgs):
    assert not rids, "the runtime was asked about a row it never scheduled"
    return [], []


def _worker():
    eng = SimpleNamespace(dropped=[], landed=[], released=[])
    eng.residency = lambda _, rid: Residency.DROPPED if rid in eng.dropped else Residency.RESIDENT
    eng.drop_request = lambda _, rid: eng.dropped.append(rid)
    eng.land_rebuild = lambda node, rid: eng.landed.append(rid)
    eng.release_chunk_inputs = lambda rid, node: eng.released.append(rid)
    eng.release_cached_prefix = lambda batch: None
    w = Worker.__new__(Worker)
    w.worker_id, w._phase_period, w.enable_nvtx, w.enable_prof, w.device = "w", 0, False, False, torch.device("cpu")
    w._in_flight_rids = set()
    w.tensor_manager = SimpleNamespace(get_tensor=None, get_tensors=None, cleanup_collectable=lambda *a: None)
    w.streams = StreamBufferManager(SimpleNamespace(per_request_info={}), w.tensor_manager, [])  # no streams
    w._rebuilds = Rebuilds({NODE: POLICY}, _graphs(), HostPool(1 << 10, torch.device("cpu")))
    w._kv_writes = {"kv": {NODE: NodeWrites(frozenset({PREFILL, DECODE}), frozenset({"loop"}))}}
    w._pending_removes, w._pending_drains, w._draining_rids = {}, set(), set()
    # keeps the rows the worker hands it in ``w.seen``, and holds no row, so any routing, stop or loop-count
    # call fails the test
    w._graph_runtime = SimpleNamespace(
        cleanup_consumed_inputs=_cleanup, clear_pending_loop_stops=lambda: None, get_rid_string=str,
        push_back_node=lambda node, rids, wgs: w.seen.extend(rids),
        set_in_flight=lambda node, rids, wgs, value: w.seen.extend(rids),
        commit_speculation=lambda spec_id, ok, dropped, node, wg_ids, scheduled_rids: w.seen.extend(scheduled_rids),
        speculate_node=lambda node, walks, rows, counts, strides: w.seen.extend(rows) or [],
    )
    w.parallel_nodes, w.parallel_leader_nodes = set(), set()
    w.engine_manager = SimpleNamespace(get_engine=lambda node: eng, engine=SimpleNamespace(paged_kv_caches=list))
    w.seen, w.failed, w.rebuilt = [], {}, []
    w.scheduler = SimpleNamespace(held_until={}, backlog={}, failed_rids=set(),
                                  note_rebuilt=lambda node, rid: w.rebuilt.append(rid))
    w.infos = {rid: CurrentForwardPassInfo(str(rid), DECODE, 0, 0, 8) for rid in (R, A)}
    w.request_state = SimpleNamespace(get_partition_for_node=lambda node: "p", get_fwd_info=lambda r, p: w.infos[r])
    w._preemption = Preemption(w.engine_manager, w._rebuilds, w.scheduler, w.request_state, w._log_throttled, str)
    w._fail_requests = w.failed.update
    return w, eng


def _rebuilding(fresh: dict[int, int] | None = None):
    """A worker rebuilding R in one step beside ``fresh`` prompt rows of its walk, and the step it builds."""
    w, eng = _worker()
    # A's prompt row carries no inputs here, so its ids count as kept already
    w._rebuilds._kept[NODE].update({R: _kept(4), A: _kept()})
    eng.dropped.append(R)
    w._rebuilds.start(R, NODE)
    batch = ScheduledBatch(NODE, PREFILL, {R: 0, **(fresh or {})}, rebuild_rids=frozenset({R}))
    return w, eng, batch, w._build_executing_batch(batch)


def _config(growth=(("loop", 1),), labels=("main",), walk=PREFILL) -> dict[str, KVReqConfig]:
    layout = StreamLayout(spans=(PrefixSpan(2, 2, walk),), growth=growth)
    return {"kv": KVReqConfig(kv_layout=dict.fromkeys(labels, layout), loop_caps={loop: 1 for loop, _ in growth})}


@pytest.mark.parametrize("talker_walk", [None, "talk", PREFILL], ids=[
    "own pool", "pool shared with a node that writes nothing for it",
    "pool shared with a node with no rebuild policy that writes for it",
])
def test_a_request_no_rebuild_can_write_again_isnt_rebuildable_in_its_pools(monkeypatch, talker_walk):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    w, _ = _worker()
    if talker_walk is not None:
        w._kv_writes["kv"]["talker"] = NodeWrites(frozenset({talker_walk}), frozenset())
    kv, w._unprocessed_messages = _pool(), {}
    # neither the publish skip nor the streams bear on the class
    w._skip_unshared_publish, w.streams = lambda rid, workers: None, SimpleNamespace(add_request=lambda rid, info: None)
    w._graph_runtime.add_request, w._graph_runtime.get_sharding_config = lambda request_id, **kw: int(request_id), id
    w._reads_done_sent, w.scheduler.note_arrival = set(), lambda rid: None
    w._graph_runtime.register_stop_limits = lambda rid, info: None
    w.engine_manager.add_request = lambda rid, configs, info=None: kv.ingest_request(rid, configs["kv"])
    w.request_state.add_request, w.request_state.per_request_info = lambda rid, info: None, defaultdict(dict)
    w.tensor_manager.register_request, w.tensor_manager.start_read_tensors = lambda *a: None, lambda *a, **k: []
    w.wakeup_event = SimpleNamespace(register_futures=lambda futures: None)
    # 12's KV grows in a loop the node's policy doesn't name, so no rebuild can write it again.
    # 13's prompt was written in a walk the node's policy doesn't prefill, so it keeps no ids.
    # 10 lays out nothing, so nothing refuses it.
    configs = {R: _config(), 12: _config(growth=(("embeds_loop", 1),)), 13: _config(walk="audio"),
               10: {"kv": KVReqConfig(kv_layout={})}}
    for rid, config in configs.items():
        info = SimpleNamespace(partition_name="p", graph_walk=PREFILL, resource_configs=config, stream_lead_items={})
        w._add_new_request(SimpleNamespace(request_id=str(rid), partition_worker_graph_ids=None, loop_caps={},
                                           worker_graph_to_workers={}, initial_inputs=[], request_info=info))
    refused = {rid for rid in configs if not kv._overrides[rid].rebuildable}
    assert refused == ({R, 12, 13} if talker_walk == PREFILL else {12, 13}), (
        "a request nothing can drop would count as one a drop frees"
    )


def test_postprocess_appends_kept_samples_and_lands_a_rebuild_row_beside_them():
    from test_dropped_row_stream_stop import _engine, _ReachedRoutingError, _Runtime, _Sub, _worker

    eng, landed = _engine(_Sub()), []
    eng.land_rebuild = lambda node, rid: landed.append(rid)
    w, node, walk = _worker(eng, _Runtime()), "vocoder", "w"
    w._rebuilds._policies, w._rebuilds._kept = {node: POLICY}, {node: {1: _kept()}}
    w._rebuilds._cursors[(node, R)] = RebuildCursor(PREFILL, array("i", [1, 2, 3]), torch.long)
    w.scheduler.note_rebuilt = lambda node, rid: None
    batch = _step((R, 1, 2), walk, node)
    eng.prepare_inputs(batch)
    pending = PendingBatch(ScheduledBatch(node, walk, {R: 0, 1: 0, 2: 0}, rebuild_rids=frozenset({R})), batch, node,
                           "p", walk, Future())
    outputs = BatchedModelOutput(per_rid_outputs={rid: {"new_token": [torch.tensor([rid + 4])]} for rid in (1, 2)})
    with pytest.raises(_ReachedRoutingError):  # the harness stops at output routing, after the append and the landing
        w._postprocess_batch(pending, outputs)
    assert list(w._rebuilds._kept[node][1].sampled) == [5], "a drop would find no samples and refuse"
    assert landed == [R], "the rebuilt request would never be ready to decode again"


@pytest.mark.parametrize("why", ["no ids", "backlog", "prefill walk", "samples short"])
def test_a_drop_is_refused_when_its_rebuild_could_not_follow(why):
    w, eng = _worker()
    w._rebuilds._kept[NODE].update({} if why == "no ids" else {R: _kept(5)})
    w.scheduler.backlog.update({(NODE, DECODE): ScheduledBatch(NODE, DECODE, {R: 0})} if why == "backlog" else {})
    w.infos[R].graph_walk = PREFILL if why == "prefill walk" else DECODE
    w.infos[R].dynamic_loop_iter_counts["loop"] = 1 if why == "samples short" else 0

    with pytest.raises(RuntimeError):
        w._preemption._drop(R, NODE)
    assert not eng.dropped, "the request would lose pages a step still reads, or come back as nothing or wrongly keyed"


def test_a_step_of_rebuild_rows_alone_routes_nothing_and_makes_them_resident():
    w, eng, batch, node_batch = _rebuilding()
    assert node_batch.step_context.rebuild_rids == {R} and node_batch.per_request_info[R].graph_walk == PREFILL
    assert node_batch.per_request_input_tensors[R]["text_inputs"][0].tolist() == [1, 2, 3]

    w._postprocess_batch(PendingBatch(batch, node_batch, NODE, "p", PREFILL, Future()), BatchedModelOutput())

    assert eng.landed == [R] and w.rebuilt == [R], "the request would never be ready to decode again"
    assert w.infos[R].graph_walk == DECODE, "the runtime's view of the request left its decode walk"


def test_a_chunk_short_of_the_prompts_end_advances_the_rebuild_and_lands_nothing():
    w, eng = _worker()
    w._rebuilds._kept[NODE][R] = _kept(4, 5)
    eng.dropped.append(R)
    w._rebuilds.start(R, NODE)
    batch = ScheduledBatch(NODE, PREFILL, {R: 0}, rebuild_rids=frozenset({R}))
    batch.chunk_ranges[R] = ChunkRange(0, 2, 4, 2)
    node_batch = w._build_executing_batch(batch)

    w._postprocess_batch(PendingBatch(batch, node_batch, NODE, "p", PREFILL, Future()), BatchedModelOutput())

    assert not (eng.landed or eng.released or w.rebuilt), "the request would decode over KV the rebuild hasn't written"
    assert w._rebuilds._cursors[(NODE, R)].done == 2, "the next chunk would start over"


@pytest.mark.parametrize(
    "chunk", [None, ChunkRange(0, 2, 4, 2), ChunkRange(2, 4, 4, 2)], ids=["whole", "first", "later"],
)
def test_a_rebuild_rows_ids_go_to_the_device_on_its_first_chunk_only(chunk):
    w, eng = _worker()
    w._rebuilds._kept[NODE][R] = _kept(4, 5)
    eng.dropped.append(R)
    w._rebuilds.start(R, NODE)
    batch = ScheduledBatch(NODE, PREFILL, {R: 0}, rebuild_rids=frozenset({R}))
    if chunk is not None:
        batch.chunk_ranges[R] = chunk

    inputs = w._build_executing_batch(batch).per_request_input_tensors[R]

    if chunk is None or chunk.start == 0:
        assert inputs["text_inputs"][0].tolist() == [1, 2, 3, 4]
    else:
        assert not inputs, "a later chunk is cut from the input the engine prepared on the first"


def test_a_skipped_rebuild_row_fails_its_request():
    w, eng, batch, node_batch = _rebuilding()
    # as the engine leaves a row whose prepare_inputs returned None
    node_batch.request_ids = []
    node_batch.skipped_rids.add(R)

    w._postprocess_batch(PendingBatch(batch, node_batch, NODE, "p", PREFILL, Future()), BatchedModelOutput())

    assert R in w.failed and not eng.landed, "the request would wait on a rebuild that never runs"


@pytest.mark.parametrize("site", ["push back", "in flight", "settle", "spec target", "unsubmitted speculation"])
def test_the_graph_runtime_never_sees_a_rebuild_row(site):
    w, _ = _worker()
    batch = ScheduledBatch(NODE, DECODE, {R: 0, A: 1}, rebuild_rids=frozenset({R}))
    speculation = Speculation(scheduled_batch=batch, node_batch=None, consumed_edges=set(), continuing_rids=set(),
                              partition="p", is_new_iter=False, is_same_node=False)
    if site == "push back":
        w._push_back_batch(batch)
    elif site == "in flight":
        w._set_in_flight_flag(batch, True)
    elif site == "settle":
        w._settle_speculation(speculation, success=True)
    elif site == "spec target":
        w._choose_spec_target(batch)
    else:
        w._handle_main_loop_error(RuntimeError("boom"), (), None, speculation)

    assert A in w.seen and R not in w.seen, "the runtime would hold a node for a row it never scheduled"
    assert not w.failed, "the rebuild of a step that never ran stays in the log for the next step"


def test_a_terminal_refusal_of_a_rebuild_row_fails_only_its_request():
    w, _ = _worker()
    batch = ScheduledBatch(NODE, PREFILL, {R: 0, A: 1}, rebuild_rids=frozenset({R}))

    w._handle_unservable(batch, SimpleNamespace(request_ids=(R,), message="too long"))

    assert w.failed.keys() == {R} and w.seen == [A], "the other row would fail, or go back as a row it never was"


def test_a_refused_rebuild_row_is_held_like_any_row():
    from test_admit_failure_handling import _alloc_failed, _HoldingWorker

    worker, held = _HoldingWorker(), []
    worker.scheduler.hold_requests = held.extend
    batch = ScheduledBatch(NODE, PREFILL, {R: 0, A: 1}, rebuild_rids=frozenset({R}))

    worker._handle_allocation_failure(batch, SimpleNamespace(node_name=NODE, failed_resource="kv",
                                                             admit_error=_alloc_failed(R)))

    assert held == [R] and worker._graph_runtime.pushed_back == [A], "the retry would be refused the same way"


def test_a_rebuild_row_stays_out_of_the_runtime_and_publishes_nothing():
    w, eng, batch, node_batch = _rebuilding(fresh={A: 0})

    assert list(node_batch.request_ids) == [R, A] and batch.graph_rows() == {A: 0}, (
        "the runtime would route the rebuild row"
    )
    assert w._publishable_request_ids(node_batch) == [A], "the rebuild row would publish with MSTAR_LAZY_PUBLISH=0"


def test_a_yield_away_to_rebuild_rows_alone_runs_them(monkeypatch):
    from test_worker_metrics import _looping_worker, _StopLoopError

    w, _, postprocessed, _ = _looping_worker(monkeypatch, schedule=[("decode", 1), None, None], speculate={},
                                             refuse=set())
    built = []

    def yield_to_rebuild(pending, consecutive, fair):
        """A chain that yields to a rebuild, once."""
        if built or pending is None:
            return None, None
        batch = ScheduledBatch(NODE, "rebuild", {R: 0}, rebuild_rids=frozenset({R}))
        built.append(batch)
        return Speculation(scheduled_batch=batch, node_batch=w._build_executing_batch(batch), consumed_edges=set(),
                           continuing_rids=set(), partition="p", is_new_iter=False, is_same_node=False,
                           is_yield_away=True), None

    w._build_speculation = yield_to_rebuild
    with pytest.raises(_StopLoopError):
        w.run()
    assert postprocessed == ["decode", "rebuild"], "the yield-away step was built and thrown away"


@pytest.mark.parametrize("where", ["being built", "in flight"])
def test_a_step_that_raises_fails_its_rebuild_rows_with_its_other_rows(where):
    w, _ = _worker()
    w._clear_in_flight_flag = lambda batch: None
    batch = ScheduledBatch(NODE, PREFILL, {A: 0, R: 0}, rebuild_rids=frozenset({R}))
    stale = PendingBatch(batch, _step(), NODE, "p", PREFILL, None) if where == "in flight" else None

    w._handle_main_loop_error(RuntimeError("boom"), (stale,), None if stale else batch)

    assert w.failed.keys() == {R, A}, "the dropped request would never be rebuilt or failed"


def _step(rids=(R, A), walk=PREFILL, node=NODE, rebuild=R, **fields) -> ExecutingBatch:
    """A step of ``rids``, where the row ``rebuild`` only recomputes KV."""
    ctx = StepContext(request_ids=rids, graph_walk=walk, slot=0, capture=False, rebuild_rids=frozenset({rebuild}))
    return ExecutingBatch(node_name=node, per_request_info={rid: SimpleNamespace(rid=rid) for rid in rids},
                          step_context=ctx, **fields)


@pytest.mark.parametrize("branch", ["rows_only", "mixed", "tensor"])
def test_a_rebuild_row_leaves_its_requests_loopback_entry(branch):
    engine = SimpleNamespace(_loopback_steps={R: (NODE, DECODE), A: (NODE, DECODE)})
    batch = _step(rows_only=branch == "rows_only", loopback_rids=frozenset({A} if branch == "mixed" else ()))
    if branch == "mixed":
        batch.step_context.request_walks = {R: PREFILL, A: DECODE}

    Engine._record_loopback(engine, batch)

    assert engine._loopback_steps[R] == (NODE, DECODE), "the resumed decode would not read the last kept id"
