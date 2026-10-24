"""The columnar speculation (``prep_spec_rows`` + ``SpeculationBuilder.assemble_rows``)
must build exactly the speculative step the per-row path builds.

A real worker (``Worker.__new__`` plus the attributes these methods read) over
a real graph runtime, Python and Rust, is driven through a decode loop. Each
step the next one is speculated twice from the same state, once per path --
prepping is side-effect free when nothing streams -- and everything the engine,
the model and the rest of the loop read off the two is compared, before and
after N's outputs are threaded in. The rows-path speculation is the one that
runs on.
"""
import copy
import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.communication.tensor_store import PythonTensorBookkeeping, TensorStore
from mstar.communication.tensors import TensorCommunicationManager
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mstar.graph.runtime.base import ColumnarEdgeSpecs, GraphRuntime, RouteInput
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.graph.special_destinations import EMPTY_DESTINATION
from mstar.model.base import WorkerGraph
from mstar.model.submodule_base import BatchedModelOutput
from mstar.utils.containers import ParallelList
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.node_manager_utils import RequestStateManager
from mstar.worker.spec_rows import DeferredRowInputs
from mstar.worker.speculation import SpeculationPreparer
from mstar.worker.worker import PendingBatch, Worker

WG, WALK, NODE, LOOP, SIG, PART, ME = 0, "decode", "LLM", "decode_loop", "text_inputs", "default", "w0"
# the second node of a two-node loop (``pair=True``): NODE -> NODE_B -> NODE
NODE_B, SIG_B = "head", "hidden"
# a prefill walk on NODE (``mixed=True``), for a step whose rows mix walks
WG_PRE, PRE, MIXED = 1, "prefill", "mixed"

def _have_rust() -> bool:
    try:
        import mstar_rust  # noqa: F401
    except ImportError:
        return False
    return True


KINDS = ["python", pytest.param("rust", marks=pytest.mark.skipif(
    not _have_rust(), reason="mstar_rust not built",
))]


class _StubTM:
    """The tensor manager surface both runtimes and the worker touch here."""

    set_output_ref_counts = TensorCommunicationManager.set_output_ref_counts
    set_persist = TensorCommunicationManager.set_persist
    dereference = TensorCommunicationManager.dereference
    dereference_batch = TensorCommunicationManager.dereference_batch
    dereference_batch_uniform = TensorCommunicationManager.dereference_batch_uniform
    cleanup_collectable = TensorCommunicationManager.cleanup_collectable

    def __init__(self, bookkeeping):
        self.tensor_store = TensorStore(bookkeeping=bookkeeping)

    def increment_ref(self, uuid, n=1):
        self.tensor_store.increment_ref(uuid, n=n)

    def get_tensor(self, uuid):
        return self.tensor_store.get_tensor(uuid)

    def _cleanup_by_uuid(self, uuid, registered=None):
        if self.tensor_store.check_uuid_presence(uuid):
            self.tensor_store.remove_tensor(uuid)


class _Scheduler:
    """The scheduler questions the speculation asks: no backlog, an optional
    cap, and the fresh rows a test queues (the same batch to both builds)."""
    failed_rids: set = frozenset()

    def __init__(self):
        self.room = None
        self.fresh: ScheduledBatch | None = None

    def backlog_splits_from(self, *a):
        return False

    def room_for_continuing(self, target):
        return self.room

    def get_next_batch(self, *a, **k):
        if self.fresh is None:
            return None
        batch = copy.copy(self.fresh)
        batch.request_to_worker_graph = dict(self.fresh.request_to_worker_graph)
        batch._by_walk = None
        return batch


def _runtime(kind: str, max_iters: int, pair: bool = False, mixed: bool = False):
    body = GraphNode(
        name=NODE, input_names=[SIG],
        outputs=[GraphEdge(next_node=NODE, name=SIG)],
    )
    if pair:
        body = Sequential([
            GraphNode(name=NODE, input_names=[SIG], outputs=[GraphEdge(next_node=NODE_B, name=SIG_B)]),
            GraphNode(name=NODE_B, input_names=[SIG_B], outputs=[GraphEdge(next_node=NODE, name=SIG)]),
        ])
    nodes = {NODE, NODE_B} if pair else {NODE}
    section = Loop(name=LOOP, section=body, max_iters=max_iters, outputs=[])
    graphs = [WorkerGraph(section=section, graph_walks={WALK}, ranks=[0], worker_graph_id=WG)]
    walks, loops, wg_nodes = {WG: {WALK}}, {WG: {LOOP}}, {WG: nodes}
    if mixed:
        # prefill hands its token to the conductor, as a walk transition does
        graphs.append(WorkerGraph(
            section=GraphNode(
                name=NODE, input_names=[SIG],
                outputs=[GraphEdge(next_node=EMPTY_DESTINATION, name=SIG)],
            ),
            graph_walks={PRE}, ranks=[0], worker_graph_id=WG_PRE,
        ))
        walks[WG_PRE], loops[WG_PRE], wg_nodes[WG_PRE] = {PRE}, set(), {NODE}
    common = dict(
        my_worker_id=ME, my_worker_graphs=graphs,
        all_wg_ids_to_graph_walks=walks, all_wg_ids_to_dyn_loops=loops,
        all_wg_ids_to_nodes=wg_nodes, node_to_partition=dict.fromkeys(nodes, PART),
        sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
    )
    if kind == "python":
        tm = _StubTM(PythonTensorBookkeeping())
        return PythonGraphRuntime(**common, tensor_manager=tm, communicator=None), tm
    from mstar.communication.tensor_store import RustTensorBookkeeping
    from mstar.graph.runtime import rust as rust_runtime
    book = RustTensorBookkeeping()
    return rust_runtime.RustGraphRuntime(**common, bookkeeping=book), _StubTM(book)


def _worker(
    kind: str, max_iters: int = 1 << 20, pair: bool = False, mixed: bool = False,
) -> Worker:
    rt, tm = _runtime(kind, max_iters, pair, mixed)
    w = Worker.__new__(Worker)
    w._graph_runtime = rt
    w.tensor_manager = tm
    w.request_state = RequestStateManager(node_to_partition={NODE: PART, NODE_B: PART})
    w.scheduler = _Scheduler()
    w._phase_period = 0
    w.enable_prof = False
    w.enable_nvtx = False
    w._stream_buffer_rids = set()
    w._pending_removes = {}
    w._pending_drains = {}
    w._draining_rids = set()
    w._consumer_node_cache = {}
    w._stream_partition = {}
    w.parallel_nodes = set()
    w.worker_id = ME
    w._rid_str = rt.get_rid_string
    return w


def _admit(w: Worker, n: int, start: int = 0, walk: str = WALK) -> list[int]:
    """``n`` requests, each with its loop-back input ingested and its decode
    node ready, as a device-loopback walk starts (or, for ``PRE``, with its
    prompt ingested)."""
    wg = WG_PRE if walk == PRE else WG
    rids = []
    for i in range(start, start + n):
        wire = f"req-{i}"
        rid = w._graph_runtime.add_request(
            request_id=wire, partition=PART, graph_walk=walk,
            partition_worker_graph_ids=[wg],
            worker_graph_to_workers=ParallelList([wg], [[ME]]), loop_caps={})
        w.request_state.add_request(rid, CurrentForwardPassInfo(
            request_id=wire, graph_walk=walk, fwd_index=0, random_seed=0,
            max_tokens=1 << 30, rid_handle=rid, partition_name=PART,
            # a stale counter from an earlier walk rides along in the copies
            dynamic_loop_iter_counts={"other_loop": 7},
        ))
        rids.append(rid)
    block = ColumnarEdgeSpecs.empty()
    for rid in rids:
        block.add(rid, SIG, [], False, next_node=NODE)
    w._graph_runtime.ingest_inputs_batch(block)
    return rids


def _pop(w: Worker, rids: list[int], walk: str = WALK) -> ScheduledBatch:
    popped = w._graph_runtime.pop_rids(NODE, walk, rids)
    return ScheduledBatch(
        node_name=NODE, graph_walk=walk,
        request_to_worker_graph=dict(zip(popped.wg_ids.keys, popped.wg_ids.values, strict=True)),
        input_edges=popped.input_edges, output_signals=list(popped.output_signals),
    )


def _first_step(w: Worker, rids: list[int]) -> PendingBatch:
    batch = _pop(w, rids)
    w._set_in_flight_flag(batch, True)
    return PendingBatch(
        batch=batch, node_batch=w._build_executing_batch(batch), node_name=NODE,
        partition=PART, graph_walk=WALK, future=None,
    )


def _mixed_first_step(w: Worker, decode: list[int], prefill: list[int]) -> PendingBatch:
    """One step whose rows mix the decode and prefill walks, as a mixing
    scheduler builds it."""
    dec, pre = _pop(w, decode), _pop(w, prefill, PRE)
    batch = ScheduledBatch(
        node_name=NODE, graph_walk=MIXED,
        request_to_worker_graph=dict(dec.request_to_worker_graph),
        input_edges=dec.input_edges, output_signals=list(dec.output_signals),
        request_walks=dict.fromkeys(decode, WALK),
        walk_output_signals={WALK: list(dec.output_signals)},
    )
    pre.request_walks = dict.fromkeys(prefill, PRE)
    pre.walk_output_signals = {PRE: list(pre.output_signals)}
    pre.graph_walk = MIXED
    batch.merge(pre)
    w._set_in_flight_flag(batch, True)
    return PendingBatch(
        batch=batch, node_batch=w._build_executing_batch(batch), node_name=NODE,
        partition=PART, graph_walk=MIXED, future=None,
    )


_ROWS_ELIGIBLE = SpeculationPreparer.rows_eligible


def _speculate(w: Worker, pending: PendingBatch, rows: bool, monkeypatch):
    # the columnar path where the worker would take it, else the per-row one
    eligible = _ROWS_ELIGIBLE if rows else (lambda *a, **k: False)
    monkeypatch.setattr(SpeculationPreparer, "rows_eligible", eligible)
    target = w._choose_spec_target(pending.batch)
    if target is None:
        return None
    return w._try_speculate_next(pending, target)


def _view(spec, read_inputs: bool = True):
    """Everything downstream reads off a speculation."""
    if spec is None:
        return None
    sb, nb = spec.scheduled_batch, spec.node_batch
    view = dict(
        node=sb.node_name, walk=sb.graph_walk,
        rtwg=list(sb.request_to_worker_graph.items()),
        request_walks=dict(sb.request_walks), output_signals=tuple(sb.output_signals),
        tp_seq=sb.tp_seq, input_edges=len(sb.input_edges),
        request_ids=list(nb.request_ids),
        info=[(r, id(i)) for r, i in nb.per_request_info.items()],
        meta={
            r: (dict(m.stream_chunks), dict(m.dynamic_loop_iter_counts))
            for r, m in nb.per_request_input_metadata.items()
        },
        wrapped={
            r: (id(x.info), dict(x.dynamic_loop_iter_counts), x.max_tokens)
            for r, x in nb.per_request_info_wrapped.items()
        },
        final=(set(nb.final_stream_rids), set(nb.stream_partition_done_rids)),
        ctx=(
            tuple(nb.step_context.request_ids), nb.step_context.graph_walk,
            dict(nb.step_context.request_walks), nb.step_context.slot,
            nb.step_context.capture,
        ),
        consumed=set(spec.consumed_edges), continuing=set(spec.continuing_rids),
        partition=spec.partition, new_iter=spec.is_new_iter, same=spec.is_same_node,
        loop=spec.loop_name, spec_id=spec.spec_id, dropped=set(spec.dropped),
        streaming=dict(spec.consumed_streaming_edges), yield_away=spec.is_yield_away,
    )
    if read_inputs:
        inputs = nb.per_request_input_tensors
        view["inputs"] = {
            r: {k: [id(t) for t in v] for k, v in inputs[r].items()} for r in inputs
        }
    return view


def _rows_only(pending: PendingBatch, missing=()) -> BatchedModelOutput:
    return BatchedModelOutput(
        rows_only=True,
        row_request_ids=tuple(r for r in pending.node_batch.request_ids if r not in missing),
    )


def _route(w: Worker, pending: PendingBatch) -> None:
    """N's postprocess as far as graph state goes: consumed inputs cleared,
    the loop-back routed with no tensor (it stays on the device), the shared
    loop counters refreshed."""
    rt, b, node = w._graph_runtime, pending.batch, pending.node_name
    rids = list(b.request_to_worker_graph)
    wg_ids = [b.request_to_worker_graph[r] for r in rids]
    w.tensor_manager.cleanup_collectable(*rt.cleanup_consumed_inputs(node, rids, wg_ids))
    rt.clear_pending_loop_stops()
    # a mixed step's prefill rows hand their token to the conductor: only the
    # loop's rows route here
    rids = b.rids_by_walk().get(WALK, [])
    wg_ids = [b.request_to_worker_graph[r] for r in rids]
    rt.complete_and_route_batch(RouteInput(
        partition=PART, graph_walk=WALK, node_name=node,
        output_signals=list(b.output_signals), wg_ids=ParallelList(rids, wg_ids),
        tensors=[], num_tensors=[0] * (len(rids) * len(b.output_signals)),
    ))
    rt.update_dynamic_loop_iters(pending.node_batch.per_request_info, PART)


def _advance(w: Worker, pending: PendingBatch, spec) -> PendingBatch:
    """Await N, thread, settle, postprocess N: the run loop's order."""
    w._clear_in_flight_flag(pending.batch)
    w._settle_speculation(spec, success=True, dropped_rids=spec.dropped)
    _route(w, pending)
    return PendingBatch(
        batch=spec.scheduled_batch, node_batch=spec.node_batch,
        node_name=spec.scheduled_batch.node_name,
        partition=PART, graph_walk=WALK, future=None,
        speculative_new_iter=spec.is_new_iter, loop_name=spec.loop_name,
    )


def _step_both(w, pending, monkeypatch, outputs=None, check_unread=True):
    """Speculate from N on both paths, compare, thread N's outputs into both,
    compare again; returns the rows-path speculation."""
    general = _speculate(w, pending, rows=False, monkeypatch=monkeypatch)
    rows = _speculate(w, pending, rows=True, monkeypatch=monkeypatch)
    if general is None or rows is None:
        assert general is None and rows is None
        return None
    assert isinstance(rows.node_batch.per_request_input_tensors, DeferredRowInputs)
    # the inputs only after threading: reading them first would build every
    # row, which a step on the uniform path never does
    assert _view(rows, read_inputs=not check_unread) == _view(
        general, read_inputs=not check_unread,
    )
    outputs = outputs if outputs is not None else _rows_only(pending)
    w._thread_outputs_to_speculative(general, outputs)
    w._thread_outputs_to_speculative(rows, outputs)
    assert _view(rows) == _view(general)
    return rows


@pytest.mark.parametrize("kind", KINDS)
def test_a_decode_loop_builds_the_same_steps(kind, monkeypatch):
    w = _worker(kind)
    rids = _admit(w, 6)
    pending = _first_step(w, rids)
    for k in range(5):
        spec = _step_both(w, pending, monkeypatch)
        assert spec is not None
        # one past the iteration in flight, and only that loop moved
        counts = {r: dict(x.dynamic_loop_iter_counts) for r, x in spec.node_batch.per_request_info_wrapped.items()}
        assert counts == dict.fromkeys(rids, {"other_loop": 7, LOOP: k + 1})
        assert dict(spec.node_batch.per_request_input_tensors[rids[0]]) == {SIG: []}
        pending = _advance(w, pending, spec)
        # routing N moved the shared counters; the step's own copy did not
        assert w.request_state.get_fwd_info(rids[0], PART).dynamic_loop_iter_counts[LOOP] == k + 1
        assert dict(spec.node_batch.per_request_info_wrapped[rids[0]].dynamic_loop_iter_counts)[LOOP] == k + 1


@pytest.mark.parametrize("kind", KINDS)
def test_a_two_node_loop_builds_the_same_steps(kind, monkeypatch):
    # every step hops to the other node: NODE -> NODE_B within an iteration,
    # NODE_B -> NODE into the next one
    w = _worker(kind, pair=True)
    rids = _admit(w, 5)
    pending = _first_step(w, rids)
    hops = []
    for _ in range(6):
        spec = _step_both(w, pending, monkeypatch)
        assert spec is not None
        hops.append((spec.scheduled_batch.node_name, spec.is_new_iter))
        pending = _advance(w, pending, spec)
    assert hops == [(NODE_B, False), (NODE, True)] * 3


@pytest.mark.parametrize("outputs", ["rows_only", "loopback", "routed"])
@pytest.mark.parametrize("kind", KINDS)
def test_a_step_that_mixes_walks_continues_its_loop_the_same(kind, outputs, monkeypatch):
    # N mixes decode rows with prefill rows; the speculation continues the
    # decode rows, which take N's outputs however N kept them
    w = _worker(kind, mixed=True)
    decode = _admit(w, 4)
    prefill = _admit(w, 3, start=4, walk=PRE)
    pending = _mixed_first_step(w, decode, prefill)
    out = {
        "rows_only": _rows_only(pending),
        "loopback": BatchedModelOutput(loopback_rids=frozenset(decode)),
        "routed": BatchedModelOutput(
            per_rid_outputs={r: {SIG: [torch.tensor([r])]} for r in decode + prefill},
        ),
    }[outputs]
    spec = _step_both(w, pending, monkeypatch, outputs=out)
    assert list(spec.node_batch.request_ids) == decode
    assert spec.scheduled_batch.graph_walk == WALK and not spec.scheduled_batch.request_walks
    pending = _advance(w, pending, spec)
    for _ in range(2):
        pending = _advance(w, pending, _step_both(w, pending, monkeypatch))


@pytest.mark.parametrize("kind", KINDS)
def test_reading_rows_before_threading_changes_nothing(kind, monkeypatch):
    w = _worker(kind)
    pending = _first_step(w, _admit(w, 4))
    for _ in range(2):
        pending = _advance(w, pending, _step_both(w, pending, monkeypatch, check_unread=False))


@pytest.mark.parametrize("kind", KINDS)
def test_rows_at_their_last_iteration_leave_on_both_paths(kind, monkeypatch):
    # staggered: the second group starts two iterations later, so the first
    # group reaches max_iters while the second still runs
    w = _worker(kind, max_iters=4)
    first = _admit(w, 3)
    pending = _first_step(w, first)
    pending = _advance(w, pending, _step_both(w, pending, monkeypatch))
    late = _admit(w, 2, start=3)
    w.scheduler.fresh = _pop(w, late)
    spec = _step_both(w, pending, monkeypatch)
    w.scheduler.fresh = None
    assert list(spec.node_batch.request_ids) == first + late
    # fresh rows run at their own counters: not incremented
    counts = {r: x.dynamic_loop_iter_counts.get(LOOP, 0) for r, x in spec.node_batch.per_request_info_wrapped.items()}
    assert counts == {**dict.fromkeys(first, 2), **dict.fromkeys(late, 0)}
    pending = _advance(w, pending, spec)
    seen = []
    while True:
        spec = _step_both(w, pending, monkeypatch)
        if spec is None:
            break
        seen.append(list(spec.node_batch.request_ids))
        pending = _advance(w, pending, spec)
    assert seen[0] == first + late and seen[-1] == late


@pytest.mark.parametrize("kind", KINDS)
def test_teardown_and_room_filter_the_same_rows(kind, monkeypatch):
    w = _worker(kind)
    rids = _admit(w, 6)
    pending = _first_step(w, rids)
    w._pending_removes = {rids[1]: object()}
    w.scheduler.room = 3
    spec = _step_both(w, pending, monkeypatch)
    assert list(spec.node_batch.request_ids) == [rids[0], rids[2], rids[3]]


@pytest.mark.parametrize("kind", KINDS)
def test_a_row_without_its_loop_back_drops_on_both_paths(kind, monkeypatch):
    w = _worker(kind)
    rids = _admit(w, 4)
    pending = _first_step(w, rids)
    spec = _step_both(w, pending, monkeypatch, outputs=_rows_only(pending, missing={rids[2]}))
    assert spec.dropped == {rids[2]}
    assert rids[2] not in spec.node_batch.request_ids
    assert not isinstance(spec.node_batch.per_request_input_tensors, DeferredRowInputs)
    _advance(w, pending, spec)


@pytest.mark.parametrize("pair", [False, True])
@pytest.mark.parametrize("kind", KINDS)
def test_routed_outputs_are_threaded_row_by_row(kind, pair, monkeypatch):
    # N kept its outputs per request (not rows-only): the rows path hands over
    # to the per-row splice, which takes N's tensor lists
    w = _worker(kind, pair=pair)
    rids = _admit(w, 3)
    pending = _first_step(w, rids)
    sig = SIG_B if pair else SIG
    toks = {r: torch.tensor([r]) for r in rids}
    outputs = BatchedModelOutput(per_rid_outputs={r: {sig: [toks[r]]} for r in rids})
    spec = _step_both(w, pending, monkeypatch, outputs=outputs)
    assert {r: spec.node_batch.per_request_input_tensors[r][sig][0] for r in rids} == toks


def _engine_facing_types(spec) -> dict:
    """The types of what the engine's resources read off a step: its context
    and its rids. A rid of another type (a numpy or pyo3 integer, say) hashes
    and compares more slowly in every per-row resource loop, and a context
    built another way can defeat its cached row sets."""
    nb = spec.node_batch
    ctx = nb.step_context
    return dict(
        ctx={k: type(v) for k, v in vars(ctx).items()},
        ctx_rids={type(r) for r in ctx.request_ids},
        info_keys={type(r) for r in nb.per_request_info},
        info_values={type(v) for v in nb.per_request_info.values()},
        wrapped={type(v) for v in nb.per_request_info_wrapped.values()},
        counts={
            (type(k), type(v))
            for m in nb.per_request_input_metadata.values()
            for k, v in m.dynamic_loop_iter_counts.items()
        },
        rtwg={(type(k), type(v)) for k, v in spec.scheduled_batch.request_to_worker_graph.items()},
    )


@pytest.mark.parametrize("kind", KINDS)
def test_what_the_engine_reads_has_the_same_types_on_both_paths(kind, monkeypatch):
    w = _worker(kind)
    pending = _first_step(w, _admit(w, 4))
    general = _speculate(w, pending, rows=False, monkeypatch=monkeypatch)
    rows = _speculate(w, pending, rows=True, monkeypatch=monkeypatch)
    assert _engine_facing_types(rows) == _engine_facing_types(general)
    assert _engine_facing_types(rows)["ctx_rids"] == {int}


@pytest.mark.parametrize("kind", KINDS)
def test_prep_spec_rows_matches_the_default_composition(kind):
    """The runtime's own ``prep_spec_rows`` against the contract's default,
    built from ``prep_spec_rids``, at every iteration of a loop."""
    w = _worker(kind, max_iters=4)
    rt = w._graph_runtime
    rids = _admit(w, 3)
    pending = _first_step(w, rids)
    for _ in range(4):
        args = (NODE, NODE, WALK, list(pending.batch.request_to_worker_graph))
        for room in (None, 1):
            assert rt.prep_spec_rows(*args, room) == GraphRuntime.prep_spec_rows(rt, *args, room)
        _route(w, pending)
        nxt = _pop(w, list(pending.batch.request_to_worker_graph))
        if not nxt.request_to_worker_graph:
            break
        pending = PendingBatch(
            batch=nxt, node_batch=w._build_executing_batch(nxt), node_name=NODE,
            partition=PART, graph_walk=WALK, future=None,
        )
