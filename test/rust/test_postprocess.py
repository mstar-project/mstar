"""The Rust runtime's native ``begin_postprocess`` / ``resume_postprocess``
must match ``PhasedPostprocess``, the Python reference, run over the same
runtime's phase methods (``check_stops``, ``stop_loops_batched``,
``complete_and_route_batch``, ``send_outputs``).

Two runtimes are driven through the same decode loop, one per path, and
everything observable is compared: where each step stops for the worker and
what it hands over, the frames each sends (read off real inboxes), and the
loop counters. One row samples EOS partway through, so steps cover the native
send, the send with stopped rows, and the completion of the worker graph.
"""
import sys
from concurrent.futures import Future
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest

pytest.importorskip("mstar_rust")

import torch
from mstar_rust import ZmqCommunicator

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.communication.tensor_store import RustTensorBookkeeping
from mstar.communication.wire import decode
from mstar.distributed.base import ShardingConfig
from mstar.engine.accelerator_graph_runner import ReplayEvents
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources.step import StepContext
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.graph.runtime import rust as rust_runtime
from mstar.graph.runtime.base import (
    ColumnarEdgeSpecs,
    InlineColumn,
    PhasedPostprocess,
    PostprocessNeed,
    PostprocessRequest,
    RouteTensors,
    SendInput,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import WorkerGraph
from mstar.model.submodule_base import BatchedModelOutput, HostRows, StopKind, StopPolicy
from mstar.testing_utils.worker import make_worker
from mstar.utils.containers import ParallelList
from mstar.worker.scheduled_batch import ScheduledBatch
from mstar.worker.step_executor import PendingBatch
from mstar.worker.step_repeats import stream_run

WALK, NODE, LOOP, SIG, PART, ME = "decode", "LLM", "decode_loop", "text_inputs", "default", "w0"
MAX_ITERS = 4
EOS = 7
POLICY = StopPolicy(StopKind.AR, token_signal=SIG, eos_tokens=(EOS,))


class _Reference(rust_runtime.RustGraphRuntime):
    """The Rust runtime with the Python reference's begin / resume."""
    begin_postprocess = PhasedPostprocess.begin_postprocess
    resume_postprocess = PhasedPostprocess.resume_postprocess


def _runtime(cls, sock_dir):
    section = Loop(
        name=LOOP,
        section=GraphNode(
            name=NODE, input_names=[SIG],
            outputs=[
                GraphEdge(next_node=NODE, name=SIG),
                GraphEdge(next_node=EMIT_TO_CLIENT, name=SIG, output_modality="text"),
            ],
        ),
        max_iters=MAX_ITERS, outputs=[],
    )
    wg = WorkerGraph(section=section, graph_walks={WALK}, ranks=[0], worker_graph_id=0)
    rt = cls(
        my_worker_id=ME, my_worker_graphs=[wg],
        all_wg_ids_to_graph_walks={0: {WALK}}, all_wg_ids_to_dyn_loops={0: {LOOP}},
        all_wg_ids_to_nodes={0: {NODE}}, node_to_partition={NODE: PART},
        sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        bookkeeping=RustTensorBookkeeping(),
        communicator=SimpleNamespace(_inner=ZmqCommunicator(ME, sock_dir)),
    )
    rt.set_stop_policies(NODE, WALK, {LOOP: POLICY})
    return rt


def _collect(box):
    out = []
    first = box.recv_timeout(1000)
    if first is None:
        return out
    out.append(decode(first))
    while (more := box.recv_timeout(50)) is not None:
        out.append(decode(more))
    return out


def _start(rt, n, limits=True):
    rids = []
    for i in range(n):
        rid = rt.add_request(
            request_id=f"r{i}", partition=PART, graph_walk=WALK,
            partition_worker_graph_ids=[0],
            worker_graph_to_workers=ParallelList([0], [[ME]]), loop_caps={})
        if limits:
            # a limit no step reaches: EOS and the loop's cap end the rows
            rt.register_stop_limits(rid, SimpleNamespace(max_tokens=1 << 20, resource_configs={}, partition_name=PART))
        rids.append(rid)
    block = ColumnarEdgeSpecs.empty()
    for rid in rids:
        block.add(rid, SIG, [], False, next_node=NODE)
    rt.ingest_inputs_batch(block)
    return rids


def _drive(rt, request, values, model_stops, dropped=()):
    """One step's postprocess as the worker runs it; each break, summarized."""
    trace = []
    step = rt.begin_postprocess(request)
    while True:
        trace.append((
            step.need, None if step.wg_ids is None else list(step.wg_ids.keys),
            sorted(step.stopped_rids), sorted(step.completed_rids),
        ))
        if step.need is PostprocessNeed.DONE:
            return trace
        if step.need is PostprocessNeed.STOPS:
            step = rt.resume_postprocess(stops=model_stops, dropped=dropped)
        elif step.need is PostprocessNeed.STORE:
            keys = step.wg_ids.keys
            step = rt.resume_postprocess(stored=RouteTensors(
                [], [0] * len(keys), SIG, [values[r] for r in keys],
            ))
        else:
            info = sorted(step.route_output.rids_needing_request_info)
            step = rt.resume_postprocess(send=SendInput(
                completion_id=step.route_output.completion_id,
                per_request_info=ParallelList(info, [None] * len(info)),
                new_token_counts=ParallelList([], []),
            ))


@pytest.mark.parametrize(
    "mode", ["runtime", "model_stops", "stored"],
    ids=["runtime_stops_inline", "model_stops_inline", "runtime_stops_stored"],
)
@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("reverse_rows", [False, True], ids=["in_order", "permuted"])
def test_native_matches_reference(tmp_path, mode, dtype, reverse_rows):
    n = 5
    boxes, rts = {}, {}
    for k, cls in (("native", rust_runtime.RustGraphRuntime), ("reference", _Reference)):
        d = tmp_path / k
        d.mkdir()
        boxes[k] = {p: ZmqCommunicator(p, str(d)) for p in ("api_server", "conductor")}
        rts[k] = _runtime(cls, str(d))
    rids = [_start(rt, n) for rt in rts.values()]
    assert rids[0] == rids[1]
    active = rids[0]
    needs = set()
    for step in range(MAX_ITERS):
        for rt in rts.values():
            assert rt.pop_rids(NODE, WALK, active) is not None
        # row 1 samples EOS on the second step
        values = {r: EOS if (step, i) == (1, 1) else 100 * step + i for i, r in enumerate(active)}
        # host rows as the stop check stages them: [rows, 1], row i for row_rids[i]
        row_rids = list(reversed(active)) if reverse_rows else list(active)
        rows = torch.tensor([[values[r]] for r in row_rids], dtype=dtype)
        column = rows.reshape(rows.shape[0], -1)[:, 0]
        # the model's stops, checked before the runtime when it does not check
        model_stops = {r: {LOOP} for r, v in values.items() if v == EOS}
        request = PostprocessRequest(
            partition=PART, node_name=NODE, graph_walk=WALK, output_signals=[SIG],
            wg_ids=ParallelList(list(active), [0] * len(active)),
            stops=model_stops if mode == "model_stops" else None,
            runtime_stops=mode != "model_stops", tokens=column, row_rids=row_rids,
            inline=None if mode == "stored" else InlineColumn(SIG, column),
        )
        traces = {k: _drive(rt, request, values, model_stops) for k, rt in rts.items()}
        assert traces["native"] == traces["reference"], step
        needs |= {t[0] for t in traces["native"]}
        stopped = set(traces["native"][-1][2])
        assert (
            rts["native"].get_dynamic_loop_iters(active, PART).values
            == rts["reference"].get_dynamic_loop_iters(active, PART).values
        )
        active = [r for r in active if r not in stopped]
    expected = {PostprocessNeed.DONE, PostprocessNeed.SEND}
    expected |= {"stored": {PostprocessNeed.STORE}}.get(mode, set())
    assert needs == expected
    for peer in ("api_server", "conductor"):
        got = {k: _collect(boxes[k][peer]) for k in boxes}
        assert got["native"] == got["reference"], peer
        assert got["native"], peer
    for rt in rts.values():
        assert rt._rust.num_parked_completions() == 0


def test_dropped_rows_are_not_routed(tmp_path):
    """When the runtime's own check cannot decide (here: no limits), a row
    ``check_stop`` then fails (STOPS' ``dropped``) leaves the step: it is
    neither stopped nor routed, in either implementation."""
    traces = {}
    for k, cls in (("native", rust_runtime.RustGraphRuntime), ("reference", _Reference)):
        d = tmp_path / k
        d.mkdir()
        rt = _runtime(cls, str(d))
        rids = _start(rt, 3, limits=False)
        assert rt.pop_rids(NODE, WALK, rids) is not None
        column = torch.tensor([1, 2, 3], dtype=torch.int64)
        request = PostprocessRequest(
            partition=PART, node_name=NODE, graph_walk=WALK, output_signals=[SIG],
            wg_ids=ParallelList(list(rids), [0] * 3), runtime_stops=True, tokens=column,
            row_rids=rids, inline=InlineColumn(SIG, column),
        )
        traces[k] = _drive(
            rt, request, dict(zip(rids, [1, 2, 3], strict=True)),
            model_stops={rids[1]: {LOOP}}, dropped=[rids[1]],
        )
    assert traces["native"] == traces["reference"]
    need, routed, stopped, _ = traces["native"][-1]
    assert need is PostprocessNeed.DONE and stopped == []
    assert traces["native"][0][0] is PostprocessNeed.STOPS


def test_a_row_with_no_host_value_routes_through_the_store(tmp_path):
    """A row that ran no forward (``ExecutingBatch.no_forward_rids``) has no
    host row: the step stores instead of riding inline, and the stored route
    takes the row's value as None (the tensor path), in either implementation."""
    traces, frames = {}, {}
    for k, cls in (("native", rust_runtime.RustGraphRuntime), ("reference", _Reference)):
        d = tmp_path / k
        d.mkdir()
        box = ZmqCommunicator("api_server", str(d))
        rt = _runtime(cls, str(d))
        rids = _start(rt, 3)
        assert rt.pop_rids(NODE, WALK, rids) is not None
        ran = rids[:2]
        column = torch.tensor([100, 101], dtype=torch.int64)
        traces[k] = _drive(rt, PostprocessRequest(
            partition=PART, node_name=NODE, graph_walk=WALK, output_signals=[SIG],
            wg_ids=ParallelList(list(rids), [0] * 3), stops={}, tokens=column, row_rids=ran,
            inline=InlineColumn(SIG, column),
        ), {ran[0]: 100, ran[1]: 101, rids[2]: None}, model_stops={})
        frames[k] = _collect(box)
    assert traces["native"] == traces["reference"]
    assert [t[0] for t in traces["native"]] == [PostprocessNeed.STORE, PostprocessNeed.DONE]
    assert frames["native"] == frames["reference"] and frames["native"]


@pytest.mark.parametrize("landing", [False, True], ids=["at_postprocess", "host_events"])
def test_missing_limits_leave_the_check_to_the_model(tmp_path, landing):
    """A row with no registered limits: the runtime decides nothing (STOPS).
    Nor while a repeated step runs: the host sends nothing and stops looking
    at once, the step still running, leaving it all to the postprocess."""
    rt = _runtime(rust_runtime.RustGraphRuntime, str(tmp_path))
    rid = rt.add_request(
        request_id="r", partition=PART, graph_walk=WALK, partition_worker_graph_ids=[0],
        worker_graph_to_workers=ParallelList([0], [[ME]]), loop_caps={})
    if landing:
        wide = torch.full((MAX_ITERS, 1), EOS, dtype=torch.int64).t()
        running = SimpleNamespace(query=lambda: pytest.fail("the host kept looking at a step it can't send"))
        landed = _landing()[0]
        assert stream_run(rt, PART, NODE, WALK, [rid], [rid], wide, InlineColumn(SIG, wide), landed, running)[0] == 0
    column = torch.tensor([EOS], dtype=torch.int64)
    step = rt.begin_postprocess(PostprocessRequest(
        partition=PART, node_name=NODE, graph_walk=WALK, output_signals=[SIG],
        wg_ids=ParallelList([rid], [0]), runtime_stops=True, tokens=column, row_rids=[rid],
    ))
    assert step.need is PostprocessNeed.STOPS


@pytest.mark.parametrize("drop", [False, True], ids=["all_rows", "a_row_dropped"])
def test_each_rows_loop_stride_advances_its_loop(tmp_path, drop):
    """A chunk-Loop step counts tokens: ``loop_strides`` moves each row's loop
    counter by its own stride, in either implementation, and a dropped row's
    stride leaves with it."""
    iters = {}
    for k, cls in (("native", rust_runtime.RustGraphRuntime), ("reference", _Reference)):
        d = tmp_path / k
        d.mkdir()
        rt = _runtime(cls, str(d))
        # without limits the runtime defers to the model's check (STOPS)
        rids = _start(rt, 3, limits=not drop)
        dropped = [rids[1]] if drop else []
        got = []
        for step, strides in enumerate(([2, 1, 3], [1, 1, 1])):
            live = [r for r in rids if r not in dropped] if step else rids
            assert rt.pop_rids(NODE, WALK, live) is not None
            values = {r: 100 * step + i for i, r in enumerate(live)}
            column = torch.tensor([values[r] for r in live], dtype=torch.int64)
            _drive(rt, PostprocessRequest(
                partition=PART, node_name=NODE, graph_walk=WALK, output_signals=[SIG],
                wg_ids=ParallelList(list(live), [0] * len(live)), runtime_stops=True,
                tokens=column, row_rids=live, inline=InlineColumn(SIG, column),
                loop_strides=[s for r, s in zip(rids, strides, strict=True) if r in live],
            ), values, model_stops={}, dropped=dropped if step == 0 else ())
            got.append([it[LOOP] for it in rt.get_dynamic_loop_iters(live, PART).values])
        iters[k] = got
    assert iters["native"] == iters["reference"]
    # the dropped row does not move; the last row's 3 + 1 reaches MAX_ITERS and resets
    assert iters["native"] == ([[2, 0, 3], [3, 0]] if drop else [[2, 1, 3], [3, 2, 0]])


# a repeated step's tokens, a row's per iteration: row 1 samples EOS at its
# second; the device ran both rows on to MAX_ITERS
RUN = [[1, 2, 3, 5], [4, EOS, 6, 8]]


def _sent(frames) -> dict:
    """Per request, its client values in order, each with its loop iteration."""
    sent = {}
    for frame in frames:
        body = frame.body
        for rid, value, idx in zip(body.request_ids, body.values, body.loop_indices, strict=True):
            sent.setdefault(rid, []).append((value, idx.loop_indices[LOOP]))
    return sent


def _counted(frames) -> dict:
    """Per request, the client chunks its completion reports count."""
    counts = {}
    for frame in frames:
        names = getattr(frame.body, "output_signal_names", None)
        if names is not None:
            counts[frame.body.request_id] = counts.get(frame.body.request_id, 0) + len(names)
    return counts


class _Event:
    """A host replay's event: done, or done once the host waits on it, when
    ``lands`` (it included) all land."""

    def __init__(self, done=False):
        self.done, self.lands = done, ()

    def query(self):
        return self.done

    def synchronize(self):
        for event in self.lands:
            event.done = True


def _landing(late=False):
    """How a repeated step's replays land as the host looks (``stream_run``):
    the first has landed when it first looks, the next two land together, then
    the last, each as its replay's event says. ``late``: the last lands while
    the host sends the two before it, so the host finds the step complete and
    leaves the last to its postprocess."""
    events = [_Event(done=True), _Event(), _Event(), _Event()]
    events[1].lands, events[3].lands = events[1:3], events[3:]
    looks = iter([False, True])

    def query():
        if not late:
            return all(e.done for e in events)
        complete = next(looks)
        events[3].done = events[3].done or complete
        return complete

    return ReplayEvents(events), SimpleNamespace(query=query)


def _through_the_worker(rt, rids, wide):
    """The repeated step as the worker postprocesses it
    (``StepExecutor.postprocess``), its replays landing as ``_landing``
    has them: its tokens go out as they land, then its route sends the rest."""
    landing, completion = _landing()
    engine = SimpleNamespace(
        inline_signals=lambda node, walk: {SIG: "new_token"},
        device_loopback_signals=lambda node, walk: (SIG,),
        step_finished=lambda *a, **k: None, after_output_copy=lambda batch: None,
        materialize_publish=lambda batch, rids: None, finalize_stopped_requests=lambda batch, rids: None,
    )
    w = make_worker(
        graph_runtime=rt, engine_manager=SimpleNamespace(get_engine=lambda node: engine),
        tensor_manager=SimpleNamespace(
            get_tensor=None, cleanup_collectable=lambda *a: None, drop_host_copies=lambda: None,
        ),
        request_state=SimpleNamespace(
            get_fwd_info=lambda rid, partition: None, get_pending_publish_info=lambda rid, partition: {},
            flush_publish_info=lambda rid, partition: None, buffer_publish_info=lambda *a: None,
        ),
        streams=SimpleNamespace(assign_consumer_walks=lambda batch: None, walk_driven_edges={}, consumer_node={}),
    )
    w.executor._stop_token_signal[(NODE, WALK)] = "new_token"
    node_batch = ExecutingBatch(
        node_name=NODE, per_request_info={r: SimpleNamespace(rid=r, dynamic_loop_iter_counts={}) for r in rids},
        step_context=StepContext(request_ids=tuple(rids), graph_walk=WALK, slot=0, capture=False, repeats=MAX_ITERS),
        completion_event=completion,
    )
    outputs = BatchedModelOutput(
        check_stop_buffers={"new_token": wide}, row_request_ids=tuple(rids), rows_only=True, landing=landing,
        host_rows=HostRows(tuple(rids), {"new_token": wide}),
    )
    batch = ScheduledBatch(
        node_name=NODE, graph_walk=WALK, request_to_worker_graph=dict.fromkeys(rids, 0), output_signals=[SIG],
    )
    w.executor.postprocess(PendingBatch(
        batch=batch, node_batch=node_batch, node_name=NODE, partition=PART, graph_walk=WALK, future=Future(),
    ), outputs)


@pytest.mark.parametrize(
    "landing", [None, "host", "late", "worker"], ids=["at_postprocess", "host_events", "last_late", "worker"],
)
def test_a_repeated_steps_client_tokens_match_one_iteration_a_step(tmp_path, landing):
    """A step that ran several iterations hands the client a token per
    iteration: a row's tokens up to its stop, in order, each with its own
    iteration's loop index and counted by the completion report, and nothing
    the device ran past the stop. The same when they go out while the step
    runs (``stream_run``) as the host learns each iteration landed (``host``:
    from its replay's event; ``late``: the step completes before the last
    lands, as far as the host saw), and its postprocess sends the rest, the worker's too (``worker``), which
    starts the route where the early sends left off. The reference runs the
    same tokens one iteration a step."""
    sent = {}
    for k in ("one_a_step", "repeated"):
        d = tmp_path / k
        d.mkdir()
        boxes = {p: ZmqCommunicator(p, str(d)) for p in ("api_server", "conductor")}
        rt = _runtime(rust_runtime.RustGraphRuntime, str(d))
        rids = _start(rt, 2)
        if k == "one_a_step":
            active = list(rids)
            for i in range(MAX_ITERS):
                assert rt.pop_rids(NODE, WALK, active) is not None
                column = torch.tensor([RUN[rids.index(r)][i] for r in active], dtype=torch.int64)
                trace = _drive(rt, PostprocessRequest(
                    partition=PART, node_name=NODE, graph_walk=WALK, output_signals=[SIG],
                    wg_ids=ParallelList(list(active), [0] * len(active)), runtime_stops=True,
                    tokens=column, row_rids=active, inline=InlineColumn(SIG, column),
                ), {}, model_stops={})
                active = [r for r in active if r not in trace[-1][2]]
        else:
            assert rt.pop_rids(NODE, WALK, rids) is not None
            # landed as a repeated step lands them: [iterations, rows], viewed
            # as a column per iteration
            wide = torch.tensor(RUN, dtype=torch.int64).t().contiguous().t()
            if landing == "worker":
                _through_the_worker(rt, rids, wide)
                sent[k] = (_sent(_collect(boxes["api_server"])), _counted(_collect(boxes["conductor"])))
                continue
            start = 0
            if landing is not None:
                start, _ = stream_run(
                    rt, PART, NODE, WALK, list(rids), rids, wide, InlineColumn(SIG, wide),
                    *_landing(late=landing == "late"),
                )
                assert start == {"host": MAX_ITERS, "late": 3}[landing]
            trace = _drive(rt, PostprocessRequest(
                partition=PART, node_name=NODE, graph_walk=WALK, output_signals=[SIG],
                wg_ids=ParallelList(list(rids), [0, 0]), runtime_stops=True,
                tokens=wide, row_rids=rids, inline=InlineColumn(SIG, wide, start),
                loop_strides=[MAX_ITERS] * 2, token_width=MAX_ITERS,
            ), {}, model_stops={})
            # row 0 runs to the Loop's cap, which ends it without a stop
            assert trace[-1][2] == [rids[1]]
        sent[k] = (_sent(_collect(boxes["api_server"])), _counted(_collect(boxes["conductor"])))
    assert sent["repeated"] == sent["one_a_step"]
    assert sent["one_a_step"] == (
        {"r0": [(1, 0), (2, 1), (3, 2), (5, 3)], "r1": [(4, 0), (EOS, 1)]}, {"r0": 4, "r1": 2},
    )


def test_a_walk_that_repeats_is_not_refused_by_another_walks_stream(tmp_path):
    """A stream goes one value a step, so a step that streams a signal by value
    can run one iteration alone. One node whose walk A streams ``SIG`` to
    another partition and whose walk B loops on itself with ``SIG`` out to the
    client: B's width-2 step is legal, though the node streams ``SIG`` in A."""
    stream_walk, loop_walk = "stream", "loop"
    streamed = WorkerGraph(
        section=GraphNode(
            name=NODE, input_names=[SIG],
            outputs=[GraphEdge(next_node="consumer", name=SIG, is_streaming=True)],
        ),
        graph_walks={stream_walk}, ranks=[0], worker_graph_id=0,
    )
    consumer = WorkerGraph(
        section=GraphNode(
            name="consumer", input_names=[SIG], outputs=[GraphEdge(next_node=EMIT_TO_CLIENT, name="audio")],
        ),
        graph_walks={"consume"}, ranks=[0], worker_graph_id=1,
    )
    loop = Loop(
        name=LOOP,
        section=GraphNode(
            name=NODE, input_names=[SIG],
            outputs=[
                GraphEdge(next_node=NODE, name=SIG),
                GraphEdge(next_node=EMIT_TO_CLIENT, name=SIG, output_modality="text"),
            ],
        ),
        max_iters=MAX_ITERS, outputs=[],
    )
    looped = WorkerGraph(section=loop, graph_walks={loop_walk}, ranks=[0], worker_graph_id=2)
    rt = rust_runtime.RustGraphRuntime(
        my_worker_id=ME, my_worker_graphs=[streamed, consumer, looped],
        all_wg_ids_to_graph_walks={0: {stream_walk}, 1: {"consume"}, 2: {loop_walk}},
        all_wg_ids_to_dyn_loops={0: set(), 1: set(), 2: {LOOP}},
        all_wg_ids_to_nodes={0: {NODE}, 1: {"consumer"}, 2: {NODE}},
        node_to_partition={NODE: PART, "consumer": "consumer"},
        sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        bookkeeping=RustTensorBookkeeping(),
        communicator=SimpleNamespace(_inner=ZmqCommunicator(ME, str(tmp_path))),
    )
    rt.set_stop_policies(NODE, loop_walk, {LOOP: POLICY})
    rt.set_inline_streams(frozenset({(NODE, SIG)}))
    rid = rt.add_request(
        request_id="r0", partition=PART, graph_walk=loop_walk, partition_worker_graph_ids=[2],
        worker_graph_to_workers=ParallelList([2], [[ME]]), loop_caps={},
    )
    rt.register_stop_limits(rid, SimpleNamespace(max_tokens=1 << 20, resource_configs={}, partition_name=PART))
    block = ColumnarEdgeSpecs.empty()
    block.add(rid, SIG, [], False, next_node=NODE)
    rt.ingest_inputs_batch(block)
    assert rt.pop_rids(NODE, loop_walk, [rid]) is not None
    column = torch.tensor([[1, 2]], dtype=torch.int64)
    trace = _drive(rt, PostprocessRequest(
        partition=PART, node_name=NODE, graph_walk=loop_walk, output_signals=[SIG],
        wg_ids=ParallelList([rid], [2]), runtime_stops=True,
        tokens=column, row_rids=[rid], inline=InlineColumn(SIG, column),
        loop_strides=[2], token_width=2,
    ), {}, model_stops={})
    assert trace[-1][0] is PostprocessNeed.DONE
