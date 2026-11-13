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
from mstar.model.submodule_base import StopKind, StopPolicy
from mstar.utils.containers import ParallelList

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


def _drive(rt, request, values, model_stops, dropped=(), withheld=None):
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
            step = rt.resume_postprocess(stops=model_stops, dropped=dropped, withheld=withheld)
        elif step.need is PostprocessNeed.STORE:
            keys = step.wg_ids.keys
            step = rt.resume_postprocess(stored=RouteTensors(
                [], [0] * len(keys), (SIG,), [values[r] for r in keys],
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
            inline=None if mode == "stored" else InlineColumn((SIG,), column),
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
            row_rids=rids, inline=InlineColumn((SIG,), column),
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
            inline=InlineColumn((SIG,), column),
        ), {ran[0]: 100, ran[1]: 101, rids[2]: None}, model_stops={})
        frames[k] = _collect(box)
    assert traces["native"] == traces["reference"]
    assert [t[0] for t in traces["native"]] == [PostprocessNeed.STORE, PostprocessNeed.DONE]
    assert frames["native"] == frames["reference"] and frames["native"]


def test_missing_limits_leave_the_check_to_the_model(tmp_path):
    """A row with no registered limits: the runtime decides nothing (STOPS)."""
    rt = _runtime(rust_runtime.RustGraphRuntime, str(tmp_path))
    rid = rt.add_request(
        request_id="r", partition=PART, graph_walk=WALK, partition_worker_graph_ids=[0],
        worker_graph_to_workers=ParallelList([0], [[ME]]), loop_caps={})
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
                tokens=column, row_rids=live, inline=InlineColumn((SIG,), column),
                loop_strides=[s for r, s in zip(rids, strides, strict=True) if r in live],
            ), values, model_stops={}, dropped=dropped if step == 0 else ())
            got.append([it[LOOP] for it in rt.get_dynamic_loop_iters(live, PART).values])
        iters[k] = got
    assert iters["native"] == iters["reference"]
    # the dropped row does not move; the last row's 3 + 1 reaches MAX_ITERS and resets
    assert iters["native"] == ([[2, 0, 3], [3, 0]] if drop else [[2, 1, 3], [3, 2, 0]])


def test_a_repeated_step_stops_a_row_at_its_iteration_and_emits_up_to_it(tmp_path):
    """A step that ran three iterations hands over a token per iteration: a row
    stops at the first that stops it, and the client gets its tokens up to that
    one, in order; what the device ran past the stop goes nowhere."""
    box = ZmqCommunicator("api_server", str(tmp_path))
    rt = _runtime(rust_runtime.RustGraphRuntime, str(tmp_path))
    rids = _start(rt, 2)
    assert rt.pop_rids(NODE, WALK, rids) is not None
    wide = torch.tensor([[1, 2, 3], [4, EOS, 5]], dtype=torch.int64)
    trace = _drive(rt, PostprocessRequest(
        partition=PART, node_name=NODE, graph_walk=WALK, output_signals=[SIG],
        wg_ids=ParallelList(list(rids), [0, 0]), runtime_stops=True,
        tokens=wide[:, 0], row_rids=rids, inline=InlineColumn((SIG,), wide[:, 0]),
        loop_strides=[3, 3], token_width=3,
    ), {}, model_stops={})
    assert trace[-1][2] == [rids[1]]
    sent = {}
    for frame in _collect(box):
        for rid, value in zip(frame.body.request_ids, frame.body.values, strict=True):
            sent.setdefault(rid, []).append(value)
    assert sent == {"r0": [1, 2, 3], "r1": [4, EOS]}


@pytest.mark.parametrize("known", [True, False], ids=["withheld_on_begin", "withheld_on_stops"])
def test_a_withheld_rows_inline_emit_goes_nowhere(tmp_path, known):
    """What the stop check withheld rides nothing inline, in either
    implementation, whether the request names it (the model's stops checked
    first) or the STOPS resume does (the runtime's check could not decide); the
    row's loop-back still feeds its next step."""
    frames, iters = {}, {}
    for k, cls in (("native", rust_runtime.RustGraphRuntime), ("reference", _Reference)):
        d = tmp_path / k
        d.mkdir()
        box = ZmqCommunicator("api_server", str(d))
        rt = _runtime(cls, str(d))
        rids = _start(rt, 3, limits=False)
        assert rt.pop_rids(NODE, WALK, rids) is not None
        column = torch.tensor([100, 101, 102], dtype=torch.int64)
        withheld = {SIG: [rids[1]]}
        trace = _drive(rt, PostprocessRequest(
            partition=PART, node_name=NODE, graph_walk=WALK, output_signals=[SIG],
            wg_ids=ParallelList(list(rids), [0] * 3), stops={} if known else None,
            runtime_stops=not known, tokens=column, row_rids=rids, inline=InlineColumn((SIG,), column),
            withheld=withheld if known else None,
        ), {}, model_stops={}, withheld=None if known else withheld)
        assert trace[-1][0] is PostprocessNeed.DONE
        assert (trace[0][0] is PostprocessNeed.STOPS) is not known
        frames[k] = _collect(box)
        assert rt.pop_rids(NODE, WALK, rids) is not None, "every row's loop-back fed its next step"
        iters[k] = [it[LOOP] for it in rt.get_dynamic_loop_iters(rids, PART).values]
    assert frames["native"] == frames["reference"]
    [tokens] = frames["native"]
    assert tokens.body.request_ids == ["r0", "r2"] and tokens.body.values == [100, 102]
    assert iters["native"] == iters["reference"] == [1, 1, 1]
