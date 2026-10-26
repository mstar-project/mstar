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


def _start(rt, n):
    rids = []
    for i in range(n):
        rid = rt.add_request(
            request_id=f"r{i}", partition=PART, graph_walk=WALK,
            partition_worker_graph_ids=[0],
            worker_graph_to_workers=ParallelList([0], [[ME]]), loop_caps={})
        # a limit no step reaches: EOS and the loop's cap end the rows
        rt.register_stop_limits(rid, SimpleNamespace(max_tokens=1 << 20, resource_configs={}))
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
        request = PostprocessRequest(
            partition=PART, node_name=NODE, graph_walk=WALK, output_signals=[SIG],
            wg_ids=ParallelList(list(active), [0] * len(active)),
            runtime_stops=mode != "model_stops", tokens=column, row_rids=row_rids,
            inline=None if mode == "stored" else InlineColumn(SIG, column),
        )
        model_stops = {r: {LOOP} for r, v in values.items() if v == EOS}
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
    expected |= {"model_stops": {PostprocessNeed.STOPS}, "stored": {PostprocessNeed.STORE}}.get(
        mode, set(),
    )
    assert needs == expected
    for peer in ("api_server", "conductor"):
        got = {k: _collect(boxes[k][peer]) for k in boxes}
        assert got["native"] == got["reference"], peer
        assert got["native"], peer
    for rt in rts.values():
        assert rt._rust.num_parked_completions() == 0


def test_dropped_rows_are_not_routed(tmp_path):
    """A row ``check_stop`` failed (STOPS' ``dropped``) leaves the step: it
    is neither stopped nor routed, in either implementation."""
    traces = {}
    for k, cls in (("native", rust_runtime.RustGraphRuntime), ("reference", _Reference)):
        d = tmp_path / k
        d.mkdir()
        rt = _runtime(cls, str(d))
        rids = _start(rt, 3)
        assert rt.pop_rids(NODE, WALK, rids) is not None
        column = torch.tensor([1, 2, 3], dtype=torch.int64)
        request = PostprocessRequest(
            partition=PART, node_name=NODE, graph_walk=WALK, output_signals=[SIG],
            wg_ids=ParallelList(list(rids), [0] * 3), row_rids=rids,
            inline=InlineColumn(SIG, column),
        )
        traces[k] = _drive(
            rt, request, dict(zip(rids, [1, 2, 3], strict=True)),
            model_stops={rids[1]: {LOOP}}, dropped=[rids[1]],
        )
    assert traces["native"] == traces["reference"]
    need, routed, stopped, _ = traces["native"][-1]
    assert need is PostprocessNeed.DONE and stopped == []
    assert traces["native"][0][0] is PostprocessNeed.STOPS


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
