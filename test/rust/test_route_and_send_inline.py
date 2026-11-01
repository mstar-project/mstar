"""``route_and_send_inline`` must be exactly ``complete_and_route_batch`` then
``send_outputs`` with the device-loopback arguments the worker builds.

Two runtimes are driven through the same decode steps, one per path, and
everything observable is compared: the frames each sends (read off real
inboxes), the next pop, and the loop counters. The loop is short, so the last
step completes the worker graph and exercises the fallback (``ROUTED``) where a
frame needs Python's per-request payloads.
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
from mstar.graph.runtime.base import ColumnarEdgeSpecs, InlineRouteStatus, RouteInput, SendInput
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import WorkerGraph
from mstar.utils.containers import ParallelList

WALK, NODE, LOOP, SIG, PART, ME = "decode", "LLM", "decode_loop", "text_inputs", "default", "w0"
MAX_ITERS = 4


def _runtime(sock_dir):
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
    return rust_runtime.RustGraphRuntime(
        my_worker_id=ME, my_worker_graphs=[wg],
        all_wg_ids_to_graph_walks={0: {WALK}}, all_wg_ids_to_dyn_loops={0: {LOOP}},
        all_wg_ids_to_nodes={0: {NODE}}, node_to_partition={NODE: PART},
        sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        bookkeeping=RustTensorBookkeeping(),
        communicator=SimpleNamespace(_inner=ZmqCommunicator(ME, sock_dir)),
    )


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
    rids = [
        rt.add_request(
            request_id=f"r{i}", partition=PART, graph_walk=WALK,
            partition_worker_graph_ids=[0],
            worker_graph_to_workers=ParallelList([0], [[ME]]), loop_caps={})
        for i in range(n)
    ]
    block = ColumnarEdgeSpecs.empty()
    for rid in rids:
        block.add(rid, SIG, [], False, next_node=NODE)
    rt.ingest_inputs_batch(block)
    return rids


def _general_step(rt, rids, values):
    out = rt.complete_and_route_batch(RouteInput(
        partition=PART, graph_walk=WALK, node_name=NODE, output_signals=[SIG],
        wg_ids=ParallelList(rids, [0] * len(rids)), tensors=[],
        num_tensors=[0] * len(rids), inline_signal=SIG, inline_values=values,
    ))
    info = sorted(out.rids_needing_request_info)
    return rt.send_outputs(SendInput(
        completion_id=out.completion_id,
        per_request_info=ParallelList(info, [None] * len(info)),
        new_token_counts=ParallelList([], []),
    ))


def _fused_step(rt, rids, column, row_rids):
    status, _cid, sent, route = rt.route_and_send_inline(
        PART, WALK, NODE, [SIG], rids, SIG, column, row_rids,
    )
    assert status in (InlineRouteStatus.SENT, InlineRouteStatus.ROUTED)
    if status is InlineRouteStatus.SENT:
        assert route is None
        return sent
    info = sorted(route.rids_needing_request_info)
    assert info, "ROUTED only when a frame needs Python's payloads"
    return rt.send_outputs(SendInput(
        completion_id=route.completion_id,
        per_request_info=ParallelList(info, [None] * len(info)),
        new_token_counts=ParallelList([], []),
    ))


@pytest.mark.parametrize("dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("reverse_rows", [False, True], ids=["in_order", "permuted"])
def test_fused_matches_route_then_send(tmp_path, dtype, reverse_rows):
    n = 5
    dirs = {k: tmp_path / k for k in ("g", "f")}
    boxes = {}
    rts = {}
    for k, d in dirs.items():
        d.mkdir()
        boxes[k] = {p: ZmqCommunicator(p, str(d)) for p in ("api_server", "conductor")}
        rts[k] = _runtime(str(d))
    rids = {k: _start(rt, n) for k, rt in rts.items()}
    assert rids["g"] == rids["f"]
    rids = rids["g"]
    statuses = []
    for step in range(MAX_ITERS):
        for rt in rts.values():
            assert rt.pop_rids(NODE, WALK, rids) is not None
        values = [1000 * step + i for i in range(n)]
        # host rows as the stop check stages them: [rows, 1], row i for row_rids[i]
        row_rids = list(reversed(rids)) if reverse_rows else list(rids)
        rows = torch.tensor([values[rids.index(r)] for r in row_rids], dtype=dtype)
        column = rows.reshape(-1, 1).reshape(rows.shape[0], -1)[:, 0]
        sent_g = _general_step(rts["g"], rids, values)
        sent_f = _fused_step(rts["f"], rids, column, row_rids)
        assert sent_g == sent_f
        statuses.append(bool(sent_f))
        assert (
            rts["g"].get_dynamic_loop_iters(rids, PART).values
            == rts["f"].get_dynamic_loop_iters(rids, PART).values
        )
    assert statuses[-1], "the last step completes the worker graph"
    for peer in ("api_server", "conductor"):
        got = {k: _collect(boxes[k][peer]) for k in boxes}
        assert got["g"] == got["f"], peer
        assert got["g"], peer
    for rt in rts.values():
        assert rt._rust.num_parked_completions() == 0


def test_a_rid_without_a_row_does_nothing(tmp_path):
    rt = _runtime(str(tmp_path))
    rids = _start(rt, 3)
    assert rt.pop_rids(NODE, WALK, rids) is not None
    column = torch.tensor([7, 8], dtype=torch.int64)
    status, _cid, sent, route = rt.route_and_send_inline(
        PART, WALK, NODE, [SIG], rids, SIG, column, rids[:2],
    )
    assert (status, sent, route) == (InlineRouteStatus.NOT_ROUTED, [], None)
    # nothing was routed: the general path still completes this step
    assert rt._rust.num_parked_completions() == 0
    assert _general_step(rt, rids, [1, 2, 3]) == []
