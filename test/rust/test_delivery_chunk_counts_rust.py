"""``test/modular/test_delivery_chunk_counts.py`` on the Rust runtime: the
frames it actually sends (read off real inboxes) must let the conductor count
every chunk, so the API server holds the request until the last one arrives.
Includes the inline-token path, and a script run on both runtimes that must
send the same frames.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, ".")
# the shared scenario lives beside the Python-runtime test; "test" itself can
# resolve to the standard library's package, so import it by directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modular"))

import pytest

pytest.importorskip("mstar_rust")

from mstar_rust import ZmqCommunicator
from test_delivery_chunk_counts import (  # noqa: E402
    DECODE,
    NODE,
    PREFILL,
    RUNTIME_KWARGS,
    SHAPES,
    WORKER,
    assert_holds_until_last_chunk,
    drive,
    python_runtime,
    tensor_info,
    worker_graphs,
)

import mstar.communication.wire_types  # noqa: F401  (registers the tags)
from mstar.api_server.request_types import ResultTokens
from mstar.communication.tensor_store import RustTensorBookkeeping
from mstar.communication.wire import decode, encode
from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.graph.base import GraphEdge
from mstar.graph.runtime import rust as rust_runtime
from mstar.graph.runtime.base import ColumnarEdgeSpecs, RequestWalks, RouteInput, SendInput
from mstar.utils.containers import ParallelList
from mstar.utils.ipc_format import ConductorMessageType

PEER = "worker_1"


def _collect(box, first_wait_ms=1000):
    out = []
    first = box.recv_timeout(first_wait_ms)
    if first is None:
        return out
    out.append(decode(first))
    while (more := box.recv_timeout(50)) is not None:
        out.append(decode(more))
    return out


def _rust(decode_steps, tmp_path, **overrides):
    """The Rust runtime, its ``put``, and a read of what it sent: the chunks
    to the API server and the WORKER_GRAPHS_DONE reports."""
    kwargs = {**RUNTIME_KWARGS, **overrides}
    inboxes = {peer: ZmqCommunicator(peer, str(tmp_path)) for peer in ("conductor", "api_server")}
    book = RustTensorBookkeeping()
    runtime = rust_runtime.RustGraphRuntime(
        my_worker_graphs=worker_graphs(decode_steps), bookkeeping=book,
        communicator=SimpleNamespace(_inner=ZmqCommunicator(kwargs["my_worker_id"], str(tmp_path))),
        **kwargs,
    )

    def put(uuid):
        book.put_tensor(uuid, tensor_info(uuid))
        book.increment_ref(uuid, 1)  # the worker's safety hold

    def sent():
        chunks = [m.body for m in _collect(inboxes["api_server"])]
        reports = [
            m.body for m in _collect(inboxes["conductor"])
            if m.message_type == ConductorMessageType.WORKER_GRAPHS_DONE
        ]
        return chunks, reports

    return runtime, put, sent


@pytest.mark.parametrize("inline", [False, True], ids=["tensor", "inline"])
@pytest.mark.parametrize("shape", list(SHAPES))
def test_request_is_held_until_its_last_chunk(shape, inline, tmp_path):
    prefills, decode_steps = SHAPES[shape]
    runtime, put, sent = _rust(decode_steps, tmp_path)
    drive(runtime, put, prefills, decode_steps, inline=inline)
    chunks, reports = sent()
    assert len(chunks) == prefills + decode_steps
    tokens = sum(isinstance(c, ResultTokens) for c in chunks)
    assert tokens == (decode_steps if inline else 0)
    assert_holds_until_last_chunk(chunks, reports)


def _admit(runtime, request_id, workers=(WORKER,)):
    return runtime.add_request(
        request_id=request_id, partition="default", graph_walk=PREFILL,
        partition_worker_graph_ids=[0, 1],
        worker_graph_to_workers=ParallelList([0, 1], [list(workers), list(workers)]),
        loop_caps={},
    )


def _ingest(runtime, rids):
    block = ColumnarEdgeSpecs.empty()
    for rid in rids:
        block.add_edge(rid, GraphEdge(name="text_inputs", next_node=NODE))
    assert runtime.ingest_inputs_batch(block, can_buffer=True) == []


def _stepper(runtime, put):
    """``step(rows, ...)`` runs NODE for ``rows``, (rid, walk) pairs, as one
    batch and sends its outputs. ``values`` ride inline as ``new_token``;
    ``stored=False`` routes no tensors, as a device loop-back step does;
    ``failed`` rids are sent nothing."""
    uuid = 100

    def step(rows, values=None, stored=True, failed=()):
        nonlocal uuid
        rids = [rid for rid, _ in rows]
        walks = list(dict.fromkeys(walk for _, walk in rows))
        walk_idx = [walks.index(walk) for _, walk in rows]
        assert runtime.pop_walk_rids(NODE, RequestWalks(rids, walks, walk_idx)) is not None
        signals = sorted({s for walk in walks for s in runtime.get_output_signals(NODE, walk)})
        tensors, counts = [], []
        for _, walk in rows:
            own = runtime.get_output_signals(NODE, walk)
            for signal in signals:
                count = 1 if stored and signal in own else 0
                for _ in range(count):
                    uuid += 1
                    put(uuid)
                    tensors.append(uuid)
                counts.append(count)
        inline = {} if values is None else dict(inline_signal="new_token", inline_values=values)
        mixed = len(walks) > 1
        out = runtime.complete_and_route_batch(RouteInput(
            partition="default", graph_walk=walks[0], node_name=NODE, output_signals=signals,
            wg_ids=ParallelList(rids, [0 if walk == PREFILL else 1 for _, walk in rows]),
            tensors=tensors, num_tensors=counts,
            walks=walks if mixed else None, rid_walk_idx=walk_idx if mixed else None,
            **inline,
        ))
        runtime.send_outputs(SendInput(
            completion_id=out.completion_id,
            per_request_info=ParallelList(rids, [None] * len(rids)),
            new_token_counts=ParallelList([], []),
            failed_rids=frozenset(failed),
        ))
        return out

    return step


def _script(runtime, put) -> list[list[int]]:
    """Three requests through prefill and a five-step decode loop, covering
    every inline case: a persisted client signal (prefill), inline decode
    rows, a step mixing a prefill row with decode rows, a device loop-back
    step that stores no tensors, values that miss a row, and a failed
    request. Returns each step's staged uuids."""
    a, b, c = (_admit(runtime, f"r{i}") for i in range(3))
    step = _stepper(runtime, put)
    _ingest(runtime, [a, b, c])
    outs = [step([(a, PREFILL), (b, PREFILL)], [1, 2])]
    for rid in (a, b):
        runtime.set_walk(rid, "default", DECODE)
    _ingest(runtime, [a, b])
    outs.append(step([(a, DECODE), (b, DECODE)], [11, 21]))
    outs.append(step([(c, PREFILL), (a, DECODE), (b, DECODE)], [3, 12, 22]))
    outs.append(step([(a, DECODE), (b, DECODE)], [13, 23], stored=False))
    outs.append(step([(a, DECODE), (b, DECODE)], [14]))
    outs.append(step([(a, DECODE), (b, DECODE)], [15, 25], failed={b}))
    return [sorted(out.register_uuids) for out in outs]


def _frames(chunks):
    """Comparable, order-free. Rust orders a RESULT_TOKENS frame's rows by a
    hash map, so the rows compare as a set."""
    out = []
    for c in chunks:
        if isinstance(c, ResultTokens):
            rows = zip(c.request_ids, c.values, c.loop_indices, strict=True)
            out.append(("tokens", c.signal, c.modality, tuple(sorted(rows, key=repr))))
        else:
            edge = c.graph_edge
            out.append((
                "tensors", c.request_id, edge.name, c.modality,
                tuple(i.uuid for i in edge.tensor_info), c.loop_indices,
            ))
    return sorted(out, key=repr)


def _reports(reports):
    return sorted(
        (r.request_id, tuple(r.worker_graph_ids), tuple(r.output_signal_names))
        for r in reports
    )


def test_both_runtimes_send_the_same_frames(tmp_path):
    py_runtime, py_put, recorder = python_runtime(decode_steps=5)
    py_staged = _script(py_runtime, py_put)
    # through the wire codec, as the API server receives them
    py_chunks = [decode(encode(m)).body for e, m in recorder.sent if e == "api_server"]

    rs_runtime, rs_put, sent = _rust(5, tmp_path)
    rs_staged = _script(rs_runtime, rs_put)
    rs_chunks, rs_reports = sent()

    assert py_staged == rs_staged
    assert _frames(py_chunks) == _frames(rs_chunks)
    # a step's tensor chunks go out before its tokens on both
    assert [type(c) for c in py_chunks] == [type(c) for c in rs_chunks]
    assert _reports(recorder.reports()) == _reports(rs_reports)
    tokens = sorted(sorted(c.values) for c in rs_chunks if isinstance(c, ResultTokens))
    assert tokens == [[11, 21], [12, 22], [13, 23], [15]]


def _tp_rank(rank):
    """Overrides for rank ``rank`` of NODE run at TP 2 by WORKER and PEER."""
    group = ShardingGroup(nodes={NODE}, tp_size=2)
    group._tp_rank = rank
    return dict(
        my_worker_id=(WORKER, PEER)[rank],
        sharding_config=ShardingConfig(groups=[group], tp_enabled_nodes={NODE}, shard_dim={}),
    )


def _tp_step(runtime, put):
    rid = _admit(runtime, "r1", workers=(WORKER, PEER))
    runtime.set_walk(rid, "default", DECODE)
    _ingest(runtime, [rid])
    _stepper(runtime, put)([(rid, DECODE)], [7])


def test_both_runtimes_emit_from_the_tp_leader_only(tmp_path):
    for rank in (0, 1):
        py_runtime, py_put, recorder = python_runtime(1, **_tp_rank(rank))
        _tp_step(py_runtime, py_put)
        ipc_dir = tmp_path / str(rank)
        ipc_dir.mkdir()
        rs_runtime, rs_put, sent = _rust(1, ipc_dir, **_tp_rank(rank))
        _tp_step(rs_runtime, rs_put)
        rs_chunks, rs_reports = sent()

        assert _frames(recorder.chunks()) == _frames(rs_chunks)
        assert _reports(recorder.reports()) == _reports(rs_reports)
        assert [type(c) for c in rs_chunks] == ([ResultTokens] if rank == 0 else [])
