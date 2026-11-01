"""The frames ``test/modular/test_delivery_chunk_counts.py``'s runtime sends for
a script covering every inline-token case, read off real inboxes.
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
# the shared scenario lives beside the modular test; "test" itself can resolve
# to the standard library's package, so import it by directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modular"))

import pytest

pytest.importorskip("mstar_rust")

from test_delivery_chunk_counts import (  # noqa: E402
    DECODE,
    LOOP,
    NODE,
    PREFILL,
    WORKER,
    new_runtime,
)

from mstar.api_server.request_types import ResultTokens
from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.graph.base import GraphEdge
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.graph.runtime.base import ColumnarEdgeSpecs, RequestWalks, RouteInput, SendInput
from mstar.utils.containers import ParallelList

PEER = "worker_1"


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
    ``failed`` rids are sent nothing; ``logprobs`` are the rows' log-probs."""
    uuid = 100

    def step(rows, values=None, stored=True, failed=(), logprobs=()):
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
            logprobs=logprobs, **inline,
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
    request. ``a`` asks for log-probs, ``b`` does not, ``c`` only at its
    prefill. Returns each step's staged uuids."""
    a, b, c = (_admit(runtime, f"r{i}") for i in range(3))
    step = _stepper(runtime, put)
    _ingest(runtime, [a, b, c])
    outs = [step([(a, PREFILL), (b, PREFILL)], [1, 2], logprobs=[-1.0, None])]
    for rid in (a, b):
        runtime.set_walk(rid, "default", DECODE)
    _ingest(runtime, [a, b])
    outs.append(step([(a, DECODE), (b, DECODE)], [11, 21], logprobs=[-1.5, None]))
    outs.append(step([(c, PREFILL), (a, DECODE), (b, DECODE)], [3, 12, 22], logprobs=[-2.0, -2.5, None]))
    outs.append(step([(a, DECODE), (b, DECODE)], [13, 23], stored=False, logprobs=[-3.0, None]))
    outs.append(step([(a, DECODE), (b, DECODE)], [14], logprobs=[-3.5, None]))
    outs.append(step([(a, DECODE), (b, DECODE)], [15, 25], failed={b}))
    return [sorted(out.register_uuids) for out in outs]


def _frames(chunks):
    """Comparable, order-free. Rust orders a RESULT_TOKENS frame's rows by a
    hash map, so the rows compare as a set."""
    out = []
    for c in chunks:
        if isinstance(c, ResultTokens):
            n = len(c.request_ids)
            rows = zip(c.request_ids, c.values, c.loop_indices, c.logprobs or [None] * n, strict=True)
            out.append(("tokens", c.signal, c.modality, tuple(sorted(rows, key=repr))))
        else:
            edge = c.graph_edge
            out.append((
                "tensors", c.request_id, edge.name, c.modality,
                tuple(i.uuid for i in edge.tensor_info), c.loop_indices, c.metadata,
            ))
    return sorted(out, key=repr)


def _reports(reports):
    return sorted(
        (r.request_id, tuple(r.worker_graph_ids), tuple(r.output_signal_names))
        for r in reports
    )


def _prefill_step():
    return NestedLoopIndices([], {}, 0)


def _decode_step(i):
    return NestedLoopIndices([LOOP], {LOOP: i}, 0)


def test_the_script_sends_these_frames(tmp_path):
    """The staged uuids, chunks and reports are the values the runtime gives
    today. This script used to diff them against a second runtime."""
    runtime, put, sent = new_runtime(5, tmp_path)
    staged = _script(runtime, put)
    chunks, reports = sent()

    assert staged == [[101, 102], [], [107], [], [112, 114], []]
    assert _frames(chunks) == [
        ("tensors", "r0", "new_token", "text", (101,), _prefill_step(), {"logprobs": [-1.0]}),
        ("tensors", "r0", "new_token", "text", (112,), _decode_step(3), {"logprobs": [-3.5]}),
        ("tensors", "r1", "new_token", "text", (102,), _prefill_step(), {}),
        ("tensors", "r1", "new_token", "text", (114,), _decode_step(3), {}),
        ("tensors", "r2", "new_token", "text", (107,), _prefill_step(), {"logprobs": [-2.0]}),
        ("tokens", "new_token", "text", (("r0", 11, _decode_step(0), -1.5), ("r1", 21, _decode_step(0), None))),
        ("tokens", "new_token", "text", (("r0", 12, _decode_step(1), -2.5), ("r1", 22, _decode_step(1), None))),
        ("tokens", "new_token", "text", (("r0", 13, _decode_step(2), -3.0), ("r1", 23, _decode_step(2), None))),
        ("tokens", "new_token", "text", (("r0", 15, _decode_step(4), None),)),
    ]
    # a step's tensor chunks go out before its tokens
    assert [type(c).__name__ for c in chunks] == [
        "ResultTensors", "ResultTensors", "ResultTokens", "ResultTensors", "ResultTokens",
        "ResultTokens", "ResultTensors", "ResultTensors", "ResultTokens",
    ]
    assert _reports(reports) == [
        ("r0", (0,), ("new_token",)), ("r0", (1,), ("new_token",) * 5),
        ("r1", (0,), ("new_token",)), ("r2", (0,), ("new_token",)),
    ]
    tokens = sorted(sorted(c.values) for c in chunks if isinstance(c, ResultTokens))
    assert tokens == [[11, 21], [12, 22], [13, 23], [15]]
    logprobs = sorted(
        (rid, lp) for c in chunks if isinstance(c, ResultTokens) and c.logprobs
        for rid, lp in zip(c.request_ids, c.logprobs, strict=True) if lp is not None
    )
    assert logprobs == [("r0", -3.0), ("r0", -2.5), ("r0", -1.5)]
    metadata = sorted(
        (c.request_id, *c.metadata["logprobs"]) for c in chunks
        if not isinstance(c, ResultTokens) and c.metadata
    )
    # the prefills, and the step whose values missed a row
    assert metadata == [("r0", -3.5), ("r0", -1.0), ("r2", -2.0)]


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


def test_only_the_tp_leader_emits(tmp_path):
    for rank, emitted in ((0, ("new_token",)), (1, ())):
        ipc_dir = tmp_path / str(rank)
        ipc_dir.mkdir()
        runtime, put, sent = new_runtime(1, ipc_dir, **_tp_rank(rank))
        _tp_step(runtime, put)
        chunks, reports = sent()

        assert [type(c) for c in chunks] == ([ResultTokens] if rank == 0 else [])
        assert _reports(reports) == [("r1", (1,), emitted)]
