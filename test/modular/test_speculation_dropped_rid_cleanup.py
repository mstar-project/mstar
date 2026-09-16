"""``_thread_outputs_to_speculative`` returns a dropped rid's streaming edges
under the dropped rid.

A rid is dropped from the spec batch when batch N produced no loop-back
output for it. Its consumed streaming edges must go back to its own stream
buffer, or the chunk is lost; a continuing rid's edges must stay consumed,
or the chunk is fed twice. The cleanup loop runs over ``dropped`` — every
line in it has to address the dropped rid, not whatever the threading loop
above it left bound.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.graph.base import GraphEdge  # noqa: E402
from mstar.worker.worker import Speculation, Worker  # noqa: E402

NODE = "decoder"


def _edge(name: str) -> GraphEdge:
    return GraphEdge(next_node=NODE, name=name, is_streaming=True)


def _speculation(rids: list[str]) -> Speculation:
    return Speculation(
        scheduled_batch=SimpleNamespace(
            request_to_worker_graph={r: "wg" for r in rids},
            node_objects={r: object() for r in rids},
        ),
        node_batch=SimpleNamespace(
            request_ids=list(rids),
            per_request_input_tensors={r: {} for r in rids},
            per_request_info={r: object() for r in rids},
        ),
        consumed_edges={("tok", NODE)},
        continuing_rids=set(rids),
        partition="p",
        is_new_iter=True,
        is_same_node=True,
        consumed_streaming_edges={r: [_edge(f"audio_{r}")] for r in rids},
    )


def test_dropped_rid_gets_its_own_edges_back():
    returned: list[tuple[str, str]] = []
    worker = SimpleNamespace(
        _return_speculative_streaming_edge=lambda rid, edge: returned.append((rid, edge.name)),
    )
    # The dropped rid goes first: the threading loop leaves its variable bound
    # to the LAST rid it visited, so a cleanup that read that leftover would
    # act on ``keep`` and look correct if ``drop`` happened to come last.
    spec = _speculation(["drop", "keep"])
    outputs_N = {"keep": {"tok": ["t"]}}  # nothing came back for ``drop``

    Worker._thread_outputs_to_speculative(worker, spec, outputs_N)

    assert spec.dropped == {"drop"}
    assert spec.continuing_rids == {"keep"}
    assert spec.node_batch.request_ids == ["keep"]
    for table in (
        spec.node_batch.per_request_input_tensors,
        spec.node_batch.per_request_info,
        spec.scheduled_batch.request_to_worker_graph,
        spec.scheduled_batch.node_objects,
    ):
        assert set(table) == {"keep"}

    # drop's chunk went back to drop's buffer; keep's stays consumed.
    assert returned == [("drop", "audio_drop")]
    assert set(spec.consumed_streaming_edges) == {"keep"}
