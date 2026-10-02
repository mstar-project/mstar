"""``_thread_outputs_to_speculative`` cleans up a dropped rid under the dropped rid.

A rid is dropped from the spec batch when batch N produced no loop-back output for
it. Its consumed streaming edges stay in the node's ready slot — only their
stream-ended marks come off (``_unend_stream``), since handing a still-ingested
chunk back to its StreamBuffer would track it twice. A continuing rid's edges must
be left alone entirely, or its chunk is fed twice. The cleanup loop runs over
``dropped`` — every line in it has to address the dropped rid, not whatever the
threading loop above it left bound.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.graph.base import GraphEdge  # noqa: E402
from mstar.streaming.stream_buffer import StreamChunkInfo, StreamingEdge  # noqa: E402
from mstar.worker.worker import Speculation, Worker  # noqa: E402

NODE = "decoder"


_CHUNK = StreamChunkInfo(start_offset=0, context_items=0, num_items=1, is_final=False)


def _edge(name: str) -> StreamingEdge:
    return StreamingEdge(GraphEdge(next_node=NODE, name=name, is_streaming=True), _CHUNK)


def _speculation(rids: list[str]) -> Speculation:
    return Speculation(
        scheduled_batch=SimpleNamespace(
            request_to_worker_graph={r: "wg" for r in rids},
        ),
        node_batch=SimpleNamespace(
            request_ids=list(rids),
            per_request_input_tensors={r: {} for r in rids},
            per_request_info={r: object() for r in rids},
            per_request_stream_chunks={r: {f"audio_{r}": None} for r in rids},
            final_stream_rids=set(rids),
            stream_partition_done_rids=set(rids),
        ),
        consumed_edges={("tok", NODE)},
        continuing_rids=set(rids),
        partition="p",
        is_new_iter=True,
        is_same_node=True,
        consumed_streaming_edges={r: [_edge(f"audio_{r}")] for r in rids},
    )


def test_dropped_rid_cleanup_addresses_its_own_edges():
    unended: list[tuple[str, str]] = []
    worker = SimpleNamespace(
        _unend_stream=lambda rid, se: unended.append((rid, se.edge.name)),
        # Must not be reached: a still-ingested chunk is never re-buffered.
        _return_streaming_edge=lambda rid, se: pytest.fail(
            f"re-buffered an ingested chunk for {rid}"
        ),
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
        spec.node_batch.per_request_stream_chunks,
        spec.scheduled_batch.request_to_worker_graph,
    ):
        assert set(table) == {"keep"}

    # drop's own chunk was un-ended; keep's is untouched and stays consumed.
    assert unended == [("drop", "audio_drop")]
    assert set(spec.consumed_streaming_edges) == {"keep"}
