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
from mstar.streaming.stream_buffer import StreamChunkInfo, StreamingEdge  # noqa: E402
from mstar.worker.micro_scheduler import ScheduledBatch  # noqa: E402
from mstar.worker.worker import Speculation, Worker  # noqa: E402

NODE = "decoder"


_CHUNK = StreamChunkInfo(start_offset=0, context_items=0, num_items=1, is_final=False)


def _edge(name: str) -> StreamingEdge:
    return StreamingEdge(GraphEdge(next_node=NODE, name=name, is_streaming=True), _CHUNK)


def _speculation(rids: list[str]) -> Speculation:
    return Speculation(
        scheduled_batch=ScheduledBatch(
            node_name=NODE, graph_walk="w",
            request_to_worker_graph={r: "wg" for r in rids},
        ),
        node_batch=SimpleNamespace(
            request_ids=list(rids),
            per_request_input_tensors={r: {} for r in rids},
            per_request_info={r: object() for r in rids},
            per_request_input_metadata={r: None for r in rids},
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


def test_dropped_rid_gets_its_own_edges_back():
    returned: list[tuple[str, str]] = []
    committed: list[tuple] = []
    worker = SimpleNamespace(
        _return_streaming_edge=lambda rid, se: returned.append((rid, se.edge.name)),
        # The staged ingests are settled per rid before the chunks go back, so
        # the chunk leaves the node's slot first; see Worker._settle_speculation.
        _graph_runtime=SimpleNamespace(
            commit_speculation=lambda sid, ok, dropped, **kw: committed.append(
                (sid, ok, sorted(dropped), kw.get('scheduled_rids'))
            ),
        ),
    )
    worker._set_in_flight_flag = lambda batch, value: None
    worker._settle_speculation = (
        lambda spec, success, dropped_rids=frozenset():
        Worker._settle_speculation(worker, spec, success, dropped_rids)
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
        spec.node_batch.per_request_input_metadata,
        spec.scheduled_batch.request_to_worker_graph,
    ):
        assert set(table) == {"keep"}

    # Settling is the caller's, keyed off ``dropped``: the threading reports
    # the rid and leaves the chunks alone.
    assert returned == [] and committed == []
    assert set(spec.consumed_streaming_edges) == {"drop", "keep"}

    # What the caller then does with it: only drop's staged ingest is undone,
    # and the undo precedes the hand-back or the chunk is in two places.
    worker._settle_speculation(spec, True, spec.dropped)
    # One crossing, two different rid sets: the undo targets the dropped rid,
    # the speculative-scheduled flag goes on the one that still runs.
    assert committed == [(spec.spec_id, True, ["drop"], ["keep"])]
    assert returned == [("drop", "audio_drop")]
    # keep's chunk now belongs to the step, so the settle forgets it.
    assert spec.consumed_streaming_edges == {} and spec.spec_id == 0


def test_failed_settle_after_success_returns_nothing():
    """An error between the success settle and submit settles again with
    ``success=False``. The runtime stage is gone by then, so the kept chunk
    stays in its slot; returning it to its buffer would track it twice.
    """
    returned: list[tuple[str, str]] = []
    worker = SimpleNamespace(
        _return_streaming_edge=lambda rid, se: returned.append((rid, se.edge.name)),
        _graph_runtime=SimpleNamespace(commit_speculation=lambda *a, **kw: None),
    )
    spec = _speculation(["keep"])
    spec.spec_id = 5

    Worker._settle_speculation(worker, spec, True)
    Worker._settle_speculation(worker, spec, False)

    assert returned == []
