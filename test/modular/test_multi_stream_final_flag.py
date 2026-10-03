"""A node fed by several streams flushes once, when the last of them ends.

One stream's final chunk means only that stream ended. The step that consumes
it is final for the node (``is_final_stream_chunk``) only if every other stream
into the node already ended, and it reports the partition done only if every
stream into the partition did. Streams end in any order, over several steps; a
speculative step that hands its final chunk back reopens that stream; a
``continue_after_producer_done`` stream never ends and holds nothing back.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.graph.base import GraphEdge  # noqa: E402
from mstar.streaming.chunk_policy import FixedChunkPolicy  # noqa: E402
from mstar.streaming.stream_buffer import (  # noqa: E402
    StreamBuffer,
    StreamChunkInfo,
    StreamingEdge,
)
from mstar.worker.worker import Worker  # noqa: E402

PARTITION = "Decoder"

# edge -> (consumer node, continue_after_done)
TOPOLOGY = {
    "codes": ("vocoder", False),
    "pitch": ("vocoder", False),
    "speaker": ("vocoder", True),
    "captions": ("captioner", False),
}


def _worker(rids=("r",)):
    per_request_info = {
        rid: SimpleNamespace(
            stream_buffers={
                name: StreamBuffer(
                    request_id=rid, edge_name=name, from_partition="LLM",
                    policy=FixedChunkPolicy(chunk_size=1, continue_after_done=cont),
                )
                for name, (_, cont) in TOPOLOGY.items()
            },
            ended_streams=set(),
        )
        for rid in rids
    }
    worker = SimpleNamespace(
        request_state=SimpleNamespace(per_request_info=per_request_info),
        _consumer_node_cache={name: node for name, (node, _) in TOPOLOGY.items()},
        _stream_partition={name: PARTITION for name in TOPOLOGY},
    )
    worker._settle_final_streams = lambda *a: Worker._settle_final_streams(worker, *a)
    return worker


def _step(worker, node, final_edges):
    batch = Worker._make_executing_batch(
        worker, node_name=node, graph_walk="chunk", request_ids=list(final_edges),
        per_request_input_tensors={}, per_request_info={}, final_edges=final_edges,
    )
    return batch.final_stream_rids, batch.stream_partition_done_rids


def _final_edge(name):
    return StreamingEdge(
        GraphEdge(
            next_node=TOPOLOGY[name][0], name=name, is_streaming=True,
            _final_stream_chunk=True,
        ),
        StreamChunkInfo(start_offset=0, context_items=0, num_items=1, is_final=True),
    )


def test_first_stream_to_end_does_not_flush_the_node():
    worker = _worker()

    assert _step(worker, "vocoder", {"r": {"codes"}}) == (set(), set())
    assert _step(worker, "vocoder", {"r": {"pitch"}}) == ({"r"}, set())


def test_streams_ending_in_one_step_flush_together():
    worker = _worker()

    assert _step(worker, "vocoder", {"r": {"codes", "pitch"}}) == ({"r"}, set())


def test_partition_is_done_only_when_every_node_in_it_is():
    worker = _worker()

    assert _step(worker, "captioner", {"r": {"captions"}}) == ({"r"}, set())
    assert _step(worker, "vocoder", {"r": {"codes"}}) == (set(), set())
    assert _step(worker, "vocoder", {"r": {"pitch"}}) == ({"r"}, {"r"})


def test_a_step_with_no_final_chunk_is_never_final():
    worker = _worker()
    _step(worker, "vocoder", {"r": {"codes", "pitch"}})

    assert _step(worker, "vocoder", {}) == (set(), set())


def test_requests_in_one_batch_are_tracked_apart():
    worker = _worker(rids=("a", "b"))
    _step(worker, "vocoder", {"a": {"codes"}})

    assert _step(worker, "vocoder", {"a": {"pitch"}, "b": {"pitch"}}) == ({"a"}, set())


def test_a_returned_final_chunk_reopens_its_stream():
    worker = _worker()
    _step(worker, "vocoder", {"r": {"codes"}})  # a speculative step takes the final chunk...
    Worker._return_streaming_edge(worker, "r", _final_edge("codes"))  # ...and is discarded

    assert _step(worker, "vocoder", {"r": {"pitch"}}) == (set(), set())
    assert _step(worker, "vocoder", {"r": {"codes"}}) == ({"r"}, set())


def test_returned_final_chunk_goes_back_to_its_buffer():
    worker = _worker()
    edge = _final_edge("codes")
    Worker._return_streaming_edge(worker, "r", edge)

    sbuf = worker.request_state.per_request_info["r"].stream_buffers["codes"]
    assert sbuf.pop_waiting_edge() is edge


def test_a_dropped_speculative_rid_neither_flushes_nor_reports_done():
    worker = _worker()
    worker._return_streaming_edge = lambda *a: Worker._return_streaming_edge(worker, *a)
    _step(worker, "captioner", {"r": {"captions"}})
    _step(worker, "vocoder", {"r": {"codes"}})
    edge = _final_edge("pitch")
    batch = Worker._make_executing_batch(
        worker, node_name="vocoder", graph_walk="chunk", request_ids=["r"],
        per_request_input_tensors={"r": {}}, per_request_info={"r": None},
        final_edges={"r": {"pitch"}},
    )
    assert (batch.final_stream_rids, batch.stream_partition_done_rids) == ({"r"}, {"r"})
    speculation = SimpleNamespace(
        node_batch=batch, continuing_rids={"r"}, consumed_edges=[("loop", None)],
        scheduled_batch=SimpleNamespace(request_to_worker_graph={"r": 0}),
        consumed_streaming_edges={"r": [edge]}, spec_id=0,
    )
    # The dropped rid's staged ingest is undone before its chunk goes back.
    worker._graph_runtime = SimpleNamespace(commit_speculation=lambda *a, **kw: None)
    worker._set_speculative_flag = lambda batch, value: None
    worker._settle_speculation = (
        lambda spec, success, dropped_rids=frozenset():
        Worker._settle_speculation(worker, spec, success, dropped_rids)
    )

    Worker._thread_outputs_to_speculative(worker, speculation, outputs_N={})
    # The caller settles, keyed off ``dropped``; that is what reopens the stream.
    worker._settle_speculation(speculation, True, speculation.dropped)

    assert speculation.dropped == {"r"}
    assert (batch.final_stream_rids, batch.stream_partition_done_rids) == (set(), set())
    assert _step(worker, "vocoder", {"r": {"pitch"}}) == ({"r"}, {"r"})
