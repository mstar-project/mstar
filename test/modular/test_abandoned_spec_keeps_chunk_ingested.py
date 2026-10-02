"""An abandoned speculative step must NOT hand its chunks back to their StreamBuffers.

``_return_streaming_edge`` does not take the edge out of the consuming node's ready
slot, so handing back a chunk the prep already ingested leaves it tracked twice: the
step that eventually runs dereferences the tensor (streamed chunks are minted with no
reference, so that frees it) and the buffer's stale edge is then re-ingested pointing
at a freed uuid -- a ``KeyError`` that kills the whole batch, since the columnar build
resolves every rid's tensors in one call.

So an abandoned step only un-marks the stream (``_unend_stream``) and leaves the chunk
in the slot, where the node consumes it when it next runs.
"""
from types import SimpleNamespace

from mstar.graph.base import GraphEdge
from mstar.streaming.stream_buffer import StreamChunkInfo, StreamingEdge
from mstar.worker.worker import Worker


def _streaming_edge(name: str = "chunk", final: bool = False) -> StreamingEdge:
    return StreamingEdge(
        edge=GraphEdge(next_node="consumer", name=name, _final_stream_chunk=final),
        chunk=StreamChunkInfo(
            start_offset=0, context_items=0, num_items=1, is_final=final,
        ),
    )


class _Buffer:
    def __init__(self):
        self.returned = []

    def store_uningested_edge(self, streaming_edge):
        self.returned.append(streaming_edge)


def _worker(ended=("chunk",)):
    sbuf = _Buffer()
    req_info = SimpleNamespace(stream_buffers={"chunk": sbuf}, ended_streams=set(ended))
    w = SimpleNamespace(
        request_state=SimpleNamespace(per_request_info={7: req_info}),
    )
    # _return_streaming_edge delegates to it, so bind the real one.
    w._unend_stream = lambda rid, se: Worker._unend_stream(w, rid, se)
    return w, sbuf, req_info


def test_unend_stream_releases_the_mark_without_returning_the_chunk():
    w, sbuf, req_info = _worker()
    Worker._unend_stream(w, 7, _streaming_edge(final=True))
    assert req_info.ended_streams == set(), "the abandoned step must un-end the stream"
    assert sbuf.returned == [], "the chunk stays in the node's ready slot"


def test_unend_stream_is_a_noop_for_a_non_final_chunk():
    w, sbuf, req_info = _worker()
    Worker._unend_stream(w, 7, _streaming_edge(final=False))
    # Nothing to un-end: only a final chunk's consumer marks the stream ended.
    assert req_info.ended_streams == {"chunk"}
    assert sbuf.returned == []


def test_unend_stream_tolerates_a_request_that_is_already_gone():
    w, _sbuf, _req = _worker()
    w.request_state.per_request_info.clear()
    Worker._unend_stream(w, 7, _streaming_edge(final=True))  # must not raise


def test_returning_a_refused_chunk_still_both_buffers_it_and_unends_it():
    """The un-ingested path is unchanged: that chunk is NOT in any ready slot."""
    w, sbuf, req_info = _worker()
    se = _streaming_edge(final=True)
    Worker._return_streaming_edge(w, 7, se)
    assert sbuf.returned == [se]
    assert req_info.ended_streams == set()
