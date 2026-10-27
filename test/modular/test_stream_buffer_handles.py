"""A StreamBuffer holds its items by handle: each item's tensor stays in the
store while the buffer or a popped chunk's edge needs it, is released once
neither does, and a chunk's items are stacked only when the consuming batch is
built (``StreamBufferManager.input_tensors``)."""
from types import SimpleNamespace

import pytest
import torch

from mstar.communication.tensor_store import PythonTensorBookkeeping, TensorStore
from mstar.communication.tensors import TensorCommunicationManager
from mstar.graph.base import TensorPointerInfo
from mstar.streaming.chunk_policy import SlidingWindowChunkPolicy
from mstar.streaming.stream_buffer import StreamBuffer
from mstar.worker.stream_buffers import StreamBufferManager

EDGE, NODE, RID = "new_token", "snac_decoder", 1


class _TM:
    """The tensor-manager surface the stream path touches, over a real store."""

    get_tensor = TensorCommunicationManager.get_tensor
    get_tensors = TensorCommunicationManager.get_tensors
    dereference_batch_uniform = TensorCommunicationManager.dereference_batch_uniform
    cleanup_collectable = TensorCommunicationManager.cleanup_collectable
    increment_ref_batch_uniform = TensorCommunicationManager.increment_ref_batch_uniform

    def __init__(self, bookkeeping):
        self.tensor_store = TensorStore(bookkeeping=bookkeeping)
        self.uuid_to_shard_dim = {}

    def _cleanup_by_uuid(self, uuid, registered=None):
        if self.tensor_store.check_uuid_presence(uuid):
            self.tensor_store.remove_tensor(uuid)


def _info(uuid):
    return TensorPointerInfo(
        dims=[1], dtype=torch.int64, nbytes=8, address=0, stride=[1], uuid=uuid,
        source_session_id="", source_entity="w0",
    )


def _bookkeeping(kind):
    if kind == "python":
        return PythonTensorBookkeeping()
    # the store the Rust graph runtime uses: it frees in place on release
    rust = pytest.importorskip("mstar_rust")
    del rust
    from mstar.communication.tensor_store import RustTensorBookkeeping
    return RustTensorBookkeeping()


def _consumer(window, stride, kind="python"):
    tm = _TM(_bookkeeping(kind))
    buf = StreamBuffer(RID, EDGE, "LLM", SlidingWindowChunkPolicy(window, stride))
    request_state = SimpleNamespace(per_request_info={RID: SimpleNamespace(stream_buffers={EDGE: buf})})
    streams = StreamBufferManager(request_state, tm, [])
    streams.consumer_node = {EDGE: NODE}
    return streams, tm, buf


def _produce(tm, buf, values, first_uuid=100):
    """Each value stored with the one reference its stream destination holds,
    then handed to the buffer as the worker does."""
    uuids = []
    for k, v in enumerate(values):
        uuid = first_uuid + k
        tm.tensor_store.put_tensor(RID, uuid, torch.tensor([v]), _info(uuid))
        tm.tensor_store.increment_ref(uuid, 1)
        buf.pre_read_register(uuid)
        buf.put(uuid, uuid)
        uuids.append(uuid)
    return uuids


def _consume(streams, tm, se):
    """Build the consuming batch's input from the chunk's edge, then release
    the edge's references as the runtime does once the node consumed it."""
    by_rid = streams.input_tensors(streams.edge_block([(RID, se)]), NODE, [RID]).by_rid
    tm.dereference_batch_uniform(se.uuids)
    return by_rid[RID].get(EDGE)


@pytest.mark.parametrize("kind", ["python", "rust"])
@pytest.mark.parametrize("window, stride, n_items, chunks, flushed", [
    # overlapping windows; the flush is the overlap item alone
    (3, 2, 7, [[0, 1, 2], [2, 3, 4], [4, 5, 6]], [6]),
    # nothing left at the flush: an empty final chunk
    (2, 2, 4, [[0, 1], [2, 3]], []),
])
def test_a_sliding_window_keeps_its_overlap_alive_and_frees_everything_once_consumed(
    kind, window, stride, n_items, chunks, flushed,
):
    streams, tm, buf = _consumer(window=window, stride=stride, kind=kind)
    uuids = _produce(tm, buf, range(n_items))
    present = tm.tensor_store.check_uuid_presence
    seen = []
    while (se := streams.pop_edge(buf, EDGE, RID)) is not None:
        chunk = _consume(streams, tm, se)
        seen.append(chunk[0].flatten().tolist())
        # whatever the buffer still holds is still in the store
        assert all(present(u) for u in uuids[buf._consumed:])
    assert seen == chunks
    # the producer's done flushes what is left: the overlap item, unstacked,
    # or an empty final chunk with no items
    buf.signal_done(len(uuids))
    se = streams.pop_edge(buf, EDGE, RID)
    assert se.chunk.is_final
    if flushed:
        (tensor,) = _consume(streams, tm, se)
        assert tensor.shape == (1,) and tensor.tolist() == flushed
    else:
        assert se.uuids == []
    assert not any(present(u) for u in uuids), "every item is released once consumed"


def test_whatever_can_ready_a_chunk_marks_the_buffer_for_the_poll():
    """The poll skips a buffer that is not ``maybe_ready``, so a missed
    setter would leave its chunk unread for good."""
    streams, tm, buf = _consumer(window=2, stride=2)
    streams.pop_edge(buf, EDGE, RID)
    _produce(tm, buf, [1])
    assert buf.maybe_ready, "an arriving item"
    streams.pop_edge(buf, EDGE, RID)
    _produce(tm, buf, [2], first_uuid=200)
    se = streams.pop_edge(buf, EDGE, RID)
    assert se is not None and buf.maybe_ready, "a pop may leave another chunk"
    streams.pop_edge(buf, EDGE, RID)
    buf.store_uningested_edge(se)
    assert buf.maybe_ready, "a refused edge coming back"
    assert streams.pop_edge(buf, EDGE, RID) is se
    streams.pop_edge(buf, EDGE, RID)
    buf.signal_done(2)
    assert buf.maybe_ready, "the producer's done signal"
