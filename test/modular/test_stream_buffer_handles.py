"""A StreamBuffer holds its items by handle: each item's tensor stays in the
store while the buffer or a popped chunk's edge needs it, is released once
neither does, and a chunk's items are stacked only when the consuming batch is
built (``Worker._input_tensors``)."""
from types import SimpleNamespace

import torch

from mstar.communication.tensor_store import PythonTensorBookkeeping, TensorStore
from mstar.communication.tensors import TensorCommunicationManager
from mstar.graph.base import TensorPointerInfo
from mstar.graph.runtime.base import ColumnarEdgeSpecs
from mstar.streaming.chunk_policy import SlidingWindowChunkPolicy
from mstar.streaming.stream_buffer import StreamBuffer
from mstar.worker.worker import Worker

EDGE, NODE, RID = "new_token", "snac_decoder", 1


class _TM:
    """The tensor-manager surface the stream path touches, over a real store."""

    get_tensor = TensorCommunicationManager.get_tensor
    dereference_batch_uniform = TensorCommunicationManager.dereference_batch_uniform
    cleanup_collectable = TensorCommunicationManager.cleanup_collectable
    increment_ref_batch_uniform = TensorCommunicationManager.increment_ref_batch_uniform

    def __init__(self):
        self.tensor_store = TensorStore(bookkeeping=PythonTensorBookkeeping())
        self.uuid_to_shard_dim = {}

    def _cleanup_by_uuid(self, uuid, registered=None):
        if self.tensor_store.check_uuid_presence(uuid):
            self.tensor_store.remove_tensor(uuid)


def _info(uuid):
    return TensorPointerInfo(
        dims=[1], dtype=torch.int64, nbytes=8, address=0, stride=[1], uuid=uuid,
        source_session_id="", source_entity="w0",
    )


def _consumer(window, stride):
    tm = _TM()
    w = Worker.__new__(Worker)
    w.tensor_manager = tm
    buf = StreamBuffer(RID, EDGE, "LLM", SlidingWindowChunkPolicy(window, stride))
    w.request_state = SimpleNamespace(per_request_info={RID: SimpleNamespace(stream_buffers={EDGE: buf})})
    w._consumer_node_cache = {EDGE: NODE}
    return w, tm, buf


def _produce(tm, buf, values, first_uuid=100):
    """Each value stored with the one reference its stream destination holds,
    then handed to the buffer as the worker does."""
    uuids = []
    for k, v in enumerate(values):
        uuid = first_uuid + k
        tm.tensor_store.put_tensor(RID, uuid, torch.tensor([v]), _info(uuid))
        tm.tensor_store.increment_ref(uuid, 1)
        buf.pre_read_register(uuid)
        buf.put(uuid, _info(uuid))
        uuids.append(uuid)
    return uuids


def _consume(w, tm, se):
    """Build the consuming batch's input from the chunk's edge, then release
    the edge's references as the runtime does once the node consumed it."""
    block = ColumnarEdgeSpecs.empty()
    block.add(RID, EDGE, [i.uuid for i in se.edge.tensor_info], se.edge._final_stream_chunk)
    by_rid = w._input_tensors(block, NODE, [RID]).by_rid
    tm.dereference_batch_uniform([i.uuid for i in se.edge.tensor_info])
    return by_rid[RID].get(EDGE)


def test_a_sliding_window_keeps_its_overlap_alive_and_frees_everything_once_consumed():
    w, tm, buf = _consumer(window=3, stride=2)
    uuids = _produce(tm, buf, range(7))
    present = tm.tensor_store.check_uuid_presence
    seen = []
    while (se := w._pop_streaming_edge(buf, EDGE, RID)) is not None:
        chunk = _consume(w, tm, se)
        seen.append(chunk[0].flatten().tolist())
        # whatever the buffer still holds is still in the store
        assert all(present(u) for u in uuids[buf._consumed:])
    assert seen == [[0, 1, 2], [2, 3, 4], [4, 5, 6]]
    # the overlap item 6 is still buffered; the producer's done flushes it
    buf.signal_done(len(uuids))
    se = w._pop_streaming_edge(buf, EDGE, RID)
    assert se.chunk.is_final
    assert _consume(w, tm, se)[0].flatten().tolist() == [6]
    assert not any(present(u) for u in uuids), "every item is released once consumed"


def test_a_one_item_chunk_is_the_item_unstacked():
    w, tm, buf = _consumer(window=1, stride=1)
    _produce(tm, buf, [5])
    se = w._pop_streaming_edge(buf, EDGE, RID)
    (tensor,) = _consume(w, tm, se)
    assert tensor.shape == (1,) and tensor.item() == 5


def test_an_empty_final_chunk_carries_no_items():
    w, tm, buf = _consumer(window=2, stride=2)
    _produce(tm, buf, [1, 2])
    _consume(w, tm, w._pop_streaming_edge(buf, EDGE, RID))
    buf.signal_done(2)
    se = w._pop_streaming_edge(buf, EDGE, RID)
    assert se.chunk.is_final and se.edge.tensor_info == []


def test_maybe_ready_is_set_by_whatever_can_ready_a_chunk_and_cleared_by_an_empty_poll():
    w, tm, buf = _consumer(window=2, stride=2)
    assert w._pop_streaming_edge(buf, EDGE, RID) is None
    assert not buf.maybe_ready, "an empty poll clears it"
    _produce(tm, buf, [1])
    assert buf.maybe_ready, "an arriving item sets it"
    assert w._pop_streaming_edge(buf, EDGE, RID) is None and not buf.maybe_ready
    _produce(tm, buf, [2], first_uuid=200)
    se = w._pop_streaming_edge(buf, EDGE, RID)
    assert se is not None and buf.maybe_ready, "a pop may leave another chunk"
    assert w._pop_streaming_edge(buf, EDGE, RID) is None and not buf.maybe_ready
    buf.store_uningested_edge(se)
    assert buf.maybe_ready, "a refused edge coming back sets it"
    assert w._pop_streaming_edge(buf, EDGE, RID) is se
    assert w._pop_streaming_edge(buf, EDGE, RID) is None and not buf.maybe_ready
    buf.signal_done(2)
    assert buf.maybe_ready, "the producer's done signal sets it"
    assert w._pop_streaming_edge(buf, EDGE, RID).chunk.is_final
