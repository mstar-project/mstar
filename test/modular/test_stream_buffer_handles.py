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
from mstar.streaming.chunk_policy import (
    FixedChunkPolicy,
    LeftContextChunkPolicy,
    RampChunkPolicy,
    ScheduledLeftContextChunkPolicy,
    SlidingWindowChunkPolicy,
)
from mstar.streaming.stream_buffer import StreamBuffer
from mstar.streaming.topology import Connection
from mstar.worker.stream_buffers import StreamBufferManager

EDGE, NODE, RID = "new_token", "snac_decoder", 1


class _TM:
    """The tensor-manager surface the stream path touches, over a real store."""

    get_tensor = TensorCommunicationManager.get_tensor
    get_tensors = TensorCommunicationManager.get_tensors
    dereference_batch_uniform = TensorCommunicationManager.dereference_batch_uniform
    cleanup_collectable = TensorCommunicationManager.cleanup_collectable
    increment_ref = TensorCommunicationManager.increment_ref
    increment_ref_batch_uniform = TensorCommunicationManager.increment_ref_batch_uniform

    def __init__(self, bookkeeping):
        self.tensor_store = TensorStore(bookkeeping=bookkeeping)
        self.uuid_to_shard_dim = {}
        self._next_uuid = 1000

    def store_and_return_tensor_info(self, request_id, tensors, skip_cuda_sync=False):
        """Mints a uuid per tensor and stores it with no references, as the
        real one does; the descriptor is a stand-in."""
        out = {}
        for name, ts in tensors.items():
            for t in ts:
                uuid, self._next_uuid = self._next_uuid, self._next_uuid + 1
                self.tensor_store.put_tensor(request_id, uuid, t, _info(uuid))
                out.setdefault(name, []).append(_info(uuid))
        return out

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


@pytest.mark.parametrize("kind", ["python", "rust"])
def test_a_chunked_producers_pieces_are_joined_into_one_stored_item(kind):
    """A buffer holding uuids joins a chunked producer's pieces through the
    manager: one new stored item, which the buffer holds instead of the
    pieces, and which is released like any other once consumed."""
    tm = _TM(_bookkeeping(kind))
    conn = Connection(
        "Thinker", "Talker", EDGE, lambda: FixedChunkPolicy(chunk_size=1), join_dim=0,
    )
    req = SimpleNamespace(stream_buffers={}, stream_buffers_by_consumer={})
    streams = StreamBufferManager(SimpleNamespace(per_request_info={RID: req}), tm, [conn])
    streams.consumer_node = {EDGE: NODE}
    streams.add_request(RID, SimpleNamespace(partition_name="Talker", stream_lead_items={}))
    buf = req.stream_buffers[EDGE]
    pieces = [torch.ones(2, 3), torch.zeros(0, 3), torch.full((1, 3), 2.0)]
    uuids = [100 + k for k in range(len(pieces))]
    buf.pre_read_register_joined(uuids)
    for uuid, piece in zip(uuids, pieces, strict=True):
        tm.tensor_store.put_tensor(RID, uuid, piece, _info(uuid))
        tm.tensor_store.increment_ref(uuid, 1)
        buf.put(uuid, uuid)
    present = tm.tensor_store.check_uuid_presence
    se = streams.pop_edge(buf, EDGE, RID)
    assert len(se.uuids) == 1 and se.uuids[0] not in uuids
    assert not any(present(u) for u in uuids), "the pieces are released once joined"
    (joined,) = _consume(streams, tm, se)
    assert joined.shape == (3, 3) and joined[:2].eq(1).all() and joined[2].eq(2).all()
    buf.signal_done(len(uuids))
    assert buf._producer_done_and_all_read()
    assert not present(se.uuids[0]), "the joined item is released once consumed"


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


def _window_consumer(kind="python"):
    tm = _TM(_bookkeeping(kind))
    buf = StreamBuffer(RID, EDGE, "LLM", FixedChunkPolicy(1))
    req_info = SimpleNamespace(stream_buffers={EDGE: buf}, ended_streams=set())
    streams = StreamBufferManager(SimpleNamespace(per_request_info={RID: req_info}), tm, [])
    streams.consumer_node = {EDGE: NODE}
    return streams, tm, buf


def test_a_stream_window_takes_every_ready_item_up_to_a_runs_count_and_ends_drained():
    """A loop that reads an item a stream each iteration: a run takes as many
    as are ready (up to its count), one an iteration, and the window that
    drains a done stream is its final chunk."""
    streams, tm, buf = _window_consumer()
    _produce(tm, buf, [10, 11, 12])
    se = streams.pop_edge(buf, EDGE, RID, window=4)
    assert se.chunk.num_items == 3 and not se.chunk.is_final
    assert _consume(streams, tm, se)[0].flatten().tolist() == [10, 11, 12]
    uuids = _produce(tm, buf, [13, 14], first_uuid=200)
    buf.signal_done(5)
    se = streams.pop_edge(buf, EDGE, RID, window=4)
    assert se.uuids == uuids and se.chunk.is_final
    assert streams.pop_edge(buf, EDGE, RID, window=4) is None


def test_a_window_a_run_ran_short_of_goes_back_first():
    """A step that settled on fewer iterations than its window hands the rest
    back ahead of anything newer, so the next step reads them in order."""
    streams, tm, buf = _window_consumer()
    _produce(tm, buf, [10, 11, 12])
    window = streams.pop_edge(buf, EDGE, RID, window=3)
    _produce(tm, buf, [13], first_uuid=200)
    assert not streams.return_unread(RID, window, used=1)
    assert _consume(streams, tm, window)[0].flatten().tolist() == [10, 11, 12]
    rest = streams.pop_edge(buf, EDGE, RID, window=3)
    assert rest.chunk.start_offset == 1 and rest.chunk.num_items == 2
    assert _consume(streams, tm, rest)[0].flatten().tolist() == [11, 12]
    assert _consume(streams, tm, streams.pop_edge(buf, EDGE, RID))[0].tolist() == [13]


@pytest.mark.parametrize("policy", [
    lambda: SlidingWindowChunkPolicy(3, 2),
    lambda: FixedChunkPolicy(2),
    lambda: LeftContextChunkPolicy(2, 3),
    lambda: RampChunkPolicy(1, 3, growth=2.0, max_chunk=6),
    lambda: ScheduledLeftContextChunkPolicy((1, 3, 8), 4, 2),
], ids=["sliding", "fixed", "left_context", "ramp", "scheduled"])
def test_a_device_queue_hands_its_consumer_the_chunks_the_host_queue_does(policy):
    """``stream_queue: device``: the buffer holds positions in the producer's
    ring and a chunk is read from it in one gather. Every chunk, its window
    and its geometry match the host queue's, for each chunk policy."""
    from mstar.engine.resources.loopback_rows import LoopbackRows

    frames = torch.arange(23 * 2).reshape(23, 2)
    tm = _TM(_bookkeeping("python"))
    host = StreamBuffer(RID, EDGE, "LLM", policy())
    device = StreamBuffer(RID, EDGE, "LLM", policy())
    request_state = SimpleNamespace(per_request_info={RID: SimpleNamespace(stream_buffers={EDGE: host})})
    streams = StreamBufferManager(request_state, tm, [])
    streams.consumer_node = {EDGE: NODE}
    ring = LoopbackRows(torch.device("cpu"), capacity=2)
    ring.ask_ring(EDGE, depth=12)
    ring.ring_table(EDGE, frames[:1])
    ring.add(RID)
    written = 0
    chunks = {"host": [], "device": []}

    def pop(name, buf):
        streams.rings = {EDGE: ring} if name == "device" else {}
        request_state.per_request_info[RID].stream_buffers = {EDGE: buf}
        while (se := streams.pop_edge(buf, EDGE, RID)) is not None:
            tensors = [tm.tensor_store.get_tensor(u) for u in se.uuids]
            items = torch.cat([t.reshape(-1, 2) for t in tensors]) if tensors else torch.zeros(0, 2)
            chunks[name].append((items.tolist(), se.chunk))

    for want in (3, 4, 1, 5, 2, 8):  # a run's frames, a step at a time
        # the ring's room caps the run, as it caps the producer's
        step = min(want, ring.ring_room(EDGE, RID))
        batch = frames[written:written + step]
        for k, frame in enumerate(batch):
            uuid = 100 + written + k
            tm.tensor_store.put_tensor(RID, uuid, frame[None], _info(uuid))
            tm.tensor_store.increment_ref(uuid, 1)
            host.pre_read_register(uuid)
            host.put(uuid, uuid)
        (base,) = ring.ring_reserve(EDGE, [RID], step)
        ring.ring_write(EDGE, [RID], batch[None], [base])
        device.put_positions(list(range(base, base + step)))
        written += step
        pop("host", host)
        pop("device", device)
    for buf in (host, device):
        buf.signal_done(written)
    pop("host", host)
    pop("device", device)
    assert chunks["device"] == chunks["host"] and len(chunks["host"]) > 3


@pytest.mark.parametrize("ran", [0, 1, 4])
@pytest.mark.parametrize("kind", ["python", "rust"])
def test_overlapping_run_windows_consume_actual_inputs_once(ran, kind):
    streams, tm, buf = _window_consumer(kind)
    uuids = _produce(tm, buf, range(6))
    buf.signal_done(6)
    first = streams.reserve_edge(buf, EDGE, RID, 4)
    second = streams.reserve_edge(buf, EDGE, RID, 4)
    assert first.chunk.reservation.bound == 4
    assert second.chunk.reservation.bound == 2
    assert second.uuids == uuids
    assert streams.reserve_edge(buf, EDGE, RID, 4) is None
    assert streams.consumption(RID)[EDGE] == 0
    assert buf.policy.items_consumed == 0 and not buf._final_chunk_emitted

    settled = streams.settle_window(RID, first, ran)
    assert settled.num_items == ran and not settled.is_final
    _consume(streams, tm, first)
    assert all(tm.tensor_store.check_uuid_presence(u) for u in uuids), "the successor still owns its snapshot"
    snapshot = _consume(streams, tm, second)[0].flatten()
    assert snapshot[ran:ran + 2].tolist() == list(range(ran, ran + 2))
    settled = streams.settle_window(RID, second, 2)
    assert settled.is_final == (ran == 4)
    assert streams.consumption(RID)[EDGE] == ran + 2
    assert buf.policy.items_consumed == ran + 2
    assert all(not tm.tensor_store.check_uuid_presence(u) for u in uuids[:ran + 2])
    assert all(tm.tensor_store.check_uuid_presence(u) for u in uuids[ran + 2:])


def test_two_short_successors_trim_the_retained_source_and_rollback_keeps_refs_balanced():
    streams, tm, buf = _window_consumer()
    uuids = _produce(tm, buf, range(8))
    first = streams.reserve_edge(buf, EDGE, RID, 4)
    second = streams.reserve_edge(buf, EDGE, RID, 4)
    streams.settle_window(RID, first, 2)
    _consume(streams, tm, first)
    third = streams.reserve_edge(buf, EDGE, RID, 4)
    assert third.chunk.start_offset == 2 and third.chunk.reservation.bound == 2
    assert third.uuids == uuids[2:]
    streams.settle_window(RID, second, 1)
    _consume(streams, tm, second)
    fourth = streams.reserve_edge(buf, EDGE, RID, 4)
    assert fourth.chunk.start_offset == 3 and fourth.chunk.reservation.bound == 3
    assert fourth.uuids == uuids[3:]
    streams.return_edge(RID, fourth)
    retry = streams.reserve_edge(buf, EDGE, RID, 4)
    assert retry.uuids == fourth.uuids
    streams.settle_window(RID, third, 2)
    _consume(streams, tm, third)
    streams.settle_window(RID, retry, 3)
    _consume(streams, tm, retry)
    assert buf._consumed == 8 and buf.policy.items_consumed == 8
    assert not buf._buffer and not buf._reservations and not buf._waiting_graph_edges
    assert not any(tm.tensor_store.check_uuid_presence(u) for u in uuids)


def test_run_reservation_never_pads_missing_live_input_or_crosses_a_walk():
    buf = StreamBuffer(RID, EDGE, "LLM", FixedChunkPolicy(1, frozenset({"decode"})), walk_tagged=True)
    buf.put_positions([0, 1], "decode")
    buf.put_positions([2], "other")
    buf.signal_done(4)  # one item has not reached this worker yet
    first = buf.reserve_window(4, "decode")
    assert first.num_items == 2 and first.reservation.bound == 2
    assert buf.reserve_window(4, "decode") is None
    buf.settle_window(first.reservation, 2)
    assert buf.reserve_window(4, "decode") is None
    second = buf.reserve_window(4, "other")
    buf.settle_window(second.reservation, 1)
    assert buf.reserve_window(4, "decode") is None
    buf.put_positions([3], "decode")
    tail = buf.reserve_window(4, "decode")
    assert tail.num_items == 1 and tail.reservation.bound == 4
    buf.settle_window(tail.reservation, 4)
    assert buf._consumed == 4 and buf.policy.items_consumed == 4
    assert buf.reserve_window(4, "other") is None


def test_a_reserved_window_narrows_before_its_successor_and_refuses_out_of_order_settlement():
    streams, tm, buf = _window_consumer()
    _produce(tm, buf, range(6))
    first = streams.reserve_edge(buf, EDGE, RID, 4)
    streams.return_unread(RID, first, 1)
    second = streams.reserve_edge(buf, EDGE, RID, 4)
    assert second.chunk.reservation.predecessor_bound == 1
    assert second.chunk.reservation.bound == 4
    with pytest.raises(ValueError, match="submission order"):
        streams.settle_window(RID, second, 1)
    with pytest.raises(ValueError, match="after its successor"):
        streams.return_unread(RID, first, 0)
    assert buf._consumed == 0
    streams.return_edge(RID, second)
    streams.return_edge(RID, first)
    assert not buf._reservations and buf._consumed == 0
    chunk = streams.pop_edge(buf, EDGE, RID, window=6)
    _consume(streams, tm, chunk)
    assert not any(tm.tensor_store.check_uuid_presence(u) for u in chunk.uuids)


def test_a_late_stream_settlement_cannot_consume_a_recycled_requests_items():
    streams, tm, old = _window_consumer()
    _produce(tm, old, range(4))
    previous = streams.reserve_edge(old, EDGE, RID, 4)
    replacement = StreamBuffer(RID, EDGE, "LLM", FixedChunkPolicy(1))
    streams._request_state.per_request_info[RID].stream_buffers[EDGE] = replacement
    _produce(tm, replacement, range(4), first_uuid=200)
    current = streams.reserve_edge(replacement, EDGE, RID, 4)
    assert streams.settle_window(RID, previous, 4) is None
    streams.return_edge(RID, previous)
    assert replacement._consumed == 0 and replacement.owns_window(current.chunk.reservation)
    streams.settle_window(RID, current, 4)
    _consume(streams, tm, current)
    assert replacement._consumed == 4


def test_ring_reservations_keep_copied_windows_after_positions_are_released_and_reused():
    from mstar.engine.resources.loopback_rows import LoopbackRows

    streams, tm, buf = _window_consumer()
    ring = LoopbackRows(torch.device("cpu"), capacity=1)
    ring.ask_ring(EDGE, depth=6)
    ring.ring_table(EDGE, torch.zeros(1, 1, dtype=torch.long))
    ring.add(RID)
    ring.ring_write(EDGE, [RID], torch.arange(6).reshape(1, 6, 1), ring.ring_reserve(EDGE, [RID], 6))
    buf.put_positions(list(range(6)))
    streams.rings[EDGE] = ring
    first = streams.reserve_edge(buf, EDGE, RID, 4)
    second = streams.reserve_edge(buf, EDGE, RID, 4)
    streams.settle_window(RID, first, 2)
    assert ring.ring_room(EDGE, RID) == 2
    ring.ring_write(EDGE, [RID], torch.tensor([[[100], [101]]]), ring.ring_reserve(EDGE, [RID], 2))
    assert _consume(streams, tm, second)[0].flatten().tolist() == list(range(6))
    _consume(streams, tm, first)
    streams.settle_window(RID, second, 2)
    assert ring.ring_room(EDGE, RID) == 2
    assert not any(tm.tensor_store.check_uuid_presence(u) for u in first.uuids + second.uuids)


@pytest.mark.parametrize("policy", [FixedChunkPolicy(2), RampChunkPolicy(1, 4)])
def test_run_reservations_refuse_policies_that_do_not_stay_one_item(policy):
    buf = StreamBuffer(RID, EDGE, "LLM", policy)
    buf.put_positions(list(range(8)), "decode")
    with pytest.raises(ValueError, match="one-item"):
        buf.reserve_window(4, "decode")
    assert buf._consumed == policy.items_consumed == 0
    assert not buf._reservations


def test_only_executed_tail_reservations_spend_pad_rows_even_at_width_one():
    buf = StreamBuffer(RID, EDGE, "LLM", FixedChunkPolicy(1, continue_after_done=True))
    buf.put_positions([0, 1])
    buf.signal_done(2)
    first = buf.reserve_window(4)
    successor = buf.reserve_window(4)
    assert first.reservation.tail_offset == successor.reservation.tail_offset == 0
    buf.settle_window(first.reservation, 3)  # two real items, then the first tail input
    assert successor.reservation.tail_offset == 0, "a submitted snapshot must not change"
    buf.settle_window(successor.reservation, 2)
    one = buf.reserve_window(1)
    assert one.num_items == 0 and one.reservation.tail_offset == 3
    buf.cancel_window(one.reservation)
    skipped = buf.reserve_window(1)
    buf.settle_window(skipped.reservation, 0)
    retry = buf.reserve_window(1)
    assert retry.reservation.tail_offset == 3, "a refused or skipped input spends no pad row"
    buf.settle_window(retry.reservation, 1)
    assert buf._tail_consumed == 4
    assert buf._consumed == buf.policy.items_consumed == 2
