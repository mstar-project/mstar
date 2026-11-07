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


@pytest.mark.parametrize("factory,bound", [(lambda: FixedChunkPolicy(2), 4), (lambda: RampChunkPolicy(1, 4), 1)])
def test_run_reservations_stop_at_changed_input_geometry(factory, bound):
    policy = factory()
    buf = StreamBuffer(RID, EDGE, "LLM", policy)
    buf.put_positions(list(range(8)), "decode")
    chunk = buf.reserve_window(4, "decode")
    assert chunk.reservation.bound == bound
    assert buf._consumed == policy.items_consumed == 0
    assert buf.window_bound(chunk.reservation) == bound


def test_only_executed_tail_reservations_spend_pad_rows_even_at_width_one():
    buf = StreamBuffer(RID, EDGE, "LLM", FixedChunkPolicy(1, continue_after_done=True))
    buf.put_positions([0, 1])
    buf.signal_done(2)
    first = buf.reserve_window(4)
    successor = buf.reserve_window(4)
    assert first.tail_offset == successor.tail_offset == 0
    buf.settle_window(first.reservation, 3)  # two real items, then the first tail input
    assert successor.tail_offset == 0, "a submitted snapshot must not change"
    buf.settle_window(successor.reservation, 2)
    one = buf.reserve_window(1)
    assert one.num_items == 0 and one.tail_offset == 3
    buf.cancel_window(one.reservation)
    skipped = buf.reserve_window(1)
    buf.settle_window(skipped.reservation, 0)
    retry = buf.reserve_window(1)
    assert retry.tail_offset == 3, "a refused or skipped input spends no pad row"
    buf.settle_window(retry.reservation, 1)
    assert buf._tail_consumed == 4
    assert buf._consumed == buf.policy.items_consumed == 2


def test_retained_iteration_inputs_require_an_independent_policy_preview():
    class Aliased(FixedChunkPolicy):
        def preview_copy(self):
            return self

    streams, _, _ = _consumer(1, 1)
    streams.connections = [Connection("Thinker", "Talker", EDGE, lambda: Aliased(1))]
    with pytest.raises(ValueError, match="independent policy preview"):
        streams.configure_windows({NODE: 4}, {NODE: frozenset({("decode", EDGE)})})


def test_only_declared_iteration_edges_reserve_and_tail_readiness_follows_the_walk():
    tm = _TM(_bookkeeping("python"))
    conn = Connection("Thinker", "Talker", EDGE, lambda: FixedChunkPolicy(1, frozenset({"decode"})))
    audio = Connection("Encoder", "Talker", "audio", lambda: FixedChunkPolicy(2))
    fwd = SimpleNamespace(graph_walk="prefill")
    req = SimpleNamespace(
        stream_buffers={}, stream_buffers_by_consumer={},
        per_partition_info={"Talker": SimpleNamespace(current_fwd_info=fwd)},
    )
    streams = StreamBufferManager(SimpleNamespace(per_request_info={RID: req}), tm, [conn, audio])
    streams.consumer_node = {EDGE: NODE, "audio": NODE}
    streams.configure_windows({NODE: 4}, {NODE: frozenset({("decode", EDGE)})})
    streams.add_request(RID, SimpleNamespace(partition_name="Talker", stream_lead_items={}))
    text, sound = req.stream_buffers[EDGE], req.stream_buffers["audio"]
    _produce(tm, sound, [10, 11], first_uuid=200)
    text.signal_done(0)
    assert streams.pop_edge(text, EDGE, RID) is None
    assert text.maybe_ready, "a later graph walk can make its empty tail ready without new producer input"
    fwd.graph_walk = "decode"
    edges = streams.poll_for(RID, NODE)
    by_name = {edge.name: edge for edge in edges}
    assert by_name[EDGE].chunk.reservation.bound == 4 and by_name[EDGE].chunk.num_items == 0
    assert by_name["audio"].chunk.reservation is None and sound._consumed == 2
    streams.return_edge(RID, by_name[EDGE])
    tm.dereference_batch_uniform(by_name["audio"].uuids)
    assert not text._reservations and text._tail_consumed == 0


@pytest.mark.parametrize("tail_only", [False, True])
def test_a_skipped_reader_keeps_its_origin_and_narrowed_bound_through_real_graph_retry(tail_only):
    from test_graph_runtime import _build

    from mstar.graph.base import GraphNode, Loop
    from mstar.graph.runtime.base import ColumnarEdgeSpecs
    from mstar.model.submodule_base import InputMetadata
    from mstar.worker.executing_batch import make_executing_batch

    graph = Loop(name="loop", section=GraphNode(name=NODE, input_names={EDGE, "state"}, outputs=[]),
                 max_iters=20, outputs=[])
    graph.section._register_streaming({EDGE})
    request_state, runtime, rid = _build(graph, 0, "decode", {NODE}, {"loop"})
    tm = _TM(runtime.tensor_store.bookkeeping)
    conn = Connection("LLM", "default", EDGE, lambda: FixedChunkPolicy(1, True))
    streams = StreamBufferManager(request_state, tm, [conn], graph_runtime=runtime)
    streams.consumer_node = {EDGE: NODE}
    streams.configure_windows({NODE: 4}, {NODE: frozenset({("decode", EDGE)})})
    streams.add_request(rid, SimpleNamespace(partition_name="default", stream_lead_items={}))
    buf = request_state.per_request_info[rid].stream_buffers[EDGE]
    uuids = []
    for i in range(0 if tail_only else 8):
        uuid = 800 + i
        tm.tensor_store.put_tensor(rid, uuid, torch.tensor([i]), _info(uuid))
        tm.tensor_store.increment_ref(uuid, 1)
        buf.pre_read_register(uuid)
        buf.put(uuid, uuid)
        uuids.append(uuid)
    if tail_only:
        buf.signal_done(0)
    first = streams.reserve_edge(buf, EDGE, rid, 4, "decode")
    retained = streams.reserve_edge(buf, EDGE, rid, 4, "decode")
    streams.return_unread(rid, retained, 2)
    assert retained.chunk.reservation.bound == 4 and streams.window_bound(rid, retained) == 2
    state = ColumnarEdgeSpecs.empty()
    state.add(rid, "state", [], False, next_node=NODE)
    assert runtime.ingest_inputs_batch(state, can_buffer=False) == []
    block = streams.edge_block([(rid, retained)])
    assert runtime.ingest_inputs_batch(block, can_buffer=False, is_streaming=True) == []
    streams.mark_ingested(rid, retained)
    runtime.pop_rids(NODE, "decode", [rid])
    streams.settle_window(rid, first, 1)
    tm.dereference_batch_uniform(first.uuids)
    assert streams.retained_prefix(rid, retained) == 1

    # A skipped launch did not consume the current graph inputs. An ordinary
    # poll's offered replacement is refused, then its newest reservation is
    # cancelled before the original reader re-admits at a narrower width.
    offered = streams.pop_edge(buf, EDGE, rid)
    assert runtime.ingest_inputs_batch(streams.edge_block([(rid, offered)]), can_buffer=False,
                                       is_streaming=True) == [0]
    streams.return_edge(rid, offered)
    runtime.push_back_node(NODE, [rid], [0])
    retry = runtime.pop_rids(NODE, "decode", [rid])
    assert {edge.signal for edge in retry.input_edges.edge_tuples()} == {EDGE, "state"}
    inputs = streams.input_tensors(retry.input_edges, NODE, [rid]).by_rid
    chunks = streams.chunks_for(rid, NODE, inputs[rid])
    assert chunks[EDGE].reservation is retained.chunk.reservation
    metadata = {rid: InputMetadata(stream_chunks=chunks, dynamic_loop_iter_counts={"loop": 1},
                                   graph_pass=0, iteration_origin=1)}
    batch = make_executing_batch(
        streams, NODE, "decode", [rid], inputs,
        {rid: request_state.get_fwd_info(rid, "default")}, per_request_input_metadata=metadata, repeats=4,
    )
    assert batch.step_context.repeats == 2
    assert batch.per_request_input_metadata[rid].iteration_origin == 0
    assert batch.stream_windows[rid][0].chunk.reservation is retained.chunk.reservation
    from mstar.worker.worker import Worker

    worker = Worker.__new__(Worker)
    worker.streams = streams
    worker._step_repeats = SimpleNamespace(
        asked=True, ceiling=lambda *args: (8, "configured"), note_ask=lambda *args: None,
    )
    worker._ring_signals = {}
    worker._walk_drivers = {}
    scheduled = SimpleNamespace(node_name=NODE, graph_walk="decode", repeats=4)
    pending = SimpleNamespace(batch=scheduled, node_batch=batch)
    speculation = SimpleNamespace(scheduled_batch=scheduled, node_batch=batch, consumed_streaming_edges={})
    worker._stamp_repeats(speculation, pending, 1)
    assert batch.step_context.repeats == 2, "a retry cannot restore the original wider bound"
    batch.step_context.repeats = 1  # a resource narrows at admission
    worker._await_admit_settled(pending)
    assert streams.window_bound(rid, retained) == 1
    assert batch.stream_windows[rid][0].chunk.reservation is retained.chunk.reservation
    worker._await_admit_settled(pending)  # an already settled fence keeps its retained reader
    from mstar.engine.engine import Engine
    from mstar.graph.loop_indices import LoopLayout
    from mstar.model.submodule_base import IterationInput

    # A retry can miss its run capture. Ordinary staging must skip the same
    # retained real/tail prefix that a device cursor would select past.
    engine = Engine.__new__(Engine)
    engine._device = torch.device("cpu")
    engine._loops = {(NODE, "decode"): LoopLayout("loop")}
    node = SimpleNamespace(ITERATION_INPUTS={"cond": IterationInput(EDGE, streamed=True, pad="tail")},
                           request_states={rid: {"tail": torch.tensor([-1., -2.])}})
    column = engine._iteration_columns(NODE, node, batch.step_context, [rid],
                                       batch.per_request_info_wrapped, batch.per_request_input_tensors)
    assert column["cond"].item() == (-2 if tail_only else 1)
    streams.settle_window(rid, retained, 1)
    tm.cleanup_collectable(*runtime.cleanup_consumed_inputs(NODE, [rid], [0]))
    assert not buf._reservations
    if tail_only:
        assert buf._tail_consumed == 2 and buf._consumed == 0
    else:
        assert buf._consumed == 2
        assert not any(tm.tensor_store.check_uuid_presence(uuid) for uuid in uuids[:2])
        assert all(tm.tensor_store.check_uuid_presence(uuid) for uuid in uuids[2:])


def test_an_ordinary_window_leaves_eos_for_its_empty_step_and_returns_the_same_tail():
    streams, tm, buf = _consumer(1, 1)
    buf.policy = FixedChunkPolicy(1, True)
    _produce(tm, buf, [5, 6])
    buf.signal_done(2)
    real = streams.pop_edge(buf, EDGE, RID, window=8)
    assert real.chunk.num_items == 2 and buf._tail_consumed == 0
    _consume(streams, tm, real)
    eos = streams.pop_edge(buf, EDGE, RID, window=8)
    assert eos.chunk.num_items == 0 and eos.chunk.tail_offset == 0
    streams.return_edge(RID, eos)
    retry = streams.pop_edge(buf, EDGE, RID, window=8)
    assert retry.chunk is eos.chunk and buf._tail_consumed == 1
    pad = streams.pop_edge(buf, EDGE, RID, window=8)
    assert pad.chunk.tail_offset == 1


@pytest.mark.parametrize("factory,prime", [
    (lambda: FixedChunkPolicy(3), 0),
    (lambda: SlidingWindowChunkPolicy(4, 2), 0),
    (lambda: LeftContextChunkPolicy(4, 2), 0),
    (lambda: ScheduledLeftContextChunkPolicy([2, 4], 4, 2), 3),
    (lambda: RampChunkPolicy(1, 2, growth=2, max_chunk=4), 0),
    (lambda: FixedChunkPolicy(2, frozenset({"decode"})), 0),
])
def test_retained_policy_transitions_match_ordinary_pops_through_context_final_and_tail(factory, prime):
    live = StreamBuffer(RID, EDGE, "LLM", factory(), walk_tagged=True)
    oracle = StreamBuffer(RID, EDGE, "LLM", factory(), walk_tagged=True)
    for buf in (live, oracle):
        if prime:
            buf.prime_context(prime)
        buf.put_positions(list(range(21)), "decode")
        buf.signal_done(21)
    executed = 0
    while executed < 20 and oracle.has_chunk_ready("decode"):
        before = (live._consumed, live.policy.items_consumed, [run.copy() for run in live._walk_runs])
        chunk = live.reserve_window(3, "decode")
        assert chunk is not None
        assert (live._consumed, live.policy.items_consumed, list(live._walk_runs)) == before
        transitions = chunk.reservation.transitions
        count = min(len(transitions), 20 - executed)
        expected_released = []
        for transition in transitions[:count]:
            expected = oracle.pop_chunk()
            assert transition.chunk == expected.info
            expected_released.extend(expected.released)
        landed = live.settle_window(chunk.reservation, count)
        assert landed.released == expected_released
        assert (live._consumed, live._tail_consumed, live._delivered_end, live._final_chunk_emitted) == (
            oracle._consumed, oracle._tail_consumed, oracle._delivered_end, oracle._final_chunk_emitted,
        )
        assert vars(live.policy) == vars(oracle.policy)
        assert live._buffer == oracle._buffer and list(live._walk_runs) == list(oracle._walk_runs)
        executed += count
    assert executed > 1


@pytest.mark.parametrize("ran", [0, 1, 2])
def test_successor_keeps_frozen_policy_prefix_when_arrivals_change_chunk_stride(ran):
    class SamplingPolicy(FixedChunkPolicy):
        # A single output item per iteration, sampling every other source item
        # when the queue is long. Policy behavior depends on arrivals.
        def next_chunk_size(self, buffer_len):
            return 2 if buffer_len >= 8 else 1

    buf = StreamBuffer(RID, EDGE, "LLM", SamplingPolicy(1), walk_tagged=True)
    buf.put_positions(list(range(6)), "decode")
    first = buf.reserve_window(2, "decode")
    assert [t.consumed for t in first.reservation.transitions] == [1, 2]
    buf.put_positions(list(range(6, 12)), "decode")
    second = buf.reserve_window(2, "decode")
    assert [t.consumed for t in second.reservation.transitions] == [1, 2, 4, 6]
    assert buf._consumed == buf.policy.items_consumed == 0
    assert list(buf._walk_runs) == [["decode", 12]]
    buf.settle_window(first.reservation, ran)
    assert buf.retained_prefix(second.info) == ran  # iterations, not consumed items
    expected = second.reservation.transitions[ran + 1].consumed
    buf.settle_window(second.reservation, 2)
    assert buf._buffer == list(range(expected, 12))
    assert buf._consumed == buf.policy.items_consumed == expected
    assert buf._executed_chunks == ran + 2


class _HistoryChunkPolicy(FixedChunkPolicy):
    """Sizing depends on both available length and the history of earlier pops."""
    def __init__(self):
        super().__init__(2)
        self.history = []
        self.last_available = 0

    def next_chunk_size(self, buffer_len):
        self.last_available = buffer_len
        return 1 if (buffer_len + len(self.history)) % 2 else 2

    def register_chunk(self, chunk_size):
        self.history.append(self.last_available)
        super().register_chunk(chunk_size)


@pytest.mark.parametrize("factory", [lambda: FixedChunkPolicy(2), lambda: SlidingWindowChunkPolicy(4, 2),
                                      _HistoryChunkPolicy])
@pytest.mark.parametrize("stop,withheld,finished", [
    (None, False, False), (1, True, True), (3, False, True), (5, True, True),
    (1, False, False), (3, False, False), (1, True, False),
])
def test_future_stream_decisions_equal_ordinary_interleaving_and_preserve_later_arrivals(
    factory, stop, withheld, finished,
):
    from mstar.engine.resources.iteration_tables import IterationTables
    from mstar.engine.resources.step import BucketKey, SlotLease

    source = StreamBuffer(RID, EDGE, "LLM", factory())
    oracle = StreamBuffer(RID, EDGE, "LLM", factory())
    source.put_positions([0, 1, 2])
    oracle.put_positions([0, 1, 2])
    future, bound = list(range(3, 9)), 8
    snapshot = source.reserve_future_window(future, bound)
    reservation = snapshot.reservation
    assert source._consumed == source._executed_chunks == 0
    assert source.policy.items_consumed == 0 and source._buffer == [0, 1, 2]
    assert source.reserve_window(1) is None
    choices = {(parent, end, done): selected for parent, end, done, selected in reservation.decisions}
    owner = IterationTables(torch.device("cpu"))
    lease = SlotLease(0, BucketKey("consume", 2, 2))
    policy_rows = torch.arange(1, lease.bucket.bs + 1, dtype=torch.long, device=torch.device("cpu"))
    window = owner.prepare_stream_window(
        lease, len(reservation.decisions), max(1, len(reservation.transitions)),
        rows=policy_rows, slots=lease.bucket.bs + 2,
    )
    owner.stage_stream_window(lease, [1], [reservation], [[True] * len(reservation.transitions)])
    window.gather()
    parent, count, produced, done = -1, 0, [], False
    chunks, released = [], []

    def pop():
        nonlocal parent, count
        selected = choices[parent, 3 + len(produced), done]
        window.select(torch.tensor([3 + len(produced), -1]), torch.tensor([done, False]),
                      torch.tensor([True, False]))
        assert window.selected.tolist() == [selected, -1]
        assert bool(window.ready.item()) == (selected >= 0)
        if not oracle.has_chunk_ready():
            assert selected == -1
            return False
        chunk = oracle.pop_chunk()
        assert selected >= 0 and reservation.transitions[selected].chunk == chunk.info
        parent, count = selected, count + 1
        assert window.metadata[0, :5].tolist() == [chunk.start_offset, chunk.context_items, chunk.num_items,
                                                 int(chunk.is_final), chunk.tail_offset]
        window.advance()
        assert window.history.tolist() == [parent, -1] and window.counts.tolist() == [count, 0]
        chunks.append(chunk.info)
        released.extend(chunk.released)
        return not chunk.is_final

    for step, item in enumerate(future, 1):
        if not (step == stop and withheld):
            produced.append(item)
            oracle.put_positions([item])
        if step == stop and finished:
            done = True
            oracle.signal_done(3 + len(produced))
        if count < bound:
            pop()
        if step == stop:
            break
    while count < bound and oracle.has_chunk_ready():
        if not pop():
            break
    # Publication and EOF are ordinary output routing, before semantic settle.
    source.put_positions(produced)
    if done:
        source.signal_done(3 + len(produced))
    else:
        source.put_positions([100, 101])  # later arrivals were not in this launch's snapshot
    result = source.settle_window(reservation, count, final_transition=parent)
    assert result.released == released
    assert source._consumed == oracle._consumed
    assert source._executed_chunks == oracle._executed_chunks == count
    assert source._delivered_end == oracle._delivered_end
    assert source._buffer == oracle._buffer + ([] if done else [100, 101])
    assert source._final_chunk_emitted == oracle._final_chunk_emitted
    assert source.policy.items_consumed == oracle.policy.items_consumed
    if isinstance(source.policy, _HistoryChunkPolicy):
        assert source.policy.history == oracle.policy.history
    if not done:
        assert not source.producer_done and not any(chunk.is_final for chunk in chunks)


@pytest.mark.parametrize("continues", [False, True])
def test_future_stream_empty_final_and_continuing_tail_use_distinct_semantic_receipts(continues):
    from mstar.engine.resources.iteration_tables import IterationTables
    from mstar.engine.resources.step import BucketKey, SlotLease

    streams, _, source = _window_consumer()
    source.policy = FixedChunkPolicy(1, continue_after_done=continues)
    source.signal_done(0)
    snapshot = source.reserve_future_window([], 3)
    reservation = snapshot.reservation
    choices = {(parent, end, done): selected for parent, end, done, selected in reservation.decisions}
    owner = IterationTables(torch.device("cpu"))
    lease = SlotLease(0, BucketKey("consume", 1, 1))
    policy_rows = torch.arange(1, lease.bucket.bs + 1, dtype=torch.long, device=torch.device("cpu"))
    window = owner.prepare_stream_window(
        lease, len(reservation.decisions), len(reservation.transitions),
        rows=policy_rows, slots=lease.bucket.bs + 2,
    )
    # These are ordinary preparation outcomes: final flush has no forward;
    # a continuing empty tail does compute. Both have zero input items.
    owner.stage_stream_window(lease, [1], [reservation], [[continues] * len(reservation.transitions)])
    window.gather()
    parent, count = -1, 0
    while count < 3:
        parent = choices[parent, 0, True]
        window.select(torch.tensor([0]), torch.tensor([True]), torch.tensor([True]))
        assert window.selected.item() == parent and window.ready.item() == 1
        assert bool(window.compute_ready.item()) == continues
        assert parent >= 0
        chunk = reservation.transitions[parent].chunk
        assert chunk.num_items == 0 and chunk.is_final is not continues
        count += 1
        window.advance()
        assert window.counts.item() == count and window.history.item() == parent
        if chunk.is_final:
            break
    window.select(torch.tensor([0]), torch.tensor([True]), torch.tensor([True]))
    assert window.selected.item() == -1 and window.ready.item() == window.compute_ready.item() == 0
    # The worker's admit fence must preserve a semantic budget when the
    # model computes fewer iterations (a final empty flush computes none).
    from mstar.streaming.stream_buffer import StreamingEdge
    from mstar.worker.worker import Worker

    worker = Worker.__new__(Worker)
    worker.streams = streams
    worker._step_repeats = SimpleNamespace(asked=True)
    edge = StreamingEdge(EDGE, NODE, [], snapshot.info)
    batch = SimpleNamespace(
        stream_windows={RID: [edge]}, step_context=SimpleNamespace(repeats=1),
        launch_started_event=None,
    )
    worker._await_admit_settled(SimpleNamespace(node_batch=batch, batch=SimpleNamespace(repeats=3)))
    assert streams.window_bound(RID, edge) == 3
    actual = {RID: count if continues else 0}
    with pytest.raises(ValueError, match="actual semantic receipt"):
        worker._settle_stream_windows(batch, actual)
    assert source.owns_window(reservation)
    worker._settle_stream_windows(batch, actual, transitions={(RID, EDGE): (count, parent)})
    assert batch.stream_windows == {} and not source.owns_window(reservation)
    assert source._final_chunk_emitted is not continues
    assert source._executed_chunks == (3 if continues else 1)
    assert source._tail_consumed == (3 if continues else 0)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_future_policy_capture_bank_reuse_clears_history_and_checks_before_staging(device_name):
    from mstar.engine.accelerator_graph_backend import get_accelerator_graph_backend
    from mstar.engine.accelerator_graph_runner import capture_into_graph
    from mstar.engine.resources.iteration_tables import IterationTables
    from mstar.engine.resources.step import BucketKey, SlotLease

    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("real resident policy owner rebinding")
    device = torch.device(device_name)
    source = StreamBuffer(RID, EDGE, "LLM", FixedChunkPolicy(1))
    reservation = source.reserve_future_window([0, 1], 2).reservation
    replacement = StreamBuffer(RID + 1, EDGE, "LLM", FixedChunkPolicy(2))
    newer = replacement.reserve_future_window([7, 8], 1).reservation
    ended = StreamBuffer(RID + 2, EDGE, "LLM", FixedChunkPolicy(1))
    ended.signal_done(0)
    empty = ended.reserve_future_window([], 1).reservation
    receipts = (reservation, newer, empty)
    owner = IterationTables(device)
    lease = SlotLease(0, BucketKey("consume", 2, 2))
    policy_rows = torch.tensor([1, 0], device=device)
    decisions = max(len(r.decisions) for r in receipts)
    transitions = max(len(r.transitions) for r in receipts)
    window = owner.prepare_stream_window(lease, decisions, transitions, 2, rows=policy_rows, slots=5)
    available = torch.tensor([1, 0], device=device)
    done = torch.tensor([False, False], device=device)
    live = torch.tensor([True, False], device=device)

    def execute():
        window.gather()
        window.select(available, done, live)
        window.advance()

    graph = None
    if device_name == "cuda":
        backend = get_accelerator_graph_backend(device)
        graph = capture_into_graph(execute, backend.graph_pool_handle(), device, None, keep_graph=True)[0]

    def run():
        if graph is None:
            execute()
        else:
            graph.replay()
            backend.synchronize()

    owner.stage_stream_window(lease, [1], [reservation], [[True] * len(reservation.transitions)], producer_limits=[2])
    run()
    assert window.counts.tolist() == [1, 0]
    old = window.bank.history[1].clone()
    before = window.bank.decisions.clone(), window.history.clone(), window.counts.clone()
    with pytest.raises(ValueError, match="prepared transitions"):
        owner.stage_stream_window(lease, [1], [reservation], [[]], producer_limits=[2])
    assert all(torch.equal(a, b) for a, b in zip(
        before, (window.bank.decisions, window.history, window.counts), strict=True,
    ))
    # A new owner is published while the physical view still belongs to the
    # old one. Staging cannot reset the old view or its durable receipt.
    owner.stage_stream_window(lease, [3], [newer], [[True] * len(newer.transitions)], producer_limits=[2])
    assert window.counts.tolist() == [1, 0] and window.bank.history[1] == old
    policy_rows.copy_(torch.tensor([3, 1], device=device))
    available[0] = 2
    run()
    assert window.metadata[0, 2] == 2 and window.payload_positions(2)[0].tolist() == [7, 8]
    assert window.bank.counts[[1, 3]].tolist() == [1, 1]
    assert window.bank.history[1] == old, "inactive real owners must not receive scratch writeback"
    policy_rows.copy_(torch.tensor([1, 0], device=device))
    run()
    assert window.bank.counts[[1, 3]].tolist() == [2, 1], "physical reuse must retain both receipts"
    # Empty EOF advances semantic state without any model output descriptor.
    owner.stage_stream_window(lease, [4], [empty], [[False] * len(empty.transitions)], producer_limits=[0])
    policy_rows[0] = 4
    available[0] = 0
    done[0] = True
    run()
    assert window.bank.counts[4] == 1 and window.bank.terminal[4]
    assert window.compute_ready.item() == 0 and window.bank.counts[1] == 2
    assert owner.prepare_stream_window(lease, decisions, transitions, 2, rows=policy_rows, slots=5) is window
    with pytest.raises(ValueError):
        owner.prepare_stream_window(lease, decisions + 1, transitions, 2, rows=policy_rows, slots=5)
    owner.discard_window(lease)
    assert owner.prepare_stream_window(lease, decisions, transitions, 2, rows=policy_rows, slots=5) is not window


@pytest.mark.parametrize("factory", [lambda: FixedChunkPolicy(2), lambda: SlidingWindowChunkPolicy(4, 2)])
@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_future_stream_policy_and_ring_payloads_run_in_one_conditional_graph(factory, device_name, monkeypatch):
    from mstar.engine.accelerator_graph_backend import GraphConditional, get_accelerator_graph_backend
    from mstar.engine.accelerator_graph_runner import capture_into_graph
    from mstar.engine.resources.iteration_tables import IterationTables
    from mstar.engine.resources.loopback_rows import LoopbackRows, LoopbackRowsStep
    from mstar.engine.resources.step import BucketKey, SlotLease, StepContext

    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("real stream-policy conditional graph")
    device = torch.device(device_name)
    ring = LoopbackRows(device, capacity=1)
    ring.add(RID)
    ring.ask_ring(EDGE, 8)
    ring.ring_table(EDGE, torch.zeros(1, 1, dtype=torch.long, device=device))
    ring.ring_reserve(EDGE, [RID], 3)
    ring.ring_skip(EDGE, RID, [1])
    assert ring.ring_reserve(EDGE, [RID], 5) == [3]
    source = StreamBuffer(RID, EDGE, "LLM", factory())
    source.put_positions([0, 2])  # logical stream positions need not equal physical ring positions
    reservation = source.reserve_future_window([3, 4, 5, 6, 7], 8).reservation
    width = max(t.chunk.num_items for t in reservation.transitions)
    lease = SlotLease(0, BucketKey("consume", 1, 1))
    policy_rows = torch.arange(1, lease.bucket.bs + 1, dtype=torch.long, device=device)
    tables = IterationTables(device)
    window = tables.prepare_stream_window(
        lease, len(reservation.decisions), len(reservation.transitions), len(reservation.positions),
     rows=policy_rows, slots=lease.bucket.bs + 2)
    # This synthetic node's ordinary preparation computes only nonempty chunks.
    computes = [t.chunk.num_items > 0 for t in reservation.transitions]
    ctx = StepContext([RID], "produce", 0, False)
    ctx.slot_lease = lease
    ring.plan(LoopbackRowsStep(), ctx)
    rows = ring.indices(lease.bucket, lease.slot)
    inputs = torch.zeros(1, width, 1, dtype=torch.long, device=device)
    a_count, emitted, b_count, done, a_ready, go, remaining = (
        torch.zeros(1, dtype=torch.long, device=device) for _ in range(7)
    )
    live = torch.ones(1, dtype=torch.bool, device=device)
    outputs = torch.zeros(8, dtype=torch.long, device=device)

    def choose():
        window.select(window.origins[:, 0] + window.origins[:, 1] + emitted, done, live)

    def more():
        choose()
        a_ready.copy_(a_count < 3)
        go.copy_(((a_ready != 0) | (window.ready != 0)) & (remaining > 0))

    def produce():
        # Third model execution stops and withholds its item. The consumer
        # sees EOF with exactly two actual future positions, not five reserved.
        keeps = a_count < 2
        ring.ring_scatter(EDGE, rows, window.produced_positions(a_count + 1), (100 + a_count)[:, None], keeps)
        emitted.add_(keeps)
        a_count.add_(1)
        done.copy_(a_count == 3)

    def consume():
        ring.ring_gather(EDGE, rows, window.payload_positions(width), inputs)
        outputs.scatter_(0, window.counts.clamp_max(7), inputs.sum(dim=(1, 2)))
        b_count.add_(1)

    def boundary():
        remaining.sub_(1)
        more()

    functions = (more, produce, choose, consume, window.advance, boundary)
    backend, graph, launches = None, None, []
    if device_name == "cuda":
        backend = get_accelerator_graph_backend(device)
        captures = [capture_into_graph(fn, backend.graph_pool_handle(), device, None, keep_graph=True)[0]
                    for fn in functions]
        graph = backend.compose((captures[0], GraphConditional("while", go, (
            GraphConditional("if", a_ready, (captures[1],)), captures[2],
            GraphConditional("if", window.compute_ready, (captures[3],)),
            GraphConditional("if", window.ready, (captures[4],)), captures[5],
        ))))
        runtime, launch = backend.while_runtime(), backend.while_runtime().launch
        monkeypatch.setattr(runtime, "launch", lambda *args: (launches.append(args[0]), launch(*args))[-1])
    tables.stage_stream_window(lease, [1], [reservation], [computes], producer_limits=[5])
    window.gather()
    for value in (a_count, emitted, b_count, done, outputs):
        value.zero_()
    remaining.fill_(12)
    ring.ring_write(EDGE, [RID], torch.tensor([[[10], [0], [12]]], device=device), [0])
    if graph is not None:
        graph.replay()
        backend.synchronize()
        assert len(launches) == 1
    else:
        more()
        while go.item():
            if a_ready.item():
                produce()
            choose()
            if window.compute_ready.item():
                consume()
            if window.ready.item():
                window.advance()
            boundary()
    oracle = StreamBuffer(RID, EDGE, "LLM", factory())
    oracle.put_positions([0, 2])
    values = {0: 10, 2: 12, 3: 100, 4: 101}
    expected, pops = [], 0
    def pop():
        nonlocal pops
        chunk = oracle.pop_chunk()
        if chunk.num_items:
            expected.append(sum(values[p] for p in chunk.data["data"]))
        pops += 1
    for position in (3, 4, None):
        if position is None:
            oracle.signal_done(4)
        else:
            oracle.put_positions([position])
        if oracle.has_chunk_ready():
            pop()
    while oracle.has_chunk_ready():
        pop()
    assert a_count.item() == 3 and emitted.item() == 2 and b_count.item() == len(expected)
    assert outputs[:len(expected)].tolist() == expected and window.counts.item() == pops
    source.put_positions([3, 4])
    source.signal_done(4)
    source.settle_window(reservation, pops, final_transition=window.history.item())
    assert source._consumed == oracle._consumed and source._buffer == oracle._buffer
    assert source._executed_chunks == oracle._executed_chunks and source._final_chunk_emitted
    assert ring.ring_room(EDGE, RID) == 0  # reads alone never release payload ownership
    if graph is not None:
        graph.reset()


def test_future_stream_settlement_requires_actual_publication_and_preserves_zero_work_retry():
    source = StreamBuffer(RID, EDGE, "LLM", FixedChunkPolicy(1))
    reservation = source.reserve_future_window([10, 11], 2).reservation
    chosen = next(selected for parent, end, done, selected in reservation.decisions
                  if parent == -1 and end == 1 and done)
    with pytest.raises(ValueError, match="actual arrivals"):
        source.settle_window(reservation, 1, final_transition=chosen)
    source.put_positions([10])
    with pytest.raises(ValueError, match="producer completion"):
        source.settle_window(reservation, 1, final_transition=chosen)
    assert source._consumed == source._executed_chunks == 0
    assert source.owns_window(reservation)
    source.settle_window(reservation, 0, final_transition=-1)
    assert source._buffer == [10] and source.policy.items_consumed == 0
    retry = source.reserve_future_window([11], 2)
    source.cancel_window(retry.reservation)
    assert source.pop_chunk().data == {"data": [10]}


def test_future_window_adopts_the_ingested_first_pop_and_restores_it_on_refusal():
    source = StreamBuffer(RID, EDGE, "LLM", _HistoryChunkPolicy())
    oracle = StreamBuffer(RID, EDGE, "LLM", _HistoryChunkPolicy())
    source.put_positions([0, 1, 2])
    oracle.put_positions([0, 1, 2])
    original = source.reserve_window(2)
    first = oracle.pop_chunk()  # ordinary polling already fixed this chunk
    assert first.num_items == 2 and first.released == [0]
    reader = original.reservation
    admitted = source.window_bound(reader)
    assigned = source._chunks_popped
    extended = source.reserve_future_window([3, 4], 4, adopt=reader)
    assert extended.chunk_index == original.chunk_index
    assert not source.owns_window(reader) and source.owns_window(extended.reservation)
    assert source._chunks_popped == assigned and source._executed_chunks == 0
    source.cancel_window(extended.reservation)
    assert source.owns_window(reader) and source.window_bound(reader) == admitted
    assert source._buffer == [0, 1, 2] and source.policy.history == []
    extended = source.reserve_future_window([3, 4], 4, adopt=reader)
    reservation = extended.reservation
    choices = {(parent, end, done): selected for parent, end, done, selected in reservation.decisions}
    # A emits and stops before B consumes its already-ingested chunk. Re-running
    # readiness against four items would choose a different, two-item stride.
    parent = choices[-1, 4, True]
    assert reservation.transitions[parent].chunk == first.info
    assert reservation.transitions[parent].consumed == 1
    assert reservation.transitions[parent].policy.history == [3]
    assert not reservation.transitions[parent].chunk.is_final
    oracle.put_positions([3])
    oracle.signal_done(4)
    count = 1
    while oracle.has_chunk_ready():
        chunk = oracle.pop_chunk()
        parent = choices[parent, 4, True]
        assert reservation.transitions[parent].chunk == chunk.info
        count += 1
    source.put_positions([3])
    source.signal_done(4)
    source.settle_window(reservation, count, final_transition=parent)
    assert source.policy.history == oracle.policy.history
    assert source._executed_chunks == oracle._executed_chunks == count
    assert source._buffer == oracle._buffer and source._consumed == oracle._consumed
    assert not source.owns_window(reader) and not source.owns_window(reservation)


def test_future_window_cannot_adopt_a_reader_after_another_reservation_or_semantic_progress():
    source = StreamBuffer(RID, EDGE, "LLM", FixedChunkPolicy(1))
    source.put_positions([0, 1, 2, 3])
    first = source.reserve_window(2).reservation
    second = source.reserve_window(2).reservation
    assert source.reserve_future_window([4], 2, adopt=first) is None
    assert source.reserve_future_window([4], 2, adopt=second) is None
    source.settle_window(first, 1)
    assert source.reserve_future_window([4], 2, adopt=second) is None
    assert source.owns_window(second) and source._executed_chunks == 1


def test_extending_an_ingested_ring_edge_preserves_its_graph_reference_and_fallback(monkeypatch):
    from mstar.engine.resources.loopback_rows import LoopbackRows

    streams, tm, source = _window_consumer()
    ring = LoopbackRows(torch.device("cpu"), capacity=1)
    ring.ask_ring(EDGE, 8)
    ring.ring_table(EDGE, torch.zeros(1, 1, dtype=torch.long))
    ring.add(RID)
    ring.ring_write(EDGE, [RID], torch.tensor([[[10], [11]]]), ring.ring_reserve(EDGE, [RID], 2))
    source.put_positions([0, 1])
    streams.rings[EDGE] = ring
    original = streams.reserve_edge(source, EDGE, RID, 1)
    streams.mark_ingested(RID, original)
    assert ring.ring_reserve(EDGE, [RID], 2) == [2]
    def no_future_read(*args):
        pytest.fail("an unexecuted producer's ring must not be materialized on the host")
    monkeypatch.setattr(ring, "ring_read", no_future_read)
    extended, snapshot = streams.reserve_connected_edge(RID, EDGE, [2, 3], 3, ingested=original)
    assert extended.uuids is original.uuids
    assert extended.chunk._replace(reservation=original.chunk.reservation) == original.chunk
    assert source.ingested_chunk is extended.chunk
    assert snapshot.data == {"data": [0, 1, 2, 3]}
    restored = streams.restore_edge(RID, extended)
    assert restored == original and source.ingested_chunk is restored.chunk
    assert source._executed_chunks == 0 and ring.ring_room(EDGE, RID) == 4
    # Original ordinary fallback consumes its existing graph reference once.
    assert _consume(streams, tm, restored)[0].flatten().tolist() == [10]
    streams.settle_window(RID, restored, 1)
    assert not any(tm.tensor_store.check_uuid_presence(uuid) for uuid in original.uuids)
    assert source._consumed == 1 and ring.ring_room(EDGE, RID) == 5


def test_a_waiting_connected_consumer_reserves_future_positions_without_fabricating_an_input(monkeypatch):
    from mstar.engine.resources.loopback_rows import LoopbackRows

    streams, tm, source = _window_consumer()
    ring = LoopbackRows(torch.device("cpu"), capacity=1)
    ring.ask_ring(EDGE, 8)
    ring.ring_table(EDGE, torch.zeros(1, 1, dtype=torch.long))
    ring.add(RID)
    streams.rings[EDGE] = ring
    assert ring.ring_reserve(EDGE, [RID], 2) == [0]
    monkeypatch.setattr(ring, "ring_read", lambda *a: pytest.fail("unproduced input was materialized"))
    receipt, snapshot = streams.reserve_connected_edge(RID, EDGE, [0, 1], 2)
    assert receipt.uuids == [] and source.ingested_chunk is None
    assert snapshot.data == {"data": [0, 1]} and not source.has_chunk_ready()
    assert streams.restore_edge(RID, receipt) is None
    assert not source._reservations and source._consumed == source._executed_chunks == 0
    receipt, snapshot = streams.reserve_connected_edge(RID, EDGE, [0, 1], 2)
    ring.ring_write(EDGE, [RID], torch.tensor([[[10], [11]]]), [0])
    streams.push_positions(RID, EDGE, [0, 1], None)
    choices = {(parent, end, done): index for parent, end, done, index in snapshot.reservation.decisions}
    first = choices[-1, 1, False]
    second = choices[first, 2, False]
    streams.settle_window(RID, receipt, 2, final_transition=second)
    assert source._consumed == source._executed_chunks == 2
    assert ring.ring_room(EDGE, RID) == 8 and not source._reservations


def test_a_full_producer_reservation_can_still_drain_consumers_and_preview_is_bounded():
    source = StreamBuffer(RID, EDGE, "LLM", FixedChunkPolicy(2))
    source.put_positions([0, 1, 2, 3])
    assert source.reserve_future_window(range(100_000), 100_000) is None
    assert source._consumed == source._executed_chunks == 0 and not source._reservations
    # No free producer positions: B can still consume existing input. Budget
    # exhaustion does not manufacture EOF or consume an extra empty chunk.
    reservation = source.reserve_future_window([], 4, may_finish=False).reservation
    choices = {(parent, end, done): selected for parent, end, done, selected in reservation.decisions}
    first = choices[-1, 4, False]
    second = choices[first, 4, False]
    assert choices[second, 4, False] == -1
    result = source.settle_window(reservation, 2, final_transition=second)
    assert result.released == [0, 1, 2, 3] and not result.is_final
    assert not source.producer_done and not source.has_chunk_ready()


@pytest.mark.parametrize("ingested", [False, True, "ordinary"])
def test_worker_reserves_and_restores_a_stream_unit_without_consuming_input(ingested, monkeypatch):
    from mstar.engine.engine import Engine, ExecutingBatch, ExecutionGroup
    from mstar.engine.resources.loopback_rows import LoopbackRows
    from mstar.engine.resources.step import StepContext
    from mstar.model.submodule_base import InputMetadata
    from mstar.worker.micro_scheduler import ScheduledBatch
    from mstar.worker.worker import Worker

    streams, tm, source = _window_consumer()
    ring = LoopbackRows(torch.device("cpu"), capacity=1)
    ring.ask_ring(EDGE, 8)
    ring.ring_table(EDGE, torch.zeros(1, 1, dtype=torch.long))
    ring.add(RID)
    streams.rings[EDGE] = ring
    original = None
    if ingested:
        ring.ring_write(EDGE, [RID], torch.tensor([[[10], [11]]]), ring.ring_reserve(EDGE, [RID], 2))
        source.put_positions([0, 1])
        original = (streams._chunk_edge(source.pop_chunk(), EDGE, NODE, RID) if ingested == "ordinary"
                    else streams.reserve_edge(source, EDGE, RID, 1))
        streams.mark_ingested(RID, original)
    monkeypatch.setattr(ring, "ring_read", lambda *a: pytest.fail("unexecuted ring was read"))
    engine = Engine.__new__(Engine)
    engine._submodules = {
        "producer": SimpleNamespace(rings={EDGE: {"source"}}, loopback_rows=ring),
        NODE: SimpleNamespace(rings={}),
    }
    members = tuple((ScheduledBatch(node, walk, {RID: 1}), ExecutingBatch(
        node, {}, StepContext([RID], walk, 0, False, repeats=bound),
    )) for node, walk, bound in (("producer", "source", 4), (NODE, "sink", 3)))
    a, b = (member for _, member in members)
    if original is not None:
        b.per_request_input_metadata = {RID: InputMetadata(stream_chunks={EDGE: original.chunk})}
        if ingested != "ordinary":
            b.stream_windows = {RID: [original]}
    group = ExecutionGroup(SimpleNamespace(
        nodes=("producer", NODE), streaming_edges=(("producer", NODE, EDGE),),
    ), 1, members)
    worker = Worker.__new__(Worker)
    worker.engine_manager = SimpleNamespace(get_engine=lambda name: engine)
    worker.streams = streams
    before = ring.ring_room(EDGE, RID)
    if ingested == "ordinary":
        assert not worker._reserve_group_stream(group)
        assert not source._reservations and not a.ring_bases
        assert source._consumed == 1 and ring.ring_room(EDGE, RID) == before
        assert _consume(streams, tm, original)[0].flatten().tolist() == [10]
        return
    assert worker._reserve_group_stream(group)
    reader = b.stream_windows[RID][0]
    assert reader.chunk.reservation.decisions
    assert reader.uuids == ([] if original is None else original.uuids)
    assert source._consumed == source._executed_chunks == 0
    assert ring.ring_room(EDGE, RID) == before - 4
    worker._restore_group_stream(group)
    assert not a.ring_bases
    assert b.stream_windows == ({} if original is None else {RID: [original]})
    assert source._consumed == source._executed_chunks == 0
    if original is not None:
        assert source.ingested_chunk.reservation is original.chunk.reservation
        assert all(tm.tensor_store.check_uuid_presence(uuid) for uuid in original.uuids)
        assert _consume(streams, tm, original)[0].flatten().tolist() == [10]
        streams.settle_window(RID, original, 1)
        monkeypatch.undo()  # an ordinary settled chunk may now read its real payload
        second = streams.reserve_edge(source, EDGE, RID, 1)
        streams.mark_ingested(RID, second)
        assert _consume(streams, tm, second)[0].flatten().tolist() == [11]
        streams.settle_window(RID, second, 1)
    assert ring.ring_room(EDGE, RID) == 8
