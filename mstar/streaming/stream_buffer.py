from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import torch

from mstar.streaming.chunk_policy import ChunkPolicy, FixedChunkPolicy


@dataclass(frozen=True, eq=False)
class StreamReservation:
    """A step's bound and the predecessor span its input snapshot covers."""
    chunk_index: int
    bound: int
    predecessor_bound: int
    # Empty tail inputs already executed when this snapshot began.
    tail_offset: int = 0


@dataclass
class StreamChunk:
    """A chunk of data popped from a StreamBuffer."""
    data: dict[str, torch.Tensor | None]
    chunk_index: int
    start_offset: int = 0  # global position of the first item in this chunk
    is_final: bool = False
    # leading items of this chunk that an earlier chunk already delivered
    # (sliding-window overlap / left context); the consumer trims their output
    context_items: int = 0
    # items in this chunk (context included); lets a consumer pick its
    # capture bucket before the chunk tensor is unpacked
    num_items: int = 0
    # items this pop took out of the buffer for good (the stride it advanced
    # by, or everything on a flush); the buffer no longer holds them
    released: list = field(default_factory=list)
    reservation: StreamReservation | None = None

    @property
    def info(self) -> "StreamChunkInfo":
        return StreamChunkInfo(
            self.start_offset, self.context_items, self.num_items, self.is_final, self.reservation,
        )


class StreamChunkInfo(NamedTuple):
    """A chunk's place in its stream, without its data."""
    start_offset: int
    context_items: int
    num_items: int
    is_final: bool
    reservation: StreamReservation | None = None


class StreamingEdge(NamedTuple):
    """A popped chunk as a synthetic input edge into its consumer node: the
    uuid of each of its items, and its place in the stream."""
    name: str
    next_node: str
    uuids: list[int]
    chunk: StreamChunkInfo


def collate(items: list[torch.Tensor]) -> torch.Tensor:
    """A chunk's items as its consumer takes them: one item as itself,
    several stacked. A consumer must not write into it: a single item is the
    stored tensor, which other destinations may read too."""
    return items[0] if len(items) == 1 else torch.stack(items)


@dataclass
class StreamBuffer:
    """Per-request, per-edge buffer on the CONSUMER worker.

    Items arrive one-by-one via normal routing: tensors, or handles to them
    (the worker puts each item's uuid and collates the tensors when the
    consuming batch is built). The buffer accumulates them
    and applies a ChunkPolicy to decide when the consuming node has enough
    data to proceed.

    For sliding-window policies the buffer keeps old items so that
    pop_chunk can return the full window while only advancing by stride.
    """
    request_id: int
    edge_name: str
    from_partition: str
    policy: ChunkPolicy
    # dim to join a chunked producer's pieces on; None keeps them separate
    join_dim: int | None = None
    # joins pieces held by handle into one item; None means they are tensors
    joiner: Callable[[list], object] | None = None
    # Items carry the consumer walk the producer assigned them, and a chunk
    # never spans two walks (set for connections that drive their consumer's
    # walk)
    walk_tagged: bool = False

    _waiting_graph_edges: deque[StreamingEdge] = field(default_factory=deque)
    # last chunk the graph accepted; read by the batch that consumes it
    ingested_chunk: StreamChunkInfo | None = None

    _buffer: list = field(default_factory=list)
    # _buffer's assigned walks as [walk, item count] runs, when walk_tagged
    _walk_runs: deque[list] = field(default_factory=deque)
    _tensor_ids_in_order: deque = field(default_factory=deque)
    _id_to_tensor: dict = field(default_factory=dict)
    _consumed: int = 0
    # Actual empty-tail reads, distinct from producer items acknowledged.
    _tail_consumed: int = 0
    _chunks_popped: int = 0
    # Submitted readers may overlap; only settlement advances the source.
    _reservations: dict[StreamReservation, int] = field(default_factory=dict)
    # global position just past the last item ever handed out; everything
    # before it in a later window is context, not new data
    _delivered_end: int = 0
    producer_done: bool = False
    # Set once a chunk has been popped with ``is_final=True`` (the terminal
    # flush). Guards the empty-buffer final flush below so it fires exactly
    # once and ``has_chunk_ready`` doesn't spin returning True forever.
    _final_chunk_emitted: bool = False

    # Whether a chunk may be ready: set by everything that can make one ready
    # (an item arriving, the producer's done signal, a refused edge coming
    # back, a pop that may leave another chunk), cleared by a poll that found
    # nothing. Lets the poll skip a buffer without asking its policy.
    maybe_ready: bool = True

    _num_tensors_registered = 0
    _num_buffer_writes = 0
    # Items the producer sent this worker in total, as the done signal reports
    # it. A partition's end comes through the conductor and its items straight
    # from the producer, so the signal can arrive first. (A client stream's end
    # follows its items on the data worker's own FIFO.)
    _num_items_expected = 0

    def prime_context(self, context_items: int) -> None:
        """The first ``context_items`` items the producer sends are context only
        (e.g. a voice clone's reference frames): the first chunk carries them
        ahead of its new items, reported as ``context_items``."""
        self.policy.prime(context_items)
        self._delivered_end = context_items

    def pre_read_register(self, tensor_id: str, walk: str | None = None):
        self._num_tensors_registered += 1
        self._tensor_ids_in_order.append((tensor_id, walk))

    def pre_read_register_joined(
        self, tensor_ids: list[str], walk: str | None = None, pieces: bool = True,
    ) -> None:
        """Register ``tensor_ids`` as the pieces of ONE item.

        A chunked producer emits a piece per chunk; without this each would be
        an item of its own and a policy counting items would hand the consumer
        only the first. Each piece still counts towards the registered total,
        which is in the producer's units (see below). Not ``pieces``: they are
        a step's iterations (``StreamBufferManager.pieces``), an item each.
        """
        if len(tensor_ids) == 1 or self.join_dim is None or not pieces:
            for tensor_id in tensor_ids:
                self.pre_read_register(tensor_id, walk)
            return
        if not tensor_ids:
            return
        # Counted in the producer's units, one per piece, because that is what
        # it reports in ``signal_done`` -- the buffer holding them as one item
        # is its own business. Counting the group as one leaves the buffer
        # permanently short of the producer's total, so it never agrees the
        # producer is finished.
        self._num_tensors_registered += len(tensor_ids)
        self._tensor_ids_in_order.append((tuple(tensor_ids), walk))

    def put(self, tensor_id: str, item) -> None:
        """Called when an item arrives via normal routing."""
        self._id_to_tensor[tensor_id] = item
        self.maybe_ready = True

    def put_positions(self, positions: list[int], walk: str | None = None) -> None:
        """Items that stay on the device, in its producer's ring
        (``step_repeats: stream_queue``): the buffer holds their positions,
        and a chunk is read from the ring when it is popped."""
        self._update_buffer()
        self._buffer.extend(positions)
        self._num_tensors_registered += len(positions)
        self._num_buffer_writes += len(positions)
        if self.walk_tagged and positions:
            if self._walk_runs and self._walk_runs[-1][0] == walk:
                self._walk_runs[-1][1] += len(positions)
            else:
                self._walk_runs.append([walk, len(positions)])
        self.maybe_ready = True

    def _update_buffer(self):
        while len(self._tensor_ids_in_order) > 0:
            entry, walk = self._tensor_ids_in_order[0]
            # a lone tensor is every item of an unjoined stream: keep it cheap
            if not isinstance(entry, tuple):
                if entry not in self._id_to_tensor:
                    return
                self._tensor_ids_in_order.popleft()
                self._buffer.append(self._id_to_tensor.pop(entry))
                # producer's units: see pre_read_register_joined
                self._num_buffer_writes += 1
            else:
                if any(tensor_id not in self._id_to_tensor for tensor_id in entry):
                    return
                self._tensor_ids_in_order.popleft()
                self._buffer.append(self._join(
                    [self._id_to_tensor.pop(tensor_id) for tensor_id in entry]
                ))
                self._num_buffer_writes += len(entry)
            if self.walk_tagged:
                if self._walk_runs and self._walk_runs[-1][0] == walk:
                    self._walk_runs[-1][1] += 1
                else:
                    self._walk_runs.append([walk, 1])

    def _join(self, pieces: list) -> torch.Tensor:
        """One item from a chunked producer's pieces, in chunk order.

        An empty piece is real -- a prompt chunk can be all system prompt,
        which the producer drops every position of -- so it is joined, not
        skipped.

        A buffer that holds uuids joins through its ``joiner``, which reads
        the pieces, concatenates them and stores the result under a fresh uuid.

        A consumer that can take the pieces as they come should do that
        instead (``ChunkingPolicy.per_chunk_outputs``), as Qwen3-Omni's Talker
        does. This is for a consumer that genuinely wants the whole thing: it
        conditions on the complete prompt, or its shapes are captured, and it
        has no use for a prefix.
        """
        if len(pieces) == 1:
            return pieces[0]
        if self.joiner is not None:
            return self.joiner(pieces)
        return torch.cat(pieces, dim=self.join_dim)

    def signal_done(self, num_items: int = 0) -> None:
        """Producer signals no more items will arrive after the ``num_items``
        it sent in total."""
        self.producer_done = True
        self._num_items_expected = num_items
        self.maybe_ready = True

    def _producer_done_and_all_read(self) -> bool:
        return (
            self.producer_done
            and self._num_tensors_registered >= self._num_items_expected
            and self._num_buffer_writes >= self._num_tensors_registered
        )

    def pop_waiting_edge(self) -> StreamingEdge | None:
        if len(self._waiting_graph_edges) > 0:
            return self._waiting_graph_edges.popleft()

    def _walk_run_len(self) -> int:
        """Leading items assigned the same walk as the first."""
        if not self._walk_runs:
            return len(self._buffer)
        return self._walk_runs[0][1]

    def peek_walk(self) -> str | None:
        """The walk the next chunk runs under, without popping it. None when
        untagged or nothing is buffered."""
        self._update_buffer()
        return self._walk_runs[0][0] if self._walk_runs else None

    def has_chunk_ready(self, consumer_walk: str | None = None) -> bool:
        self._update_buffer()
        buf_len = len(self._buffer)

        # An item for a later walk is buffered, so the leading run is complete
        if self._walk_run_len() < buf_len:
            return True

        if not self._producer_done_and_all_read():
            return self.policy.is_ready(buf_len)

        # When continue_after_producer_done is set, keep producing empty
        # chunks after the producer finishes and all items are consumed.
        # This allows the consumer to keep running (e.g., Talker continues
        # generating codec tokens after the Thinker hits text EOS).
        # Producer done and the buffer already drained to empty (all items
        # were consumed in earlier chunks. Emit exactly one final
        # (empty) chunk so ``is_final`` propagates and the stream closes.

        if self.policy.continue_after_producer_done():
            return buf_len > 0 or self.policy.continues_in(consumer_walk)
        return buf_len > 0 or not self._final_chunk_emitted

    def pop_chunk(self) -> StreamChunk:
        """Pop the next chunk. Only call when has_chunk_ready() is True.

        For sliding-window: returns `window_size` items, advances by
        `stride` items, discards items that have fallen out of the window.
        start_offset is the global position of the first item in the chunk.
        """
        if self._reservations:
            raise ValueError("cannot pop a stream with unsettled run reservations")
        self._update_buffer()
        buf_len = len(self._buffer)
        window = self.policy.window_size()
        offset = self._consumed  # global position of buffer[0]

        run = self._walk_run_len()

        if run < buf_len and not self.policy.is_ready(run):
            # The next items run under another walk: flush this walk's run
            items = self._buffer[:run]
            released = items
            self._buffer = self._buffer[run:]
            self._consumed += run
            stride = run
        elif self._producer_done_and_all_read() and not self.policy.is_ready(buf_len):
            # Flush remainder — return whatever is left (may be empty)
            items = list(self._buffer)
            released = items
            self._buffer.clear()
            self._consumed += len(items)
            stride = len(items)
        else:
            stride = self.policy.next_chunk_size(buf_len)
            # Return the first `window` items (overlapping sliding window)
            items = self._buffer[:window]
            released = self._buffer[:stride]
            # Advance by stride — discard items that fell out of the window
            self._buffer = self._buffer[stride:]
            self._consumed += stride
        self.policy.register_chunk(stride)
        if self._walk_runs and stride:
            # A chunk never spans two walks, so the stride is within one run
            run = self._walk_runs[0]
            run[1] -= stride
            if run[1] == 0:
                self._walk_runs.popleft()

        is_final = self._producer_done_and_all_read() and len(self._buffer) == 0
        # When continue_after_producer_done, never mark as final — the
        # consumer decides when it's done via its own model logic.
        if self.policy.continue_after_producer_done():
            is_final = False
        if is_final:
            self._final_chunk_emitted = True

        chunk = StreamChunk(
            data=self._collate(items),
            chunk_index=self._chunks_popped,
            start_offset=offset,
            is_final=is_final,
            context_items=min(max(self._delivered_end - offset, 0), len(items)),
            num_items=len(items),
            released=released,
        )
        self._delivered_end = max(self._delivered_end, offset + len(items))
        self._chunks_popped += 1
        return chunk

    def pop_window(self, n: int) -> StreamChunk:
        """Up to ``n`` ready chunks as one, for a step that repeats and takes
        an item each iteration: only a stream of one-item chunks merges, and
        never across a walk change or past the final chunk. Only call when
        ``has_chunk_ready()``."""
        walk = self.peek_walk()
        first = self.pop_chunk()
        if n == 1 or first.num_items != 1 or first.is_final or self.policy.window_size() != 1:
            return first
        chunks = [first]
        while len(chunks) < n and self.has_chunk_ready() and self.peek_walk() == walk:
            chunk = self.pop_chunk()
            chunks.append(chunk)
            if chunk.is_final or chunk.num_items != 1:
                break
        if len(chunks) == 1:
            return first
        data = [chunk.data.get("data") for chunk in chunks]
        held = [d for d in data if d is not None]
        if held and isinstance(held[0], torch.Tensor):
            # a one-item chunk holds its item as itself (`collate`)
            merged = {"data": collate(held)}
        else:
            merged = {"data": [u for d in held for u in d]}
        last = chunks[-1]
        return StreamChunk(
            data=merged, chunk_index=first.chunk_index, start_offset=first.start_offset, is_final=last.is_final,
            context_items=first.context_items, num_items=sum(c.num_items for c in chunks),
            released=[item for c in chunks for item in c.released],
        )

    def reserve_window(self, bound: int, consumer_walk: str | None = None) -> StreamChunk | None:
        """Snapshot inputs without consuming them, including every possible
        predecessor exit. Only one-item, non-overlapping chunks may reserve.
        The caller gives the snapshot its own graph-reader references.
        """
        if type(bound) is not int or bound < 1:
            raise ValueError("a stream run bound must be a positive integer")
        self._update_buffer()
        ready = self._walk_run_len()
        # A ramp can begin at one item and grow after the first execution.
        # Reservation settlement must not change the policy's chunk geometry.
        if type(self.policy) is not FixedChunkPolicy or self.policy.window_size() != 1:
            raise ValueError("stream run reservations need fixed one-item chunks")
        if self._waiting_graph_edges:
            raise ValueError("cannot reserve past an unconsumed graph edge")
        if len(self._reservations) >= 2:
            return None
        predecessor = sum(self._reservations.values())
        ended = self._producer_done_and_all_read() and ready == len(self._buffer)
        tail = ended and self.policy.continues_in(consumer_walk)
        if not tail:
            bound = min(bound, ready - predecessor)
        if bound <= 0:
            return None
        if ready and self.walk_tagged and self.peek_walk() != consumer_walk:
            return None
        reservation = StreamReservation(self._chunks_popped, bound, predecessor, self._tail_consumed)
        self._chunks_popped += 1
        self._reservations[reservation] = bound
        items = self._buffer[:min(ready, predecessor + bound)]
        return StreamChunk(
            data=self._collate(items), chunk_index=reservation.chunk_index,
            start_offset=self._consumed, num_items=len(items), reservation=reservation,
        )

    def owns_window(self, reservation: StreamReservation) -> bool:
        """A recycled request's new buffer cannot settle an old reader."""
        return reservation in self._reservations

    def narrow_window(self, reservation: StreamReservation, bound: int) -> None:
        """Admission can narrow the newest reader before a successor reserves."""
        previous = self._reservations.get(reservation)
        if previous is None or type(bound) is not int or not 0 <= bound <= previous:
            raise ValueError("invalid stream reservation bound")
        if next(reversed(self._reservations)) is not reservation:
            raise ValueError("cannot narrow a stream reservation after its successor reserved")
        self._reservations[reservation] = bound

    def cancel_window(self, reservation: StreamReservation) -> None:
        """After undoing graph ingest, release an unsubmitted reader. The
        buffer still owns every item, so no input is returned or duplicated.
        """
        if not self._reservations or next(reversed(self._reservations)) is not reservation:
            raise ValueError("stream reservations must roll back from the newest reader")
        del self._reservations[reservation]
        self.maybe_ready = True

    def settle_window(self, reservation: StreamReservation, executed: int) -> StreamChunk:
        """Retire actual input consumption in submission order. ``released``
        holds the buffer's references; overlapping graph readers keep theirs.
        Empty tail iterations advance policy but acknowledge no producer data.
        """
        bound = self._reservations.get(reservation)
        if bound is None or type(executed) is not int or not 0 <= executed <= bound:
            raise ValueError("invalid stream reservation execution count")
        if next(iter(self._reservations)) is not reservation:
            raise ValueError("stream reservations must settle in submission order")
        self._update_buffer()
        consumed = min(executed, self._walk_run_len())
        items = self._buffer[:consumed]
        self._buffer = self._buffer[consumed:]
        origin = self._consumed
        self._consumed += consumed
        self._tail_consumed += executed - consumed
        for i in range(executed):
            self.policy.register_chunk(int(i < consumed))
        if self._walk_runs and consumed:
            self._walk_runs[0][1] -= consumed
            if self._walk_runs[0][1] == 0:
                self._walk_runs.popleft()
        self._delivered_end = max(self._delivered_end, self._consumed)
        final = bool(executed and self._producer_done_and_all_read() and not self._buffer
                     and not self.policy.continue_after_producer_done())
        if final:
            self._final_chunk_emitted = True
        del self._reservations[reservation]
        self.maybe_ready = True
        return StreamChunk(
            data=self._collate(items), chunk_index=reservation.chunk_index, start_offset=origin,
            num_items=consumed, is_final=final, released=items,
        )

    def store_uningested_edge(self, streaming_edge: StreamingEdge, first: bool = False):
        """Hand back a chunk the graph didn't take; ``first`` ahead of any
        other waiting one (the rest of a window a step ran short of)."""
        if first:
            self._waiting_graph_edges.appendleft(streaming_edge)
        else:
            self._waiting_graph_edges.append(streaming_edge)
        self.maybe_ready = True

    def _collate(self, items: list) -> dict[str, torch.Tensor | None]:
        if not items:
            return {"data": None}
        if isinstance(items[0], torch.Tensor):
            return {"data": collate(items)}
        return {"data": items}
