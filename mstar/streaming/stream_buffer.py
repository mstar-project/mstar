from __future__ import annotations

from collections import deque
from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import dataclass, field
from typing import NamedTuple

import torch

from mstar.streaming.chunk_policy import ChunkPolicy


@dataclass(frozen=True, eq=False)
class StreamReservation:
    """A step's bound and the predecessor span its input snapshot covers."""
    chunk_index: int
    bound: int
    predecessor_bound: int
    start_iteration: int = 0
    transitions: tuple[StreamTransition, ...] = ()
    # Future producer decisions form a tree of ordinary policy pops. The key
    # is (previous pop, absolute available end, producer done); -1 means no pop.
    # Empty columns retain the linear, already-arrived predecessor window.
    parents: tuple[int, ...] = ()
    decisions: tuple[tuple[int, int, bool, int], ...] = ()
    # An unsubmitted graph-ingested reader replaced by a future window. Its
    # first frozen pop stays authoritative; cancellation restores this reader.
    replaced: tuple[StreamReservation, int] | None = None


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
    tail_offset: int = 0

    @property
    def info(self) -> "StreamChunkInfo":
        return StreamChunkInfo(
            self.start_offset, self.context_items, self.num_items, self.is_final, self.reservation, self.tail_offset,
        )


class StreamChunkInfo(NamedTuple):
    """A chunk's place in its stream, without its data."""
    start_offset: int
    context_items: int
    num_items: int
    is_final: bool
    reservation: StreamReservation | None = None
    tail_offset: int = 0


class StreamTransition(NamedTuple):
    """One frozen ordinary policy pop and its resulting semantic progress."""
    chunk: StreamChunkInfo
    policy: ChunkPolicy
    consumed: int
    tail_consumed: int
    delivered_end: int
    final_chunk_emitted: bool


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
    # Tail position: ordinary pops assign it; retained windows advance it only
    # at settlement. Returned ordinary chunks keep their snapshotted offset.
    _tail_consumed: int = 0
    _chunks_popped: int = 0
    # Semantic iterations, distinct from the IDs assigned to retained readers.
    _executed_chunks: int = 0
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
        tail_offset = self._tail_consumed

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
        if not items and self.policy.continue_after_producer_done():
            self._tail_consumed += 1
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
            tail_offset=tail_offset,
        )
        self._delivered_end = max(self._delivered_end, offset + len(items))
        self._chunks_popped += 1
        self._executed_chunks += 1
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
        while len(chunks) < n and self.has_chunk_ready() and self.peek_walk() == walk and self._walk_run_len():
            # Legacy windows execute only their real item count. Leave an
            # empty tail for its own step so an unexecuted EOS is not spent.
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
            released=[item for c in chunks for item in c.released], tail_offset=first.tail_offset,
        )

    def _preview(self) -> StreamBuffer:
        """Copy semantic containers only; arrived items keep their identities.

        Pending arrivals are not replayed, so preview never invokes a joiner or
        touches transport/graph references. Registered/written totals still
        prevent an early producer-done flush while arrivals are outstanding.
        """
        preview = copy(self)
        preview.policy = self.policy.preview_copy()
        if preview.policy is self.policy:
            raise ValueError("stream reservations need independent policy preview state")
        preview._buffer = list(self._buffer)
        preview._walk_runs = deque(run.copy() for run in self._walk_runs)
        preview._tensor_ids_in_order = deque()
        preview._id_to_tensor = {}
        preview._waiting_graph_edges = deque()
        preview._reservations = {}
        return preview

    def _apply_transition(self, transition: StreamTransition) -> list:
        """Apply frozen consumption while preserving later producer arrivals."""
        consumed = transition.consumed - self._consumed
        if consumed < 0 or consumed > len(self._buffer):
            raise ValueError("stream transition no longer matches its retained source")
        policy = transition.policy.preview_copy()
        released, self._buffer = self._buffer[:consumed], self._buffer[consumed:]
        left = consumed
        while self._walk_runs and left:
            run = self._walk_runs[0]
            take = min(run[1], left)
            run[1] -= take
            left -= take
            if run[1] == 0:
                self._walk_runs.popleft()
        self.policy = policy
        self._consumed = transition.consumed
        self._tail_consumed = transition.tail_consumed
        self._delivered_end = transition.delivered_end
        self._final_chunk_emitted = transition.final_chunk_emitted
        self._executed_chunks += 1
        return released

    def reserve_window(self, bound: int, consumer_walk: str | None = None) -> StreamChunk | None:
        """Retain ordinary policy decisions over every possible predecessor exit.

        Only shape-compatible pops share a run. A changed window/context needs
        its ordinary next capture; no policy type is special to this backend.
        The graph separately retains every source item used by the snapshot.
        """
        if type(bound) is not int or bound < 1:
            raise ValueError("a stream run bound must be a positive integer")
        self._update_buffer()
        if self._waiting_graph_edges:
            raise ValueError("cannot reserve past an unconsumed graph edge")
        if len(self._reservations) >= 2 or any(r.decisions for r in self._reservations):
            return None
        if self._buffer and self.walk_tagged and self.peek_walk() != consumer_walk:
            return None
        predecessor = sum(self._reservations.values())
        preview = self._preview()
        transitions = []
        if self._reservations:
            earlier = next(iter(self._reservations))
            prefix = self._executed_chunks - earlier.start_iteration
            # Do not recompute this prefix against newly arrived items: sizing
            # may depend on buffer_len even when its captured shape is stable.
            transitions.extend(earlier.transitions[prefix:prefix + predecessor])
            for transition in transitions:
                preview._apply_transition(transition)
        geometry = next(((t.chunk.num_items, t.chunk.context_items) for t in transitions if t.chunk.num_items), None)
        while len(transitions) < predecessor + bound and preview.has_chunk_ready(consumer_walk):
            if preview._buffer and preview.walk_tagged and preview.peek_walk() != consumer_walk:
                break
            chunk = preview.pop_chunk()
            shape = (chunk.num_items, chunk.context_items)
            if chunk.num_items and geometry is not None and shape != geometry:
                break
            if chunk.num_items:
                geometry = shape
            transitions.append(StreamTransition(
                chunk.info, preview.policy.preview_copy(), preview._consumed, preview._tail_consumed,
                preview._delivered_end, preview._final_chunk_emitted,
            ))
            if chunk.is_final:
                break
        bound = min(bound, len(transitions) - predecessor)
        if bound <= 0:
            return None
        reservation = StreamReservation(
            self._chunks_popped, bound, predecessor, self._executed_chunks, tuple(transitions),
        )
        self._chunks_popped += 1
        self._reservations[reservation] = bound
        end = max(t.chunk.start_offset + t.chunk.num_items for t in transitions)
        items = self._buffer[:end - self._consumed]
        return StreamChunk(
            data=self._collate(items), chunk_index=reservation.chunk_index,
            start_offset=self._consumed, context_items=transitions[0].chunk.context_items,
            num_items=len(items), reservation=reservation, tail_offset=self._tail_consumed,
        )

    def reserve_future_window(
        self, future_positions: Sequence[int], bound: int, consumer_walk: str | None = None,
        *, may_finish: bool = True, adopt: StreamReservation | None = None,
    ) -> StreamChunk | None:
        """Freeze ordinary policy pops for an A-then-B connected sweep.

        Each producer iteration emits one item, or ends with/without that item.
        Its owner reserves these positions for the whole launch; the buffer
        acquires only actual emitted items at ordinary output publication.
        Once the producer stops or exhausts its reservation, B may drain.
        Exhausting the reservation never signals producer done.

        ``adopt`` requires the caller's exclusive unsubmitted graph claim. It
        replaces one retained reader without changing its first frozen pop;
        cancel_window restores that reader for ordinary fallback.
        """
        if type(bound) is not int or bound < 1:
            raise ValueError("a stream run bound must be a positive integer")
        self._update_buffer()
        if adopt is not None and (
            list(self._reservations) != [adopt] or adopt.decisions or adopt.predecessor_bound
            or adopt.start_iteration != self._executed_chunks or not adopt.transitions
            or self._reservations[adopt] <= 0
        ):
            return None
        if ((self._reservations and adopt is None) or self._waiting_graph_edges or self._tensor_ids_in_order
                or self._num_buffer_writes < self._num_tensors_registered
                or (self.walk_tagged and self._walk_run_len() != len(self._buffer))
                or (self._buffer and self.walk_tagged and self.peek_walk() != consumer_walk)):
            return None
        if self.producer_done and (future_positions or not self._producer_done_and_all_read()):
            return None
        # Bound compilation work independently of the device capture budget.
        # This bounds preview calls and table bytes, not a policy's private
        # storage inside its existing preview_copy implementation.
        max_previews = 4096
        producers = len(future_positions)
        # Each producer has two terminal drain branches and one continuation;
        # include their independent snapshots as well as every possible pop.
        preview_work = 3 * producers + 1 + (2 * producers + 1) * bound
        if preview_work > max_previews:
            return None
        transitions: list[StreamTransition] = []
        parents: list[int] = []
        decisions: dict[tuple[int, int, bool], int] = {}
        origin_end = self._consumed + len(self._buffer)

        def pop(preview: StreamBuffer, parent: int, produced: int, done: bool) -> int | None:
            key = parent, origin_end + produced, done
            if key in decisions:
                selected = decisions[key]
                if selected >= 0:
                    preview._apply_transition(transitions[selected])
                return None if selected < 0 else selected
            if parent == -1 and adopt is not None:
                # The graph already ingested this first chunk. New producer
                # items/EOF cannot change its frozen size, history or finality.
                transition = adopt.transitions[0]
                preview._apply_transition(transition)
            elif not preview.has_chunk_ready(consumer_walk):
                decisions[key] = -1
                return None
            else:
                chunk = preview.pop_chunk()
                transition = StreamTransition(
                    chunk.info, preview.policy.preview_copy(), preview._consumed, preview._tail_consumed,
                    preview._delivered_end, preview._final_chunk_emitted,
                )
            selected = len(transitions)
            transitions.append(transition)
            parents.append(parent)
            decisions[key] = selected
            return selected

        def drain(preview: StreamBuffer, parent: int, produced: int, count: int, done: bool) -> None:
            if done:
                preview.signal_done(preview._num_tensors_registered)
            while count < bound:
                selected = pop(preview, parent, produced, done)
                if selected is None:
                    break
                parent, count = selected, count + 1
                if transitions[selected].chunk.is_final:
                    break

        primary = self._preview()
        parent, count = -1, 0
        if self._producer_done_and_all_read():
            drain(primary, parent, 0, count, True)
        else:
            for produced, position in enumerate(future_positions):
                if may_finish:
                    # A stop withholds this iteration's item, or emits it.
                    drain(primary._preview(), parent, produced, count, True)
                    final = primary._preview()
                    final.put_positions([position], consumer_walk)
                    drain(final, parent, produced + 1, count, True)
                primary.put_positions([position], consumer_walk)
                if count < bound:
                    selected = pop(primary, parent, produced + 1, False)
                    if selected is not None:
                        parent, count = selected, count + 1
            drain(primary, parent, len(future_positions), count, False)
        reservation = StreamReservation(
            self._chunks_popped if adopt is None else adopt.chunk_index,
            bound, 0, self._executed_chunks, tuple(transitions), tuple(parents),
            tuple((*key, selected) for key, selected in decisions.items()),
            None if adopt is None else (adopt, self._reservations[adopt]),
        )
        if adopt is None:
            self._chunks_popped += 1
        else:
            del self._reservations[adopt]
        self._reservations[reservation] = bound
        items = [*self._buffer, *future_positions]
        return StreamChunk(
            data=self._collate(items), chunk_index=reservation.chunk_index,
            start_offset=self._consumed, num_items=len(items), reservation=reservation,
            tail_offset=self._tail_consumed,
        )

    def owns_window(self, reservation: StreamReservation) -> bool:
        """A recycled request's new buffer cannot settle an old reader."""
        return reservation in self._reservations

    def window_bound(self, reservation: StreamReservation) -> int:
        """The current admitted capacity, including when a skipped reader retries."""
        if reservation not in self._reservations:
            raise ValueError("a stream window no longer belongs to this buffer")
        return self._reservations[reservation]

    def retained_prefix(self, chunk: StreamChunkInfo) -> int:
        """Executed real or tail inputs still present in this retained snapshot."""
        reservation = chunk.reservation
        if reservation is None or reservation not in self._reservations:
            raise ValueError("a retained stream snapshot needs its live reservation")
        prefix = self._executed_chunks - reservation.start_iteration
        if not 0 <= prefix <= reservation.predecessor_bound:
            raise ValueError("stream progress is outside its retained predecessor span")
        return prefix

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
        if reservation.replaced is not None:
            original, bound = reservation.replaced
            self._reservations[original] = bound
        self.maybe_ready = True

    def settle_window(
        self, reservation: StreamReservation, executed: int, *, final_transition: int | None = None,
    ) -> StreamChunk:
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
        prefix = self._executed_chunks - reservation.start_iteration
        if reservation.decisions:
            if type(final_transition) is not int or not -1 <= final_transition < len(reservation.transitions):
                raise ValueError("a future stream reservation needs its selected policy transition")
            chain = []
            index = final_transition
            while index >= 0:
                chain.append(index)
                index = reservation.parents[index]
            chain.reverse()
            selected = tuple(reservation.transitions[index] for index in chain)
            selected_ids = set(chain)
            available_end = self._consumed + len(self._buffer)
            for _, end, done, target in reservation.decisions:
                if target in selected_ids and (
                    end > available_end or (done and not self._producer_done_and_all_read())
                ):
                    raise ValueError("selected stream transition precedes its actual arrivals or producer completion")
        else:
            if final_transition is not None:
                raise ValueError("an arrived stream window selects by actual iteration count")
            selected = reservation.transitions[prefix:prefix + executed]
        if prefix < 0 or prefix > reservation.predecessor_bound or len(selected) != executed:
            raise ValueError("stream execution is outside its frozen policy transitions")
        origin, items = self._consumed, []
        for transition in selected:
            items.extend(self._apply_transition(transition))
        final = bool(selected and selected[-1].chunk.is_final)
        del self._reservations[reservation]
        self.maybe_ready = True
        return StreamChunk(
            data=self._collate(items), chunk_index=reservation.chunk_index, start_offset=origin,
            num_items=len(items), is_final=final, released=items,
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
