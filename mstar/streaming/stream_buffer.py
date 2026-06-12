from collections import deque
from dataclasses import dataclass, field

import torch

from mstar.graph.base import GraphEdge
from mstar.streaming.chunk_policy import ChunkPolicy


@dataclass
class StreamChunk:
    """A chunk of data popped from a StreamBuffer."""
    data: dict[str, torch.Tensor | None]
    chunk_index: int
    start_offset: int = 0  # global position of the first item in this chunk
    is_final: bool = False
    graph_walk_transition: str | None = None


@dataclass
class StreamingTensor:
    index: int
    tensor: torch.Tensor
    graph_walk: str | None = None

@dataclass
class WaitingEdge:
    edge: GraphEdge
    walk_transition: str | None = None


@dataclass
class StreamBuffer:
    """Per-request, per-edge buffer on the CONSUMER worker.

    Tensors arrive one-by-one via normal RDMA routing.
    The buffer accumulates them and applies a ChunkPolicy to decide
    when the consuming node has enough data to proceed.

    For sliding-window policies the buffer keeps old items so that
    pop_chunk can return the full window while only advancing by stride.
    """
    request_id: str
    edge_name: str
    from_partition: str
    policy: ChunkPolicy

    # graph edges of chunks that have been popped but not ingested
    _waiting_graph_edges: deque = field(default_factory=deque)

    # edge index -> tensor and metadata
    _tensors: dict[int, StreamingTensor] = field(default_factory=dict)
    _buffer: list[StreamingTensor] = field(default_factory=list)
    _current_index: int = 0
    _consumed: int = 0
    _chunks_popped: int = 0
    producer_done: bool = False

    _num_tensors_registered = 0
    _num_buffer_writes = 0

    def pre_read_register(self):
        """
        Register that we are reading a tensor so we don't prematurely declare
        the producer as done.
        """
        self._num_tensors_registered += 1

    def _update_buffer(self):
        while self._current_index in self._tensors:
            self._buffer.append(self._tensors.pop(self._current_index))
            self._current_index += 1

    def put(self, item: torch.Tensor, index: int, graph_walk: str | None = None) -> None:
        """Called when a tensor arrives via normal RDMA routing.

        Idempotent by index: if this index has already been buffered or has
        already been drained into ``_buffer`` (``index < _current_index``),
        the duplicate is dropped (first-arrival-wins). This handles the case
        where multiple colocated producer ranks emit the same streaming item.
        """
        # Counts put attempts, not unique items: incremented before the dedup
        # return so it stays balanced with ``_num_tensors_registered`` (which
        # ``pre_read_register`` also bumps once per registered tensor, including
        # duplicates). ``_producer_done_and_all_read`` relies on that symmetry.
        self._num_buffer_writes += 1
        if index < self._current_index or index in self._tensors:
            return
        self._tensors[index] = StreamingTensor(
            index=index,
            tensor=item,
            graph_walk=graph_walk,
        )

    def signal_done(self) -> None:
        """Producer signals no more items will arrive."""
        self.producer_done = True

    def set_index(self, index: int):
        """Seed the next index to drain (e.g. when a new consumer worker takes
        over a partition after a prefill->decode handoff).

        Only ever advances: indices are monotonic, and the conductor's tracked
        value lags (it is refreshed only at WorkerGraphsDone). Rewinding a live
        buffer would point ``_current_index`` at items already popped out of
        ``_tensors``, deadlocking the drain.
        """
        self._current_index = max(self._current_index, index)

    def _producer_done_and_all_read(self) -> bool:
        return self.producer_done and \
            self._num_buffer_writes >= self._num_tensors_registered

    def pop_waiting_edge(self) -> WaitingEdge | None:
        if len(self._waiting_graph_edges) > 0:
            return self._waiting_graph_edges.popleft()

    def has_chunk_ready(self, graph_walk: str) -> bool:
        self._update_buffer()
        buf_len = len(self._buffer)
        if self._producer_done_and_all_read() and buf_len > 0:
            return True
        # When continue_after_producer_done is set, keep producing empty
        # chunks after the producer finishes and all items are consumed.
        # This allows the consumer to keep running (e.g., Talker continues
        # generating codec tokens after the Thinker hits text EOS).
        if (self._producer_done_and_all_read()
                and buf_len == 0
                and self.policy.continue_after_producer_done(graph_walk)):
            return True
        return self.policy.is_ready(buf_len, graph_walk)

    def _chunk_boundary(
        self, current_walk: str, max_len: int
    ) -> tuple[int, str | None]:
        """Bound a chunk so a producer-triggered walk transition starts fresh.

        A producer-triggered graph-walk transition must mark a chunk boundary:
        the consumer runs one forward pass per walk, so a chunk cannot straddle
        two walks.

        Returns ``(boundary, transition)`` where:
          - ``transition`` is the walk this chunk runs under, taken from the
            *first* buffered item if it carries a transition to a walk other
            than ``current_walk`` (else ``None`` — walk unchanged).
          - ``boundary`` is the number of leading items that share this chunk's
            walk, clamped to ``max_len``. Any later item carrying a transition
            to a *different* walk forces the boundary before it, so it becomes
            the leading item of the next chunk.
        """
        if not self._buffer or max_len <= 0:
            return max(max_len, 0), None

        # (1) Leading transition: the walk this chunk runs under.
        transition = None
        chunk_walk = current_walk
        first = self._buffer[0].graph_walk
        if first is not None and first != current_walk:
            transition = first
            chunk_walk = first

        # (2) Cut before any later item that transitions to a different walk.
        boundary = min(max_len, len(self._buffer))
        for j in range(1, boundary):
            gw = self._buffer[j].graph_walk
            if gw is not None and gw != chunk_walk:
                boundary = j
                break
        return boundary, transition

    def pop_chunk(self, graph_walk: str) -> StreamChunk:
        """Pop the next chunk. Only call when has_chunk_ready() is True.

        For sliding-window: returns `window_size` items, advances by
        `stride` items, discards items that have fallen out of the window.
        start_offset is the global position of the first item in the chunk.

        A producer-triggered walk transition forces a chunk boundary (see
        ``_chunk_boundary``): the returned chunk never straddles two walks, and
        ``graph_walk_transition`` carries the walk this chunk runs under.
        """
        self._update_buffer()
        buf_len = len(self._buffer)
        offset = self._consumed  # global position of buffer[0]

        if self._producer_done_and_all_read() and not self.policy.is_ready(buf_len, graph_walk):
            # Flush remainder — return whatever is left (may be empty), still
            # cut at the first walk transition so each walk gets its own pass.
            boundary, transition = self._chunk_boundary(graph_walk, len(self._buffer))
            items = self._buffer[:boundary]
            self._buffer = self._buffer[boundary:]
            self._consumed += boundary
            stride = boundary
        else:
            stride = self.policy.next_chunk_size(buf_len, graph_walk)
            window = self.policy.window_size(graph_walk)
            # Bound the window/stride so a transition starts a fresh chunk.
            boundary, transition = self._chunk_boundary(graph_walk, window)
            stride = min(stride, boundary)
            # Return the first `boundary` items (overlapping sliding window).
            items = self._buffer[:boundary]
            # Advance by stride — discard items that fell out of the window
            self._buffer = self._buffer[stride:]
            self._consumed += stride
        self.policy.register_chunk(stride)

        is_final = self._producer_done_and_all_read() and len(self._buffer) == 0
        # When continue_after_producer_done, never mark as final — the
        # consumer decides when it's done via its own model logic.
        if self.policy.continue_after_producer_done(graph_walk):
            is_final = False

        chunk = StreamChunk(
            data=self._collate([it.tensor for it in items]),
            chunk_index=self._chunks_popped,
            start_offset=offset,
            is_final=is_final,
            graph_walk_transition=transition,
        )
        self._chunks_popped += 1
        return chunk

    def store_uningested_edge(self, edge: GraphEdge, walk_transition: str | None=None):
        self._waiting_graph_edges.append(WaitingEdge(
            edge=edge,
            walk_transition=walk_transition
        ))

    def _collate(self, items: list) -> dict[str, torch.Tensor | None]:
        if not items:
            return {"data": None}
        if isinstance(items[0], torch.Tensor):
            if len(items) == 1:
                return {"data": items[0]}
            return {"data": torch.stack(items)}
        return {"data": items}
