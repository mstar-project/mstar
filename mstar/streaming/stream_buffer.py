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

    _waiting_graph_edges: deque = field(default_factory=deque)

    _buffer: list = field(default_factory=list)
    _tensor_ids_in_order: deque = field(default_factory=deque)
    _id_to_tensor: dict = field(default_factory=dict)
    _consumed: int = 0
    _chunks_popped: int = 0
    producer_done: bool = False
    # Set once a chunk has been popped with ``is_final=True`` (the terminal
    # flush). Guards the empty-buffer final flush below so it fires exactly
    # once and ``has_chunk_ready`` doesn't spin returning True forever.
    _final_chunk_emitted: bool = False

    # MSTAR_CODEC_CHUNK_EMIT: producer-side staging of arrived frames, flushed
    # into the buffer in one batched put per coalesce boundary. list[(id,item)]
    # in arrival order. Empty (and unused) unless the producer routes via the
    # coalesced path.
    _coalesce_pending: list = field(default_factory=list)

    _num_tensors_registered = 0
    _num_buffer_writes = 0

    def pre_read_register(self, tensor_id: str):
        self._num_tensors_registered += 1
        self._tensor_ids_in_order.append(tensor_id)

    def put(self, tensor_id: str, item: torch.Tensor) -> None:
        """Called when a tensor arrives via normal RDMA routing."""
        self._id_to_tensor[tensor_id] = item

    def stage(self, tensor_id: str, item: torch.Tensor) -> None:
        """MSTAR_CODEC_CHUNK_EMIT: hold an arrived frame for a later batched
        put instead of writing it into the buffer immediately. Ordering /
        registration (``pre_read_register``) are unchanged; only the buffer
        write is deferred to ``flush_pending``."""
        self._coalesce_pending.append((tensor_id, item))

    def num_pending(self) -> int:
        return len(self._coalesce_pending)

    def flush_pending(self) -> int:
        """MSTAR_CODEC_CHUNK_EMIT: write all staged frames into the buffer in
        one batched put. Returns the number of frames flushed (0 if none).

        Byte-identical to having called ``put`` for each staged frame in
        arrival order: the buffered item sequence, and hence every popped
        window, is the same. The batched fast path skips the per-frame
        ``_id_to_tensor`` insert+delete churn when the staged ids are exactly
        the next-in-order registered ids; otherwise it falls back to the
        per-frame ``put`` path, which is identical by construction."""
        pending = self._coalesce_pending
        if not pending:
            return 0
        self._coalesce_pending = []
        ids = [tid for tid, _ in pending]
        # Fast path: staged ids are exactly the head of the registration order
        # and nothing earlier is still awaiting arrival. Then draining them is
        # exactly what _update_buffer would do after N puts, so append straight
        # to _buffer and skip the id->tensor dict round-trip.
        n = len(ids)
        head_matches = (
            not self._id_to_tensor
            and len(self._tensor_ids_in_order) >= n
            and all(self._tensor_ids_in_order[i] == ids[i] for i in range(n))
        )
        if head_matches:
            for _ in range(n):
                self._tensor_ids_in_order.popleft()
            for _, item in pending:
                self._buffer.append(item)
                self._num_buffer_writes += 1
        else:
            # Safe fallback: identical to arrival-order puts + lazy drain.
            for tid, item in pending:
                self._id_to_tensor[tid] = item
            self._update_buffer()
        return n

    def _update_buffer(self):
        while len(self._tensor_ids_in_order) > 0:
            tensor_id = self._tensor_ids_in_order[0]
            if tensor_id not in self._id_to_tensor:
                return
            self._tensor_ids_in_order.popleft()
            self._buffer.append(self._id_to_tensor[tensor_id])
            self._num_buffer_writes += 1
            del self._id_to_tensor[tensor_id]

    def signal_done(self) -> None:
        """Producer signals no more items will arrive."""
        # Flush any coalesced remainder (< coalesce_size frames) before marking
        # done, so the tail chunk isn't stranded in staging.
        self.flush_pending()
        self.producer_done = True

    def _producer_done_and_all_read(self) -> bool:
        return self.producer_done and self._num_buffer_writes >= self._num_tensors_registered

    def pop_waiting_edge(self) -> GraphEdge | None:
        if len(self._waiting_graph_edges) > 0:
            return self._waiting_graph_edges.popleft()

    def has_chunk_ready(self) -> bool:
        self._update_buffer()
        buf_len = len(self._buffer)

        if not self._producer_done_and_all_read():
            return self.policy.is_ready(buf_len)

        # When continue_after_producer_done is set, keep producing empty
        # chunks after the producer finishes and all items are consumed.
        # This allows the consumer to keep running (e.g., Talker continues
        # generating codec tokens after the Thinker hits text EOS).
        # Producer done and the buffer already drained to empty (all items
        # were consumed in earlier chunks. Emit exactly one final
        # (empty) chunk so ``is_final`` propagates and the stream closes.

        return (
            buf_len > 0
            or self.policy.continue_after_producer_done()
            or not self._final_chunk_emitted
        )

    def pop_chunk(self) -> StreamChunk:
        """Pop the next chunk. Only call when has_chunk_ready() is True.

        For sliding-window: returns `window_size` items, advances by
        `stride` items, discards items that have fallen out of the window.
        start_offset is the global position of the first item in the chunk.
        """
        self._update_buffer()
        buf_len = len(self._buffer)
        window = self.policy.window_size()
        offset = self._consumed  # global position of buffer[0]

        if self._producer_done_and_all_read() and not self.policy.is_ready(buf_len):
            # Flush remainder — return whatever is left (may be empty)
            items = list(self._buffer)
            self._buffer.clear()
            self._consumed += len(items)
            stride = len(items)
        else:
            stride = self.policy.next_chunk_size(buf_len)
            # Return the first `window` items (overlapping sliding window)
            items = self._buffer[:window]
            # Advance by stride — discard items that fell out of the window
            self._buffer = self._buffer[stride:]
            self._consumed += stride
        self.policy.register_chunk(stride)

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
        )
        self._chunks_popped += 1
        return chunk

    def store_uningested_edge(self, edge: GraphEdge):
        self._waiting_graph_edges.append(edge)

    def _collate(self, items: list) -> dict[str, torch.Tensor | None]:
        if not items:
            return {"data": None}
        if isinstance(items[0], torch.Tensor):
            if len(items) == 1:
                return {"data": items[0]}
            return {"data": torch.stack(items)}
        return {"data": items}
