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

    _num_tensors_registered = 0
    _num_buffer_writes = 0

    def pre_read_register(self, tensor_id: str):
        self._num_tensors_registered += 1
        self._tensor_ids_in_order.append(tensor_id)

    def put(self, tensor_id: str, item: torch.Tensor) -> None:
        """Called when a tensor arrives via normal RDMA routing."""
        self._id_to_tensor[tensor_id] = item

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
        self.producer_done = True

    def _producer_done_and_all_read(self) -> bool:
        return self.producer_done and self._num_buffer_writes >= self._num_tensors_registered

    def pop_waiting_edge(self) -> GraphEdge | None:
        if len(self._waiting_graph_edges) > 0:
            return self._waiting_graph_edges.popleft()

    def has_chunk_ready(self) -> bool:
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
                and self.policy.continue_after_producer_done()):
            return True
        return self.policy.is_ready(buf_len)

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
