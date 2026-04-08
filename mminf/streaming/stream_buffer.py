from collections import deque

import torch
from dataclasses import dataclass, field

from mminf.streaming.chunk_policy import ChunkPolicy


@dataclass
class StreamChunk:
    """A chunk of data popped from a StreamBuffer."""
    data: dict[str, torch.Tensor]
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

    def producer_done_and_all_read(self) -> bool:
        return self.producer_done and self._num_buffer_writes >= self._num_tensors_registered

    def has_chunk_ready(self) -> bool:
        self._update_buffer()
        buf_len = len(self._buffer)
        if self.producer_done_and_all_read() and buf_len > 0:
            return True
        return self.policy.is_ready(buf_len, self._consumed)

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

        if self.producer_done_and_all_read() and not self.policy.is_ready(buf_len, self._consumed):
            # Flush remainder — return whatever is left
            items = list(self._buffer)
            self._buffer.clear()
            self._consumed += len(items)
        else:
            stride = self.policy.next_chunk_size(buf_len, self._consumed)
            # Return the first `window` items (overlapping sliding window)
            items = self._buffer[:window]
            # Advance by stride — discard items that fell out of the window
            self._buffer = self._buffer[stride:]
            self._consumed += stride

        is_final = self.producer_done_and_all_read() and len(self._buffer) == 0

        chunk = StreamChunk(
            data=self._collate(items),
            chunk_index=self._chunks_popped,
            start_offset=offset,
            is_final=is_final,
        )
        self._chunks_popped += 1
        return chunk

    def _collate(self, items: list) -> dict[str, torch.Tensor]:
        if not items:
            return {"data": torch.tensor([])}
        if isinstance(items[0], torch.Tensor):
            return {"data": torch.stack(items)}
        return {"data": items}
