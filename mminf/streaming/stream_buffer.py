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
    _consumed: int = 0
    _chunks_popped: int = 0
    producer_done: bool = False

    def put(self, item: torch.Tensor) -> None:
        """Called when a tensor arrives via normal RDMA routing."""
        self._buffer.append(item)

    def signal_done(self) -> None:
        """Producer signals no more items will arrive."""
        self.producer_done = True

    def has_chunk_ready(self) -> bool:
        buf_len = len(self._buffer)
        if self.producer_done and buf_len > 0:
            return True
        return self.policy.is_ready(buf_len, self._consumed)

    def pop_chunk(self) -> StreamChunk:
        """Pop the next chunk. Only call when has_chunk_ready() is True.

        For sliding-window: returns `window_size` items, advances by
        `stride` items, discards items that have fallen out of the window.
        start_offset is the global position of the first item in the chunk.
        """
        buf_len = len(self._buffer)
        window = self.policy.window_size()
        offset = self._consumed  # global position of buffer[0]

        if self.producer_done and not self.policy.is_ready(buf_len, self._consumed):
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

        is_final = self.producer_done and len(self._buffer) == 0

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
