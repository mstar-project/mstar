from abc import ABC, abstractmethod


class ChunkPolicy(ABC):
    """Determines when a StreamBuffer has enough items for the consumer node."""

    @abstractmethod
    def is_ready(self, buffer_len: int, items_consumed: int) -> bool:
        """Return True if the buffer has enough items for a chunk."""
        ...

    @abstractmethod
    def next_chunk_size(self, buffer_len: int, items_consumed: int) -> int:
        """Return the number of items to consume for the next chunk.

        Only called when is_ready() returns True.
        For sliding-window policies this is the stride, not the window.
        """
        ...

    @abstractmethod
    def window_size(self) -> int:
        """Return the full window of items to include in the chunk.

        For non-overlapping policies, equals next_chunk_size.
        For sliding-window policies, this is larger than the stride —
        the buffer retains older items so the chunk contains the full window.
        """
        ...


class SlidingWindowChunkPolicy(ChunkPolicy):
    """Fixed-size sliding window that advances by a stride.

    Each pop_chunk returns `window` items and advances the consumed
    pointer by `stride`. Old items before the window are discarded.

    Example (Orpheus SNAC): window=28 tokens (4 frames), stride=7 (1 frame).
    """

    def __init__(self, window: int, stride: int):
        self._window = window
        self._stride = stride

    def is_ready(self, buffer_len: int, items_consumed: int) -> bool:
        return buffer_len >= self._window

    def next_chunk_size(self, buffer_len: int, items_consumed: int) -> int:
        return self._stride

    def window_size(self) -> int:
        return self._window
