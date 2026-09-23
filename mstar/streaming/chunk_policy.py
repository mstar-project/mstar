from abc import ABC, abstractmethod
from collections.abc import Sequence


class ChunkPolicy(ABC):
    """Determines when a StreamBuffer has enough items for the consumer node."""
    def __init__(self):
        self.first_chunk_read = False
        self.items_consumed = 0

    def register_chunk(self, chunk_size: int):
        self.first_chunk_read = True
        self.items_consumed += chunk_size

    @abstractmethod
    def is_ready(self, buffer_len: int) -> bool:
        """Return True if the buffer has enough items for a chunk."""
        ...

    @abstractmethod
    def next_chunk_size(self, buffer_len: int) -> int:
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

    def continue_after_producer_done(self) -> bool:
        """Whether the buffer should keep producing (empty) chunks after the
        producer signals done and all buffered items have been consumed.

        Default ``False``: partition-done is propagated to the conductor after
        the last item is flushed.

        Set to ``True`` for connections where the consumer must keep running
        after the producer finishes (e.g., Thinker→Talker: the Talker
        continues generating codec tokens after the Thinker hits text EOS).
        In this case the buffer produces empty chunks (``_collate([])`` →
        ``{"data": None}``), and the consumer's partition-done is determined
        by its own model logic, not by the StreamBuffer.
        """
        return False


class SlidingWindowChunkPolicy(ChunkPolicy):
    """Fixed-size sliding window that advances by a stride.

    Each pop_chunk returns `window` items and advances the consumed
    pointer by `stride`. Old items before the window are discarded.

    Example (Orpheus SNAC): window=28 tokens (4 frames), stride=7 (1 frame).
    """

    def __init__(self, window: int, stride: int):
        super().__init__()
        self._window = window
        self._stride = stride

    def is_ready(self, buffer_len: int) -> bool:
        return buffer_len >= self._window

    def next_chunk_size(self, buffer_len: int) -> int:
        return self._stride

    def window_size(self) -> int:
        return self._window


class LeftContextChunkPolicy(ChunkPolicy):
    """Chunk policy for streaming vocoders with left-context overlap.

    Matches HuggingFace's ``Qwen3OmniMoeCode2Wav.chunked_decode`` pattern:

        Iter 0: codes[0 : chunk]                → emit all (no context)
        Iter 1: codes[chunk-ctx : 2*chunk]       → trim first ctx, emit rest
        Iter 2: codes[2*chunk-ctx : 3*chunk]     → trim first ctx, emit rest

    The first pop returns ``chunk`` items (no context).  Subsequent pops
    return ``chunk + left_context`` items, where the leading ``left_context``
    items OVERLAP with the tail of the previous chunk.  This overlap allows
    the causal ConvNet vocoder to "warm up" its internal state on frames
    it has already processed, ensuring a smooth transition at chunk
    boundaries.

    The key invariant: the first pop advances by ``chunk - left_context``
    (not ``chunk``), so the last ``left_context`` items of the first chunk
    remain in the buffer as overlap for the second pop.  All subsequent
    pops advance by ``chunk``.
    """

    def __init__(self, chunk: int, left_context: int):
        super().__init__()
        self._chunk = chunk
        self._left_context = left_context
        self._window = chunk + left_context

    def is_ready(self, buffer_len: int) -> bool:
        if not self.first_chunk_read:
            return buffer_len >= self._chunk
        return buffer_len >= self._window

    def next_chunk_size(self, buffer_len: int) -> int:
        # First pop: advance by (chunk - left_context) so the tail of the
        # first chunk stays in the buffer as overlap for the next pop.
        if not self.first_chunk_read:
            return self._chunk - self._left_context
        return self._chunk

    def window_size(self) -> int:
        if not self.first_chunk_read:
            return self._chunk
        return self._window


class FixedChunkPolicy(ChunkPolicy):
    """Release non-overlapping chunks of fixed size.

    Each pop_chunk returns exactly `chunk_size` items and advances by
    `chunk_size`. No overlap, no sliding window.

    Args:
        chunk_size: number of items per chunk.
        continue_after_done: if True, keep producing empty chunks after
            the producer finishes and all buffered items are consumed.
    """

    def __init__(self, chunk_size: int, continue_after_done: bool = False):
        super().__init__()
        self._chunk_size = chunk_size
        self._continue_after_done = continue_after_done

    def is_ready(self, buffer_len) -> bool:
        return buffer_len >= self._chunk_size

    def next_chunk_size(self, buffer_len: int) -> int:
        return self._chunk_size

    def window_size(self) -> int:
        return self._chunk_size

    def continue_after_producer_done(self) -> bool:
        return self._continue_after_done


class ScheduledLeftContextChunkPolicy(ChunkPolicy):
    """Left-context chunking whose chunk sizes follow a ramp.

    Streaming vocoders want the first audio out as early as possible and
    larger chunks once the stream is running. Chunk ``k`` delivers
    ``schedule[k]`` new items (``chunk`` once the schedule is exhausted) with
    up to ``left_context`` already-delivered items in front of them, so a
    causal decoder can warm up on frames it has processed before. Unlike
    ``LeftContextChunkPolicy`` the first chunk may be smaller than the
    context: the context is whatever has been delivered so far, capped.

    Example (Qwen3-TTS, 12 Hz frames): ``schedule=(4, 8, 16)``, ``chunk=25``,
    ``left_context=25`` pops windows of 4, 4+8, 12+16, 25+25, 25+25, ...
    items and the first audio leaves after four frames instead of 300.

    The consumer learns how many leading items of a window are context from
    ``StreamChunk.context_items`` (the worker passes it along as
    ``step_metadata["stream_chunks"][edge]["context_items"]``), so it can trim
    the duplicated output without re-deriving this schedule.
    """

    def __init__(self, schedule: Sequence[int], chunk: int, left_context: int):
        super().__init__()
        if chunk <= 0 or any(size <= 0 for size in schedule) or left_context < 0:
            raise ValueError("chunk sizes must be positive and left_context non-negative")
        self._schedule = tuple(int(size) for size in schedule)
        self._chunk = int(chunk)
        self._left_context = int(left_context)
        self._chunks_popped = 0
        self._delivered = 0  # new items handed to the consumer so far

    def _new_items(self) -> int:
        if self._chunks_popped < len(self._schedule):
            return self._schedule[self._chunks_popped]
        return self._chunk

    def _context(self) -> int:
        return min(self._left_context, self._delivered)

    def is_ready(self, buffer_len: int) -> bool:
        return buffer_len >= self.window_size()

    def window_size(self) -> int:
        return self._context() + self._new_items()

    def next_chunk_size(self, buffer_len: int) -> int:
        # The buffer pointer sits ``context`` items before the first new item.
        # After this pop it must sit ``next context`` items before the next
        # chunk's first new item.
        delivered_after = self._delivered + self._new_items()
        next_context = min(self._left_context, delivered_after)
        return (delivered_after - next_context) - (self._delivered - self._context())

    def register_chunk(self, chunk_size: int):
        super().register_chunk(chunk_size)
        # A regular pop delivers exactly this chunk's new items. The only
        # other caller is the terminal flush (producer done, window not
        # full), after which no data-carrying chunk follows, so treating it
        # the same keeps the bookkeeping trivially correct where it matters.
        self._delivered += self._new_items()
        self._chunks_popped += 1
