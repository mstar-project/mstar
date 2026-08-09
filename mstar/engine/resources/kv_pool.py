"""KV cache storage resources.

Storage divides in two. The arena is the physical pool: the backing tensor
and the page allocator over it. The pool is per-request accounting against
an arena: which pages a stream holds, its stored length, its position
counter. Several pools may share one arena; nothing above a pool sees the
arena.
"""

import torch

from mstar.engine.kv_store import PageAllocator, PagedAllocationManager
from mstar.engine.resources.base import Reservation, Segment, SequenceView


class PageArena:
    """Physical page storage: a backing K/V tensor and the allocator over
    its pages. Pools draw pages from here and return them here."""

    def __init__(
        self,
        tensor: torch.Tensor | None,
        allocator: PageAllocator,
        page_size: int,
    ):
        self.tensor = tensor
        self.allocator = allocator
        self.page_size = page_size

    def allocate(self, n: int) -> list[int]:
        return self.allocator.allocate(n)

    def try_allocate(self, n: int) -> list[int] | None:
        return self.allocator.try_allocate(n)

    def free(self, pages: list[int]) -> None:
        self.allocator.free(pages)

    @property
    def num_free(self) -> int:
        return self.allocator.num_free

    @property
    def total_pages(self) -> int:
        return self.allocator.max_num_pages


class KVCachePool:
    """Per-request cache accounting behind the segment lifecycle.

    ``admit`` reserves capacity for a segment and reports what is already
    resident; ``view`` describes the stream the segment extends; ``commit``
    advances the stream's stored length and position counter. The
    ``PagedAllocationManager`` this pool fronts remains the storage owner
    (its lock, its request states, its transfer machinery); the pool is the
    surface planning code goes through, so callers stop reaching into
    request-state internals.
    """

    def __init__(self, manager: PagedAllocationManager):
        self._manager = manager
        self._arena = PageArena(
            tensor=manager.kv_cache,
            allocator=manager.page_allocator,
            page_size=manager.config.page_size,
        )

    @property
    def page_size(self) -> int:
        return self._arena.page_size

    @property
    def num_free_pages(self) -> int:
        return self._arena.num_free

    @property
    def total_pages(self) -> int:
        return self._arena.total_pages

    def admit(self, segment: Segment) -> Reservation:
        """Reserve pages so the segment's stream can hold its history plus
        this segment's span. Raises ``AllocationFailedError`` when the arena
        cannot supply the pages; a zero-span segment reserves nothing."""
        state = self._manager.get_state(segment.request_id, segment.label)
        resident = state.seq_len
        self._manager.alloc(
            segment.request_id, segment.label, resident + segment.span
        )
        return Reservation(
            resident=resident,
            to_compute=segment.span,
            pending=state.read_in_progress,
        )

    def view(self, segment: Segment) -> SequenceView:
        """The stream as this step's plans must see it: every page backing
        it and the extent those pages cover once the segment's span lands.
        Call after ``admit`` for spans that need new pages."""
        state = self._manager.get_state(segment.request_id, segment.label)
        return SequenceView(
            pool=self,
            page_indices=tuple(state.page_indices),
            start=0,
            length=state.seq_len + segment.span,
        )

    def commit(self, segment: Segment, pos_advance: int | None = None) -> None:
        """Record that the segment's span was computed: stored length grows
        by the span, the position counter by ``pos_advance`` (defaults to
        the span)."""
        state = self._manager.get_state(segment.request_id, segment.label)
        state.seq_len += segment.span
        state.position_id_start += (
            segment.span if pos_advance is None else pos_advance
        )

    def positions(self, request_id: str, label: str) -> int:
        """Current position counter for one stream, read-only."""
        return self._manager.get_state(request_id, label).position_id_start
