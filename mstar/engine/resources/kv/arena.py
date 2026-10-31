"""The page pool's bookkeeping: free list, per-page ownership, and every
stream's page list.

`new_page_arena` returns the Rust arena (`rust/src/page_arena.rs`) when the
extension has one, and this module's `PythonPageArena` otherwise. Both hand
out the same pages in the same order and expose the same calls; a call that
takes ``pages`` accepts a list or a stream's `StreamPages`. The pages' storage
and copies are `KVCache`'s.

A page goes back to the allocator when its last owner releases it, not its
first. A sealed page is never written again, so a second owner may read it;
freeing clears the seal. Every caller holds the KV manager's `_lock`, so the
counts and seals need none of their own.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

from mstar.engine.resources.kv.cache import PageAllocator
from mstar.engine.resources.kv.page_table import PageTable, PlanColumns, StreamPages, columns_from_raw

try:
    from mstar_rust import PageArena as RustPageArena
except ImportError:  # an older build of the extension, or none
    RustPageArena = None


class PythonPageArena:
    def __init__(self, max_num_pages: int):
        self.allocator = PageAllocator(max_num_pages)
        self.num_owners = [0] * max_num_pages
        self.sealed = [False] * max_num_pages
        # the prefix index to tell when a page gains or loses its second owner
        self.index = None
        # queues each page a release leaves to the index alone, for a copy to the host
        self.writes = None
        self.trail: tuple[int, int] | None = None
        self._table = PageTable()

    @property
    def max_num_pages(self) -> int:
        return self.allocator.max_num_pages

    @property
    def num_free(self) -> int:
        return self.allocator.num_free

    def free_pages(self) -> list[int]:
        """The free pages, in the order they will be handed out."""
        return list(self.allocator.free_pages.queue)

    def acquire(self, n: int) -> list[int] | None:
        pages = self.allocator.try_allocate(n)
        if pages is not None:
            for page in pages:
                self.num_owners[page] = 1
            if self.trail is not None:
                self.trail = (self.trail[0] + 1, hash((self.trail, *pages)))
        return pages

    def retain(self, pages: Iterable[int]) -> None:
        for page in pages:
            self.num_owners[page] += 1
            if self.num_owners[page] == 2 and self.index is not None:
                self.index.set_shared(page, True)

    def release(self, pages: Iterable[int]) -> None:
        freed = []
        for page in pages:
            assert self.num_owners[page] > 0, f"page {page} released with no owner"
            self.num_owners[page] -= 1
            if self.num_owners[page] == 0:
                self.sealed[page] = False
                freed.append(page)
            elif self.num_owners[page] == 1 and self.index is not None:
                self.index.set_shared(page, False)
                if self.writes is not None:
                    self.writes.queue(self.index.key_of(page))
        self.allocator.free(freed)

    def seal(self, pages: Iterable[int]) -> None:
        for page in pages:
            self.sealed[page] = True

    def any_sealed(self, pages: Iterable[int]) -> bool:
        return any(self.sealed[page] for page in pages)

    def is_sealed(self, page: int) -> bool:
        return self.sealed[page]

    def owners(self, page: int) -> int:
        return self.num_owners[page]

    def owners_of(self, pages: Iterable[int]) -> list[int]:
        owners = self.num_owners
        return [owners[page] for page in pages]

    def new_stream(self) -> StreamPages:
        return self._table.new()

    def plan_columns(self, sids: list[int], lens: list[int], spans: list[int], page_size: int) -> PlanColumns:
        streams = self._table.streams
        rows = [streams[sid] for sid in sids]
        return self._table.plan_columns(
            [pages.off for pages in rows], [pages.n for pages in rows], lens, spans, page_size,
        )


# either backend
PageArena: TypeAlias = "PythonPageArena | RustPageArena"


def new_page_arena(max_num_pages: int, backend: str | None = None):
    """The Rust arena when the extension has one, else `PythonPageArena`;
    ``backend`` ("rust" or "python") picks one."""
    if backend is None:
        backend = "python" if RustPageArena is None else "rust"
    if backend == "python":
        return PythonPageArena(max_num_pages)
    if backend != "rust":
        raise ValueError(f"no page arena backend {backend!r}")
    if RustPageArena is None:
        raise RuntimeError("the Rust page arena needs the mstar_rust extension")
    return RustPageArena(max_num_pages)


def plan_columns(arena, sids: list[int], lens: list[int], spans: list[int], page_size: int) -> PlanColumns:
    """A plan label's index arrays over the page lists of the streams
    ``sids``, one per row (`PageTable.plan_columns`)."""
    if type(arena) is PythonPageArena:
        return arena.plan_columns(sids, lens, spans, page_size)
    return columns_from_raw(arena.plan_columns(sids, lens, spans, page_size), len(sids))
