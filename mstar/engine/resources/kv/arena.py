"""A KV pool's page ownership: who holds each page, and when it goes back to the
free list.

Free of the cache and its kernels, so the conductor's routing mirror
(``kv/routing.py``) keeps each replica's pages by the worker's own rules.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mstar.engine.resources.kv.cache import KVCache, PageAllocator
    from mstar.engine.resources.kv.prefix_index import PrefixIndex


@dataclass
class PageArena:
    """physical storage, free list, and per-page ownership

    A page goes back to the allocator when its last owner releases it, not its
    first. A sealed page is never written again, so a second owner may read it;
    freeing clears the seal. Every caller holds the manager's `_lock`, or is the
    conductor's one thread, so the counts and seals need none of their own.
    """
    kv_cache: "KVCache | None"
    allocator: "PageAllocator"
    num_owners: list[int] = field(init=False, repr=False)
    sealed: list[bool] = field(init=False, repr=False)
    index: "PrefixIndex | None" = field(default=None, init=False, repr=False)
    trail: tuple[int, int] | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.num_owners = [0] * self.allocator.max_num_pages
        self.sealed = [False] * self.allocator.max_num_pages

    def acquire(self, n: int) -> list[int] | None:
        pages = self.allocator.try_allocate(n)
        if pages is not None:
            for page in pages:
                self.num_owners[page] = 1
            if self.trail is not None:
                self.trail = (self.trail[0] + 1, hash((self.trail, *pages)))
        return pages

    def retain(self, pages: list[int]) -> None:
        for page in pages:
            self.num_owners[page] += 1
            if self.num_owners[page] == 2 and self.index is not None:
                self.index.set_shared(page, True)

    def seal(self, pages: list[int]) -> None:
        for page in pages:
            self.sealed[page] = True

    def any_sealed(self, pages: list[int]) -> bool:
        return any(self.sealed[page] for page in pages)

    def release(self, pages: list[int]) -> None:
        freed = []
        for page in pages:
            assert self.num_owners[page] > 0, f"page {page} released with no owner"
            self.num_owners[page] -= 1
            if self.num_owners[page] == 0:
                self.sealed[page] = False
                freed.append(page)
            elif self.num_owners[page] == 1 and self.index is not None:
                self.index.set_shared(page, False)
        self.allocator.free(freed)

    def copy_pages(self, src: list[int], dst: list[int]) -> None:
        self.kv_cache.copy_pages(src, dst)

    @property
    def num_free(self):
        return self.allocator.num_free
