"""The prefix index: which cached page holds which key, and which to drop first.

Inserting makes the index a page's second owner — it seals the page and retains
it through the arena — so evicting releases that one reference and the page
reaches the free list only once no live request holds it too.

A page is removed only as a leaf, because a parent link is a physical page id: a
page removed from the middle of a chain could be reused under a descendant that
still names it. The child counts are what make "is this a leaf" a read rather
than a walk.

Leaves are ordered by a clock that ticks on every insert and every matching
lookup. A page that is re-stamped or gains a child leaves a stale heap entry
behind rather than being hunted down, so an entry's stamp is checked against the
page's own before it is used. Every caller holds the manager's lock, so none of
this takes one of its own.
"""

import heapq
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mstar.engine.resources.kv.manager import PageArena

_NO_PARENT = -1


class PrefixIndex:
    def __init__(self, arena: "PageArena") -> None:
        num_pages = arena.allocator.max_num_pages
        self._arena = arena
        self._by_key: dict[bytes, int] = {}
        self._key: list[bytes | None] = [None] * num_pages
        self._parent: list[int] = [_NO_PARENT] * num_pages
        self._children: list[int] = [0] * num_pages
        self._stamp: list[int] = [0] * num_pages
        self._leaves: list[tuple[int, int]] = []
        self._clock = 0

    def lookup(self, keys: Sequence[bytes]) -> list[int]:
        """Walk ``keys`` from the root and stop at the first one not indexed."""
        pages: list[int] = []
        for key in keys:
            page = self._by_key.get(key)
            if page is None:
                break
            pages.append(page)
        if pages:
            # one stamp for the whole run, so recency stays monotone along it
            self._clock += 1
            for page in pages:
                self._stamp[page] = self._clock
                if self._children[page] == 0:
                    heapq.heappush(self._leaves, (self._clock, page))
        return pages

    def insert(self, key: bytes, page: int, parent: int | None = None) -> bool:
        """Name ``page`` by ``key`` under ``parent``; the first writer wins."""
        if parent is None:
            parent = _NO_PARENT
        assert parent == _NO_PARENT or self._key[parent] is not None, (
            f"page {page} indexed under parent page {parent}, which the index "
            "does not hold; a parent must outlive its children"
        )
        if key in self._by_key:
            return False
        self._clock += 1
        self._by_key[key] = page
        self._key[page] = key
        self._parent[page] = parent
        self._children[page] = 0
        self._stamp[page] = self._clock
        self._arena.seal([page])
        self._arena.retain([page])
        if parent != _NO_PARENT:
            self._children[parent] += 1
        heapq.heappush(self._leaves, (self._clock, page))
        return True

    def page_for(self, key: bytes) -> int | None:
        """The page under ``key``, without counting it as a hit."""
        return self._by_key.get(key)

    def pages(self) -> list[int]:
        """Every page the index is holding a reference to."""
        return list(self._by_key.values())

    def evict(self, n: int) -> int:
        """Drop the oldest leaves until ``n`` pages are back on the free list.

        Returns how many pages reached the free list, which is fewer than the
        entries dropped whenever a live request still owns one of them.
        """
        freed = 0
        while freed < n:
            page = self._pop_leaf()
            if page is None:
                break
            self._remove(page)
            if self._arena.num_owners[page] == 0:
                freed += 1
        return freed

    def _pop_leaf(self) -> int | None:
        while self._leaves:
            stamp, page = heapq.heappop(self._leaves)
            if (
                self._key[page] is not None
                and self._children[page] == 0
                and self._stamp[page] == stamp
            ):
                return page
        return None

    def _remove(self, page: int) -> None:
        del self._by_key[self._key[page]]
        self._key[page] = None
        parent = self._parent[page]
        self._parent[page] = _NO_PARENT
        self._arena.release([page])
        if parent != _NO_PARENT:
            self._children[parent] -= 1
            if self._children[parent] == 0 and self._key[parent] is not None:
                heapq.heappush(self._leaves, (self._stamp[parent], parent))
