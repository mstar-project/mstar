"""The prefix index: which cached page holds which key, and which to drop first.

Inserting makes the index a page's second owner, so evicting releases that one
reference and the page reaches the free list only once no live request holds it
too. Leaves are ordered by a clock that ticks on every insert and every matching
lookup. A hit re-stamps a page without pushing it again, so a leaf keeps one heap
entry however often it is read, and an entry that surfaces under an older stamp
is pushed back under the page's own.

Every caller holds the manager's lock, so none of this takes one of its own.
It needs of its arena only the page owner counts and ``seal``, ``retain`` and
``release``, so the conductor's routing mirror runs it over a counting arena
(``kv/routing.py``).
"""

import heapq
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mstar.engine.resources.kv.manager import PageArena


class PrefixIndex:
    def __init__(self, arena: "PageArena") -> None:
        num_pages = len(arena.num_owners)
        self._arena = arena
        self._by_key: dict[bytes, int] = {}
        self._key: list[bytes | None] = [None] * num_pages
        self._parent: list[int | None] = [None] * num_pages
        self._children: list[int] = [0] * num_pages
        self._stamp: list[int] = [0] * num_pages
        self._leaves: list[tuple[int, int]] = []
        self._clock = 0

    def lookup(self, keys: Sequence[bytes]) -> list[int]:
        """Walk ``keys`` from the root and stop at the first one not indexed."""
        pages = self.pages_for(keys)
        self.touch(pages)
        return pages

    def touch(self, pages: Sequence[int]) -> None:
        pages = [page for page in pages if self._key[page] is not None]
        if pages:
            # one stamp for the whole run, so recency stays monotone along it
            self._clock += 1
            for page in pages:
                self._stamp[page] = self._clock

    def insert(self, key: bytes, page: int, parent: int | None = None) -> bool:
        """Name ``page`` by ``key`` under ``parent``; the first writer wins."""
        assert parent is None or self._key[parent] is not None, (
            f"page {page} indexed under parent page {parent}, which the index "
            "does not hold; a parent must outlive its children"
        )
        if key in self._by_key:
            return False
        assert self._key[page] is None, (
            f"page {page} indexed under {key.hex()[:16]} while it is still "
            f"indexed under {self._key[page].hex()[:16]}; evicting one entry "
            "would free a page the other still names"
        )
        self._clock += 1
        self._by_key[key] = page
        self._key[page] = key
        self._parent[page] = parent
        self._children[page] = 0
        self._stamp[page] = self._clock
        self._arena.seal([page])
        self._arena.retain([page])
        if parent is not None:
            self._children[parent] += 1
        heapq.heappush(self._leaves, (self._clock, page))
        return True

    def insert_run(self, keys: Sequence[bytes], pages: Sequence[int], parent: int | None) -> int:
        """Name ``pages`` by ``keys``, each under the one before and the first
        under ``parent``. A key already indexed keeps its page, and the next key
        goes under that page. Returns how many pages were inserted."""
        inserted = 0
        for key, page in zip(keys, pages, strict=True):
            if self.insert(key, page, parent):
                inserted += 1
            else:
                page = self._by_key[key]
            parent = page
        return inserted

    def page_for(self, key: bytes) -> int | None:
        # not `lookup`: this is not a hit, and must not re-stamp the page
        return self._by_key.get(key)

    def pages_for(self, keys: Sequence[bytes]) -> list[int]:
        """`lookup` without the stamp, for a walk that is not a hit."""
        pages: list[int] = []
        for key in keys:
            page = self._by_key.get(key)
            if page is None:
                break
            pages.append(page)
        return pages

    def holds(self, page: int) -> bool:
        return self._key[page] is not None

    def pages(self) -> list[int]:
        """Every page the index is holding a reference to."""
        return list(self._by_key.values())

    def num_pages(self) -> int:
        """How many pages the index holds a reference to (``len(pages())``)."""
        return len(self._by_key)

    def num_sole_owned(self) -> int:
        """Indexed pages nothing else holds: an upper bound on what ``evict`` can free."""
        return sum(
            1 for page in self._by_key.values() if self._arena.num_owners[page] == 1
        )

    def evict(self, n: int) -> list[bytes]:
        """Drop the oldest leaves the index alone holds, until ``n`` are free.

        A leaf a request or a lease also holds is passed over and put back:
        dropping it would free nothing and lose an entry that is still good.
        Returns the keys of the pages that reached the free list.
        """
        dropped = []
        passed_over = []
        while len(dropped) < n:
            entry = self._pop_leaf()
            if entry is None:
                break
            stamp, page = entry
            if self._arena.num_owners[page] > 1:
                passed_over.append(entry)
                continue
            dropped.append(self._key[page])
            self._remove(page)
        for entry in passed_over:
            heapq.heappush(self._leaves, entry)
        return dropped

    def drop(self, pages: Sequence[int]) -> None:
        """Remove these pages' entries, last first, each only while it is a
        leaf the index alone holds: what the routing mirror predicted a request
        would index, taken back once the request ended without running."""
        for page in reversed(pages):
            if self._key[page] is not None and not self._children[page] and self._arena.num_owners[page] == 1:
                self._remove(page)

    def _pop_leaf(self) -> tuple[int, int] | None:
        while self._leaves:
            stamp, page = heapq.heappop(self._leaves)
            if self._key[page] is None or self._children[page]:
                continue
            if self._stamp[page] != stamp:
                heapq.heappush(self._leaves, (self._stamp[page], page))
                continue
            return stamp, page
        return None

    def _remove(self, page: int) -> None:
        del self._by_key[self._key[page]]
        self._key[page] = None
        parent = self._parent[page]
        self._parent[page] = None
        self._arena.release([page])
        if parent is not None:
            self._children[parent] -= 1
            if self._children[parent] == 0 and self._key[parent] is not None:
                heapq.heappush(self._leaves, (self._stamp[parent], parent))
