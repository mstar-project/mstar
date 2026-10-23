"""The prefix index: which cached page holds which key, and which to drop first.

Inserting makes the index a page's second owner, so evicting releases that one
reference and the page reaches the free list only once no live request holds it
too. Leaves are ordered by a clock that ticks on every insert and every matching
lookup. A hit re-stamps a page without pushing it again, so a leaf keeps one heap
entry however often it is read, and an entry that surfaces under an older stamp
is pushed back under the page's own.

Every caller holds the manager's lock, so none of this takes one of its own.
"""

import copy
import heapq
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mstar.engine.resources.kv.manager import PageArena


class PrefixIndex:
    def __init__(self, arena: "PageArena") -> None:
        num_pages = arena.allocator.max_num_pages
        self._arena = arena
        self._by_key: dict[bytes, int] = {}
        self._key: list[bytes | None] = [None] * num_pages
        self._parent: list[int | None] = [None] * num_pages
        self._children: list[int] = [0] * num_pages
        # how many of a page's children are held, or have a held page below them
        self._held_below: list[int] = [0] * num_pages
        self._stamp: list[int] = [0] * num_pages
        self._leaves: list[tuple[int, int]] = []
        self._clock = 0
        self._evictable: set[int] = set()
        arena.index = self

    def lookup(self, keys: Sequence[bytes]) -> list[int]:
        """Walk ``keys`` from the root and stop at the first one not indexed."""
        pages = self.peek(keys)
        if pages:
            # one stamp for the whole run, so recency stays monotone along it
            self._clock += 1
            for page in pages:
                self._stamp[page] = self._clock
        return pages

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
        self._held_below[page] = 0
        self._stamp[page] = self._clock
        self._arena.seal([page])
        self._arena.retain([page])
        if parent is not None:
            self._children[parent] += 1
        heapq.heappush(self._leaves, (self._clock, page))
        return True

    def peek(self, keys: Sequence[bytes]) -> list[int]:
        """Like `lookup`, but leaves the matched pages' stamps alone."""
        pages: list[int] = []
        for key in keys:
            page = self._by_key.get(key)
            if page is None:
                break
            pages.append(page)
        return pages

    def page_for(self, key: bytes) -> int | None:
        # not `lookup`: this is not a hit, and must not re-stamp the page
        return self._by_key.get(key)

    def evictable(self) -> set[int]:
        """Return the live set of pages `evict` could free, cascading from the leaves.
        Callers read it and don't keep it.

        A page only the index holds is out of reach while a page below it is held,
        since eviction takes leaves only.
        """
        return self._evictable

    def set_shared(self, page: int, shared: bool) -> None:
        """Called when ``page`` gains or loses its one owner besides the index."""
        if self._key[page] is None or self._held_below[page]:
            # nothing changes for a page that isn't indexed or is pinned from below
            return
        if shared:
            self._evictable.discard(page)
            self._pin_above(page)
        else:
            self._evictable.add(page)
            self._unpin_above(page)

    def _pin_above(self, page: int) -> None:
        """Mark the pages above a newly held ``page`` as not evictable."""
        parent = self._parent[page]
        while parent is not None:
            self._held_below[parent] += 1
            if self._held_below[parent] > 1 or self._arena.num_owners[parent] > 1:
                return
            self._evictable.discard(parent)
            parent = self._parent[parent]

    def _unpin_above(self, page: int) -> None:
        """Mark the pages above ``page`` evictable again once nothing below them is held."""
        parent = self._parent[page]
        while parent is not None:
            self._held_below[parent] -= 1
            if self._held_below[parent] or self._arena.num_owners[parent] > 1:
                return
            self._evictable.add(parent)
            parent = self._parent[parent]

    def evictable_after(self, released: Mapping[int, int]) -> int:
        """How many pages would join the evictable set if each page in ``released`` lost the number of
        owners it maps to.
        Replays `set_shared` and `_unpin_above` without changing anything."""
        owners = self._arena.num_owners
        left = {page: owners[page] - n for page, n in released.items()}
        below: dict[int, int] = {}
        joined: set[int] = set()
        for page in released:
            if self._key[page] is None or left[page] != 1 or page in joined or below.get(page, self._held_below[page]):
                continue
            while True:
                joined.add(page)
                parent = self._parent[page]
                if parent is None:
                    break
                below[parent] = below.get(parent, self._held_below[parent]) - 1
                if below[parent] or left.get(parent, owners[parent]) > 1:
                    break
                page = parent
        return len(joined)

    def assert_evictable(self) -> None:
        """Check that `evict`, run on a copy of the index and the owner counts, frees every page
        in the evictable set."""
        index = copy.copy(self)
        for name in ("_by_key", "_key", "_parent", "_children", "_stamp", "_leaves", "_evictable"):
            setattr(index, name, copy.copy(getattr(self, name)))
        index._arena = _Owners(self._arena.num_owners)
        freed = index.evict(len(self._evictable))
        assert freed == len(self._evictable), (
            f"eviction reaches {freed} of the {len(self._evictable)} pages the evictable set counts"
        )

    def pages(self) -> list[int]:
        """Every page the index is holding a reference to."""
        return list(self._by_key.values())

    def num_pages(self) -> int:
        """How many pages the index holds a reference to (``len(pages())``)."""
        return len(self._by_key)

    def evict(self, n: int) -> int:
        """Drop the oldest leaves the index alone holds, until ``n`` are free.

        A leaf a request or a lease also holds is passed over and put back:
        dropping it would free nothing and lose an entry that is still good.
        Returns how many pages reached the free list.
        """
        freed = 0
        passed_over = []
        while freed < n:
            entry = self._pop_leaf()
            if entry is None:
                break
            stamp, page = entry
            if self._arena.num_owners[page] > 1:
                passed_over.append(entry)
                continue
            self._remove(page)
            freed += 1
        for entry in passed_over:
            heapq.heappush(self._leaves, entry)
        return freed

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
        self._evictable.discard(page)
        del self._by_key[self._key[page]]
        self._key[page] = None
        parent = self._parent[page]
        self._parent[page] = None
        self._arena.release([page])
        if parent is not None:
            self._children[parent] -= 1
            if self._children[parent] == 0 and self._key[parent] is not None:
                heapq.heappush(self._leaves, (self._stamp[parent], parent))


class _Owners:
    """A copy of the arena's owner counts, which `evict` on a copied index releases."""

    def __init__(self, num_owners: list[int]) -> None:
        self.num_owners = list(num_owners)

    def release(self, pages: list[int]) -> None:
        for page in pages:
            self.num_owners[page] -= 1
