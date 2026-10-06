"""The prefix index: which cached page holds which key, and which to drop first.

Inserting makes the index a page's second owner, so evicting releases that one
reference and the page reaches the free list only once no live request holds it
too. Leaves are ordered by a clock that ticks on every insert and every matching
lookup. A hit re-stamps a page without pushing it again, so a leaf keeps one heap
entry however often it is read, and an entry that surfaces under an older stamp
is pushed back under the page's own.

The pages eviction could reach are kept as the arena reports owners added and
dropped (see `evictable`), so asking for them is not a walk over every page indexed.

Every caller holds the manager's lock, so none of this takes one of its own.
"""

import heapq
from collections.abc import Sequence
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
        self._stamp: list[int] = [0] * num_pages
        self._leaves: list[tuple[int, int]] = []
        self._clock = 0
        # a page has an owner besides the index, as of the arena's last word
        self._shared: list[bool] = [False] * num_pages
        # how many of a page's children have a shared page at or below them
        self._busy: list[int] = [0] * num_pages
        # the pages eviction could reach, and a frozen copy of them for whoever asks
        self._evictable: set[int] = set()
        self._frozen: frozenset[int] | None = frozenset()
        arena.owner_hook = self._owners_changed

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
        self._stamp[page] = self._clock
        self._shared[page] = False
        self._busy[page] = 0
        self._arena.seal([page])
        # the arena reports this one: the page is shared (its request still owns it),
        # which keeps it, and every page above it that is not already kept, out of the set
        self._arena.retain([page])
        if parent is not None:
            self._children[parent] += 1
        heapq.heappush(self._leaves, (self._clock, page))
        return True

    def peek(self, keys: Sequence[bytes]) -> list[int]:
        """`lookup` without the hit: asking what would match re-stamps nothing."""
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

    def evictable(self) -> frozenset[int]:
        """The pages `evict` could free, cascading from the leaves.

        A page only the index holds is still out of reach while any page below
        it is held by someone else: eviction takes leaves only, and that one is
        never a leaf it can drop.

        Kept as the arena reports a page's owners changing, rather than walked
        for: a request waiting at the head of the admission queue asks on every
        scheduling pass, and the walk is over every page indexed. With the free
        list empty every page a request is granted evicts one and every page it
        fills is inserted, so the owners change between nearly every two asks.
        """
        if self._frozen is None:
            self._frozen = frozenset(self._evictable)
        return self._frozen

    def evictable_count(self, leasing: Sequence[int] = ()) -> int:
        """``len(evictable())`` less the ``leasing`` pages in it, without making the set."""
        pages = self._evictable
        return len(pages) - len(pages.intersection(leasing))

    def _find_evictable(self) -> frozenset[int]:
        """The set by walking every page indexed: what the one kept has to equal."""
        pinned: set[int] = set()
        for page in self._by_key.values():
            if self._arena.num_owners[page] > 1:
                parent = self._parent[page]
                while parent is not None and parent not in pinned:
                    pinned.add(parent)
                    parent = self._parent[parent]
        return frozenset(
            page for page in self._by_key.values()
            if self._arena.num_owners[page] == 1 and page not in pinned
        )

    def _owners_changed(self, pages: list[int]) -> None:
        """The arena's word that ``pages`` have gained or lost an owner."""
        owners = self._arena.num_owners
        for page in pages:
            if self._key[page] is not None and (owners[page] > 1) != self._shared[page]:
                self._share(page, owners[page] > 1)

    def _share(self, page: int, shared: bool) -> None:
        """``page`` has gained or lost an owner besides the index.

        A page with another owner is out of the set, and so is every page above it
        for as long as any page below it has one: eviction takes leaves only, and the
        index's own reference is all that is left of a page nobody else holds.
        ``_busy`` counts, for each page, the children with a shared page at or below
        them, so one above is let back in when the last of them is. That count moves
        only when this page is the first or the last shared one in its subtree, and
        the walk up stops at the first page that was kept out, or stays out, for some
        other reason.
        """
        busy = self._busy
        was = self._shared[page] or busy[page] > 0
        self._shared[page] = shared
        while True:
            held = self._shared[page] or busy[page] > 0
            self._classify(page)
            if held == was:
                return
            page = self._parent[page]
            if page is None:
                return
            was = self._shared[page] or busy[page] > 0
            busy[page] += 1 if held else -1

    def _classify(self, page: int) -> None:
        """Put ``page`` in the set, or take it out, by whether it or a page below it is shared."""
        pages = self._evictable
        if self._shared[page] or self._busy[page] > 0:
            if page in pages:
                pages.remove(page)
                self._frozen = None
        elif page not in pages:
            pages.add(page)
            self._frozen = None

    def pages(self) -> list[int]:
        """Every page the index is holding a reference to."""
        return list(self._by_key.values())

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
        # a leaf, and only the index holds it: nothing above it was kept out by it
        if self._shared[page]:
            self._share(page, False)
        self._evictable.discard(page)
        self._frozen = None
        del self._by_key[self._key[page]]
        self._key[page] = None
        parent = self._parent[page]
        self._parent[page] = None
        self._arena.release([page])
        if parent is not None:
            self._children[parent] -= 1
            if self._children[parent] == 0 and self._key[parent] is not None:
                heapq.heappush(self._leaves, (self._stamp[parent], parent))
