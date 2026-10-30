"""A paged KV cache's routing mirror: which prompt pages each replica's prefix
cache is predicted to hold, so a request goes where its prefix already is.

Each replica runs the worker's own ``PrefixIndex`` and ``PageArena``, with no
storage, and a request takes the steps ``KVManager`` takes for it, all at
dispatch: the lease of its matched pages, its own pages (evicting on a
shortfall), the index entries of its whole keyed pages, and the page of its
last item that ends inside one, up to that end. Its end releases its pages, and
takes back the entries of a request whose walk never ran.

It predicts from what the conductor sees, a request's layout and page keys
when it is placed and its end, and is approximate for that reason:

- the worker's pool and index also move with admit timing, decode growth,
  scratch positions, chunked commits and preemption, and it keys a reply's
  tokens, which the conductor never sees: a later turn's hit on an earlier
  reply is under-predicted;
- a request the pool cannot hold yet waits on the worker, but here indexes only
  the pages it could take;
- each replica's pool is charged every label of the request's layout, also
  those whose pages another node group's pool holds, so it evicts early;
- ``ran`` is per partition: a pass of a walk that skips the cache counts as
  run, and a pass that wrote the prompt but was aborted before it came back
  does not;
- an entry belongs to the request placed first, where the worker's belongs to
  the one that ran first: requests sharing a prefix that all end unrun leave
  the entries a later one leased, and when only a later one runs, the first
  one's end takes back an entry the worker keeps.
- on a node with a recurrent state pool a hit is served only up to the longest
  snapshot of that state the pool holds under it (``cap_cached_prefix``), and
  the pool keeps a few per prefill in ``prefix_slots``; the mirror counts the
  KV hit whole, so it over-predicts there.

Only the GPU pool is modelled; the host pool (``cpu_offload_pages``) holds
offloaded live requests, not a cache.

Kernel-free, since the conductor builds it.
"""

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, NamedTuple

from mstar.engine.resources.kv.arena import PageArena
from mstar.engine.resources.kv.prefix_index import PrefixIndex
from mstar.engine.resources.spec import RoutingFacts

if TYPE_CHECKING:
    from mstar.engine.resources.kv.config import KVReqConfig


class _FreePages:
    """The free list a replica's `PageArena` takes from: `PageAllocator`'s,
    without the queue and lock that cost the conductor's one thread about 3 µs
    a page."""

    def __init__(self, num_pages: int):
        self.max_num_pages = num_pages
        # every page but SINK_PAGE (0), which the pool never hands out
        self._free = list(range(1, num_pages))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def try_allocate(self, n: int) -> list[int] | None:
        if n > len(self._free):
            return None
        pages = self._free[len(self._free) - n:]
        del self._free[len(self._free) - n:]
        return pages

    def free(self, pages: list[int]) -> None:
        self._free.extend(pages)


class _Keyed(NamedTuple):
    """One label's prompt as the worker's ``PrefixChain`` keys it."""
    keys: Sequence[bytes]
    # how many of its pages are whole: the only ones the worker indexes (``PrefixChain.keyed_pages``)
    whole: int
    # its keyed positions
    total: int
    # where each item that ends inside a page ends, and the key of that page up to there
    ends: list[tuple[int, bytes]]


class _Placed(NamedTuple):
    replica: int
    # every page the request holds: its lease and its own
    pages: list[int]
    # the (key, page) entries it filed, which the worker files only once the walk runs.
    # An item's end page is the index's alone, so it may be evicted and refiled under another key
    indexed: list[tuple[bytes, int]]


class KVRoutingMirror:
    """`RoutingMirror` of one paged KV resource, for the keyed ``labels`` whose
    walks run on these replicas."""

    def __init__(self, num_replicas: int, num_pages: int, page_size: int, labels: Collection[str]):
        self._page_size = page_size
        self._labels = labels
        self._arenas = [PageArena(kv_cache=None, allocator=_FreePages(num_pages)) for _ in range(num_replicas)]
        self._indexes = [PrefixIndex(arena) for arena in self._arenas]
        for arena, index in zip(self._arenas, self._indexes, strict=True):
            # as `KVManager.enable_prefix_cache` at one rank, the only size a mirror is built for
            arena.index = index

    def _keyed(self, config: "KVReqConfig") -> dict[str, _Keyed]:
        if not config.prefix_cache:
            return {}
        layouts = config.kv_layout or {}
        keyed = {}
        for label, keys in (config.prefix_keys or {}).items():
            layout = layouts.get(label)
            if not keys or layout is None or label not in self._labels:
                continue
            total, ends = 0, []
            for span in layout.spans[:layout.keyed_spans]:
                total += span.length
                if span.end_key is not None:
                    ends.append((total, span.end_key))
            whole = total // self._page_size
            if layout.retention is not None:
                whole = min(whole, layout.retention.protected_prefix // self._page_size)
            keyed[label] = _Keyed(keys, whole, total, ends)
        return keyed

    def _image_end(self, keyed: _Keyed, held: int) -> tuple[int, bytes] | None:
        """`KVManager._image_end`: the last item that ends inside the page after ``held`` positions."""
        for end, key in reversed(keyed.ends):
            if held < end < min(held + self._page_size, keyed.total):
                return end, key
        return None

    def _hit(self, index: PrefixIndex, keyed: _Keyed) -> int:
        """The positions `KVManager._lease` would hold: the matched pages but the
        prompt's last, so the walk has input left, then up to the end of an item
        cached inside the next page."""
        held = min(len(index.pages_for(keyed.keys)), len(keyed.keys) - 1) * self._page_size
        end = self._image_end(keyed, held)
        return end[0] if end is not None and index.page_for(end[1]) is not None else held

    def facts(self, config: "KVReqConfig") -> list[RoutingFacts]:
        keyed = self._keyed(config)
        # every prompt position of every label the request opens, keyed or not:
        # a hit saves that share of its prefill
        work = sum(span.length for layout in (config.kv_layout or {}).values() for span in layout.spans)
        return [
            RoutingFacts(work, sum(self._hit(index, label) for label in keyed.values()))
            for index in self._indexes
        ]

    def add_request(self, config: "KVReqConfig", replica: int) -> _Placed | None:
        keyed = self._keyed(config)
        pages = {
            label: -(-sum(span.length for span in layout.spans) // self._page_size)
            for label, layout in (config.kv_layout or {}).items()
        }
        if not pages:
            return None
        arena, index = self._arenas[replica], self._indexes[replica]
        held, indexed = [], []
        for label, total in pages.items():
            prompt = keyed.get(label, _Keyed((), 0, 0, []))
            keys = prompt.keys
            # `KVManager._lease`, then its `_alloc`, `_index_filled_pages` and `_cache_image_end`
            leased = index.lookup(keys)[:len(keys) - 1] if keys else []
            end = self._image_end(prompt, len(leased) * self._page_size)
            if end is not None:
                # the lease reads the cached item end, and copies it into a page of its own
                index.lookup([end[1]])
            arena.retain(leased)
            need = total - len(leased)
            if need > arena.num_free:
                index.evict(need - arena.num_free)
            own = arena.acquire(min(need, arena.num_free))
            run = max(0, min(prompt.whole - len(leased), len(own)))
            if run:
                filed = keys[len(leased):len(leased) + run]
                index.insert_run(filed, own[:run], leased[-1] if leased else None)
                indexed += [(key, page) for key, page in zip(filed, own[:run]) if index.page_for(key) == page]
            if prompt.ends:
                indexed += self._file_item_end(index, arena, prompt)
            held += leased + own
        return _Placed(replica, held, indexed)

    def _file_item_end(self, index: PrefixIndex, arena: PageArena, prompt: _Keyed) -> list[tuple[bytes, int]]:
        """`KVManager._cache_image_end` for the prompt's last item that ends inside
        a page: that page up to the item's end, in a free page, never an evicted
        one, filed under the page before it. Returns the entry it filed."""
        end, key = prompt.ends[-1]
        page = end // self._page_size
        parent = index.page_for(prompt.keys[page - 1]) if page else None
        if (page and parent is None) or index.page_for(key) is not None or not arena.num_free:
            return []
        copy = arena.acquire(1)
        index.insert(key, copy[0], parent)
        # the index is its one owner, as on the worker
        arena.release(copy)
        return [(key, copy[0])]

    def remove_request(self, handle: _Placed, ran: bool) -> None:
        self._arenas[handle.replica].release(handle.pages)
        if not ran:
            self._indexes[handle.replica].drop(handle.indexed)
