"""A paged KV cache's routing mirror: which prompt pages each replica's prefix
cache is predicted to hold, so a request goes where its prefix already is.

Each replica runs the worker's own ``PrefixIndex`` over a counting arena, and a
request takes the steps ``KVManager`` takes for it, all at dispatch: the lease
of its matched pages, its own pages (evicting on a shortfall), and the index
entries of its whole keyed pages. Its end releases its pages, and takes back
the entries of a request whose walk never ran.

It predicts from what the conductor sees, a request's layout and page keys
when it is placed and its end, and is approximate for that reason:

- the worker's pool and index also move with admit timing, decode growth,
  scratch positions, chunked commits and preemption, and it keys a reply's
  tokens and an image ending inside a page, which the conductor never sees: a
  later turn's hit on an earlier reply is under-predicted;
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

Only the GPU pool is modelled; the host pool (``cpu_offload_pages``) holds
offloaded live requests, not a cache.

Kernel-free, since the conductor builds it.
"""

from collections.abc import Collection, Sequence
from typing import TYPE_CHECKING, NamedTuple

from mstar.engine.resources.kv.prefix_index import PrefixIndex
from mstar.engine.resources.spec import RoutingFacts

if TYPE_CHECKING:
    from mstar.engine.resources.kv.config import KVReqConfig


class _Arena:
    """The page owner counts and free list of one replica's pool, with no storage."""

    def __init__(self, num_pages: int):
        self.num_owners = [0] * num_pages
        # every page but SINK_PAGE (0), which the pool never hands out
        self._free = list(range(1, num_pages))

    @property
    def num_free(self) -> int:
        return len(self._free)

    def acquire(self, n: int) -> list[int]:
        """Up to ``n`` free pages."""
        pages = self._free[len(self._free) - min(n, len(self._free)):]
        del self._free[len(self._free) - len(pages):]
        self.retain(pages)
        return pages

    def retain(self, pages: Sequence[int]) -> None:
        for page in pages:
            self.num_owners[page] += 1

    def seal(self, pages: Sequence[int]) -> None:
        pass

    def release(self, pages: Sequence[int]) -> None:
        for page in pages:
            self.num_owners[page] -= 1
            if not self.num_owners[page]:
                self._free.append(page)


class _Placed(NamedTuple):
    replica: int
    # every page the request holds: its lease and its own
    pages: list[int]
    # its own pages it indexed, which the worker indexes only once the walk runs
    indexed: list[int]


class KVRoutingMirror:
    """`RoutingMirror` of one paged KV resource, for the keyed ``labels`` whose
    walks run on these replicas."""

    def __init__(self, num_replicas: int, num_pages: int, page_size: int, labels: Collection[str]):
        self._page_size = page_size
        self._labels = labels
        self._arenas = [_Arena(num_pages) for _ in range(num_replicas)]
        self._indexes = [PrefixIndex(arena) for arena in self._arenas]

    def _keyed(self, config: "KVReqConfig") -> dict[str, tuple[Sequence[bytes], int]]:
        """Each label's page keys, and how many of its pages are whole: the
        only ones the worker indexes (``PrefixChain.keyed_pages``)."""
        if not config.prefix_cache:
            return {}
        layouts = config.kv_layout or {}
        keyed = {}
        for label, keys in (config.prefix_keys or {}).items():
            layout = layouts.get(label)
            if not keys or layout is None or label not in self._labels:
                continue
            whole = sum(span.length for span in layout.spans[:layout.keyed_spans]) // self._page_size
            if layout.retention is not None:
                whole = min(whole, layout.retention.protected_prefix // self._page_size)
            keyed[label] = keys, whole
        return keyed

    def facts(self, config: "KVReqConfig") -> list[RoutingFacts]:
        keyed = self._keyed(config)
        # every prompt position of every label the request opens, keyed or not:
        # a hit saves that share of its prefill
        work = sum(span.length for layout in (config.kv_layout or {}).values() for span in layout.spans)
        # the worker never matches a prompt's last page, so the walk has input left
        return [
            RoutingFacts(work, self._page_size * sum(
                min(len(index.pages_for(keys)), len(keys) - 1) for keys, _ in keyed.values()
            ))
            for index in self._indexes
        ]

    def add_request(self, config: "KVReqConfig", replica: int) -> _Placed | None:
        layouts = config.kv_layout or {}
        if not layouts:
            return None
        keyed = self._keyed(config)
        arena, index = self._arenas[replica], self._indexes[replica]
        held, indexed = [], []
        for label, layout in layouts.items():
            keys, whole = keyed.get(label, ((), 0))
            # `KVManager._lease`, then its `_alloc` and `_index_filled_pages`
            leased = index.lookup(keys)[:len(keys) - 1] if keys else []
            arena.retain(leased)
            need = -(-sum(span.length for span in layout.spans) // self._page_size) - len(leased)
            if need > arena.num_free:
                index.evict(need - arena.num_free)
            own = arena.acquire(need)
            run = max(0, min(whole - len(leased), len(own)))
            if run:
                index.insert_run(keys[len(leased):len(leased) + run], own[:run], leased[-1] if leased else None)
                indexed += [page for page in own[:run] if index.holds(page)]
            held += leased + own
        return _Placed(replica, held, indexed)

    def remove_request(self, handle: _Placed, ran: bool) -> None:
        self._arenas[handle.replica].release(handle.pages)
        if not ran:
            self._indexes[handle.replica].drop(handle.indexed)
