"""A paged KV cache's routing mirror: which prompt pages each replica's prefix
cache is predicted to hold, so a request goes where its prefix already is.

It predicts from what the conductor sees, a request's page keys when it is
placed and its end, and is approximate for that reason. The worker's index
also moves with admit timing, decode growth, chunked commits and preemption,
and it keys a reply's tokens, which the conductor never sees: a later turn's
hit on an earlier reply is under-predicted. Only the GPU pool is modelled; the
host pool (``cpu_offload_pages``) holds offloaded live requests, not a cache.

Kernel-free, since the conductor builds it.
"""

from collections import OrderedDict
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from mstar.engine.resources.kv.config import KVReqConfig

Chains = tuple[tuple[bytes, ...], ...]


class _Placed(NamedTuple):
    replica: int
    # one per label, so each releases leaves first
    chains: Chains


class _ReplicaPages:
    """One replica's predicted index: the pages live requests hold, and the
    cached pages nothing holds, least recently released first. A simplified
    copy of ``PrefixIndex``'s eviction order (oldest first, leaves before their
    prefix), so a change to one belongs in the other."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.held: dict[bytes, int] = {}
        self.idle: OrderedDict[bytes, None] = OrderedDict()

    def matched(self, keys: Sequence[bytes]) -> int:
        n = 0
        for key in keys:
            if key not in self.held and key not in self.idle:
                break
            n += 1
        return n

    def hold(self, chains: Chains) -> None:
        for chain in chains:
            for key in chain:
                if key in self.idle:
                    del self.idle[key]
                self.held[key] = self.held.get(key, 0) + 1
        # the worker evicts what nothing holds, oldest first, when a request needs pages
        over = len(self.held) + len(self.idle) - self.capacity
        while over > 0 and self.idle:
            self.idle.popitem(last=False)
            over -= 1

    def release(self, chains: Chains) -> None:
        # the deepest pages of every chain first, so a prefix outlives the pages
        # that extend it, as the worker's index evicts leaves first
        for depth in range(max(map(len, chains)) - 1, -1, -1):
            for chain in chains:
                if depth >= len(chain):
                    continue
                key = chain[depth]
                n = self.held[key] - 1
                if n:
                    self.held[key] = n
                else:
                    del self.held[key]
                    self.idle[key] = None


class KVRoutingMirror:
    """`RoutingMirror` of one paged KV resource. A replica's cost is minus the
    share of the request's prompt positions it is predicted to hold already, so
    a hit is worth less than one live request. A request reserves its prompt
    pages on its replica until it ends, and they stay cached after."""

    def __init__(self, num_replicas: int, capacity: int, page_size: int):
        self._page_size = page_size
        self._replicas = [_ReplicaPages(capacity) for _ in range(num_replicas)]

    @staticmethod
    def _chains(config: "KVReqConfig") -> dict[str, tuple[bytes, ...]]:
        """The page keys of each label the request opens and keys, as the worker reads them."""
        if not config.prefix_cache:
            return {}
        layouts = config.kv_layout or {}
        return {
            label: tuple(keys) for label, keys in (config.prefix_keys or {}).items()
            if keys and label in layouts
        }

    def score(self, config: "KVReqConfig") -> list[float]:
        chains = self._chains(config)
        if not chains:
            return [0.0] * len(self._replicas)
        # every prompt position of every label the request opens, keyed or not:
        # a hit saves that share of its prefill
        tokens = sum(span.length for layout in config.kv_layout.values() for span in layout.spans)
        # the worker never matches a prompt's last page, so the walk has input left
        return [
            -self._page_size * sum(min(pages.matched(keys), len(keys) - 1) for keys in chains.values()) / tokens
            for pages in self._replicas
        ]

    def add_request(self, config: "KVReqConfig", replica: int) -> _Placed | None:
        chains = tuple(self._chains(config).values())
        if not chains:
            return None
        self._replicas[replica].hold(chains)
        return _Placed(replica, chains)

    def remove_request(self, handle: _Placed) -> None:
        self._replicas[handle.replica].release(handle.chains)
