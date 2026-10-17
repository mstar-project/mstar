"""Small generic containers with no engine or worker knowledge"""

from __future__ import annotations

import time
from collections import OrderedDict, deque
from collections.abc import Callable, Hashable, Iterator
from typing import Generic, NamedTuple, TypeVar

T = TypeVar("T", bound=Hashable)


class RecentSet(Generic[T]):
    """Set of the ``maxlen`` most recently added items: O(1) add / ``in``, FIFO
    eviction. Readding a present item is a no-op (age not refreshed)."""

    __slots__ = ("_items", "_order", "maxlen")

    def __init__(self, maxlen: int) -> None:
        if maxlen < 1:
            raise ValueError(f"maxlen must be >= 1, got {maxlen}")
        self.maxlen = maxlen
        self._items: set[T] = set()
        self._order: deque[T] = deque()

    def add(self, item: T) -> None:
        if item in self._items:
            return
        self._items.add(item)
        self._order.append(item)
        if len(self._order) > self.maxlen:
            self._items.discard(self._order.popleft())

    def __contains__(self, item: object) -> bool:
        return item in self._items

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        """Oldest to newest."""
        return iter(self._order)

    def __repr__(self) -> str:
        return f"RecentSet(maxlen={self.maxlen}, items={list(self._order)!r})"


class ExpiringSet(Generic[T]):
    """Set whose items expire ``ttl_s`` after they were last added: O(1)
    amortized add / ``in`` / ``discard``. Expired items are pruned oldest first
    on add and on lookup, so it holds at most what was added in the last
    ``ttl_s``."""

    __slots__ = ("_clock", "_expiry", "ttl_s")

    def __init__(
        self, ttl_s: float, clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if ttl_s <= 0:
            raise ValueError(f"ttl_s must be > 0, got {ttl_s}")
        self.ttl_s = ttl_s
        self._clock = clock
        # Insertion order is expiry order (one TTL, a monotonic clock, and a
        # readd moves to the end). OrderedDict rather than dict: finding a
        # dict's first key after deleting from the front walks the holes.
        self._expiry: OrderedDict[T, float] = OrderedDict()

    def _prune(self, now: float) -> None:
        expiry = self._expiry
        while expiry:
            oldest = next(iter(expiry))
            if expiry[oldest] > now:
                return
            del expiry[oldest]

    def add(self, item: T) -> None:
        now = self._clock()
        self._prune(now)
        self._expiry[item] = now + self.ttl_s
        self._expiry.move_to_end(item)

    def discard(self, item: T) -> None:
        self._expiry.pop(item, None)

    def __contains__(self, item: object) -> bool:
        self._prune(self._clock())
        return item in self._expiry

    def __len__(self) -> int:
        self._prune(self._clock())
        return len(self._expiry)

    def __repr__(self) -> str:
        return f"ExpiringSet(ttl_s={self.ttl_s}, items={list(self._expiry)!r})"


TK = TypeVar('TK')
TV = TypeVar('TV')

class ParallelList(NamedTuple, Generic[TK, TV]):
    """Struct-of-arrays pairing, so a batch crosses a language boundary as two
    flat lists instead of a dict of objects.

    Two sharp edges come from the field names plus the custom ``__iter__``:

    * ``dict(pl)`` does NOT work. ``dict`` probes for a ``.keys()`` method to
      decide whether it was handed a mapping; here ``keys`` is a list
      attribute, so it raises ``'list' object is not callable``. Use
      ``dict(iter(pl))``.
    * ``k, v = pl`` does not give the two lists -- see ``__iter__``.
    """

    keys: list[TK]
    values: list[TV]

    def __len__(self) -> int:
        return len(self.keys)

    def __iter__(self) -> Iterator[tuple[TK, TV]]:
        """Yields (key, value) pairs.

        NOTE: this overrides NamedTuple's field iteration, so ``k, v = pl``
        does NOT give you the two lists -- it unpacks pairs, and silently
        succeeds with exactly two entries. Use ``pl.keys`` / ``pl.values``.

        strict: the two lists being the same length is the invariant of the
        whole type; a silent truncation here would drop requests.
        """
        return zip(self.keys, self.values, strict=True)

    @classmethod
    def from_dict(cls, d: dict[TK, TV]) -> "ParallelList[TK, TV]":
        keys = list(d.keys())
        values = [d[k] for k in keys]
        return cls(keys, values)
