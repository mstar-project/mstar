"""Small generic containers with no engine or worker knowledge"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterator
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
