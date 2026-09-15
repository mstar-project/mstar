"""Small generic containers with no engine or worker knowledge"""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable, Iterator
from typing import Generic, TypeVar

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
