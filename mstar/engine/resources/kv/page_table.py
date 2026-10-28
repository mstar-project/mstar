"""Every cache stream's page list, as extents of one int32 pool.

A plan reads the pages of every row of a step. Held as one Python list per
stream, that is a slice and a list walk per row; held as extents of one array,
it is one gather over the step's rows (``PageTable.gather``), whatever the
rows' order, their number or how many pages each holds.

An extent's capacity is a power of two, so a stream that grows moves at most
log2(pages) times, and a freed extent is reused by the next stream of that
size. Memory follows the pages streams actually hold, not the longest context
a model allows.
"""
from __future__ import annotations

import numpy as np

_MIN_CAPACITY = 8


def _capacity_for(n: int) -> int:
    return max(_MIN_CAPACITY, 1 << max(n - 1, 0).bit_length())


class PageTable:
    """The pool and its free extents. A stream's ``StreamPages`` owns one
    extent; ``gather`` reads many at once.

    Concurrency is the KV manager's, as it was for the lists: a stream is not
    mutated while a step that holds it is planned, so a gather never reads an
    extent being moved. Growing the pool swaps in a new array, and a gather
    that took the old one still reads the same pages from it.
    """

    def __init__(self, capacity: int = 1024):
        self.pool = np.empty(capacity, np.int32)
        self._top = 0
        self._free: dict[int, list[int]] = {}

    def new(self) -> StreamPages:
        return StreamPages(self)

    def _alloc(self, capacity: int) -> int:
        free = self._free.get(capacity)
        if free:
            return free.pop()
        off = self._top
        if off + capacity > self.pool.shape[0]:
            size = self.pool.shape[0]
            while off + capacity > size:
                size *= 2
            pool = np.empty(size, np.int32)
            pool[:off] = self.pool[:off]
            self.pool = pool
        self._top = off + capacity
        return off

    def _release(self, off: int, capacity: int) -> None:
        if capacity:
            self._free.setdefault(capacity, []).append(off)

    def gather(self, offs: np.ndarray, counts: np.ndarray, indptr: np.ndarray) -> np.ndarray:
        """The first ``counts[i]`` pages of the extent at ``offs[i]``, for every
        row, concatenated (``indptr`` is ``counts``' exclusive prefix sum, with
        the total last)."""
        total = int(indptr[-1])
        src = np.repeat(offs - indptr[:-1], counts)
        src += np.arange(total)
        return self.pool[src]


class StreamPages:
    """One stream's page list, read and written like the list it replaces.

    Slices and iteration hand out Python lists; ``tolist`` is the whole list.
    """

    __slots__ = ("_table", "off", "cap", "n")

    def __init__(self, table: PageTable | None = None):
        # a stream built outside a KV manager (tests) gets a table of its own
        self._table = table if table is not None else PageTable(_MIN_CAPACITY)
        self.off = 0
        self.cap = 0
        self.n = 0

    def __len__(self) -> int:
        return self.n

    def tolist(self) -> list[int]:
        return self._table.pool[self.off:self.off + self.n].tolist()

    def __iter__(self):
        return iter(self.tolist())

    def __reversed__(self):
        return reversed(self.tolist())

    def __getitem__(self, i):
        if isinstance(i, slice):
            start, stop, step = i.indices(self.n)
            if step != 1:
                return self.tolist()[i]
            if stop <= start:
                return []
            return self._table.pool[self.off + start:self.off + stop].tolist()
        if i < 0:
            i += self.n
        if not 0 <= i < self.n:
            raise IndexError("page index out of range")
        return int(self._table.pool[self.off + i])

    def __eq__(self, other) -> bool:
        if isinstance(other, StreamPages):
            other = other.tolist()
        if isinstance(other, (list, tuple)):
            return self.tolist() == list(other)
        return NotImplemented

    __hash__ = None

    def __repr__(self) -> str:
        return f"StreamPages({self.tolist()})"

    def _reserve(self, n: int) -> None:
        """Room for ``n`` pages, moving to a larger extent if needed."""
        if n <= self.cap:
            return
        table = self._table
        cap = _capacity_for(n)
        off = table._alloc(cap)
        pool = table.pool
        pool[off:off + self.n] = pool[self.off:self.off + self.n]
        table._release(self.off, self.cap)
        self.off, self.cap = off, cap

    def extend(self, pages) -> None:
        k = len(pages)
        if not k:
            return
        self._reserve(self.n + k)
        self._table.pool[self.off + self.n:self.off + self.n + k] = pages
        self.n += k

    def assign(self, pages) -> None:
        """Make ``pages`` the whole list."""
        self.n = 0
        self.extend(pages)

    def clear(self) -> None:
        self.n = 0

    def __delitem__(self, i) -> None:
        if not isinstance(i, slice):
            i = slice(i, i + 1 if i != -1 else None)
        start, stop, step = i.indices(self.n)
        if step != 1:
            raise ValueError("StreamPages deletes contiguous ranges only")
        k = stop - start
        if k <= 0:
            return
        pool, off = self._table.pool, self.off
        # numpy copies an overlapping slice correctly
        pool[off + start:off + self.n - k] = pool[off + stop:off + self.n]
        self.n -= k

    def free(self) -> None:
        """Return the extent to the table; the list is empty after."""
        self._table._release(self.off, self.cap)
        self.off = self.cap = self.n = 0
