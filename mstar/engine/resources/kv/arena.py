"""The page pool's bookkeeping: free list, per-page ownership, and every
stream's page list.

`new_page_arena` returns the Rust arena (`rust/src/page_arena.rs`) when the
extension has one, and this module's `PythonPageArena` otherwise. Both hand
out the same pages in the same order and expose the same calls; a call that
takes ``pages`` accepts a list or a stream's `StreamPages`. The pages' storage
and copies are `KVCache`'s.

A page goes back to the allocator when its last owner releases it, not its
first. A sealed page is never written again, so a second owner may read it;
freeing clears the seal. Every caller holds the KV manager's `_lock`, so the
counts and seals need none of their own.
"""
from __future__ import annotations

import weakref
from collections.abc import Iterable, Iterator
from typing import NamedTuple, Protocol, overload

import numpy as np

from mstar.engine.resources.kv.cache import PageAllocator

try:
    from mstar_rust import PageArena as RustPageArena
except ImportError:  # an older build of the extension, or none
    RustPageArena = None


class PlanColumns(NamedTuple):
    """A plan label's index arrays (int32), from per-row columns."""
    qo_indptr: np.ndarray
    kv_indptr: np.ndarray
    indices: np.ndarray
    last_page_len: np.ndarray
    kv_lens: np.ndarray
    # each row's write page and slot; meaningful only when every row has a page
    decode_pages: np.ndarray
    decode_offsets: np.ndarray
    all_have_pages: bool


def columns_from_raw(raw: tuple, n: int) -> PlanColumns:
    """The columns of ``n`` rows from an arena's ``plan_columns``,
    ``(bytearray, total, all_have_pages)``, as views of its one buffer
    (laid out as `rust/src/kv_plan.rs` says)."""
    raw, _, all_have_pages = raw
    buf = np.frombuffer(raw, np.int32)
    a, b = n + 1, 2 * (n + 1)
    return PlanColumns(
        qo_indptr=buf[:a], kv_indptr=buf[a:b], indices=buf[b + 4 * n:],
        last_page_len=buf[b:b + n], kv_lens=buf[b + n:b + 2 * n],
        decode_pages=buf[b + 2 * n:b + 3 * n], decode_offsets=buf[b + 3 * n:b + 4 * n],
        all_have_pages=all_have_pages,
    )


class StreamPages(Protocol):
    """One stream's pages, in order, read and written like a list of ints."""

    @property
    def sid(self) -> int: ...
    @property
    def n(self) -> int: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[int]: ...
    @overload
    def __getitem__(self, i: int) -> int: ...
    @overload
    def __getitem__(self, i: slice) -> list[int]: ...
    def __delitem__(self, i: int | slice) -> None: ...
    def tolist(self) -> list[int]: ...
    def extend(self, pages: Iterable[int]) -> None: ...
    def assign(self, pages: Iterable[int]) -> None: ...
    def clear(self) -> None: ...
    def free(self) -> None:
        """Drop the stream; its id may be reused."""


class PageArena(Protocol):
    """What the KV manager, the prefix index and the host pool use of an
    arena. The Rust arena satisfies it structurally: a PyO3 class can't
    subclass a Python one, and a wrapper would cost a frame per call."""

    # told when a page gains or loses its second owner (`PrefixIndex.set_shared`)
    index: object | None
    # (acquires, hash of the pages handed out) once set to (0, 0); None is off
    trail: tuple[int, int] | None

    @property
    def max_num_pages(self) -> int: ...
    @property
    def num_free(self) -> int: ...
    def free_pages(self) -> list[int]:
        """The free pages, in the order they will be handed out. Linear in
        the pool: for consistency checks and tests, not a step path."""
    def acquire(self, n: int) -> list[int] | None: ...
    def retain(self, pages: Iterable[int]) -> None: ...
    def release(self, pages: Iterable[int]) -> None: ...
    def seal(self, pages: Iterable[int]) -> None: ...
    def any_sealed(self, pages: Iterable[int]) -> bool: ...
    def is_sealed(self, page: int) -> bool: ...
    def owners(self, page: int) -> int: ...
    def owners_of(self, pages: Iterable[int]) -> list[int]: ...
    def new_stream(self) -> StreamPages: ...
    def plan_columns(
        self, sids: list[int], lens: list[int], spans: list[int], page_size: int,
    ) -> tuple[bytearray, int, bool]:
        """`PlanColumns` of the streams ``sids`` as one int32 buffer
        (`columns_from_raw`)."""


class _ListStreamPages(list):
    """`PythonPageArena`'s stream: a plain list, plus the stream's id."""

    __slots__ = ("_arena", "sid", "__weakref__")

    def __init__(self, arena: PythonPageArena, sid: int):
        super().__init__()
        self._arena = arena
        self.sid = sid

    @property
    def n(self) -> int:
        return len(self)

    def tolist(self) -> list[int]:
        return list(self)

    def assign(self, pages: Iterable[int]) -> None:
        self[:] = pages

    def free(self) -> None:
        self.clear()
        self._arena._streams.pop(self.sid, None)


class PythonPageArena:
    """The arena in Python. ``allocator`` is the free list it takes from, a
    `PageAllocator` by default (the conductor's routing mirror passes a
    lock-free one)."""

    def __init__(self, max_num_pages: int, allocator=None):
        self.allocator = PageAllocator(max_num_pages) if allocator is None else allocator
        self.num_owners = [0] * max_num_pages
        self.sealed = [False] * max_num_pages
        # the prefix index to tell when a page gains or loses its second owner
        self.index = None
        # queues each page a release leaves to the index alone, for a copy to the host
        self.writes = None
        self.trail: tuple[int, int] | None = None
        # held weakly: a stream nothing holds is gone, as a Rust one is
        self._streams: weakref.WeakValueDictionary[int, _ListStreamPages] = weakref.WeakValueDictionary()
        self._next_sid = 0

    @property
    def max_num_pages(self) -> int:
        return self.allocator.max_num_pages

    @property
    def num_free(self) -> int:
        return self.allocator.num_free

    def free_pages(self) -> list[int]:
        """The free pages, in the order they will be handed out."""
        return list(self.allocator.free_pages.queue)

    def acquire(self, n: int) -> list[int] | None:
        pages = self.allocator.try_allocate(n)
        if pages is not None:
            for page in pages:
                self.num_owners[page] = 1
            if self.trail is not None:
                self.trail = (self.trail[0] + 1, hash((self.trail, *pages)))
        return pages

    def retain(self, pages: Iterable[int]) -> None:
        for page in pages:
            self.num_owners[page] += 1
            if self.num_owners[page] == 2 and self.index is not None:
                self.index.set_shared(page, True)

    def release(self, pages: Iterable[int]) -> None:
        freed = []
        for page in pages:
            assert self.num_owners[page] > 0, f"page {page} released with no owner"
            self.num_owners[page] -= 1
            if self.num_owners[page] == 0:
                self.sealed[page] = False
                freed.append(page)
            elif self.num_owners[page] == 1 and self.index is not None:
                self.index.set_shared(page, False)
                if self.writes is not None:
                    self.writes.queue(self.index.key_of(page))
        self.allocator.free(freed)

    def seal(self, pages: Iterable[int]) -> None:
        for page in pages:
            self.sealed[page] = True

    def any_sealed(self, pages: Iterable[int]) -> bool:
        return any(self.sealed[page] for page in pages)

    def is_sealed(self, page: int) -> bool:
        return self.sealed[page]

    def owners(self, page: int) -> int:
        return self.num_owners[page]

    def owners_of(self, pages: Iterable[int]) -> list[int]:
        owners = self.num_owners
        return [owners[page] for page in pages]

    def new_stream(self) -> StreamPages:
        pages = self._streams[self._next_sid] = _ListStreamPages(self, self._next_sid)
        self._next_sid += 1
        return pages

    def plan_columns(
        self, sids: list[int], lens: list[int], spans: list[int], page_size: int,
    ) -> tuple[bytearray, int, bool]:
        """`rust/src/kv_plan.rs`'s ``plan_columns``, op for op, in numpy."""
        rows = [self._streams[sid] for sid in sids]
        n = len(rows)
        lens_a = np.array(lens, np.int32).reshape(n)
        # a stream can hold more pages than its length needs, and never more
        # is read than it holds
        npages = lens_a + (page_size - 1)
        npages //= page_size
        np.minimum(npages, [len(pages) for pages in rows], out=npages)
        kv_indptr = np.zeros(n + 1, np.int32)
        np.cumsum(npages, out=kv_indptr[1:])
        total = int(kv_indptr[-1])
        indices = np.fromiter(
            (page for pages, k in zip(rows, npages.tolist(), strict=True) for page in pages[:k]),
            np.int32, total,
        )
        # `SequenceView.last_page_len(page_size) or page_size`
        last = lens_a - 1
        last %= page_size
        last += 1
        # FlashInfer's `get_seq_lens`: (pages - 1) * page_size + last
        kv_lens = np.maximum(npages, 1)
        kv_lens -= 1
        kv_lens *= page_size
        kv_lens += last
        qo = np.zeros(n + 1, np.int32)
        np.cumsum(spans, out=qo[1:])
        all_have_pages = bool(npages.all())
        decode_pages = np.zeros(n, np.int32)
        has = npages > 0
        decode_pages[has] = indices[kv_indptr[1:][has] - 1]
        buf = np.concatenate((qo, kv_indptr, last, kv_lens, decode_pages, last - 1, indices), dtype=np.int32)
        return bytearray(buf), total, all_have_pages


def new_page_arena(max_num_pages: int, backend: str | None = None) -> PageArena:
    """The Rust arena when the extension has one, else `PythonPageArena`;
    ``backend`` ("rust" or "python") picks one."""
    if backend is None:
        backend = "python" if RustPageArena is None else "rust"
    if backend == "python":
        return PythonPageArena(max_num_pages)
    if backend != "rust":
        raise ValueError(f"no page arena backend {backend!r}")
    if RustPageArena is None:
        raise RuntimeError("the Rust page arena needs the mstar_rust extension")
    return RustPageArena(max_num_pages)


def plan_columns(arena: PageArena, sids: list[int], lens: list[int], spans: list[int], page_size: int) -> PlanColumns:
    """A plan label's index arrays over the page lists of the streams
    ``sids``, one per row."""
    return columns_from_raw(arena.plan_columns(sids, lens, spans, page_size), len(sids))
