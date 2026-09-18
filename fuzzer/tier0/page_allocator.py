"""PageAllocator: the KV page free list. This machine uses one thread.

Ops: alloc, try_alloc, free, free_part. The generator makes well-formed
sequences only. Thus an accounting error comes from the allocator.

Invariants
----------
alloc.in_range                     returned pages lie in 0..max-1
alloc.no_repeat_within_call        one call never returns a page twice
alloc.no_double_issue              a returned page is not already held
alloc.failed_call_is_atomic        a raising allocate() consumes nothing
alloc.try_failure_is_atomic        a None from try_allocate() consumes nothing
alloc.try_failure_only_when_short  try_allocate() returns None only when short
pages.num_free_matches_queue       num_free equals the length of the free list
pages.no_duplicates_in_free_list   the free list holds no page twice
pages.free_and_held_disjoint       a page is free or held, never both
pages.conservation                 free + held is exactly 0..max-1
pages.quiesce_restores_full_pool   freeing everything restores every page
pages.quiesce_pool_is_exact        the restored pool is exactly 0..max-1

Not covered:

* concurrent access: the `alloc_concurrent` machine drives that path
* a double free: `PageAllocator` does not detect one
* the contents of a page: a page is an integer here"""

from __future__ import annotations

import random
from collections.abc import Iterator

from fuzzer.common.case import Op
from fuzzer.common.machine import StateMachine, require
from fuzzer.tier0 import _stubs  # noqa: F401  (import-order side effect)
from mstar.engine.resources.kv.cache import PageAllocator


class PageAllocatorMachine(StateMachine):
    name = "page_allocator"

    @classmethod
    def gen_config(cls, rng: random.Random) -> dict:
        return {"max_pages": rng.choice([1, 2, 3, 4, 8, 16, 64])}

    def __init__(self, config: dict) -> None:
        self.max_pages: int = config["max_pages"]
        self.alloc = PageAllocator(self.max_pages)
        # handle -> pages this "request" currently holds
        self.live: dict[int, list[int]] = {}
        self._next_handle = 0

    # -- generation ----------------------------------------------------------

    def gen_op(self, rng: random.Random) -> Op:
        choice = rng.random()
        if choice < 0.40:
            return Op("alloc", (rng.randint(0, self.max_pages + 1),))
        if choice < 0.70:
            return Op("try_alloc", (rng.randint(0, self.max_pages + 1),))
        if choice < 0.90:
            return Op("free", (rng.randrange(8),))
        return Op("free_part", (rng.randrange(8), rng.randint(0, 4)))

    @classmethod
    def shrink_config(cls, config: dict) -> Iterator[dict]:
        for candidate in (1, 2, config["max_pages"] // 2):
            if 1 <= candidate < config["max_pages"]:
                yield {"max_pages": candidate}

    # -- execution -----------------------------------------------------------

    def _handle_at(self, index: int) -> int | None:
        """Find a holder by its position in the sorted list of holders.

        The op does not name a handle. The shrinker can remove the op that
        made a given handle. A position always points at a holder that exists.
        """
        handles = sorted(self.live)
        if not handles:
            return None
        return handles[index % len(handles)]

    def _take(self, pages: list[int]) -> None:
        """Record that a new holder now has ``pages``."""
        self._check_fresh(pages)
        self.live[self._next_handle] = list(pages)
        self._next_handle += 1

    def _check_fresh(self, pages: list[int]) -> None:
        """Check the pages that one call returned, before a holder takes them."""
        require(
            "alloc.in_range",
            all(0 <= p < self.max_pages for p in pages),
            f"allocator returned out-of-range pages {pages} "
            f"(max_pages={self.max_pages})",
        )
        require(
            "alloc.no_repeat_within_call",
            len(set(pages)) == len(pages),
            f"allocator returned the same page twice in one call: {pages}",
        )
        held = {p for owned in self.live.values() for p in owned}
        overlap = held & set(pages)
        require(
            "alloc.no_double_issue",
            not overlap,
            f"allocator handed out page(s) {sorted(overlap)} that are already held",
        )

    def execute(self, op: Op) -> None:
        if op.kind == "alloc":
            (count,) = op.args
            before = self.alloc.num_free
            if count > before:
                try:
                    self.alloc.allocate(count)
                except RuntimeError:
                    require(
                        "alloc.failed_call_is_atomic",
                        self.alloc.num_free == before,
                        f"allocate({count}) raised but consumed "
                        f"{before - self.alloc.num_free} page(s)",
                    )
                    return
                raise AssertionError(
                    f"allocate({count}) succeeded with only {before} free pages"
                )
            self._take(self.alloc.allocate(count))

        elif op.kind == "try_alloc":
            (count,) = op.args
            before = self.alloc.num_free
            pages = self.alloc.try_allocate(count)
            if pages is None:
                require(
                    "alloc.try_failure_is_atomic",
                    self.alloc.num_free == before,
                    f"try_allocate({count}) returned None but consumed "
                    f"{before - self.alloc.num_free} page(s)",
                )
                require(
                    "alloc.try_failure_only_when_short",
                    count > before,
                    f"try_allocate({count}) returned None with {before} free",
                )
                return
            self._take(pages)

        elif op.kind == "free":
            handle = self._handle_at(op.args[0])
            if handle is None:
                return
            self.alloc.free(self.live.pop(handle))

        elif op.kind == "free_part":
            index, count = op.args
            handle = self._handle_at(index)
            if handle is None:
                return
            pages = self.live[handle]
            count = min(count, len(pages))
            if count == 0:
                return
            self.alloc.free(pages[:count])
            remaining = pages[count:]
            if remaining:
                self.live[handle] = remaining
            else:
                del self.live[handle]

        else:
            raise AssertionError(f"unknown op {op.kind}")

    # -- invariants ----------------------------------------------------------

    def _free_pages(self) -> list[int]:
        """Read the free list of the allocator.

        ``free_pages`` is a ``queue.Queue``. Read its deque directly. This
        reads the contents. It does not change them.
        """
        return list(self.alloc.free_pages.queue)

    def check(self) -> None:
        free = self._free_pages()
        held = [p for owned in self.live.values() for p in owned]

        require(
            "pages.num_free_matches_queue",
            self.alloc.num_free == len(free),
            f"num_free={self.alloc.num_free} but the free list holds {len(free)}",
        )
        require(
            "pages.no_duplicates_in_free_list",
            len(set(free)) == len(free),
            f"the same page appears twice in the free list: {sorted(free)}",
        )
        require(
            "pages.free_and_held_disjoint",
            not (set(free) & set(held)),
            f"page(s) {sorted(set(free) & set(held))} are free and held at once",
        )
        require(
            "pages.conservation",
            sorted(free + held) == list(range(self.max_pages)),
            f"pages leaked or multiplied: free={sorted(free)} held={sorted(held)} "
            f"expected exactly 0..{self.max_pages - 1}",
        )

    def final_check(self) -> None:
        """Give every page back. The allocator must then be in its first state."""
        for pages in list(self.live.values()):
            self.alloc.free(pages)
        self.live.clear()
        require(
            "pages.quiesce_restores_full_pool",
            self.alloc.num_free == self.max_pages,
            f"after freeing everything only {self.alloc.num_free} of "
            f"{self.max_pages} pages came back",
        )
        require(
            "pages.quiesce_pool_is_exact",
            sorted(self._free_pages()) == list(range(self.max_pages)),
            f"the recovered pool is not 0..{self.max_pages - 1}: "
            f"{sorted(self._free_pages())}",
        )
