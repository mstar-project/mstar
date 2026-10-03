"""Eviction takes the coldest leaf and never strands a chain.

A parent link is a physical page id, so a page removed from the middle of a
chain could be handed out again under a descendant that still names it. Removal
is therefore leaf-only, which also means a branch that was just hit keeps its
parent: after a request hits A->B->D and finishes, evicting C stops at B, which
still has D.
"""

from __future__ import annotations

import sys
from collections import Counter

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.kv.cache import KVCache, PageAllocator
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.manager import PageArena
from mstar.engine.resources.kv.prefix_index import PrefixIndex

PAGE_SIZE = 16


def _arena(max_num_pages: int = 16) -> PageArena:
    """A page arena over a CPU cache, with no manager above it."""
    cfg = PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=8, max_seq_len=4096,
        max_num_pages=max_num_pages, page_size=PAGE_SIZE,
    )
    return PageArena(
        kv_cache=KVCache(cfg, torch.device("cpu"), torch.float32),
        allocator=PageAllocator(max_num_pages),
    )


def _assert_pages_partition(arena: PageArena) -> None:
    """Every page is either free or owned, never both and never neither.

    `KVManager.assert_pages_conserved` cannot stand in here: it counts a page's
    owners against the streams naming it, and this arena has no streams.
    """
    free = list(arena.allocator.free_pages.queue)
    owned = [
        page for page in range(arena.allocator.max_num_pages)
        if arena.num_owners[page] > 0
    ]
    assert not set(free) & set(owned), (
        f"pages both free and owned: {sorted(set(free) & set(owned))}"
    )
    assert len(free) + len(owned) == arena.allocator.max_num_pages, (
        f"{arena.allocator.max_num_pages} pages in the pool, but "
        f"{len(free)} free and {len(owned)} owned"
    )


def _chain(index: PrefixIndex, arena: PageArena) -> dict[str, int]:
    """A->B, B->C and B->D, each page indexed and then let go by its request."""
    a, b, c, d = arena.acquire(4)
    index.insert(b"A", a)
    index.insert(b"B", b, a)
    index.insert(b"C", c, b)
    index.insert(b"D", d, b)
    arena.release([a, b, c, d])
    return {"A": a, "B": b, "C": c, "D": d}


# ── the index as an owner ───────────────────────────────────────────────


def test_an_indexed_page_is_sealed_and_kept_by_a_second_owner():
    arena = _arena()
    index = PrefixIndex(arena)
    page, = arena.acquire(1)

    index.insert(b"k", page)

    assert arena.num_owners[page] == 2, "the index did not take a reference"
    assert arena.sealed[page], "an indexed page was left writable"
    _assert_pages_partition(arena)


def test_a_page_its_request_still_owns_is_passed_over():
    arena = _arena()
    index = PrefixIndex(arena)
    page, = arena.acquire(1)
    index.insert(b"k", page)

    freed = index.evict(1)

    assert freed == 0, "a page a request still owns was counted as freed"
    assert arena.num_owners[page] == 2, "the index let go of a page it still names"
    assert index.lookup([b"k"]) == [page], (
        "an entry that is still good was dropped to free nothing"
    )
    _assert_pages_partition(arena)


def test_two_inserts_of_one_key_keep_one_entry_and_one_count():
    arena = _arena()
    index = PrefixIndex(arena)
    first, second = arena.acquire(2)
    assert index.insert(b"k", first), "the first writer of a key was refused"

    assert index.insert(b"k", second) is False, (
        "a second page took a key the index already holds"
    )

    assert index.lookup([b"k"]) == [first], "the second writer took the key"
    assert arena.num_owners[second] == 1, "a refused insert still retained"
    _assert_pages_partition(arena)


# ── the walk ────────────────────────────────────────────────────────────


def test_a_lookup_stops_at_the_first_key_the_index_does_not_hold():
    arena = _arena()
    index = PrefixIndex(arena)
    pages = _chain(index, arena)

    assert index.lookup([b"A", b"B", b"nope", b"D"]) == [pages["A"], pages["B"]], (
        "the walk carried on past a key the index does not hold"
    )
    _assert_pages_partition(arena)


# ── eviction ────────────────────────────────────────────────────────────


def test_eviction_after_a_hit_takes_the_cold_branch_and_stops_at_the_fork():
    arena = _arena()
    index = PrefixIndex(arena)
    pages = _chain(index, arena)
    assert index.lookup([b"A", b"B", b"D"]) == [pages["A"], pages["B"], pages["D"]], (
        "the branch this test warms was not the one the index walked"
    )

    freed = index.evict(1)

    assert freed == 1, "the eviction stopped before it had given a page back"
    assert index.lookup([b"A", b"B", b"D"]) == [pages["A"], pages["B"], pages["D"]], (
        "evicting the cold branch disturbed the branch that was hit"
    )
    assert index.lookup([b"A", b"B", b"C"]) == [pages["A"], pages["B"]], (
        "the cold leaf survived the eviction"
    )
    _assert_pages_partition(arena)


def test_eviction_walks_up_a_chain_once_it_is_childless():
    arena = _arena()
    index = PrefixIndex(arena)
    pages = _chain(index, arena)

    freed = index.evict(4)

    assert freed == 4, f"a whole chain of 4 freed only {freed} pages"
    assert index.lookup([b"A"]) == [], "the root outlived every child"
    _assert_pages_partition(arena)


def test_eviction_stops_when_no_leaf_is_left():
    arena = _arena()
    index = PrefixIndex(arena)
    _chain(index, arena)

    assert index.evict(99) == 4, "eviction claimed more pages than it held"
    _assert_pages_partition(arena)


# ── how big the heap gets ───────────────────────────────────────────────


def _one_entry_per_indexed_page(index: PrefixIndex) -> None:
    entries = Counter(page for _, page in index._leaves)
    assert set(entries) <= set(index.pages()), (
        f"the heap still names pages the index let go of: "
        f"{sorted(set(entries) - set(index.pages()))}"
    )
    assert max(entries.values(), default=0) <= 1, (
        f"a page holds more than one heap entry: "
        f"{sorted(page for page, count in entries.items() if count > 1)}"
    )


def test_lookups_of_one_leaf_leave_at_most_one_heap_entry_per_page():
    arena = _arena()
    index = PrefixIndex(arena)
    (page,) = arena.acquire(1)
    index.insert(b"k", page)
    arena.release([page])

    for _ in range(10_000):
        index.lookup([b"k"])

    assert len(index._leaves) <= len(index.pages()), (
        f"{len(index._leaves)} heap entries for {len(index.pages())} indexed "
        "page: every hit on a leaf pushed another one"
    )


def test_children_churning_under_a_held_parent_leave_it_one_heap_entry():
    arena = _arena()
    index = PrefixIndex(arena)
    (parent,) = arena.acquire(1)
    index.insert(b"parent", parent)  # and a request keeps holding it

    for turn in range(200):
        (child,) = arena.acquire(1)
        key = f"child {turn}".encode()
        index.insert(key, child, parent)
        arena.release([child])
        index.lookup([b"parent", key])

        assert index.evict(1) == 1, "the one child nobody holds was not given back"
        _one_entry_per_indexed_page(index)
    _assert_pages_partition(arena)


# ── one key a page ──────────────────────────────────────────────────────


def test_a_page_already_indexed_cannot_take_a_second_key():
    arena = _arena()
    index = PrefixIndex(arena)
    (page,) = arena.acquire(1)
    index.insert(b"first key", page)

    with pytest.raises(AssertionError, match=f"page {page} indexed under .* while it is still indexed under"):
        index.insert(b"second key", page)
