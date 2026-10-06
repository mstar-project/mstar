"""The evictable pages the index keeps are what walking it finds, and it never walks.

A pool that admits by reservation counts its supply (the free pages and the cached ones
eviction reaches) for every request that asks. The count of the cached ones used to be a
walk over every page indexed, kept until a page's owners changed; once the free list is
empty every page a request is granted evicts a cached one and every page it fills is
inserted, so the owners changed between nearly every two asks and so did the walk. The
index now keeps the set as the arena reports owners added and dropped (a page is out of it
while it, or a page below it, has an owner besides the index), and walks only to check.

What is kept must never differ from what a walk finds, whatever the pool did in between,
so these compare the two, and count the walks to show there are none.
"""

from __future__ import annotations

import random
import sys

sys.path.insert(0, ".")

import pytest
import torch
from test_kv_admission_peak import (
    DECODE,
    PAGE_SIZE,
    PREFILL,
    ROOT,
    _finish,
    _ingest,
    _prefill,
    _ready,
    _request,
    _run,
    _StubTransfer,
)
from test_kv_prefix_index import _arena

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.prefix_index import PrefixIndex

SEED = 20261006


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    # the check walks too, and these count the walks
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", False)
    for name in ("MSTAR_KV_ADMISSION_FIT", "MSTAR_KV_ADMISSION_ORDER", "MSTAR_STEP_TELEMETRY_DIR"):
        monkeypatch.delenv(name, raising=False)


class _Walks:
    """Counts the walks over every page indexed."""

    def __init__(self, index: PrefixIndex):
        self.count = 0
        # a walk made to check the answer, which is not one made to find it
        self.fresh = walk = index._find_evictable

        def counting():
            self.count += 1
            return walk()

        index._find_evictable = counting


def _manager(max_num_pages: int) -> KVManager:
    kv = KVManager(
        cfg=PagedKVConfig(
            num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096,
            max_num_pages=max_num_pages, page_size=PAGE_SIZE,
        ),
        name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.enable_prefix_cache(ROOT, {"main": (PREFILL, DECODE)})
    return kv


def _cached(kv: KVManager, pages: int) -> None:
    """Run requests to their ends until ``pages`` of the pool are cached pages."""
    for i in range(pages):
        _ingest(kv, f"fill{i}", PAGE_SIZE, 1)
        assert _ready(kv, f"fill{i}").ready and _prefill(kv, f"fill{i}", PAGE_SIZE).ok
        kv.remove_request(f"fill{i}")
    assert len(kv._index.evictable()) == pages


def _fresh(kv: KVManager, walks: _Walks) -> int:
    return kv._arena.num_free + len(walks.fresh())


# ── in a pool ───────────────────────────────────────────────────────────


def test_a_page_granted_from_the_free_list_does_not_change_the_evictable_set():
    kv = _manager(32)
    walks = _Walks(kv._index)
    _ingest(kv, "a", prompt=2 * PAGE_SIZE, max_tokens=8 * PAGE_SIZE)
    assert _ready(kv, "a").ready and _prefill(kv, "a", 2 * PAGE_SIZE).ok
    kept, owner_changes, free = kv._index.evictable(), kv._arena.owner_changes, kv._arena.num_free

    for _ in range(4):
        assert _run(kv, {"a": PAGE_SIZE}).ok

    assert kv._arena.num_free == free - 4, "the pages were not taken from the free list"
    assert kv._arena.owner_changes == owner_changes, (
        "a page handed out from the free list is one the index does not hold, "
        "and counted as a change of owners"
    )
    assert kv._index.evictable() is kept, "a grant of free pages made the set again"
    assert kv._supply() == _fresh(kv, walks)
    assert walks.count == 0


def test_a_grant_that_evicts_and_a_page_it_fills_do_not_have_the_index_walked():
    kv = _manager(16)
    _cached(kv, 15)
    assert kv._arena.num_free == 0
    walks = _Walks(kv._index)

    # a prompt of three pages takes three cached pages, and indexes the three it fills
    _ingest(kv, "a", prompt=3 * PAGE_SIZE, max_tokens=PAGE_SIZE)
    assert _ready(kv, "a").ready and _prefill(kv, "a", 3 * PAGE_SIZE).ok

    assert len(kv._index.pages()) == 15, "the prompt did not evict three and index three"
    assert kv._supply() == _fresh(kv, walks) == 12
    assert walks.count == 0


def test_a_page_a_request_gives_back_is_counted_again_without_a_walk():
    kv = _manager(16)
    _cached(kv, 12)
    _ingest(kv, "a", prompt=2 * PAGE_SIZE, max_tokens=PAGE_SIZE)
    assert _ready(kv, "a").ready and _prefill(kv, "a", 2 * PAGE_SIZE).ok
    walks = _Walks(kv._index)
    held = kv._supply()
    assert held == _fresh(kv, walks)

    kv.remove_request("a")

    assert kv._supply() == _fresh(kv, walks) == held + 2, "the pages the request held were not counted again"
    assert walks.count == 0
    _finish(kv)


def test_a_request_that_is_leased_cached_pages_takes_them_out_of_the_supply_without_a_walk():
    kv = _manager(32)
    prompt = list(range(4 * PAGE_SIZE))
    kv.ingest_request("x", _request(prompt, max_tokens=1))
    assert _ready(kv, "x").ready and _prefill(kv, "x", len(prompt)).ok
    kv.remove_request("x")
    walks = _Walks(kv._index)
    cached = kv._supply()
    assert cached == _fresh(kv, walks)

    kv.ingest_request("y", _request(prompt + [7] * PAGE_SIZE, max_tokens=PAGE_SIZE))
    assert _ready(kv, "y").ready

    leased = len(kv._streams["y"]["main"].lease)
    assert leased == 4, "the cached prompt was not leased"
    assert kv._supply() == _fresh(kv, walks) == cached - leased
    assert walks.count == 0
    _finish(kv)


def test_a_pool_that_checks_what_it_keeps_finds_it_the_same(monkeypatch):
    """`_supply` compares the kept set with a walk when ``MSTAR_KV_DEBUG_ASSERTS`` is on."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    kv = _manager(16)
    _cached(kv, 15)
    assert kv._supply() == 15
    kv._index._evictable.discard(next(iter(kv._index._evictable)))
    kv._index._frozen = None

    with pytest.raises(AssertionError, match="evictable pages kept by the index"):
        kv._supply()


# ── in the index ────────────────────────────────────────────────────────


def _chain(index: PrefixIndex, arena, held: bool = False) -> dict[str, int]:
    """A->B->C, each indexed and then let go by its request, unless ``held``."""
    pages = {}
    parent = None
    for name in "ABC":
        page, = arena.acquire(1)
        assert index.insert(name.encode(), page, parent)
        pages[name] = parent = page
    if not held:
        arena.release(list(pages.values()))
    return pages


def test_an_insert_under_a_page_only_the_index_holds_takes_it_out_of_the_set():
    arena = _arena()
    index = PrefixIndex(arena)
    pages = _chain(index, arena)
    walks = _Walks(index)
    assert index.evictable() == set(pages.values())

    # the new page is held by its request, so the chain above it can no longer be evicted
    page, = arena.acquire(1)
    index.insert(b"D", page, parent=pages["C"])

    assert index.evictable() == set(), "pages that a held page sits under were left in the set"
    assert walks.count == 0


def test_an_insert_under_a_page_that_is_already_held_leaves_the_set_as_it_was():
    arena = _arena()
    index = PrefixIndex(arena)
    index.insert(b"root", arena.acquire(1)[0])
    cold, = arena.acquire(1)
    index.insert(b"cold", cold)
    arena.release([cold])
    held, = arena.acquire(1)
    index.insert(b"held", held)
    walks = _Walks(index)
    kept = index.evictable()
    assert kept == {cold}

    child, = arena.acquire(1)
    index.insert(b"child", child, parent=held)
    page, = arena.acquire(1)
    index.insert(b"unparented", page)

    assert index.evictable() is kept and kept == {cold} == walks.fresh()
    assert walks.count == 0


def test_an_eviction_takes_out_what_it_evicted_and_nothing_else():
    arena = _arena()
    index = PrefixIndex(arena)
    pages = _chain(index, arena)
    other, = arena.acquire(1)
    index.insert(b"other", other)
    arena.release([other])
    walks = _Walks(index)
    assert index.evictable() == {*pages.values(), other}

    assert index.evict(2) == 2

    assert index.evictable() == walks.fresh()
    assert len(index.evictable()) == 2
    # the leaf of the chain is gone, so what is above it is now the one to go
    assert index.evict(10) == 2
    assert index.evictable() == set() == walks.fresh()
    assert walks.count == 0


def test_an_eviction_that_frees_nothing_leaves_the_set_kept():
    arena = _arena()
    index = PrefixIndex(arena)
    page, = arena.acquire(1)
    index.insert(b"k", page)
    kept = index.evictable()
    assert kept == set()

    assert index.evict(1) == 0

    assert index.evictable() is kept


def test_a_page_stays_out_of_the_set_until_the_last_page_held_below_it_is_let_go():
    arena = _arena()
    index = PrefixIndex(arena)
    root, = arena.acquire(1)
    index.insert(b"root", root)
    first, second = arena.acquire(2)
    index.insert(b"first", first, parent=root)
    index.insert(b"second", second, parent=root)
    arena.release([root, first, second])
    assert index.evictable() == {root, first, second}

    # two requests hold the one page, and a third the other: the root is out until all are gone
    arena.retain([first])
    arena.retain([first])
    arena.retain([second])
    assert index.evictable() == set()
    arena.release([first])
    assert index.evictable() == set()
    arena.release([first])
    assert index.evictable() == {first}, "a page was kept out by a request that had let it go"
    arena.release([second])
    assert index.evictable() == {root, first, second}
    assert index.evictable() == index._find_evictable()


# ── under random operations ─────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(8))
def test_the_evictable_pages_kept_are_what_walking_finds_under_random_operations(seed):
    """Requests that fill, lease, finish and are evicted from at random, over trees of any
    shape, asking for the set at random: it is the walk's, every time, and is never walked for."""
    rng = random.Random(SEED + seed)
    arena = _arena(max_num_pages=40)
    index = PrefixIndex(arena)
    walks = _Walks(index)
    requests: list[list[int]] = []
    leases: list[int] = []
    keys = iter(range(10**9))
    asked = 0

    def insert_some(pages: list[int]) -> None:
        parent = None
        for page in pages:
            if rng.random() < 0.8:
                pick = rng.random()
                if pick < 0.5 and parent is not None:
                    under = parent
                elif pick < 0.7:
                    under = rng.choice(index.pages() or [None])
                else:
                    under = None
                if index.insert(f"k{next(keys)}".encode(), page, under):
                    parent = page

    for step in range(4000):
        roll = rng.random()
        if roll < 0.25:
            pages = arena.acquire(rng.randint(1, 4))
            if pages is None:
                index.evict(4)
                continue
            insert_some(pages)
            requests.append(pages)
        elif roll < 0.45 and requests:
            arena.release(requests.pop(rng.randrange(len(requests))))
        elif roll < 0.55 and index.pages():
            held = rng.sample(index.pages(), k=min(len(index.pages()), rng.randint(1, 3)))
            arena.retain(held)
            leases.extend(held)
        elif roll < 0.65 and leases:
            arena.release([leases.pop(rng.randrange(len(leases)))])
        elif roll < 0.85:
            index.evict(rng.randint(1, 6))
        elif roll < 0.9:
            index.lookup([f"k{rng.randrange(next(keys) or 1)}".encode()])
        if rng.random() < 0.7:
            asked += 1
            assert index.evictable() == walks.fresh(), (
                f"seed {seed}, step {step}: the evictable pages kept are not what a walk finds"
            )
            assert index.evictable_count() == len(index.evictable())
            some = rng.sample(sorted(index.evictable()), k=min(3, len(index.evictable())))
            assert index.evictable_count(some) == len(index.evictable()) - len(some)
    assert asked > 2000
    assert walks.count == 0
