"""CPU ownership sequences checked against logical owners, not arena counters."""

from __future__ import annotations

import random
from collections import Counter, deque

import pytest
import torch

from mstar.engine.resources.kv.cache import KVCache, PageAllocator
from mstar.engine.resources.kv.config import PagedKVConfig
from mstar.engine.resources.kv.manager import PageArena


@pytest.mark.parametrize(
    ("seed", "capacity"),
    [(0, 4), (1, 7), (20260920, 11), (20261017, 17)],
    ids=["seed-0-pool-4", "seed-1-pool-7", "seed-20260920-pool-11", "seed-20261017-pool-17"],
)
def test_random_ownership_matches_logical_owners(seed: int, capacity: int):
    cfg = PagedKVConfig(
        num_layers=1, num_kv_heads=1, head_dim=2, max_seq_len=64,
        max_num_pages=capacity, page_size=4, cpu_offload_pages=0,
    )
    arena = PageArena(
        kv_cache=KVCache(cfg, torch.device("cpu"), torch.float32),
        allocator=PageAllocator(capacity),
    )
    rng = random.Random(seed)
    all_pages = set(range(capacity))
    owners: dict[int, set[int]] = {}
    sealed: set[int] = set()
    history: deque[str] = deque(maxlen=32)
    operation_index = 0

    def references() -> Counter:
        # Owner memberships are the oracle; never read the arena to update it.
        return Counter(page for pages in owners.values() for page in pages)

    def check():
        expected = references()
        free = list(arena.allocator.free_pages.queue)
        assert len(free) == len(set(free)), f"duplicate free pages: {free}"
        assert set(free) == all_pages - expected.keys(), "free page membership differs"
        assert arena.num_free == capacity - len(expected), "free page count differs"
        assert arena.num_owners == [expected[page] for page in range(capacity)], (
            "reference counts differ from logical owners"
        )
        assert arena.sealed == [page in sealed for page in range(capacity)], (
            "sealed pages differ from the model"
        )
        assert sealed <= expected.keys(), "the model left a free page sealed"
        assert all(count == 1 or page in sealed for page, count in expected.items()), (
            "the model shared an unsealed page"
        )
        assert arena.any_sealed([]) is False
        assert arena.any_sealed(sorted(all_pages)) == bool(sealed)
        for page in all_pages:
            assert arena.any_sealed([page]) == (page in sealed)

    def apply(operation: str, owner: int = -1, pages=(), count: int = 0):
        nonlocal operation_index
        operation_index += 1
        pages = list(pages)
        history.append(
            f"{operation_index}: {operation} owner={owner} pages={pages} count={count}"
        )
        try:
            if operation == "acquire":
                assert owner not in owners
                free_before = all_pages - references().keys()
                queue_before = list(arena.allocator.free_pages.queue)
                got = arena.acquire(count)
                history[-1] += f" -> {got}"
                if count > len(free_before):
                    assert got is None, "an over-capacity allocation succeeded"
                    assert list(arena.allocator.free_pages.queue) == queue_before, (
                        "a failed allocation changed the free queue"
                    )
                else:
                    assert got is not None and len(got) == count
                    assert len(set(got)) == count, "an allocation repeated a page"
                    assert set(got) <= free_before, "an allocation took an owned page"
                    if got:
                        owners[owner] = set(got)
            elif operation == "seal":
                assert set(pages) <= references().keys()
                arena.seal(pages)
                sealed.update(pages)
            elif operation == "retain":
                assert owner not in owners and set(pages) <= sealed
                arena.retain(pages)
                if pages:
                    owners[owner] = set(pages)
            elif operation == "release":
                assert set(pages) <= owners[owner]
                arena.release(pages)
                owners[owner].difference_update(pages)
                if not owners[owner]:
                    del owners[owner]
                sealed.intersection_update(references())
            else:
                raise AssertionError(f"unknown operation: {operation}")
            check()
        except Exception as exc:
            exc.add_note(
                f"PageArena seed={seed}, capacity={capacity}, operation={operation_index}\n"
                "Last operations (recorded before execution):\n" + "\n".join(history)
            )
            raise

    # Force sharing, exhaustion, mixed final/non-final releases and reuse;
    # these boundaries must run even when a random sequence avoids them.
    apply("acquire", owner=0, count=3)
    apply("release", owner=0)
    shared, second, exclusive = sorted(owners[0])
    apply("seal", pages=[shared, second, exclusive])
    apply("retain", owner=1, pages=[shared, second])
    apply("retain", owner=2, pages=[shared])
    apply("acquire", owner=3, count=capacity - 3)
    apply("acquire", owner=4, count=1)
    apply("release", owner=0, pages=[shared, exclusive])
    apply("acquire", owner=4, count=2)
    # The sole free page was sealed: reacquiring it must give a writable page.
    apply("acquire", owner=5, count=1)
    apply("release", owner=1, pages=[shared, second])
    apply("release", owner=2, pages=[shared])
    apply("release", owner=0, pages=[second])
    apply("release", owner=3, pages=sorted(owners[3]))
    apply("release", owner=5, pages=sorted(owners[5]))
    apply("acquire", owner=6, count=0)
    apply("seal")
    apply("retain", owner=7)

    for i in range(400):
        roll = rng.randrange(10)
        if not owners or roll < 3:
            apply("acquire", owner=10 + i, count=rng.randint(1, capacity + 1))
        elif roll < 6:
            owner = rng.choice(sorted(owners))
            pages = rng.sample(sorted(owners[owner]), rng.randint(1, len(owners[owner])))
            apply("release", owner=owner, pages=pages)
        elif roll < 8 or not sealed:
            owned = sorted(references())
            pages = rng.sample(owned, rng.randint(1, len(owned)))
            apply("seal", pages=pages)
        else:
            pages = rng.sample(sorted(sealed), rng.randint(1, len(sealed)))
            apply("retain", owner=10 + i, pages=pages)

    for owner in sorted(owners):
        apply("release", owner=owner, pages=sorted(owners[owner]))
    # Exact full-pool reuse also checks that final releases neither lose nor
    # duplicate a page and that no previous lifetime's seal survives.
    apply("acquire", owner=1000, count=capacity)
    apply("release", owner=1000, pages=sorted(owners[1000]))
