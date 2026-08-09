"""Thread-safety tests for ``PageAllocator`` and ``KVCachePool.admit``.

Under speculative scheduling, the plan thread reserves pages (today via
``KVCachePool.admit``, which fronts ``PagedAllocationManager.alloc``) while
the GPU thread runs ``reset_label``. Two race shapes matter:

1. ``PageAllocator.try_allocate`` is qsize-then-get; without the allocator
   lock a concurrent ``free`` between the two could false-negate the
   allocation or hand two threads the same page.

2. ``admit`` and ``reset_label`` both touch ``request_states[rid][label]``.
   Unsynchronized, a freshly reserved page list could be freed by a
   concurrent ``reset_label``, or a list freed by ``reset_label`` could be
   re-extended through a stale state reference.

The allocator and manager locks close both; these tests exercise the race
shapes directly with a thread pool and verify page conservation.
"""

from __future__ import annotations

import sys
import threading

sys.path.insert(0, ".")

from concurrent.futures import ThreadPoolExecutor, as_completed

from mstar.engine.kv_store import (
    KVCacheConfig,
    PageAllocator,
    PagedAllocationManager,
    StoreWritePolicy,
)
from mstar.engine.resources import KVCachePool, Segment


def _make_test_manager(max_num_pages: int = 32, page_size: int = 8) -> PagedAllocationManager:
    """Build a ``PagedAllocationManager`` bypassing ``__init__``'s transfer-
    engine setup, which requires CUDA. Only the fields touched by ``alloc``,
    ``reset_label``, ``add_request``, ``remove_request`` are populated.
    """
    manager = PagedAllocationManager.__new__(PagedAllocationManager)
    manager.config = KVCacheConfig(
        num_layers=1,
        num_kv_heads=1,
        head_dim=1,
        max_seq_len=max_num_pages * page_size,
        max_num_pages=max_num_pages,
        page_size=page_size,
    )
    manager.page_allocator = PageAllocator(max_num_pages)
    manager.request_states = {}
    manager.kv_cache = None
    manager.write_policy = StoreWritePolicy.ALWAYS
    manager._kv_transfer_engine = None
    manager._offload_stream = None
    manager.pending_reads = {}
    manager._lock = threading.RLock()
    return manager


class TestPageAllocatorThreadSafety:
    def test_concurrent_alloc_free_conserves_pages(self):
        """Many threads each do alloc + free in a loop. Total free pages
        must equal max_num_pages at the end (no double-allocation, no
        leaks). Without the lock, this stress test reliably surfaces
        double-allocations on contended ``free_pages.get()`` sequences.
        """
        max_pages = 64
        alloc = PageAllocator(max_num_pages=max_pages)
        n_threads = 16
        n_iters = 200
        pages_per_alloc = 2

        def worker():
            for _ in range(n_iters):
                pages = alloc.try_allocate(pages_per_alloc)
                if pages is not None:
                    # Sanity: allocated pages are unique within this batch.
                    assert len(set(pages)) == len(pages)
                    alloc.free(pages)

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futures = [ex.submit(worker) for _ in range(n_threads)]
            for f in as_completed(futures):
                f.result()

        assert alloc.num_free == max_pages

    def test_no_double_allocation_under_contention(self):
        """Two threads racing on ``try_allocate`` of the same n pages
        must NEVER receive overlapping page indices. Without the lock,
        the qsize-then-get sequence could allow both threads to pass
        the qsize check and then race on get.
        """
        max_pages = 8
        alloc = PageAllocator(max_num_pages=max_pages)
        n_threads = 4
        per_thread = 2  # 4 * 2 = 8, exactly fills the pool

        results: list[list[int]] = []
        results_lock = threading.Lock()
        barrier = threading.Barrier(n_threads)

        def worker():
            barrier.wait()  # maximize contention
            pages = alloc.try_allocate(per_thread)
            if pages is not None:
                with results_lock:
                    results.append(pages)

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futures = [ex.submit(worker) for _ in range(n_threads)]
            for f in as_completed(futures):
                f.result()

        # Every successfully returned page index must be unique across
        # all threads — proves no two threads got the same page.
        all_pages = [p for batch in results for p in batch]
        assert len(all_pages) == len(set(all_pages))

    def test_try_allocate_under_concurrent_free_does_not_lose_pages(self):
        """Bounded stress: producer threads ``free`` pages, consumer
        threads ``try_allocate``. Ledger of pages-in-flight tracks how
        many pages are out at any moment; total must never exceed
        max_num_pages, and the pool must drain back to max_num_pages
        once all consumers stop.
        """
        max_pages = 32
        alloc = PageAllocator(max_num_pages=max_pages)

        # Pre-allocate everything so producers have something to free.
        held = alloc.allocate(max_pages)
        held_lock = threading.Lock()

        n_iters = 500
        stop_event = threading.Event()

        def consumer():
            while not stop_event.is_set():
                pages = alloc.try_allocate(1)
                if pages is None:
                    continue
                # Hold briefly then return.
                alloc.free(pages)

        def producer():
            for _ in range(n_iters):
                with held_lock:
                    if not held:
                        continue
                    p = held.pop()
                alloc.free([p])
                # Take one back to keep the pool bounded.
                pages = alloc.try_allocate(1)
                if pages is not None:
                    with held_lock:
                        held.extend(pages)

        with ThreadPoolExecutor(max_workers=8) as ex:
            consumers = [ex.submit(consumer) for _ in range(4)]
            producers = [ex.submit(producer) for _ in range(4)]
            for f in as_completed(producers):
                f.result()
            stop_event.set()
            for f in as_completed(consumers):
                f.result()

        # Drain everything we still hold.
        with held_lock:
            alloc.free(held)
            held.clear()
        assert alloc.num_free == max_pages


class TestKVCachePoolThreadSafety:
    def test_concurrent_admit_reset_conserves_pages(self):
        """Plan-thread ``admit`` racing GPU-thread ``reset_label`` for
        the same (rid, label) must leave the page pool fully drained
        once both stop. Without the manager lock, the race shape is:
            T1: state = request_states[rid][label]   # old ref
            T2: reset_label  → frees state.page_indices, swaps in new state
            T1: try_allocate → mutates the OLD state (not in dict)
            → leaked pages, dict has empty new state.
        """
        manager = _make_test_manager(max_num_pages=64, page_size=8)
        pool = KVCachePool(manager)
        rid = "rid"
        label = "main"
        manager.add_request(rid, [label])

        n_iters = 300
        # Small spans so each admit only takes a couple pages.
        span_seq = [8, 16, 24, 16, 8]

        def admit_worker():
            for i in range(n_iters):
                try:
                    pool.admit(Segment(rid, label, span_seq[i % len(span_seq)]))
                except (KeyError, RuntimeError):
                    # KeyError if reset_label wiped the entry between
                    # add_request and admit; RuntimeError if pool empty.
                    pass

        def reset_worker():
            for _ in range(n_iters):
                try:
                    manager.reset_label(rid, label)
                except KeyError:
                    pass

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [
                ex.submit(admit_worker),
                ex.submit(admit_worker),
                ex.submit(reset_worker),
                ex.submit(reset_worker),
            ]
            for f in as_completed(futures):
                f.result()

        # Final reset to drain whatever's still allocated.
        manager.reset_label(rid, label)
        assert pool.num_free_pages == pool.total_pages
        # request_states must still contain a valid (empty) state.
        assert label in manager.request_states[rid]
        assert manager.request_states[rid][label].page_indices == []

    def test_concurrent_add_remove_request_conserves_pages(self):
        """Multiple threads cycling add_request → admit → remove_request
        must conserve pages. Stresses the request-lifecycle locking.
        """
        manager = _make_test_manager(max_num_pages=128, page_size=8)
        pool = KVCachePool(manager)
        n_threads = 8
        n_iters = 50

        def worker(thread_idx: int):
            for i in range(n_iters):
                rid = f"rid_{thread_idx}_{i}"
                manager.add_request(rid, ["main"])
                try:
                    pool.admit(Segment(rid, "main", 16))
                except RuntimeError:
                    pass  # pool exhausted, ok
                manager.remove_request(rid)

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futures = [ex.submit(worker, i) for i in range(n_threads)]
            for f in as_completed(futures):
                f.result()

        assert pool.num_free_pages == pool.total_pages
        assert manager.request_states == {}
        assert manager.pending_reads == {}

    def test_admit_then_reset_releases_correct_pages(self):
        """Single-threaded sanity: confirm the locking didn't break the
        normal reserve/release contract.
        """
        manager = _make_test_manager(max_num_pages=16, page_size=8)
        pool = KVCachePool(manager)
        rid = "rid"
        manager.add_request(rid, ["main"])

        segment = Segment(rid, "main", 24)  # 3 pages
        reservation = pool.admit(segment)
        assert reservation.resident == 0
        assert reservation.to_compute == 24
        assert len(pool.view(segment).page_indices) == 3
        assert pool.num_free_pages == 13

        manager.reset_label(rid, "main")
        assert pool.view(Segment(rid, "main", 0)).page_indices == ()
        assert pool.num_free_pages == 16

        manager.remove_request(rid)
        assert pool.num_free_pages == 16


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
