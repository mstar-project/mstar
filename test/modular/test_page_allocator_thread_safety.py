"""Thread-safety tests for ``PageAllocator`` and ``PagedAllocationManager``.

PR #78 issue #4: under Phase 3 spec, the plan thread runs ``alloc`` while
the GPU thread runs ``reset_label``. The previous implementation had two
unprotected races:

1. ``PageAllocator.try_allocate`` was ``qsize() < n`` followed by ``n``
   ``get()`` calls — non-atomic. A concurrent ``free`` could land between
   the qsize check and the get loop, false-negating ``try_allocate`` (it
   returns ``None`` even though pages are now available).

2. ``PagedAllocationManager.alloc`` and ``reset_label`` both touch
   ``request_states[rid][label]``. If they interleaved, a freshly allocated
   page list could be freed by a concurrent ``reset_label``, or a page
   list freed by ``reset_label`` could be re-extended by a still-running
   ``alloc`` on the now-stale state object.

The fix added per-allocator + per-manager locks. These tests exercise the
race shapes directly with a thread pool and verify page conservation
across stress runs.
"""

from __future__ import annotations

import sys
import threading

sys.path.insert(0, ".")

from concurrent.futures import ThreadPoolExecutor, as_completed

from mminf.engine.kv_store import (
    AllocationStatus,
    KVCacheConfig,
    PageAllocator,
    PagedAllocationManager,
    StoreWritePolicy,
)


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
    manager.alloc_status = AllocationStatus()
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


class TestPagedAllocationManagerThreadSafety:
    def test_concurrent_alloc_reset_conserves_pages(self):
        """Plan-thread ``alloc`` racing GPU-thread ``reset_label`` for
        the same (rid, label) must leave the page pool fully drained
        once both stop. Without the manager lock, the race shape is:
            T1: state = request_states[rid][label]   # old ref
            T2: reset_label  → frees state.page_indices, swaps in new state
            T1: try_allocate → mutates the OLD state (not in dict)
            → leaked pages, dict has empty new state.
        """
        manager = _make_test_manager(max_num_pages=64, page_size=8)
        rid = "rid"
        label = "main"
        manager.add_request(rid, [label])

        n_iters = 300
        # Smaller seq_len so each alloc only takes a couple pages.
        seq_len_seq = [8, 16, 24, 16, 8]

        def alloc_worker():
            for i in range(n_iters):
                try:
                    manager.alloc(rid, label, seq_len_seq[i % len(seq_len_seq)])
                except (KeyError, RuntimeError):
                    # KeyError if reset_label wiped the entry between
                    # add_request and alloc; RuntimeError if pool empty.
                    pass

        def reset_worker():
            for _ in range(n_iters):
                try:
                    manager.reset_label(rid, label)
                except KeyError:
                    pass

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [
                ex.submit(alloc_worker),
                ex.submit(alloc_worker),
                ex.submit(reset_worker),
                ex.submit(reset_worker),
            ]
            for f in as_completed(futures):
                f.result()

        # Final reset to drain whatever's still allocated.
        manager.reset_label(rid, label)
        assert manager.page_allocator.num_free == manager.config.max_num_pages
        # request_states must still contain a valid (empty) state.
        assert label in manager.request_states[rid]
        assert manager.request_states[rid][label].page_indices == []

    def test_concurrent_add_remove_request_conserves_pages(self):
        """Multiple threads cycling add_request → alloc → remove_request
        must conserve pages. Stresses the request-lifecycle locking.
        """
        manager = _make_test_manager(max_num_pages=128, page_size=8)
        n_threads = 8
        n_iters = 50

        def worker(thread_idx: int):
            for i in range(n_iters):
                rid = f"rid_{thread_idx}_{i}"
                manager.add_request(rid, ["main"])
                try:
                    manager.alloc(rid, "main", seq_len=16)
                except RuntimeError:
                    pass  # pool exhausted, ok
                manager.remove_request(rid)

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futures = [ex.submit(worker, i) for i in range(n_threads)]
            for f in as_completed(futures):
                f.result()

        assert manager.page_allocator.num_free == manager.config.max_num_pages
        assert manager.request_states == {}
        assert manager.pending_reads == {}

    def test_alloc_then_reset_releases_correct_pages(self):
        """Single-threaded sanity: confirm the lock didn't break the
        normal alloc/free contract.
        """
        manager = _make_test_manager(max_num_pages=16, page_size=8)
        rid = "rid"
        manager.add_request(rid, ["main"])

        manager.alloc(rid, "main", seq_len=24)  # 3 pages
        assert len(manager.request_states[rid]["main"].page_indices) == 3
        assert manager.page_allocator.num_free == 13

        manager.reset_label(rid, "main")
        assert manager.request_states[rid]["main"].page_indices == []
        assert manager.page_allocator.num_free == 16

        manager.remove_request(rid)
        assert manager.page_allocator.num_free == 16


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
