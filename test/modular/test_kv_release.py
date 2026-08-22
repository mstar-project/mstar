"""Tests for partial KV release on a live request: the
``PagedAllocationManager.protect_prefix`` / ``release_oldest`` primitives and
their offload round-trip.

The load-bearing invariant throughout: ``page_indices`` stays a contiguous
logical stream — ``len(page_indices) == ceil(seq_len / page_size)`` — because
release removes whole pages from the front of the unprotected region and
drops ``seq_len`` by exactly the freed token count. The planner and the
retrieve path both index pages as ``token // page_size``, so any drift here
corrupts attention.
"""

from __future__ import annotations

import random
import sys
import threading

sys.path.insert(0, ".")

from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from mstar.engine.cpu_page_pool import OffloadedState
from mstar.engine.kv_store import (
    KVCacheConfig,
    PageAllocator,
    PagedAllocationManager,
    StoreWritePolicy,
)

PS = 8  # page_size used across these tests


def _make_manager(max_num_pages: int = 64) -> PagedAllocationManager:
    """Build a ``PagedAllocationManager`` bypassing ``__init__``'s
    transfer-engine setup (requires CUDA); only fields the tested paths
    touch are populated."""
    manager = PagedAllocationManager.__new__(PagedAllocationManager)
    manager.config = KVCacheConfig(
        num_layers=1,
        num_kv_heads=1,
        head_dim=1,
        max_seq_len=max_num_pages * PS,
        max_num_pages=max_num_pages,
        page_size=PS,
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


def _commit(manager, rid, label, seq_len):
    """Emulate what a prefill/commit pass does: grow pages, advance seq_len."""
    manager.alloc(rid, label, seq_len)
    manager.request_states[rid][label].seq_len = seq_len


def _assert_stream_coherent(state):
    assert len(state.page_indices) == -(-state.seq_len // PS), (
        f"page list {len(state.page_indices)} != ceil({state.seq_len}/{PS})"
    )


class TestReleaseOldest:
    def test_basic_release_compacts_front(self):
        m = _make_manager()
        m.add_request("r", ["main"])
        _commit(m, "r", "main", 10 * PS)
        st = m.request_states["r"]["main"]
        original = list(st.page_indices)

        m.protect_prefix("r", "main", 2 * PS)
        freed = m.release_oldest("r", "main", 3 * PS)

        assert freed == 3 * PS
        assert st.page_indices == original[:2] + original[5:]
        assert st.seq_len == 7 * PS
        assert st.released_tokens == 3 * PS
        assert st.prefix_epoch == 1
        _assert_stream_coherent(st)
        # Freed pages went back to the pool.
        assert m.page_allocator.num_free == m.config.max_num_pages - 7

    def test_release_floors_to_whole_pages(self):
        m = _make_manager()
        m.add_request("r", ["main"])
        _commit(m, "r", "main", 10 * PS)
        m.protect_prefix("r", "main", PS)

        assert m.release_oldest("r", "main", PS - 1) == 0
        assert m.release_oldest("r", "main", 2 * PS + 3) == 2 * PS

    def test_protection_boundary_page_never_freed(self):
        # Protect 1.5 pages: the straddling page (index 1) must survive even
        # though its tail tokens are unprotected.
        m = _make_manager()
        m.add_request("r", ["main"])
        _commit(m, "r", "main", 6 * PS)
        st = m.request_states["r"]["main"]
        original = list(st.page_indices)

        m.protect_prefix("r", "main", PS + PS // 2)
        freed = m.release_oldest("r", "main", 100 * PS)

        # Releasable pages start at page 2; the partially-written tail rule
        # doesn't bite here (seq_len is page-aligned), so pages 2..5 free.
        assert freed == 4 * PS
        assert st.page_indices == original[:2]
        _assert_stream_coherent(st)

    def test_partial_tail_page_never_freed(self):
        m = _make_manager()
        m.add_request("r", ["main"])
        _commit(m, "r", "main", 3 * PS + 1)  # 4 pages, last holds 1 token
        st = m.request_states["r"]["main"]
        original = list(st.page_indices)

        freed = m.release_oldest("r", "main", 100 * PS)

        assert freed == 3 * PS
        assert st.page_indices == [original[3]]
        assert st.seq_len == 1
        _assert_stream_coherent(st)

    def test_release_without_protection_frees_from_front(self):
        m = _make_manager()
        m.add_request("r", ["main"])
        _commit(m, "r", "main", 4 * PS)
        st = m.request_states["r"]["main"]
        original = list(st.page_indices)

        assert m.release_oldest("r", "main", 2 * PS) == 2 * PS
        assert st.page_indices == original[2:]

    def test_release_invalidates_dense_prefix_and_bumps_epoch(self):
        m = _make_manager()
        m.add_request("r", ["main"])
        _commit(m, "r", "main", 4 * PS)
        st = m.request_states["r"]["main"]
        st.dense_prefix_kv = {0: "sentinel"}

        m.release_oldest("r", "main", PS)
        assert st.dense_prefix_kv is None
        assert st.prefix_epoch == 1
        # A no-op release must not bump the epoch or touch the cache field.
        st.dense_prefix_kv = {0: "sentinel"}
        assert m.release_oldest("r", "main", PS - 1) == 0
        assert st.dense_prefix_kv == {0: "sentinel"}
        assert st.prefix_epoch == 1

    def test_position_id_start_untouched(self):
        m = _make_manager()
        m.add_request("r", ["main"])
        _commit(m, "r", "main", 6 * PS)
        st = m.request_states["r"]["main"]
        st.position_id_start = 123
        m.release_oldest("r", "main", 2 * PS)
        assert st.position_id_start == 123

    def test_grow_after_release_stays_coherent(self):
        m = _make_manager()
        m.add_request("r", ["main"])
        _commit(m, "r", "main", 6 * PS)
        m.protect_prefix("r", "main", PS)
        m.release_oldest("r", "main", 3 * PS)

        # Next window commits: alloc sees the compacted stream and grows it.
        st = m.request_states["r"]["main"]
        _commit(m, "r", "main", st.seq_len + 2 * PS + 3)
        _assert_stream_coherent(st)

    def test_remove_request_after_partial_release_conserves_pages(self):
        m = _make_manager()
        m.add_request("r", ["main", "uncond"])
        _commit(m, "r", "main", 10 * PS)
        _commit(m, "r", "uncond", 7 * PS)
        m.protect_prefix("r", "main", PS)
        m.release_oldest("r", "main", 4 * PS)
        m.remove_request("r")
        assert m.page_allocator.num_free == m.config.max_num_pages


class TestProtectPrefix:
    def test_protect_validations(self):
        m = _make_manager()
        m.add_request("r", ["main"])
        _commit(m, "r", "main", 4 * PS)

        with pytest.raises(ValueError, match="outside committed seq_len"):
            m.protect_prefix("r", "main", 5 * PS)

        m.protect_prefix("r", "main", 2 * PS)
        m.protect_prefix("r", "main", 2 * PS)  # idempotent at same value
        with pytest.raises(ValueError, match="already"):
            m.protect_prefix("r", "main", 3 * PS)

        m.release_oldest("r", "main", PS)
        fresh = _make_manager()
        fresh.add_request("r", ["main"])
        _commit(fresh, "r", "main", 4 * PS)
        fresh.release_oldest("r", "main", PS)
        with pytest.raises(ValueError, match="precede"):
            fresh.protect_prefix("r", "main", PS)

    def test_fully_protected_stream_releases_nothing(self):
        m = _make_manager()
        m.add_request("r", ["main"])
        _commit(m, "r", "main", 3 * PS)
        m.protect_prefix("r", "main", 3 * PS)
        assert m.release_oldest("r", "main", 100 * PS) == 0


class TestReleaseRandomized:
    def test_random_grow_release_sequences_hold_invariants(self):
        rng = random.Random(0)
        for trial in range(50):
            m = _make_manager(max_num_pages=128)
            m.add_request("r", ["main"])
            protect = rng.randrange(0, 3 * PS)
            _commit(m, "r", "main", max(protect, rng.randrange(1, 6 * PS)))
            if protect:
                m.protect_prefix("r", "main", protect)
            st = m.request_states["r"]["main"]

            for _ in range(rng.randrange(3, 12)):
                if rng.random() < 0.5:
                    grow = rng.randrange(1, 4 * PS)
                    _commit(m, "r", "main", st.seq_len + grow)
                else:
                    before = st.seq_len
                    freed = m.release_oldest(
                        "r", "main", rng.randrange(0, 6 * PS)
                    )
                    assert freed % PS == 0
                    assert st.seq_len == before - freed
                    # Protected tokens always survive.
                    assert st.seq_len >= st.protected_prefix_tokens
                _assert_stream_coherent(st)
                assert len(set(st.page_indices)) == len(st.page_indices)

            m.remove_request("r")
            assert m.page_allocator.num_free == m.config.max_num_pages, (
                f"trial {trial} leaked pages"
            )


class _StubCpuPool:
    """Duck-typed stand-in for CPUPagePool: records offloads, returns the
    stored ``OffloadedState`` on reload. Lets the kv_store round-trip run
    without CUDA (the real pool pins memory and uses CUDA streams)."""

    def __init__(self):
        self.offloaded = {}

    def offload_pages(
        self, request_id, label, gpu_kv_cache, gpu_page_indices, seq_len,
        position_id_start, protected_prefix_tokens=0, released_tokens=0,
        prefix_epoch=0,
    ):
        self.offloaded.setdefault(request_id, {})[label] = OffloadedState(
            cpu_page_indices=list(range(len(gpu_page_indices))),
            seq_len=seq_len,
            position_id_start=position_id_start,
            protected_prefix_tokens=protected_prefix_tokens,
            released_tokens=released_tokens,
            prefix_epoch=prefix_epoch,
        )

    def reload_pages(self, request_id, label, gpu_kv_cache, gpu_page_indices):
        state = self.offloaded[request_id][label]
        del self.offloaded[request_id][label]
        if not self.offloaded[request_id]:
            del self.offloaded[request_id]
        return state

    def sync(self):
        pass


class TestOffloadRoundTrip:
    def test_release_state_survives_offload_reload(self):
        m = _make_manager()
        m.add_request("r", ["main"])
        _commit(m, "r", "main", 8 * PS)
        m.protect_prefix("r", "main", 2 * PS)
        m.release_oldest("r", "main", 3 * PS)
        st = m.request_states["r"]["main"]
        st.position_id_start = 42
        expect = (
            st.seq_len, st.position_id_start, st.protected_prefix_tokens,
            st.released_tokens, st.prefix_epoch,
        )

        pool = _StubCpuPool()
        freed = m.offload_request("r", pool)
        assert freed == 5
        assert st.page_indices == [] and st.seq_len == 0

        # Even if the live state object were replaced while offloaded, the
        # pool record restores the full stream bookkeeping.
        m.request_states["r"]["main"] = m._new_state()

        m.reload_request("r", pool)
        st = m.request_states["r"]["main"]
        assert (
            st.seq_len, st.position_id_start, st.protected_prefix_tokens,
            st.released_tokens, st.prefix_epoch,
        ) == expect
        _assert_stream_coherent(st)

        # Lifecycle continues where it left off after reload.
        assert m.release_oldest("r", "main", PS) == PS


class TestReleaseThreadSafety:
    def test_concurrent_grow_release_conserves_pages(self):
        m = _make_manager(max_num_pages=128)
        rid, label = "r", "main"
        m.add_request(rid, [label])
        _commit(m, rid, label, 4 * PS)
        m.protect_prefix(rid, label, PS)

        n_iters = 300

        def grow_worker():
            for _ in range(n_iters):
                try:
                    with m._lock:
                        st = m.request_states[rid][label]
                        _commit(m, rid, label, st.seq_len + PS)
                except (KeyError, RuntimeError):
                    pass

        def release_worker():
            for _ in range(n_iters):
                try:
                    m.release_oldest(rid, label, 2 * PS)
                except KeyError:
                    pass

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [
                ex.submit(grow_worker),
                ex.submit(grow_worker),
                ex.submit(release_worker),
                ex.submit(release_worker),
            ]
            for f in as_completed(futures):
                f.result()

        st = m.request_states[rid][label]
        _assert_stream_coherent(st)
        assert len(set(st.page_indices)) == len(st.page_indices)
        m.remove_request(rid)
        assert m.page_allocator.num_free == m.config.max_num_pages


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
