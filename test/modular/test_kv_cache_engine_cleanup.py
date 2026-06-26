"""Regression tests for ``KVCacheEngine.remove_request`` skipping the submodule
cleanup hook.

``KVCacheEngine.remove_request`` tears down cache / sampler / cuda-graph-runner
state when a request completes, but it never invoked the submodule's own
``cleanup_request`` hook -- unlike ``StatelessEngine.remove_request``, which does.
Any KV_CACHE-engine submodule that holds per-request state in ``cleanup_request``
therefore leaked it: per-request bookkeeping sets/dicts grew without bound, and a
submodule managing a *bounded* resource pool (a fixed set of decode slots) drained
its pool and then silently reused one slot across concurrent requests -- i.e.
per-request state bleed.

These tests pin the contract: ``remove_request`` MUST call
``submodule.cleanup_request`` for every managed submodule, and a pooled submodule
must therefore reclaim its resource and serve unbounded requests without falling
back to a shared slot.
"""
from unittest.mock import MagicMock

from mstar.engine.kv_cache_engine import KVCacheEngine, SubmoduleManagement


def _mgmt(submodule):
    """A SubmoduleManagement with mocked infra and the given submodule. Uses the
    real dataclass so a field rename breaks the test loudly."""
    kv = MagicMock(name="kv_management")
    kv.cpu_page_pool = None  # exercise the None branch (skip cpu pool teardown)
    return SubmoduleManagement(
        submodule=submodule,
        kv_management=kv,
        tp_group=MagicMock(name="tp_group"),
        default_sampling_config=MagicMock(name="sampling_config"),
        sampler=MagicMock(name="sampler"),
        cuda_graph_runner=None,
    )


def _engine(mgmt_map):
    engine = object.__new__(KVCacheEngine)  # skip heavy __init__; remove_request
    engine.submodule_management = mgmt_map  # only touches submodule_management
    return engine


def test_remove_request_invokes_submodule_cleanup():
    """The contract: remove_request must call cleanup_request(rid) on EVERY managed
    submodule, alongside the cache / sampler teardown that already happened."""
    a, b = MagicMock(name="submodule_a"), MagicMock(name="submodule_b")
    engine = _engine({"n0": _mgmt(a), "n1": _mgmt(b)})

    engine.remove_request("req-1")

    a.cleanup_request.assert_called_once_with("req-1")
    b.cleanup_request.assert_called_once_with("req-1")
    # the teardown that already worked must still fire
    engine.submodule_management["n0"].sampler.remove_request.assert_called_once_with("req-1")
    engine.submodule_management["n0"].kv_management.alloc_manager.remove_request.assert_called_once_with("req-1")


class _PooledSubmodule:
    """Stand-in for a KV_CACHE submodule that owns a BOUNDED per-request resource
    pool -- the pattern that makes the missing cleanup catastrophic. Each request
    takes a slot at prepare time and returns it in ``cleanup_request``; on
    exhaustion it falls back to a shared slot 0 (so two live requests would then
    share one slot's state)."""

    def __init__(self, size):
        self.free = list(range(size))
        self.assigned = {}
        self.exhaustion_fallbacks = 0

    def acquire(self, rid):
        if self.free:
            slot = self.free.pop()
        else:
            slot = 0
            self.exhaustion_fallbacks += 1
        self.assigned[rid] = slot
        return slot

    def cleanup_request(self, rid):
        if rid in self.assigned:
            self.free.append(self.assigned.pop(rid))


def test_pooled_submodule_reclaims_slots_through_remove_request():
    """Consequence: serve many more requests than the pool holds, going through
    engine.remove_request each time. The cleanup call must reclaim the slot so the
    pool never drains; without it the pool empties and every later request collapses
    onto the shared slot 0 (state bleed at concurrency)."""
    POOL = 4
    sub = _PooledSubmodule(POOL)
    engine = _engine({"decoder": _mgmt(sub)})

    for i in range(POOL * 5):              # 20 requests through a 4-slot pool
        rid = f"req-{i}"
        sub.acquire(rid)                   # prepare
        engine.remove_request(rid)         # complete -> must reclaim the slot

    assert sub.exhaustion_fallbacks == 0, \
        "pool drained -> requests fell back to the shared slot 0 (cleanup not called)"
    assert not sub.assigned, "no per-request state should linger after completion"
    assert sorted(sub.free) == list(range(POOL)), "pool must be fully reclaimed"
