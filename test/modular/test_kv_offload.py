"""Unit tests for the KV cache's CPU offload/reload path.

``offload`` moves a request's streams to pinned host memory and returns their
device pages to the arena; ``reload`` brings them back. Both drop the manager
lock around the copies, so the interesting cases are the ones where another
thread touches the streams in that window — the plan thread pre-planning the
next step is the real one, and it is what the race tests here simulate.

Every test ends on ``_assert_pages_conserved``: a page is owned by exactly one
of the arena's free list, a live stream, or the sink. That single invariant is
what both a leak and a double-free break.
"""

from __future__ import annotations

import sys
import threading

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources import (
    AllocationFailed,
    RequestOffloading,
    Segment,
    StepContext,
)
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVConfig, KVStep
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.plan import SINK_PAGE

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="offload needs pinned host memory"
)

PAGE_SIZE = 8


class _StubTransferManager:
    """No engine, no bytes moved: retrieves complete immediately."""

    def __init__(self, transfer_engine_info, kv_cache):
        del transfer_engine_info, kv_cache

    def get_kv_transfer_info(self):
        return None

    def start_async_retrieve(self, **kwargs):
        # no future: nothing to wait on
        del kwargs

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransferManager)


def _make_manager(max_num_pages: int = 16, cpu_offload_pages: int = 16):
    cfg = KVConfig(
        num_layers=2,
        num_kv_heads=1,
        head_dim=4,
        max_seq_len=max_num_pages * PAGE_SIZE,
        max_num_pages=max_num_pages,
        page_size=PAGE_SIZE,
        cpu_offload_pages=cpu_offload_pages,
    )
    return KVManager(
        cfg=cfg, name="kv", joint_comm_group=None, transfer_engine_info=None,
        device=torch.device("cuda"), dtype=torch.float32,
    )


def _ctx(*rids: str) -> StepContext:
    return StepContext(
        request_ids=tuple(rids), graph_walk="walk", slot=0, capture=False
    )


def _grow(mgr: KVManager, rid: str, label: str, span: int, **step_kw):
    """Admit and commit one step extending ``label`` by ``span`` tokens."""
    step = KVStep(segments=(Segment(rid, label, span),), **step_kw)
    ctx = _ctx(rid)
    outcome = mgr.admit(step, ctx)
    if not outcome.ok:
        return outcome
    mgr.plan(step, ctx)
    mgr.commit(step, ctx)
    return outcome


def _stream(mgr: KVManager, rid: str, label: str = "main"):
    return mgr._streams[rid][label]


def _paint(mgr: KVManager, pages: list[int], base: float) -> None:
    """Give every page a distinct value, so a copy can be checked exactly."""
    for i, page in enumerate(pages):
        mgr.kv_cache.tensor[:, page] = base + i


def _snapshot(mgr: KVManager, pages: list[int]) -> torch.Tensor:
    return mgr.kv_cache.tensor[:, pages].clone()


def _scribble(mgr: KVManager) -> None:
    """Poison the whole device cache, so a reload has to do real work."""
    mgr.kv_cache.tensor.fill_(-99.0)


def _assert_pages_conserved(mgr: KVManager) -> None:
    free = list(mgr._arena.allocator.free_pages.queue)
    held = [
        page
        for streams in mgr._streams.values()
        for stream in streams.values()
        for page in stream.page_indices
    ]
    owned = free + held + [SINK_PAGE]
    duplicated = {p for p in owned if owned.count(p) > 1}
    assert not duplicated, f"pages owned twice (double free): {sorted(duplicated)}"
    missing = set(range(mgr.config.max_num_pages)) - set(owned)
    assert not missing, f"pages orphaned (leaked): {sorted(missing)}"


def _hook_offload_stream(mgr: KVManager, hook, when: str | None = None):
    """Run ``hook`` on another thread from inside a host copy.

    The manager lock is not held there, so this reproduces exactly what the
    plan thread can do mid-offload. Joining with a timeout means a regression
    that takes the lock across the copy fails the test instead of hanging it.
    """
    real = mgr._cpu_pool.offload_stream
    fired = False

    def wrapper(**kwargs):
        nonlocal fired
        moved = real(**kwargs)
        # once only, so a test can retry the offload without racing it again
        if not fired and (when is None or kwargs["label"] == when):
            fired = True
            thread = threading.Thread(target=hook)
            thread.start()
            thread.join(timeout=10)
            assert not thread.is_alive(), (
                "the racing thread blocked: offload is holding the manager "
                "lock across the host copy"
            )
        return moved

    mgr._cpu_pool.offload_stream = wrapper


# --- round trip ------------------------------------------------------------


@requires_cuda
def test_offload_reload_round_trip():
    mgr = _make_manager()
    mgr.ingest_request("r0")
    _grow(mgr, "r0", "main", 3 * PAGE_SIZE)

    stream = _stream(mgr, "r0")
    pages = list(stream.page_indices)
    assert len(pages) == 3
    _paint(mgr, pages, base=1.0)
    expected = _snapshot(mgr, pages)
    free_before = mgr._arena.num_free

    assert mgr.offload("r0") == 3
    assert mgr.is_offloaded("r0")
    assert mgr.reclaimable("r0") == 0
    assert mgr._arena.num_free == free_before + 3
    _assert_pages_conserved(mgr)

    # nothing may be read back off the device pages we just gave up
    _scribble(mgr)

    assert mgr.reload("r0") is True
    assert not mgr.is_offloaded("r0")
    stream = _stream(mgr, "r0")
    assert stream.stored_len == 3 * PAGE_SIZE
    assert len(stream.page_indices) == 3
    torch.testing.assert_close(_snapshot(mgr, stream.page_indices), expected)
    _assert_pages_conserved(mgr)


@requires_cuda
def test_offload_reload_restores_geometry():
    mgr = _make_manager()
    mgr.ingest_request("r0")
    _grow(mgr, "r0", "main", 2 * PAGE_SIZE)
    stream = _stream(mgr, "r0")
    stream.position = 17
    stream.released = 5

    assert mgr.offload("r0") == 2
    assert mgr.reload("r0") is True

    stream = _stream(mgr, "r0")
    assert (stream.stored_len, stream.position, stream.released) == (
        2 * PAGE_SIZE, 17, 5
    )
    _assert_pages_conserved(mgr)


@requires_cuda
def test_offload_reload_multi_label():
    """The bagel CFG shape: a forked second label, both offloaded together."""
    mgr = _make_manager()
    mgr.ingest_request("r0")
    _grow(mgr, "r0", "main", 2 * PAGE_SIZE)
    _grow(mgr, "r0", "main", PAGE_SIZE, pre_forks=(("main", "cfg"),))

    _paint(mgr, _stream(mgr, "r0", "main").page_indices, base=10.0)
    _paint(mgr, _stream(mgr, "r0", "cfg").page_indices, base=50.0)
    expected = {
        label: _snapshot(mgr, _stream(mgr, "r0", label).page_indices)
        for label in ("main", "cfg")
    }
    held = mgr.reclaimable("r0")

    assert mgr.offload("r0") == held
    _scribble(mgr)
    assert mgr.reload("r0") is True

    for label, want in expected.items():
        got = _snapshot(mgr, _stream(mgr, "r0", label).page_indices)
        torch.testing.assert_close(got, want)
    _assert_pages_conserved(mgr)


@requires_cuda
def test_offload_noop_without_cpu_pool():
    mgr = _make_manager(cpu_offload_pages=0)
    mgr.ingest_request("r0")
    _grow(mgr, "r0", "main", PAGE_SIZE)

    assert mgr.supports_eviction is False
    assert mgr.offload("r0") == 0
    assert mgr.reload("r0") is False
    assert mgr.reclaimable("r0") == 1
    _assert_pages_conserved(mgr)


# --- capacity limits -------------------------------------------------------


@requires_cuda
def test_offload_partial_when_host_pool_full():
    """A stream that doesn't fit on the host keeps its device pages."""
    mgr = _make_manager(cpu_offload_pages=2)
    mgr.ingest_request("r0")
    _grow(mgr, "r0", "main", 2 * PAGE_SIZE)
    _grow(mgr, "r0", "other", 2 * PAGE_SIZE)

    # only one of the two 2-page streams fits
    assert mgr.offload("r0") == 2
    on_device = [
        label for label in ("main", "other") if _stream(mgr, "r0", label).page_indices
    ]
    assert len(on_device) == 1
    _assert_pages_conserved(mgr)

    # the stream that stayed is not left claimed: it must still be allocatable
    kept = _stream(mgr, "r0", on_device[0])
    assert kept.offloaded is False
    assert _grow(mgr, "r0", on_device[0], PAGE_SIZE).ok
    _assert_pages_conserved(mgr)


@requires_cuda
def test_failed_offload_leaves_stream_usable():
    """The host pool is full, so nothing moves — and nothing is left claimed."""
    mgr = _make_manager(cpu_offload_pages=1)
    mgr.ingest_request("r0")
    _grow(mgr, "r0", "main", 2 * PAGE_SIZE)
    held = mgr.reclaimable("r0")

    assert mgr.offload("r0") == 0
    assert mgr.is_offloaded("r0") is False
    assert _stream(mgr, "r0").offloaded is False
    assert mgr.reclaimable("r0") == held
    assert _grow(mgr, "r0", "main", PAGE_SIZE).ok
    _assert_pages_conserved(mgr)


@requires_cuda
def test_reload_declines_when_device_full():
    mgr = _make_manager(max_num_pages=8)
    mgr.ingest_request("r0")
    mgr.ingest_request("r1")
    _grow(mgr, "r0", "main", 3 * PAGE_SIZE)

    assert mgr.offload("r0") == 3
    # take everything the offload just freed
    while mgr._arena.num_free:
        assert _grow(mgr, "r1", "main", PAGE_SIZE).ok

    assert mgr.reload("r0") is False
    assert mgr.is_offloaded("r0")
    assert _stream(mgr, "r0").page_indices == []
    _assert_pages_conserved(mgr)

    # once room is returned it lands
    mgr.remove_request("r1")
    assert mgr.reload("r0") is True
    _assert_pages_conserved(mgr)


@requires_cuda
def test_remove_request_returns_host_pages():
    mgr = _make_manager(cpu_offload_pages=4)
    mgr.ingest_request("r0")
    _grow(mgr, "r0", "main", 3 * PAGE_SIZE)
    assert mgr.offload("r0") == 3
    assert mgr._cpu_pool.num_free_pages == 1

    mgr.remove_request("r0")
    assert mgr._cpu_pool.num_free_pages == 4
    assert mgr.is_offloaded("r0") is False
    _assert_pages_conserved(mgr)


# --- races against the plan thread -----------------------------------------


@requires_cuda
def test_alloc_during_offload_is_refused():
    """The regression: an admit landing mid-offload used to orphan its pages.

    ``_alloc`` extends the live ``page_indices`` the offload already snapshot,
    so those pages were neither released nor reachable afterwards.
    """
    mgr = _make_manager()
    mgr.ingest_request("r0")
    _grow(mgr, "r0", "main", 2 * PAGE_SIZE)
    raced: list = []

    _hook_offload_stream(mgr, lambda: raced.append(_grow(mgr, "r0", "main", PAGE_SIZE)))

    assert mgr.offload("r0") == 2
    outcome = raced[0]
    assert not outcome.ok
    assert isinstance(outcome.reason, RequestOffloading)
    assert outcome.reason.request_id == "r0"
    assert _stream(mgr, "r0").page_indices == []
    _assert_pages_conserved(mgr)

    # and the request still comes back intact
    assert mgr.reload("r0") is True
    assert _stream(mgr, "r0").stored_len == 2 * PAGE_SIZE
    _assert_pages_conserved(mgr)


@requires_cuda
def test_admit_refused_for_every_label_of_an_offloading_request():
    mgr = _make_manager()
    mgr.ingest_request("r0")
    _grow(mgr, "r0", "main", PAGE_SIZE)
    _grow(mgr, "r0", "other", PAGE_SIZE)
    seen: list = []

    def race():
        seen.append(_grow(mgr, "r0", "other", PAGE_SIZE))
        # a fork whose source is claimed must be refused at admit, where it
        # can still be reported
        seen.append(_grow(mgr, "r0", "main", PAGE_SIZE, pre_forks=(("main", "cfg"),)))

    _hook_offload_stream(mgr, race, when="main")

    mgr.offload("r0")
    assert all(isinstance(o.reason, RequestOffloading) for o in seen), seen
    assert "cfg" not in mgr._streams["r0"] or not mgr._streams["r0"]["cfg"].page_indices
    _assert_pages_conserved(mgr)


@requires_cuda
def test_fork_during_offload_declines_the_offload():
    """A fork copy is the one writer `_alloc`'s guard can't see.

    It lands in pages already copied to the host, so the host copy may be torn;
    the offload declines rather than freeing pages whose contents it can't
    vouch for.
    """
    mgr = _make_manager()
    mgr.ingest_request("r0")
    _grow(mgr, "r0", "main", 2 * PAGE_SIZE)
    _grow(mgr, "r0", "cfg", 2 * PAGE_SIZE)
    held = mgr.reclaimable("r0")
    host_free = mgr._cpu_pool.num_free_pages

    # the fork bumps `cfg`'s generation while its pages are mid-copy
    _hook_offload_stream(
        mgr, lambda: mgr._apply_fork("r0", "main", "cfg"), when="main"
    )

    assert mgr.offload("r0") == 0
    assert mgr.is_offloaded("r0") is False
    assert mgr.reclaimable("r0") == held
    assert mgr._cpu_pool.num_free_pages == host_free, "host pages not returned"
    assert all(
        not stream.offloaded for stream in mgr._streams["r0"].values()
    ), "streams left claimed after a declined offload"
    _assert_pages_conserved(mgr)

    # declining is not terminal: the next attempt goes through
    assert mgr.offload("r0") == held
    _assert_pages_conserved(mgr)


@requires_cuda
def test_remove_during_offload_leaks_nothing():
    mgr = _make_manager(cpu_offload_pages=4)
    mgr.ingest_request("r0")
    _grow(mgr, "r0", "main", 2 * PAGE_SIZE)

    _hook_offload_stream(mgr, lambda: mgr.remove_request("r0"))

    assert mgr.offload("r0") == 0
    assert mgr._streams.get("r0") is None
    assert mgr._cpu_pool.num_free_pages == 4, "host pages leaked with the request"
    _assert_pages_conserved(mgr)


@requires_cuda
def test_raise_mid_copy_leaves_no_half_offloaded_request():
    """A copy that dies partway must not strand the labels already on the host.

    `reload` would hand those streams fresh pages while they still hold their
    own, so the host copies have to go back with the claims.
    """
    mgr = _make_manager(cpu_offload_pages=8)
    mgr.ingest_request("r0")
    _grow(mgr, "r0", "main", 2 * PAGE_SIZE)
    _grow(mgr, "r0", "cfg", 2 * PAGE_SIZE)
    held = mgr.reclaimable("r0")
    real = mgr._cpu_pool.offload_stream

    def blow_up_on_the_second(**kwargs):
        if kwargs["label"] == "cfg":
            raise RuntimeError("copy died")
        return real(**kwargs)

    mgr._cpu_pool.offload_stream = blow_up_on_the_second

    with pytest.raises(RuntimeError, match="copy died"):
        mgr.offload("r0")

    assert mgr.is_offloaded("r0") is False
    assert mgr.reclaimable("r0") == held
    assert mgr._cpu_pool.num_free_pages == 8, "host pages stranded"
    _assert_pages_conserved(mgr)

    # and the request is still usable afterwards
    mgr._cpu_pool.offload_stream = real
    assert mgr.offload("r0") == held
    assert mgr.reload("r0") is True
    _assert_pages_conserved(mgr)


@requires_cuda
def test_is_offloaded_true_while_the_copy_is_in_flight():
    """`check_ready` gates on this: the window must not look schedulable."""
    mgr = _make_manager()
    mgr.ingest_request("r0")
    _grow(mgr, "r0", "main", PAGE_SIZE)
    seen: list[bool] = []

    _hook_offload_stream(mgr, lambda: seen.append(mgr.is_offloaded("r0")))

    mgr.offload("r0")
    assert seen == [True]
    _assert_pages_conserved(mgr)


@requires_cuda
def test_alloc_for_other_requests_still_works_during_offload():
    """The lock is dropped for the copy precisely so this stays possible."""
    mgr = _make_manager()
    mgr.ingest_request("r0")
    mgr.ingest_request("r1")
    _grow(mgr, "r0", "main", 2 * PAGE_SIZE)
    raced: list = []

    _hook_offload_stream(mgr, lambda: raced.append(_grow(mgr, "r1", "main", PAGE_SIZE)))

    assert mgr.offload("r0") == 2
    assert raced[0].ok
    assert len(_stream(mgr, "r1").page_indices) == 1
    _assert_pages_conserved(mgr)


@requires_cuda
def test_oom_still_reports_allocation_failed():
    """The offload refusal must not have swallowed the real OOM signal."""
    mgr = _make_manager(max_num_pages=4)
    mgr.ingest_request("r0")

    outcome = _grow(mgr, "r0", "main", 8 * PAGE_SIZE)
    assert not outcome.ok
    assert isinstance(outcome.reason, AllocationFailed)
    assert outcome.reason.pages_short > 0
    _assert_pages_conserved(mgr)
