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

import ctypes
import sys
import threading
import time
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources import (
    AllocationFailed,
    RequestOffloading,
    Segment,
    StepContext,
)
from mstar.engine.resources.kv import cpu_page_pool, page_copier
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.cache import GroupPages
from mstar.engine.resources.kv.config import KVLayout, KVStep, PagedKVConfig
from mstar.engine.resources.kv.cpu_page_pool import PageCopies
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.plan import SINK_PAGE
from mstar.platforms import current_platform

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="offload needs pinned host memory"
)
requires_nvidia = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="checks the CUDA copier chain"
)

PAGE_SIZE = 8


class _StubTransferManager:
    """No engine, no bytes moved: retrieves complete immediately."""

    def __init__(self, transfer_engine_info, kv_cache, **kwargs):
        del transfer_engine_info, kv_cache, kwargs

    def get_kv_transfer_info(self, **kwargs):
        del kwargs

    def start_async_retrieve(self, **kwargs):
        # no future: nothing to wait on
        del kwargs

    def owns_transfer_info(self, transfer_info, **kwargs):
        del transfer_info, kwargs
        return False

    def cleanup(self):
        pass

    def remove_request(self, request_id):
        del request_id


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransferManager)


def _make_manager(
    max_num_pages: int = 16, cpu_offload_pages: int = 16,
    layout: KVLayout = KVLayout.NHD,
):
    mla = (dict(kv_lora_rank=3, qk_rope_head_dim=1, num_qo_heads=2)
           if layout == KVLayout.MLA else {})
    cfg = PagedKVConfig(
        num_layers=2,
        num_kv_heads=1,
        head_dim=4,
        max_seq_len=max_num_pages * PAGE_SIZE,
        max_num_pages=max_num_pages,
        page_size=PAGE_SIZE,
        cpu_offload_pages=cpu_offload_pages,
        layout=layout,
        **mla,
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


def _hook_host_copy(mgr: KVManager, hook):
    """Run ``hook`` on another thread from inside a host copy.

    The manager lock is not held there, so this reproduces exactly what the
    plan thread can do mid-offload. Joining with a timeout means a regression
    that takes the lock across the copy fails the test instead of hanging it.
    """
    real = mgr._cpu_pool.copy_out
    fired = False

    def wrapper(gpu_kv_cache, copies):
        nonlocal fired
        real(gpu_kv_cache, copies)
        # once only, so a test can retry the offload without racing it again
        if not fired:
            fired = True
            thread = threading.Thread(target=hook)
            thread.start()
            thread.join(timeout=10)
            assert not thread.is_alive(), (
                "the racing thread blocked: offload is holding the manager "
                "lock across the host copy"
            )

    mgr._cpu_pool.copy_out = wrapper


# --- round trip ------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("layout", [KVLayout.NHD, KVLayout.MLA])
def test_offload_reload_round_trip(layout):
    mgr = _make_manager(layout=layout)
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
def test_can_reload_tracks_free_pages_without_a_prefix_index():
    """The index is off at world size > 1, so under TP this is the only case."""
    mgr = _make_manager(max_num_pages=8)
    assert mgr._index is None
    mgr.ingest_request("r0")
    mgr.ingest_request("r1")
    assert mgr.can_reload("r0") is False
    _grow(mgr, "r0", "main", 3 * PAGE_SIZE)

    assert mgr.offload("r0") == 3
    assert mgr.can_reload("r0") is True
    while mgr._arena.num_free:
        assert _grow(mgr, "r1", "main", PAGE_SIZE).ok
    assert mgr.can_reload("r0") is False

    mgr.remove_request("r1")
    assert mgr.can_reload("r0") is True
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

    _hook_host_copy(mgr, lambda: raced.append(_grow(mgr, "r0", "main", PAGE_SIZE)))

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

    _hook_host_copy(mgr, race)

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
    _hook_host_copy(mgr, lambda: mgr._apply_fork("r0", "main", "cfg"))

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

    _hook_host_copy(mgr, lambda: mgr.remove_request("r0"))

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
    real = mgr._cpu_pool.copy_out
    copied = []

    def blow_up_on_the_second(gpu_kv_cache, copies):
        copied.append(copies)
        if len(copied) == 2:
            raise RuntimeError("copy died")
        real(gpu_kv_cache, copies)

    mgr._cpu_pool.copy_out = blow_up_on_the_second

    with pytest.raises(RuntimeError, match="copy died"):
        mgr.offload("r0")

    assert mgr.is_offloaded("r0") is False
    assert mgr.reclaimable("r0") == held
    assert mgr._cpu_pool.num_free_pages == 8, "host pages stranded"
    _assert_pages_conserved(mgr)

    # and the request is still usable afterwards
    mgr._cpu_pool.copy_out = real
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

    _hook_host_copy(mgr, lambda: seen.append(mgr.is_offloaded("r0")))

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

    _hook_host_copy(mgr, lambda: raced.append(_grow(mgr, "r1", "main", PAGE_SIZE)))

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


# --- a refused admit gives its pages back -----------------------------------
#
# Nothing else does: ``commit`` is what normally releases a reservation, and a
# step that never runs never commits. Left alone, every refusal shrinks the free
# pool, and two ranks refusing a different number of times end up with different
# page counts — which under TP is a deadlock.


def _ctx_pre(*rids: str) -> StepContext:
    ctx = _ctx(*rids)
    ctx.is_preplan = True
    return ctx


@requires_cuda
def test_a_rolled_back_admit_returns_its_pages():
    mgr = _make_manager(max_num_pages=8)
    mgr.ingest_request("r0")
    before = mgr._arena.num_free

    step = KVStep(segments=(Segment("r0", "main", 2 * PAGE_SIZE),))
    ctx = _ctx("r0")
    assert mgr.admit(step, ctx).ok
    assert mgr._arena.num_free < before, "admit took no pages, so nothing to test"

    mgr.rollback_admit(step, ctx)

    assert mgr._arena.num_free == before
    assert _stream(mgr, "r0").page_indices == [], (
        "pages went back to the arena but the stream still addresses them"
    )


@requires_cuda
def test_rolling_back_twice_releases_once():
    """The runner unwinds the refusing resource as well as the ones ahead of it,
    so a double call has to be harmless rather than double-free the arena."""
    mgr = _make_manager(max_num_pages=8)
    mgr.ingest_request("r0")
    before = mgr._arena.num_free

    step = KVStep(segments=(Segment("r0", "main", 2 * PAGE_SIZE),))
    ctx = _ctx("r0")
    mgr.admit(step, ctx)
    mgr.rollback_admit(step, ctx)
    mgr.rollback_admit(step, ctx)

    assert mgr._arena.num_free == before


@requires_cuda
def test_a_committed_admit_is_not_rolled_back():
    """Once the step has run its reservation is real, and unwinding it would pull
    pages out from under a stream that is addressing them."""
    mgr = _make_manager(max_num_pages=8)
    mgr.ingest_request("r0")

    step = KVStep(segments=(Segment("r0", "main", 2 * PAGE_SIZE),))
    ctx = _ctx("r0")
    mgr.admit(step, ctx)
    mgr.plan(step, ctx)
    mgr.commit(step, ctx)
    held = list(_stream(mgr, "r0").page_indices)
    free_after_commit = mgr._arena.num_free

    mgr.rollback_admit(step, ctx)

    assert _stream(mgr, "r0").page_indices == held
    assert mgr._arena.num_free == free_after_commit


@requires_cuda
def test_a_preplanned_step_unwinds_what_the_preplan_reserved():
    """The case the log keeps hitting. A pre-planned step is reserved on the
    pre-plan pass; its later ``admit`` no-ops. So when a resource further down the
    order refuses, the unwind arrives with the GPU pass's ctx and still has to
    release what the pre-plan took — otherwise those pages leak on exactly the
    path that refuses most often."""
    mgr = _make_manager(max_num_pages=8)
    mgr.ingest_request("r0")
    before = mgr._arena.num_free

    step = KVStep(segments=(Segment("r0", "main", 2 * PAGE_SIZE),))
    pre_ctx = _ctx_pre("r0")
    assert mgr.admit(step, pre_ctx).ok             # the plan thread reserves
    # ``plan`` is what marks it pre-planned, and records WHICH step for -- the
    # GPU pass's admit only no-ops if that key matches, so staging it by hand
    # would take the "a different step arrived first" path instead
    mgr.plan(step, pre_ctx)
    assert mgr._preplanned
    assert mgr._arena.num_free < before

    gpu_ctx = _ctx("r0")
    assert mgr.admit(step, gpu_ctx).ok              # no-ops over the reservation

    mgr.rollback_admit(step, gpu_ctx)               # another resource refused

    assert mgr._arena.num_free == before, (
        "the pre-plan's pages leaked: reserved under is_preplan, unwound without"
    )
    assert _stream(mgr, "r0").page_indices == []


@requires_cuda
def test_unwinding_one_step_leaves_another_streams_pages_alone():
    mgr = _make_manager(max_num_pages=16)
    mgr.ingest_request("r0")
    mgr.ingest_request("r1")

    committed = KVStep(segments=(Segment("r1", "main", 2 * PAGE_SIZE),))
    ctx1 = _ctx("r1")
    mgr.admit(committed, ctx1)
    mgr.plan(committed, ctx1)
    mgr.commit(committed, ctx1)
    kept = list(_stream(mgr, "r1").page_indices)

    refused = KVStep(segments=(Segment("r0", "main", 2 * PAGE_SIZE),))
    ctx0 = _ctx("r0")
    assert mgr.admit(refused, ctx0).ok
    mgr.rollback_admit(refused, ctx0)

    assert _stream(mgr, "r0").page_indices == []
    assert _stream(mgr, "r1").page_indices == kept, (
        "unwinding one step released another stream's pages"
    )


@requires_cuda
def test_removing_a_request_forgets_its_unwind_record():
    """Handles are RECYCLED (see ``mstar.worker.rid_table``), so a reservation
    record outliving its request attaches to whichever request gets that handle
    next -- and ``rollback_admit`` would then resolve it against that request's
    stream.

    The clear at the top of ``admit`` does not cover it: a pre-planned step's
    admit returns before reaching the clear, which is deliberate (the record is
    what a later refusal unwinds). So the release has to purge it.
    """
    mgr = _make_manager(max_num_pages=8)
    mgr.ingest_request("r0")

    # admit, then neither commit nor roll back -- a step that raised in
    # prepare_inputs, or a batch dropped before it ran
    step = KVStep(segments=(Segment("r0", "main", 2 * PAGE_SIZE),))
    mgr.admit(step, _ctx("r0"))
    assert mgr._admit_reserved_pages, "nothing was reserved, so nothing to test"

    mgr.remove_request("r0")

    assert mgr._admit_reserved_pages == {}, (
        "the record outlived the request; the next holder of this handle would "
        "have its pages unwound"
    )


@requires_cuda
def test_resetting_a_request_forgets_its_unwind_record():
    """A reset releases the stream's pages, so a surviving record names pages the
    arena has already handed to someone else."""
    mgr = _make_manager(max_num_pages=8)
    mgr.ingest_request("r0")

    step = KVStep(segments=(Segment("r0", "main", 2 * PAGE_SIZE),))
    mgr.admit(step, _ctx("r0"))
    assert mgr._admit_reserved_pages

    mgr.reset_request("r0", free=True)

    assert mgr._admit_reserved_pages == {}


@requires_cuda
def test_an_alloc_that_needs_no_new_pages_reports_none():
    """The common decode case: the stream already holds enough pages, so nothing
    is acquired and there is nothing to unwind. ``_alloc`` still has to answer —
    returning an unset ``new_pages`` raises, on the hottest path there is."""
    mgr = _make_manager(max_num_pages=8)
    mgr.ingest_request("r0")
    _grow(mgr, "r0", "main", PAGE_SIZE)          # one full page
    held = list(_stream(mgr, "r0").page_indices)

    # ask for fewer tokens than the stream already covers
    result = mgr._alloc("r0", "main", PAGE_SIZE // 2)

    assert result.success
    assert not result.new_pages
    assert _stream(mgr, "r0").page_indices == held


@requires_cuda
def test_offload_reload_keeps_the_protected_prefix():
    mgr = _make_manager()
    mgr.ingest_request("r0")
    _grow(mgr, "r0", "main", 4 * PAGE_SIZE)
    mgr.protect_prefix("r0", 2 * PAGE_SIZE)
    assert mgr.offload("r0") == 4
    assert mgr.reload("r0") is True
    assert _stream(mgr, "r0").protected_prefix == 2 * PAGE_SIZE
    assert mgr.release_oldest("r0", 4 * PAGE_SIZE) == 2 * PAGE_SIZE


@requires_cuda
def test_pool_copies_queue_behind_the_compute_stream():
    """Both directions wait for the current stream before copying: the pages
    they touch may still be in use by the step that last ran (a commit hands
    pages on right after the launch)."""
    mgr = _make_manager()
    waited = []

    class _Spy(torch.cuda.Stream):
        def wait_stream(self, stream):
            waited.append(stream.cuda_stream)
            super().wait_stream(stream)

    mgr._cpu_pool._stream = _Spy()
    mgr.ingest_request("r0")
    _grow(mgr, "r0", "main", 2 * PAGE_SIZE)
    compute = torch.cuda.current_stream().cuda_stream
    assert mgr.offload("r0") == 2
    assert waited == [compute]
    assert mgr.reload("r0") is True
    assert waited == [compute, compute]


def _loadable(cls):
    try:
        return cls()
    except (ImportError, OSError, AttributeError, RuntimeError) as e:
        pytest.skip(f"{cls.name} unavailable: {e}")


@requires_cuda
@pytest.mark.parametrize(
    "copier_cls",
    [page_copier.CudaBindingsCopier, page_copier.CudartCopier, page_copier.HipCopier,
     page_copier.TorchCopier],
    ids=lambda cls: cls.name,
)
def test_pool_copies_keep_every_layer_and_element(copier_cls):
    """A page is one strided block per layer, so a wrong pitch or page stride
    mixes layers or pages; distinct values in every element catch either, and
    pages nobody named must come through untouched."""
    mgr = _make_manager()
    pool = mgr._cpu_pool
    pool._copier = _loadable(copier_cls)
    device = mgr.kv_cache.tensor
    device.copy_(torch.randn_like(device))
    expected = device.clone()
    src, dst, host = [5, 6, 9], [3, 4, 7], [0, 1, 2]

    pool.copy_out(device, PageCopies(gpu=src, cpu=host))
    pool.sync()
    device.fill_(-99.0)
    pool.copy_in(device, PageCopies(gpu=dst, cpu=host))
    pool.sync()
    torch.cuda.synchronize()

    for s, d in zip(src, dst, strict=True):
        assert torch.equal(device[:, d], expected[:, s])
    others = [p for p in range(device.shape[1]) if p not in dst]
    assert (device[:, others] == -99.0).all()


def test_a_copier_checks_each_pair_of_caches_once():
    """Checking a pair costs the host more than the copies the copier then queues, so the copier keeps
    each pair's pitches and strides."""

    class _Counting(page_copier._Memcpy2DCopier):
        _max_pitch = 1 << 62

        def _memcpy_2d(self, dst, dpitch, src, spitch, width, height, stream):
            return 0

    copier = _Counting()
    checked = []
    check = copier._check
    copier._check = lambda dst, src: checked.append((dst, src)) or check(dst, src)
    host, device, other = (torch.zeros(2, 4, 3) for _ in range(3))
    stream = SimpleNamespace(cuda_stream=0)

    copier.copy(host, [0], device, [1], stream)
    copier.copy(host, [2], device, [3], stream)
    assert len(checked) == 1, "a second copy between the same two caches checked them again"
    copier.copy(other, [0], device, [1], stream)
    assert len(checked) == 2, "a copy into another cache used the first pair's pitches and strides"
    # as if `later` had taken the id of a freed cache
    later = torch.zeros(2, 4, 3)
    copier._layouts[id(later), id(device)] = copier._layouts.pop((id(other), id(device)))
    copier.copy(later, [0], device, [1], stream)
    assert len(checked) == 3, "a cache at a reused id got the freed cache's pitches and strides"


def test_pages_adjacent_in_both_caches_go_as_one_copy():
    """A call per page costs a 60-page reload about 160 µs of host time, and adjacent pages are one block
    per layer."""
    calls = []

    class _Recording(page_copier._Memcpy2DCopier):
        _max_pitch = 1 << 62

        def _memcpy_2d(self, dst, dpitch, src, spitch, width, height, stream):
            calls.append((dst, dpitch, src, spitch, width, height))
            return 0

    # float32, [layers, pages, 3]: a page is 12 bytes, a layer 96 bytes on the host and 72 on the device
    host, device = torch.zeros(2, 8, 3), torch.zeros(2, 6, 3)
    # pages as (host, device): (5, 2) to (6, 4) is adjacent on the host only, and (6, 4) to (0, 5) on the device only
    _Recording().copy(host, [4, 5, 6, 0], device, [1, 2, 4, 5], SimpleNamespace(cuda_stream=0))

    h, d = host.data_ptr(), device.data_ptr()
    want = [(h + 48, 96, d + 12, 72, 24, 2), (h + 72, 96, d + 48, 72, 12, 2), (h, 96, d + 60, 72, 12, 2)]
    assert calls == want, "a copy didn't cover exactly a run of pages adjacent in both caches"


@requires_cuda
@requires_nvidia
def test_copies_on_cuda_keep_the_gil():
    """Beside a thread running Python, queueing four page copies took 15 ms dropping the GIL and 28 µs keeping it."""
    _loadable(page_copier.CudartCopier)
    copier = page_copier.make_page_copier.__wrapped__()
    fn = getattr(copier, "_fn", None)
    assert fn is not None and fn._flags_ & ctypes._FUNCFLAG_PYTHONAPI, f"{copier.name} drops the GIL on every copy"


@requires_cuda
def test_pool_copies_do_not_block_the_host():
    """The copies queue behind the compute stream on the GPU; the host must
    not wait for that stream to drain before it can go on scheduling."""
    mgr = _make_manager()
    pool = mgr._cpu_pool
    assert not isinstance(pool._copier, page_copier.TorchCopier), (
        "no 2D async memcpy binding loaded on a CUDA or ROCm build of torch"
    )
    device = mgr.kv_cache.tensor
    pool.copy_out(device, PageCopies(gpu=[1], cpu=[0]))
    torch.cuda.synchronize()

    torch.cuda._sleep(500_000_000)  # a few hundred ms of compute ahead of the copies
    start = time.perf_counter()
    pool.copy_out(device, PageCopies(gpu=[1, 2, 3], cpu=[0, 1, 2]))
    pool.copy_in(device, PageCopies(gpu=[4, 5, 6], cpu=[0, 1, 2]))
    pool.sync()
    elapsed = time.perf_counter() - start
    torch.cuda.synchronize()
    assert elapsed < 0.05, f"the host waited {elapsed * 1e3:.0f} ms on the copies"


@requires_cuda
def test_a_full_host_pool_logs_a_summary_not_every_refusal(monkeypatch, caplog):
    """A full pool refuses every eviction attempt; one warning per interval,
    counting the refusals it stood for, keeps the log readable."""
    mgr = _make_manager(cpu_offload_pages=1)
    pool = mgr._cpu_pool
    assert pool.place("held", "main", (GroupPages([1]),), [], position=0)
    clock = [100.0]
    monkeypatch.setattr(cpu_page_pool.time, "monotonic", lambda: clock[0])

    with caplog.at_level("WARNING", logger=cpu_page_pool.__name__):
        for _ in range(50):
            assert pool.place("r0", "main", (GroupPages([2]),), [], position=0) is None
        clock[0] += cpu_page_pool._EXHAUSTED_LOG_INTERVAL
        assert pool.place("r0", "main", (GroupPages([2]),), [], position=0) is None

    warnings = [r.getMessage() for r in caplog.records if "exhausted" in r.getMessage()]
    assert len(warnings) == 2
    assert "0 more offloads refused" in warnings[0]
    assert "49 more offloads refused" in warnings[1]


@requires_cuda
@requires_nvidia
def test_copier_falls_through_to_the_next_that_works(monkeypatch, caplog):
    """A binding that won't load, or loads and copies wrong, is skipped; with
    none left the pool keeps copy_ and warns once per process, not per pool."""

    class _Missing(page_copier.CudaBindingsCopier):
        def __init__(self):
            raise ImportError("no cuda.bindings")

    class _Wrong(page_copier._Memcpy2DCopier):
        name = "wrong"
        _max_pitch = 1 << 62

        def _memcpy_2d(self, dst, dpitch, src, spitch, width, height, stream):
            return 0  # claims success, copies nothing

    monkeypatch.setattr(page_copier, "CudaBindingsCopier", _Missing)
    monkeypatch.setattr(page_copier, "CudartCopier", _Wrong)
    page_copier.make_page_copier.cache_clear()
    try:
        with caplog.at_level("INFO", logger=page_copier.__name__):
            pools = [_make_manager()._cpu_pool for _ in range(2)]
        assert all(isinstance(pool._copier, page_copier.TorchCopier) for pool in pools)
        messages = [r.getMessage() for r in caplog.records]
        assert any("unavailable (no cuda.bindings)" in m for m in messages)
        assert any("returned wrong data" in m for m in messages)
        assert sum("fall back to copy_" in m for m in messages) == 1

        monkeypatch.undo()
        page_copier.make_page_copier.cache_clear()
        assert not isinstance(page_copier.make_page_copier(), page_copier.TorchCopier)
    finally:
        page_copier.make_page_copier.cache_clear()
