"""Partial KV release on a live request: the retention a committing step
declares (``KVStep.retention``) and the explicit ``KVManager.protect_prefix``
/ ``release_oldest`` pair (ported from #198's allocator-level tests onto the
pool).

The invariant everything rests on: a stream's ``page_indices`` stays a
contiguous logical stream over its committed tokens — release removes whole
pages from the front of the unprotected region and drops ``stored_len`` by
exactly the freed token count — because the planner indexes pages as
``token // page_size``. Everything here runs on CPU.
"""

from __future__ import annotations

import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.resources.kv.config import KVConfig, KVStep, RetentionPolicy
from mstar.engine.resources.kv.cpu_page_pool import OffloadedStream
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.kv.plan import SINK_PAGE
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.step import Segment, StepContext
from mstar.engine.windowing import WindowSchedule

PS = 8  # page size used throughout


def _make_manager(max_num_pages: int = 64) -> KVManager:
    cfg = KVConfig(
        num_layers=1, num_kv_heads=1, head_dim=4, max_seq_len=max_num_pages * PS,
        max_num_pages=max_num_pages, page_size=PS,
    )
    return KVManager(
        cfg=cfg, name="kv", joint_comm_group=None,
        transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
        device=torch.device("cpu"), dtype=torch.float32,
    )


def _ctx(*rids: str) -> StepContext:
    return StepContext(request_ids=tuple(rids), graph_walk="walk", slot=0, capture=False)


def _grow(
    mgr: KVManager, rid: str, label: str, span: int, commit: bool = True,
    retention: RetentionPolicy | None = None,
):
    """Admit, plan and commit one step extending ``label`` by ``span`` tokens,
    declaring ``retention`` for the stream when given (as a window commit does)."""
    step = KVStep(
        segments=(Segment(rid, label, span),), commit=commit,
        retention={} if retention is None else {(rid, label): retention},
    )
    ctx = _ctx(rid)
    outcome = mgr.admit(step, ctx)
    assert outcome.ok, outcome.reason
    mgr.plan(step, ctx)
    mgr.commit(step, ctx)
    return step, ctx


def _stream(mgr: KVManager, rid: str, label: str = "main"):
    return mgr._streams[rid][label]


def _assert_coherent(stream) -> None:
    assert len(stream.page_indices) >= -(-stream.stored_len // PS)


def _assert_pages_conserved(mgr: KVManager) -> None:
    free = list(mgr._arena.allocator.free_pages.queue)
    held = [p for streams in mgr._streams.values() for s in streams.values() for p in s.page_indices]
    owned = free + held + [SINK_PAGE]
    duplicated = {p for p in owned if owned.count(p) > 1}
    assert not duplicated, f"pages owned twice: {sorted(duplicated)}"
    missing = set(range(mgr.config.max_num_pages)) - set(owned)
    assert not missing, f"pages leaked: {sorted(missing)}"


def test_basic_release_compacts_front() -> None:
    m = _make_manager()
    m.ingest_request("r")
    _grow(m, "r", "main", 10 * PS)
    st = _stream(m, "r")
    original = list(st.page_indices)
    gen0 = st.generation
    free0 = m._arena.num_free

    m.protect_prefix("r", 2 * PS, label="main")
    freed = m.release_oldest("r", 3 * PS, label="main")

    assert freed == 3 * PS
    assert st.page_indices == original[:2] + original[5:]
    assert st.stored_len == 7 * PS
    assert st.released == 3 * PS
    assert st.generation == gen0 + 1
    assert m._arena.num_free == free0 + 3
    _assert_coherent(st)
    _assert_pages_conserved(m)


def test_release_floors_to_whole_pages() -> None:
    m = _make_manager()
    m.ingest_request("r")
    _grow(m, "r", "main", 10 * PS)
    m.protect_prefix("r", PS, label="main")
    assert m.release_oldest("r", PS - 1, label="main") == 0
    assert m.release_oldest("r", 2 * PS + 3, label="main") == 2 * PS
    assert _stream(m, "r").stored_len == 8 * PS


def test_protection_boundary_page_never_freed() -> None:
    m = _make_manager()
    m.ingest_request("r")
    _grow(m, "r", "main", 6 * PS)
    st = _stream(m, "r")
    original = list(st.page_indices)
    # Protect 1.5 pages: the straddling page (index 1) survives.
    m.protect_prefix("r", PS + PS // 2, label="main")
    assert m.release_oldest("r", 100 * PS, label="main") == 4 * PS
    assert st.page_indices == original[:2]
    assert st.stored_len == 2 * PS


def test_partial_tail_page_never_freed() -> None:
    m = _make_manager()
    m.ingest_request("r")
    _grow(m, "r", "main", 4 * PS + 3)  # 5 pages, the last one partial
    st = _stream(m, "r")
    original = list(st.page_indices)
    m.protect_prefix("r", PS, label="main")
    assert m.release_oldest("r", 100 * PS, label="main") == 3 * PS
    assert st.page_indices == [original[0], original[4]]
    assert st.stored_len == PS + 3


def test_protect_validation() -> None:
    m = _make_manager()
    m.ingest_request("r")
    _grow(m, "r", "main", 4 * PS)
    with pytest.raises(ValueError, match="outside the committed"):
        m.protect_prefix("r", 5 * PS, label="main")
    m.protect_prefix("r", 2 * PS, label="main")
    m.protect_prefix("r", 2 * PS, label="main")  # idempotent at the same value
    with pytest.raises(ValueError, match="already"):
        m.protect_prefix("r", PS, label="main")
    m.release_oldest("r", PS, label="main")
    with pytest.raises(ValueError, match="must precede"):
        m.protect_prefix("r", 3 * PS, label="main")


def test_release_refused_under_an_admitted_step() -> None:
    m = _make_manager()
    m.ingest_request("r")
    _grow(m, "r", "main", 6 * PS)
    m.protect_prefix("r", PS, label="main")
    step = KVStep(segments=(Segment("r", "main", PS),))
    ctx = _ctx("r")
    assert m.admit(step, ctx).ok
    with pytest.raises(RuntimeError, match="admitted step"):
        m.release_oldest("r", PS, label="main")
    m.plan(step, ctx)
    m.commit(step, ctx)
    assert m.release_oldest("r", PS, label="main") == PS


def test_later_steps_plan_over_the_compacted_stream() -> None:
    m = _make_manager()
    m.ingest_request("r")
    _grow(m, "r", "main", 8 * PS)
    st = _stream(m, "r")
    m.protect_prefix("r", 2 * PS, label="main")
    m.release_oldest("r", 4 * PS, label="main")
    kept = list(st.page_indices)
    # The next step extends the compacted stream: its view spans the kept
    # pages plus the new span, indexed from token 0.
    step = KVStep(segments=(Segment("r", "main", 2 * PS),))
    ctx = _ctx("r")
    assert m.admit(step, ctx).ok
    views = m._sequence_views(list(step.segments))
    assert views[0].length == 6 * PS and views[0].to_compute == 2 * PS
    assert views[0].page_idxs[:len(kept)] == kept
    m.plan(step, ctx)
    m.commit(step, ctx)
    assert st.stored_len == 6 * PS
    _assert_pages_conserved(m)


def test_remove_request_returns_every_page() -> None:
    m = _make_manager()
    free_at_start = m._arena.num_free  # the pool keeps its sink page
    m.ingest_request("r")
    _grow(m, "r", "main", 10 * PS)
    m.protect_prefix("r", 2 * PS, label="main")
    m.release_oldest("r", 3 * PS, label="main")
    m.remove_request("r")
    assert m._arena.num_free == free_at_start
    _assert_pages_conserved(m)


class _FakePool:
    """The CPUPagePool surface `offload` / `reload` use, minus the copies
    (those need a device)."""

    def __init__(self):
        self.states: dict = {}

    def offload_stream(
        self, rid, label, gpu_kv_cache, gpu_page_indices, stored_len, position,
        released=0, protected_prefix=0,
    ):
        self.states.setdefault(rid, {})[label] = OffloadedStream(
            cpu_page_indices=list(gpu_page_indices), stored_len=stored_len,
            position=position, released=released, protected_prefix=protected_prefix,
        )
        return True

    def reload_stream(self, rid, label, gpu_kv_cache, gpu_page_indices):
        state = self.states[rid].pop(label)
        if not self.states[rid]:
            del self.states[rid]
        return state

    def sync(self):
        pass

    def is_offloaded(self, rid):
        return bool(self.states.get(rid))

    def labels(self, rid):
        return list(self.states.get(rid, {}))

    def num_pages(self, rid, label):
        return len(self.states[rid][label].cpu_page_indices)

    def discard(self, rid, label):
        self.states.get(rid, {}).pop(label, None)

    def remove_request(self, rid):
        self.states.pop(rid, None)


def test_retention_releases_at_commit() -> None:
    """A committing step that declares a retention for its stream sheds the
    oldest unprotected pages as part of the commit that pushes it past the
    budget: the prefix stays, ``stored_len`` drops by whole pages, the
    generation moves, pages return to the arena. The policy rides on the step;
    a step that declares none releases nothing."""
    m = _make_manager(max_num_pages=64)
    m.ingest_request("r")
    prefix = 3 * PS
    _grow(m, "r", "main", prefix)
    policy = RetentionPolicy(context_budget=4 * PS, protected_prefix=prefix)
    st = _stream(m, "r")
    assert st.retention is None and st.protected_prefix == 0
    free0 = m._arena.num_free
    # Four pages of generation fit the budget exactly: nothing released.
    for _ in range(4):
        _grow(m, "r", "main", PS, retention=policy)
    assert st.protected_prefix == prefix and st.retention is policy
    assert st.released == 0 and st.stored_len == prefix + 4 * PS
    gen = st.generation
    # The fifth page is one over budget: one page (the oldest) goes.
    _grow(m, "r", "main", PS, retention=policy)
    assert st.released == PS and st.stored_len == prefix + 4 * PS
    assert st.generation > gen
    assert len(st.page_indices) == 7 and m._arena.num_free == free0 - 4
    # Committing two pages at once releases two.
    _grow(m, "r", "main", 2 * PS, retention=policy)
    assert st.released == 3 * PS and st.stored_len == prefix + 4 * PS
    _assert_coherent(st)
    _assert_pages_conserved(m)
    # A non-committing step (a denoise read) never triggers a release.
    _grow(m, "r", "main", PS, commit=False, retention=policy)
    assert st.released == 3 * PS
    # Neither does a commit that declares no retention.
    _grow(m, "r", "main", PS)
    assert st.released == 3 * PS and st.stored_len == prefix + 5 * PS


def test_retention_shortfall_carries_over() -> None:
    """Sub-page commits accumulate until a whole page is over budget; the
    realized context never exceeds the budget by a full page."""
    m = _make_manager(max_num_pages=64)
    m.ingest_request("r")
    prefix = PS
    _grow(m, "r", "main", prefix)
    budget = 3 * PS
    policy = RetentionPolicy(context_budget=budget, protected_prefix=prefix)
    st = _stream(m, "r")
    unit = 3  # tokens per unit, not page aligned
    for _ in range(40):
        _grow(m, "r", "main", unit, retention=policy)
        assert st.stored_len - prefix <= budget + PS - 1
        _assert_coherent(st)
    assert st.released > 0
    _assert_pages_conserved(m)


def test_retention_validation() -> None:
    with pytest.raises(ValueError):
        RetentionPolicy(context_budget=0)
    with pytest.raises(ValueError):
        RetentionPolicy(context_budget=-1)
    with pytest.raises(ValueError):
        RetentionPolicy(context_budget=PS, protected_prefix=-1)
    m = _make_manager()
    m.ingest_request("r")
    _grow(m, "r", "main", 2 * PS)
    # A prefix that has not committed yet is a declaration error.
    with pytest.raises(ValueError, match="protects"):
        _grow(m, "r", "main", PS, retention=RetentionPolicy(context_budget=PS, protected_prefix=4 * PS))
    # 32 committed against an 8-token prefix + 8-token budget: two pages go.
    _grow(m, "r", "main", PS, retention=RetentionPolicy(context_budget=PS, protected_prefix=PS))
    st = _stream(m, "r")
    assert st.protected_prefix == PS and st.released == 2 * PS
    # The prefix cannot move afterwards; the budget can.
    with pytest.raises(ValueError, match="already"):
        _grow(m, "r", "main", PS, retention=RetentionPolicy(context_budget=PS, protected_prefix=2 * PS))
    _grow(m, "r", "main", PS, retention=RetentionPolicy(context_budget=3 * PS, protected_prefix=PS))
    assert st.released == 2 * PS
    # Reset clears the policy with the rest of the stream state.
    m.reset_request("r", free=True)
    st = _stream(m, "r")
    assert st.retention is None and st.protected_prefix == 0 and st.released == 0


def test_retention_survives_offload_and_reload() -> None:
    """The policy rides on each committing step, so an offloaded stream comes
    back with nothing to re-install: its next window commit releases per the
    budget as before. The explicit protected prefix and the released count
    travel with the host copy."""
    m = _make_manager(max_num_pages=64)
    m._cpu_pool = _FakePool()
    m.ingest_request("r")
    prefix = 2 * PS
    _grow(m, "r", "main", prefix)
    policy = RetentionPolicy(context_budget=2 * PS, protected_prefix=prefix)
    for _ in range(4):
        _grow(m, "r", "main", PS, retention=policy)
    st = _stream(m, "r")
    assert st.released == 2 * PS and st.stored_len == prefix + 2 * PS
    assert m.offload("r") == 4
    assert m.is_offloaded("r") and not st.page_indices
    assert m.reload("r")
    st = _stream(m, "r")
    assert (st.stored_len, st.released, st.protected_prefix) == (prefix + 2 * PS, 2 * PS, prefix)
    assert st.retention is None
    _grow(m, "r", "main", PS, retention=policy)
    assert st.released == 3 * PS and st.stored_len == prefix + 2 * PS
    _assert_coherent(st)
    _assert_pages_conserved(m)


def test_protected_prefix_survives_offload_and_reload() -> None:
    m = _make_manager(max_num_pages=64)
    m._cpu_pool = _FakePool()
    m.ingest_request("r")
    _grow(m, "r", "main", 4 * PS)
    m.protect_prefix("r", 2 * PS)
    assert m.offload("r") == 4 and m.reload("r")
    assert _stream(m, "r").protected_prefix == 2 * PS
    # only the two unprotected pages go
    assert m.release_oldest("r", 4 * PS) == 2 * PS
    _assert_pages_conserved(m)


def test_window_schedule_drives_the_manager() -> None:
    """The schedule's budget, declared on every window commit and applied by
    the real pool: the retained context never exceeds the horizon plus one
    page."""
    m = _make_manager(max_num_pages=128)
    m.ingest_request("r")
    prefix = 3 * PS
    _grow(m, "r", "main", prefix)
    schedule = WindowSchedule(total_units=48, window_units=8, context_units=16)
    tpu = PS // 2
    policy = schedule.retention(tpu, prefix)
    assert policy.context_budget == 16 * tpu and policy.protected_prefix == prefix
    for w in schedule.windows():
        _grow(m, "r", "main", (w.commit_end - w.commit_start) * tpu, retention=policy)
        st = _stream(m, "r")
        retained_units = (st.stored_len - prefix) // tpu
        # Never more than the context horizon plus one page's worth of slack.
        assert retained_units <= schedule.context_units + PS // tpu
        _assert_pages_conserved(m)
    st = _stream(m, "r")
    assert st.released == (schedule.total_units - 16) * tpu
