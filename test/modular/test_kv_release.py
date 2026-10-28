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

from types import SimpleNamespace

import pytest
import torch

from mstar.communication.tensors import LocalTransferEngine
from mstar.engine.resources.base import KeyedWalks
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.cache import PageAllocator
from mstar.engine.resources.kv.config import KVStep, PagedKVConfig, RetentionPolicy
from mstar.engine.resources.kv.cpu_page_pool import CPUPagePool
from mstar.engine.resources.kv.manager import KVManager, PageArena
from mstar.engine.resources.kv.plan import SINK_PAGE
from mstar.engine.resources.kv.prefix_index import PrefixIndex
from mstar.engine.resources.kv.transfer import TransferEngineInfo
from mstar.engine.resources.step import Segment, StepContext
from mstar.engine.windowing import WindowSchedule

PS = 8  # page size used throughout


def _make_manager(max_num_pages: int = 64) -> KVManager:
    cfg = PagedKVConfig(
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


class _FakePool(CPUPagePool):
    def __init__(self, max_cpu_pages: int = 64):
        self.arena = PageArena(kv_cache=None, allocator=PageAllocator(max_cpu_pages))
        self.index = PrefixIndex(self.arena)
        self.offloaded = {}
        self._refused_logged_at = None
        self._refused_unlogged = 0
        self.dropped = 0
        self._dropped_logged_at = None

    def copy_out(self, gpu_kv_cache, copies):
        pass

    def copy_in(self, gpu_kv_cache, copies):
        pass

    def sync(self):
        pass


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


# --- sliding layers: the group with the window keeps the last tokens --------

# Layer 0 is full and layers 1 and 2 slide, so a sliding page holds PS/2 tokens
# of two layers: the same bytes as a full page.
WINDOW = 10


def _window_manager(
    sliding=(1, 2), max_num_pages: int = 32, prefix_cache: bool = True,
    joint_comm_group=None, config: dict | None = None, **kw,
) -> KVManager:
    cfg = PagedKVConfig(
        num_layers=3, num_kv_heads=1, head_dim=4, max_seq_len=4096,
        max_num_pages=max_num_pages, page_size=PS, prefix_cache=prefix_cache,
        sliding_window=WINDOW, sliding_layers=sliding, **(config or {}),
    )
    return KVManager(
        cfg=cfg, name="kv", joint_comm_group=joint_comm_group,
        transfer_engine_info=TransferEngineInfo("h", "h", LocalTransferEngine("h")),
        device=torch.device("cpu"), dtype=torch.float32, **kw,
    )


def _grow_writing(m: KVManager, rid: str, start: int, span: int) -> None:
    """One step of ``span`` tokens whose K, in every layer, is the token's position."""
    step = KVStep(segments=(Segment(rid, "main", span),))
    ctx = _ctx(rid)
    assert m.admit(step, ctx).ok
    m.plan(step, ctx)
    k = torch.arange(start, start + span, dtype=torch.float32)[:, None, None].expand(span, 1, 4)
    for layer in range(3):
        m.write_kv(k, -k, layer_idx=layer, label="main")
    m.commit(step, ctx)


def _held_positions(m: KVManager, stream, layer: int, page_size: int) -> list[int]:
    view = m.layer_view(layer)
    return [
        int(view[stream.page_indices[i // page_size], 0, i % page_size, 0, 0])
        for i in range(stream.stored_len)
    ]


@pytest.mark.parametrize("sliding", [(1, 2), (0, 1, 2)], ids=["some", "every"])
def test_sliding_layers_keep_a_window(sliding) -> None:
    """A long request's sliding layers hold about the window, its newest tokens, while
    its full layer holds every token; what the window drops goes back to the pool. Where
    every layer slides, the stream itself keeps the window."""
    m = _window_manager(sliding)
    kept = m.config.groups[-1]
    budget, page_size = kept.retention.context_budget, kept.page_size
    free = m._arena.num_free
    m.ingest_request("r")
    total = 50
    _grow_writing(m, "r", 0, total)
    for _ in range(30):
        _grow_writing(m, "r", total, 1)
        total += 1
        st = _stream(m, "r")
        window = st if st.window is None else st.window
        assert budget <= window.stored_len < budget + page_size
        assert window.released + window.stored_len == total
        assert len(window.page_indices) == -(-window.stored_len // page_size)
        m.assert_pages_conserved()
    # in pages of every layer, as max_num_pages counts them
    assert m.page_counts().total == m.config.max_num_pages - 1
    if st.window is not None:
        assert st.stored_len == total and st.released == 0
        assert _held_positions(m, st, 0, PS) == list(range(total))
    for layer in sliding:
        assert _held_positions(m, window, layer, page_size) == list(range(window.released, total))
    m.remove_request("r")
    assert m._arena.num_free == free


class _CopyingPool(_FakePool):
    """Moves each page's bytes, so a round trip can be checked."""

    def __init__(self):
        super().__init__()
        self.host: dict[int, torch.Tensor] = {}

    def copy_out(self, gpu_kv_cache, copies):
        for gpu, cpu in zip(copies.gpu, copies.cpu, strict=True):
            self.host[cpu] = gpu_kv_cache[:, gpu].clone()

    def copy_in(self, gpu_kv_cache, copies):
        for gpu, cpu in zip(copies.gpu, copies.cpu, strict=True):
            gpu_kv_cache[:, gpu] = self.host[cpu]


def test_window_survives_offload_and_reload() -> None:
    """Both groups' pages go to the host and come back, bytes and window intact."""
    m = _window_manager()
    m._cpu_pool = _CopyingPool()
    m.ingest_request("r")
    _grow_writing(m, "r", 0, 50)
    st = _stream(m, "r")
    held = len(st.page_indices) + len(st.window.page_indices)
    window = (st.window.stored_len, st.window.released)
    expected = [_held_positions(m, st, 0, PS)] + [_held_positions(m, st.window, i, PS // 2) for i in (1, 2)]
    assert m.offload("r") == held
    assert not st.window.page_indices and m._cpu_pool.arena.num_free == 64 - held
    m.kv_cache.tensor.fill_(-1)
    assert m.reload("r")
    assert (st.window.stored_len, st.window.released) == window
    assert [_held_positions(m, st, 0, PS)] + [_held_positions(m, st.window, i, PS // 2) for i in (1, 2)] == expected
    assert m._cpu_pool.arena.num_free == 64
    m.assert_pages_conserved()


@pytest.mark.parametrize("kw", [
    dict(sliding_window=4),
    dict(sliding_layers=(1,)),
    dict(sliding_window=4, sliding_layers=(1, 2), page_size=7),
])
def test_sliding_layers_validation(kw) -> None:
    geometry = dict(num_layers=3, num_kv_heads=1, head_dim=4, max_seq_len=64, page_size=PS)
    with pytest.raises(ValueError):
        PagedKVConfig(**{**geometry, **kw})
    cfg = PagedKVConfig(**geometry, sliding_window=4, sliding_layers=(1, 2))
    with pytest.raises(ValueError):
        cfg.apply_yaml_overrides(page_size=7)


def test_groups_follow_a_page_size_set_after_construction() -> None:
    cfg = PagedKVConfig(num_layers=3, num_kv_heads=1, head_dim=4, max_seq_len=64, page_size=PS,
                        max_num_pages=8, sliding_window=4, sliding_layers=(1, 2))
    cfg.page_size, cfg.max_num_pages = 2 * PS, 16
    assert [group.page_size for group in cfg.groups] == [2 * PS, PS]
    assert cfg.num_units == 48


@pytest.mark.parametrize(("layers", "sliding", "pages"), [
    (52, tuple(i for i in range(52) if i % 4 != 3), (192, 64)),   # 13 full, 39 sliding
    (62, tuple(i for i in range(62) if i % 4 != 1), (184, 64)),   # 16 full, 46 sliding
    (11, tuple(range(3, 11)), (176, 66)),                         # 3 full, 8 sliding
    (11, tuple(range(8, 11)), (66, 176)),                         # 8 full, 3 sliding
    (3, (0, 1, 2), (128,)),                                       # every layer slides
])
def test_an_unset_page_size_is_whole_for_both_groups(layers, sliding, pages) -> None:
    cfg = PagedKVConfig(num_layers=layers, num_kv_heads=1, head_dim=4, max_seq_len=64,
                        sliding_window=4, sliding_layers=sliding)
    assert tuple(group.page_size for group in cfg.groups) == pages


@pytest.mark.parametrize("sliding", [(1, 2), (0, 1, 2)], ids=["some", "every"])
def test_window_pool_refusals_and_tp_offload(sliding, monkeypatch) -> None:
    with pytest.raises(ValueError, match="reads this pool"):
        _window_manager(sliding, needs_remote_transfer=True)
    keyed = {"main": KeyedWalks(("prefill",))}
    with pytest.raises(ValueError):
        _window_manager(sliding).enable_prefix_cache(b"root", keyed)
    assert not _window_manager(sliding, prefix_cache=False).enable_prefix_cache(b"root", keyed)
    m = _window_manager(sliding)
    assert not m.enable_prefix_cache(b"root", {})
    m.ingest_request("r")
    free = m._arena.num_free
    step = KVStep(segments=(Segment("r", "main", 4),), pre_forks=(("main", "other"),))
    outcome = m.admit(step, _ctx("r"))
    assert not outcome.ok and outcome.reason.terminal
    assert m._arena.num_free == free
    if len(m.config.groups) > 1:  # full layers too
        with pytest.raises(ValueError):
            m.write_slots()
    else:
        # the stream is the window, which a protected prefix would hold past the window
        _grow_writing(m, "r", 0, 4)
        with pytest.raises(ValueError):
            m.protect_prefix("r", 4)
        step = KVStep(segments=(Segment("r", "main", 1),),
                      retention={("r", "main"): RetentionPolicy(context_budget=64, protected_prefix=4)})
        outcome = m.admit(step, _ctx("r"))
        assert not outcome.ok and outcome.reason.terminal
    # no prefix cache, so prefix_cache_tp has no hit order for an offload to break
    monkeypatch.setattr(manager_mod, "CPUPagePool", lambda **kw: _FakePool())
    _window_manager(
        sliding, joint_comm_group=SimpleNamespace(rank=0, world_size=2),
        config=dict(num_qo_heads=2, prefix_cache_tp=True, cpu_offload_pages=4),
    )
