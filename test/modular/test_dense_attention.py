"""Unit tests for the dense attention backend and the stream epoch it rides on.

The dense backend is the first resource to cache something *derived from the
contents* of a KV stream across steps. Nothing else in v1 does that, so the two
things worth pinning are the layout it builds off the KV plan, and the
invalidation: a cached gather must not survive a fork, a reset, or a page-table
move. Both run on CPU with no kernel — the FlashAttention-3 call itself is the
only part that needs a GPU.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    AttentionStep,
    AttnBackend,
    KVConfig,
    KVStep,
    Segment,
    SlotLease,
    StepContext,
)
from mstar.engine.resources.attn.manager import (
    AttentionManager,
    DenseAttentionManager,
    FlashInferManager,
    _fa3_unavailable_reason,
)
from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.kv.manager import KVPlanOutput, SequenceView

PAGE_SIZE = 4
NUM_KV_HEADS = 2
HEAD_DIM = 3
MAX_PAGES = 8


def _kv_config() -> KVConfig:
    return KVConfig(
        num_layers=1,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        max_seq_len=64,
        max_num_pages=MAX_PAGES,
        page_size=PAGE_SIZE,
    )


def _manager() -> DenseAttentionManager:
    return DenseAttentionManager(
        kv_cache="kv",
        device=torch.device("cpu"),
        dtype=torch.float32,
        kv_config=_kv_config(),
    )


def _ctx(views: list[SequenceView], **kwargs) -> StepContext:
    return StepContext(
        request_ids=tuple(dict.fromkeys(view.request_id for view in views)),
        graph_walk="gen",
        slot=0,
        capture=False,
        plan_results={
            "kv": {"main": KVPlanOutput(cpu_indptrs=None, cuda_indptrs=None, views=views)}
        },
        **kwargs,
    )


def _view(rid: str, pages: list[int], prefix: int, fresh: int, generation: int = 0):
    return SequenceView(
        request_id=rid,
        label="main",
        page_idxs=pages,
        length=prefix + fresh,
        to_compute=fresh,
        generation=generation,
    )


def _layer_tensor(fill: float = 0.0) -> torch.Tensor:
    """A KV layer whose every slot is a distinct number, so a gather that
    picked the wrong page or the wrong offset shows up as a value mismatch."""
    n = MAX_PAGES * 2 * PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM
    return (torch.arange(n, dtype=torch.float32) + fill).reshape(
        MAX_PAGES, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM
    )


def _expected_prefix(layer: torch.Tensor, pages: list[int], prefix_len: int, kv: int):
    rows = layer[pages][:, kv].reshape(-1, NUM_KV_HEADS, HEAD_DIM)
    return rows[:prefix_len]


class TestDensePlan:
    def test_prefix_is_what_the_step_does_not_compute(self):
        """The frozen prefix is `length - to_compute`: the fresh tokens are
        declared as ordinary spans (with commit=False), so the KV plan hands
        over both halves without the step saying anything backend-specific."""
        manager = _manager()
        # 6 resident + 5 fresh, over 3 pages of 4
        views = [_view("r0", [1, 2, 3], prefix=6, fresh=5)]
        manager.plan(AttentionStep(causal=False), _ctx(views))

        (segment,) = manager._current_plans["main"].segments
        assert segment.prefix_len == 6
        assert segment.q_len == 5
        # only the pages the prefix actually covers, not the ones reserved for
        # the fresh span
        assert segment.pages.tolist() == [1, 2]

    def test_cumulative_lengths_pack_in_kv_plan_order(self):
        manager = _manager()
        views = [
            _view("r0", [0, 1], prefix=3, fresh=2),
            _view("r1", [2, 3, 4], prefix=9, fresh=4),
        ]
        manager.plan(AttentionStep(causal=False), _ctx(views))
        plan = manager._current_plans["main"]

        assert plan.cu_q.tolist() == [0, 2, 6]
        assert plan.cu_k.tolist() == [0, 5, 18]
        assert plan.max_q == 4
        assert plan.max_k == 13
        assert [s.request_id for s in plan.segments] == ["r0", "r1"]
        assert plan.causal is False

    def test_causal_rides_on_the_step(self):
        manager = _manager()
        manager.plan(AttentionStep(causal=True), _ctx([_view("r0", [0], prefix=1, fresh=1)]))
        assert manager._current_plans["main"].causal is True

    def test_planning_under_a_graph_slot_is_rejected(self):
        """Eager-only: there is no captured replay of a gather whose shape the
        step decides, so a lease is a config error rather than a silent
        fall-through to a wrong path."""
        manager = _manager()
        ctx = _ctx([_view("r0", [0], prefix=2, fresh=2)])
        ctx.slot_lease = SlotLease(slot=0, bucket=None)
        with pytest.raises(AssertionError, match="eager-only"):
            manager.plan(AttentionStep(causal=False), ctx)

    def test_missing_kv_plan_is_named(self):
        manager = _manager()
        ctx = _ctx([])
        ctx.plan_results.clear()
        with pytest.raises(AssertionError, match="expected plan result from kv"):
            manager.plan(AttentionStep(causal=False), ctx)


class TestPrefixGather:
    def _planned(self, manager, views):
        manager.plan(AttentionStep(causal=False), _ctx(views))
        return manager._current_plans["main"].segments

    def test_gather_reads_the_prefix_rows_and_stops_at_its_length(self):
        manager = _manager()
        layer = _layer_tensor()
        (segment,) = self._planned(manager, [_view("r0", [5, 6], prefix=6, fresh=2)])

        k, v = manager._prefix_kv(segment, 0, layer)
        assert k.shape == (6, NUM_KV_HEADS, HEAD_DIM)
        assert torch.equal(k, _expected_prefix(layer, [5, 6], 6, 0))
        assert torch.equal(v, _expected_prefix(layer, [5, 6], 6, 1))

    def test_gather_is_reused_across_steps_and_layers(self):
        manager = _manager()
        layer = _layer_tensor()
        (segment,) = self._planned(manager, [_view("r0", [0], prefix=3, fresh=2)])
        first_k, _ = manager._prefix_kv(segment, 0, layer)

        # each layer is its own gather off its own layer view...
        other_k, _ = manager._prefix_kv(segment, 1, _layer_tensor(fill=1000.0))
        assert torch.equal(other_k, first_k + 1000.0)

        # ...but the same (stream, layer) is served from the cache, which is
        # the point: the pages are read once, not once per denoise step
        layer.mul_(-1)
        (again,) = self._planned(manager, [_view("r0", [0], prefix=3, fresh=2)])
        cached_k, _ = manager._prefix_kv(again, 0, layer)
        assert cached_k is first_k

    def test_a_bumped_generation_invalidates_the_gather(self):
        """The stream epoch is the whole invalidation story: same request,
        same label, same length, same pages — different contents."""
        manager = _manager()
        layer = _layer_tensor()
        (segment,) = self._planned(manager, [_view("r0", [0], prefix=3, fresh=2)])
        stale, _ = manager._prefix_kv(segment, 0, layer)

        layer.mul_(-1)
        (moved,) = self._planned(
            manager, [_view("r0", [0], prefix=3, fresh=2, generation=1)]
        )
        fresh, _ = manager._prefix_kv(moved, 0, layer)
        assert not torch.equal(fresh, stale)
        assert torch.equal(fresh, _expected_prefix(layer, [0], 3, 0))

    def test_a_grown_prefix_invalidates_the_gather(self):
        manager = _manager()
        layer = _layer_tensor()
        (segment,) = self._planned(manager, [_view("r0", [0, 1], prefix=3, fresh=1)])
        manager._prefix_kv(segment, 0, layer)

        (grown,) = self._planned(manager, [_view("r0", [0, 1], prefix=5, fresh=1)])
        k, _ = manager._prefix_kv(grown, 0, layer)
        assert torch.equal(k, _expected_prefix(layer, [0, 1], 5, 0))

    def test_request_lifecycle_drops_the_gather(self):
        manager = _manager()
        layer = _layer_tensor()
        (segment,) = self._planned(manager, [_view("r0", [0], prefix=3, fresh=1)])
        manager._prefix_kv(segment, 0, layer)
        assert manager._prefix_cache

        manager.remove_request("r0")
        assert not manager._prefix_cache

    def test_run_without_the_fresh_kv_is_an_error(self):
        manager = _manager()
        self._planned(manager, [_view("r0", [0], prefix=3, fresh=1)])
        with pytest.raises(AssertionError, match="dense attention runs on"):
            manager.run(torch.zeros(1, 1, 1), "main", _layer_tensor())

    def test_running_under_capture_is_refused(self, monkeypatch):
        """`plan` guards every path that plans, which leaves the piecewise
        region that declares `reuses_outer_plan` and never plans. Its capture
        would bake in this step's shapes and the address of a cached prefix,
        so `run` refuses too."""
        manager = _manager()
        self._planned(manager, [_view("r0", [0], prefix=3, fresh=1)])
        monkeypatch.setattr(manager, "_on_cuda", True)
        monkeypatch.setattr(
            torch.cuda, "is_current_stream_capturing", lambda: True
        )
        fresh = torch.zeros(1, NUM_KV_HEADS, HEAD_DIM)
        with pytest.raises(RuntimeError, match="cannot be captured"):
            manager.run(
                fresh, "main", _layer_tensor(), k=fresh, v=fresh, layer_idx=0,
            )


class TestBackendSelection:
    def test_dense_spec_builds_dense_when_fa3_is_there(self):
        """...and degrades to the paged backend when it is not: the two are
        drop-in for each other, so a missing wheel costs speed, not serving."""
        spec = AttentionSpec(
            resource_key="attn",
            nodes={"llm"},
            config=AttentionConfig(kv_cache="kv", backend=AttnBackend.DENSE),
            kv_config=_kv_config(),
        )
        manager = AttentionManager.build(
            spec, EngineResourceInfo(device=torch.device("cpu"))
        )
        expected = (
            DenseAttentionManager if _fa3_unavailable_reason() is None
            else FlashInferManager
        )
        assert isinstance(manager, expected)
        assert manager.depends_on() == {"kv"}

    def test_only_the_dense_backend_skips_the_kv_write(self):
        assert _manager().requires_kv_write is False
        assert FlashInferManager(
            kv_cache="kv",
            device=torch.device("cpu"),
            dtype=torch.float32,
            kv_config=_kv_config(),
        ).requires_kv_write is True


@pytest.fixture
def kv_manager(monkeypatch):
    """A KVManager over a CPU cache. Its transfer engine registers the cache
    for CUDA IPC at construction, which needs a GPU and has nothing to do with
    page bookkeeping, so it is stubbed out."""
    from mstar.engine.resources.kv import manager as kv_manager_module

    monkeypatch.setattr(
        kv_manager_module, "KVTransferManager", lambda info, kv_cache: None
    )
    return kv_manager_module.KVManager(
        cfg=_kv_config(),
        name="kv",
        joint_comm_group=None,
        transfer_engine_info=None,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


class TestStreamGeneration:
    """`CacheStream.generation` exists for exactly one reader, so its contract
    is worth pinning here: it moves when the resident bytes may have moved
    under that reader, and stands still when they cannot have."""

    def _plan_views(self, kv, segments, **step_kwargs):
        step = KVStep(segments=tuple(segments), **step_kwargs)
        ctx = StepContext(
            request_ids=tuple(s.request_id for s in segments),
            graph_walk="gen", slot=0, capture=False,
        )
        assert kv.admit(step, ctx).ok
        out = kv.plan(step, ctx)
        kv.commit(step, ctx)
        return out, step, ctx

    def test_appending_does_not_invalidate_a_resident_prefix(self, kv_manager):
        kv = kv_manager
        kv.ingest_request("r0")
        out, _, _ = self._plan_views(kv, [Segment("r0", "main", 6)])
        first = out["main"].views[0].generation

        # a second step appends within the pages already held: same pages,
        # same resident bytes, so a gather of the first 6 tokens stays good
        out, _, _ = self._plan_views(kv, [Segment("r0", "main", 1)])
        assert out["main"].views[0].generation == first

    def test_taking_new_pages_moves_the_epoch(self, kv_manager):
        kv = kv_manager
        kv.ingest_request("r0")
        out, _, _ = self._plan_views(kv, [Segment("r0", "main", 4)])
        first = out["main"].views[0].generation

        out, _, _ = self._plan_views(kv, [Segment("r0", "main", 4)])
        assert out["main"].views[0].generation > first

    def test_a_fork_moves_the_target_epoch(self, kv_manager):
        """The case a page-identity fingerprint would miss: the fork copies
        over the target's existing pages, so nothing about the page list or
        the length changes — only the contents."""
        kv = kv_manager
        kv.ingest_request("r0")
        self._plan_views(kv, [Segment("r0", "main", 4)])
        # give the target a stream of its own at the same length
        self._plan_views(kv, [Segment("r0", "cfg", 4)])
        out, _, _ = self._plan_views(kv, [Segment("r0", "cfg", 0)])
        before = out["cfg"].views[0]
        # a view holds the stream's page list by reference, so snapshot it
        before_pages, before_len = list(before.page_idxs), before.length

        self._plan_views(kv, [Segment("r0", "cfg", 0)], post_forks=(("main", "cfg"),))
        out, _, _ = self._plan_views(kv, [Segment("r0", "cfg", 0)])
        after = out["cfg"].views[0]
        assert list(after.page_idxs) == before_pages
        assert after.length == before_len
        assert after.generation > before.generation

    def test_a_reset_moves_the_epoch(self, kv_manager):
        kv = kv_manager
        kv.ingest_request("r0")
        out, _, _ = self._plan_views(kv, [Segment("r0", "main", 4)])
        before = out["main"].views[0].generation

        kv.reset_request("r0")
        out, _, _ = self._plan_views(kv, [Segment("r0", "main", 4)])
        assert out["main"].views[0].generation > before
