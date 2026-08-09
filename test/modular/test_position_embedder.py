"""Unit tests for the positional embedder and pool-committed advances.

``RopeEmbedder.plan`` turns a segment list plus the pool's counters into a
``PositionPlan``; the cache manager's advance paths resolve each step's
position delta (default span, explicit ``pos_id_ns``, or the
``custom_pos_advance`` side channel) and record it only through
``KVCachePool.commit``.
"""

from __future__ import annotations

import sys
import threading

sys.path.insert(0, ".")

import torch

from mstar.engine.cache_manager import BatchedCfgInfo, FlashInferCacheManager
from mstar.engine.kv_store import (
    KVCacheConfig,
    PageAllocator,
    PagedAllocationManager,
    StoreWritePolicy,
)
from mstar.engine.resources import KVCachePool, RopeEmbedder, Segment


def _make_manager(max_num_pages: int = 16, page_size: int = 8) -> PagedAllocationManager:
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


def _make_cache_manager(request_ids, labels=("main",)):
    alloc = _make_manager()
    for rid in request_ids:
        alloc.add_request(rid, list(labels))
    cm = FlashInferCacheManager(
        request_ids=list(request_ids),
        active_labels_per_request={rid: labels[0] for rid in request_ids},
        kv_cache=None,
        alloc_manager=alloc,
        buffer_manager=None,
        kv_cache_config=alloc.config,
        device="cpu",
    )
    return cm


def _seed_planned_seq_lens(cm, label, seq_lens):
    """Set a label's planned seq_lens through the pre-planned fast path
    (the same shortcut the worker's plan thread uses), which records them
    without building a FlashInfer wrapper."""
    cm.plan_rope(seq_lens=seq_lens, label=label)
    cm._pre_planned_labels.add(label)
    cm.plan_attention(seq_lens=seq_lens, label=label)


class TestRopeEmbedderPlan:
    def test_ids_follow_segment_order_and_counters(self):
        alloc = _make_manager()
        alloc.add_request("a", ["main"])
        alloc.add_request("b", ["main"])
        pool = KVCachePool(alloc)
        pool.commit(Segment("a", "main", 5))  # counter at 5
        embedder = RopeEmbedder()

        plan = embedder.plan(
            [Segment("a", "main", 3), Segment("b", "main", 2)], pool
        )
        assert plan.pos_ids.tolist() == [5, 6, 7, 0, 1]
        assert plan.advance == (3, 2)
        assert plan.pos_ids.device.type == "cpu"

    def test_zero_span_segment_contributes_no_ids(self):
        alloc = _make_manager()
        alloc.add_request("a", ["main"])
        pool = KVCachePool(alloc)
        embedder = RopeEmbedder()

        plan = embedder.plan([Segment("a", "main", 0)], pool)
        assert plan.pos_ids.numel() == 0
        assert plan.advance == (0,)


class TestAdvanceCommitsThroughPool:
    def test_plain_decode_advances_by_span(self):
        cm = _make_cache_manager(["a", "b"])
        _seed_planned_seq_lens(cm, "main", [1, 1])

        cm.advance_seq_lens()
        assert cm.kv_pool.positions("a", "main") == 1
        assert cm.kv_pool.positions("b", "main") == 1
        assert cm.kv_pool.view(Segment("a", "main", 0)).length == 1

    def test_custom_pos_advance_overrides_span(self):
        """The vision-prefill shape: position span larger than the token
        count, delivered through the side channel."""
        cm = _make_cache_manager(["a"])
        _seed_planned_seq_lens(cm, "main", [4])
        cm.set_custom_pos_advance([90], label="main")

        cm.advance_seq_lens()
        assert cm.kv_pool.view(Segment("a", "main", 0)).length == 4
        assert cm.kv_pool.positions("a", "main") == 90
        # The side channel is consumed, not persistent.
        _seed_planned_seq_lens(cm, "main", [1])
        cm.advance_seq_lens()
        assert cm.kv_pool.positions("a", "main") == 91

    def test_pos_id_ns_scalar_override(self):
        """The image-block shape: many tokens, one position."""
        cm = _make_cache_manager(["a"])
        _seed_planned_seq_lens(cm, "main", [16])

        cm.advance_seq_lens(pos_id_ns=1)
        assert cm.kv_pool.view(Segment("a", "main", 0)).length == 16
        assert cm.kv_pool.positions("a", "main") == 1

    def test_pos_id_ns_per_request_list(self):
        cm = _make_cache_manager(["a", "b"])
        _seed_planned_seq_lens(cm, "main", [2, 2])

        cm.advance_seq_lens(pos_id_ns=[5, 7])
        assert cm.kv_pool.positions("a", "main") == 5
        assert cm.kv_pool.positions("b", "main") == 7

    def test_batched_cfg_advances_every_label(self):
        cm = _make_cache_manager(["a"], labels=("main", "uncond"))
        cm._batched_cfg_info = BatchedCfgInfo(
            per_label_seq_len={"main": [3], "uncond": [3]}
        )

        cm.advance_seq_lens()
        for label in ("main", "uncond"):
            assert cm.kv_pool.view(Segment("a", label, 0)).length == 3
            assert cm.kv_pool.positions("a", label) == 3

    def test_advance_seq_len_singular(self):
        cm = _make_cache_manager(["a"])

        cm.advance_seq_len(4, pos_id_n=2)
        assert cm.kv_pool.view(Segment("a", "main", 0)).length == 4
        assert cm.kv_pool.positions("a", "main") == 2


class TestPlanRopeUsesEmbedder:
    def test_plan_rope_builds_ids_from_pool_counters(self):
        cm = _make_cache_manager(["a"])
        cm.kv_pool.commit(Segment("a", "main", 3))  # counter at 3

        cm.plan_rope(seq_lens=[2], label="main")
        assert cm._plan_states["main"].pos_ids.tolist() == [3, 4]

    def test_explicit_pos_ids_pass_through(self):
        cm = _make_cache_manager(["a"])
        explicit = torch.tensor([7, 9], dtype=torch.long)

        cm.plan_rope(seq_lens=[2], pos_ids=explicit, label="main")
        assert cm._plan_states["main"].pos_ids.tolist() == [7, 9]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
