"""Unit tests for step declarations and the runner driving them.

A declaring submodule hands the runner a ``StepDeclaration``; the runner's
``drive`` must make exactly the plan calls a facade-driven model would, in
declaration order, and ``commit`` must advance exactly the declared spans
on the pool (with declared position advances), then apply post forks.
"""

from __future__ import annotations

import dataclasses
import sys
import threading

sys.path.insert(0, ".")

import pytest

from mstar.engine.kv_store import (
    KVCacheConfig,
    PageAllocator,
    PagedAllocationManager,
    StoreWritePolicy,
)
from mstar.engine.resources import KVCachePool, PlanSpec, StepDeclaration
from mstar.engine.resources.step import StepRunner


def _make_pool(max_num_pages: int = 16, page_size: int = 8):
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
    return KVCachePool(manager), manager


class _RecordingManager:
    """Step-surface double that records every call with its kwargs."""

    def __init__(self, request_ids):
        self.request_ids = request_ids
        self.calls = []

    def snapshot_all(self, from_label, to_label):
        self.calls.append(("snapshot_all", from_label, to_label))

    def plan_attention(self, **kwargs):
        self.calls.append(("plan_attention", kwargs))

    def plan_attention_batched_cfg(self, **kwargs):
        self.calls.append(("plan_attention_batched_cfg", kwargs))

    def plan_rope(self, **kwargs):
        self.calls.append(("plan_rope", kwargs))

    def plan_rope_batched_cfg(self, **kwargs):
        self.calls.append(("plan_rope_batched_cfg", kwargs))


class _PoolManager:
    """Commit-side double: a real pool plus the step addressing commit
    reads, recording fork calls so ordering is checkable."""

    def __init__(self, pool, request_ids):
        self.kv_pool = pool
        self.request_ids = request_ids
        self.forks = []

    def snapshot_all(self, from_label, to_label):
        self.forks.append((from_label, to_label))


class TestDeclarationTypes:
    def test_plan_spec_is_frozen(self):
        plan = PlanSpec(labels=("main",), spans={"main": (4,)})
        with pytest.raises(dataclasses.FrozenInstanceError):
            plan.is_causal = False

    def test_declaration_is_frozen(self):
        decl = StepDeclaration(
            plans=(PlanSpec(labels=("main",), spans={"main": (1,)}),)
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            decl.plans = ()

    def test_defaults(self):
        plan = PlanSpec(labels=("main",), spans={"main": (2, 3)})
        assert plan.is_causal and plan.write_store and plan.commit
        assert not plan.dense_gen and not plan.rope and not plan.combined
        assert plan.pos_advance is None
        decl = StepDeclaration(plans=(plan,))
        assert decl.pre_forks == () and decl.post_forks == ()


class TestDrive:
    def test_plain_plan_with_rope(self):
        cm = _RecordingManager(["r0", "r1"])
        decl = StepDeclaration(plans=(
            PlanSpec(
                labels=("main",), spans={"main": (3, 1)},
                is_causal=True, write_store=True, rope=True,
            ),
        ))
        StepRunner().drive(decl, cm)
        assert [c[0] for c in cm.calls] == ["plan_attention", "plan_rope"]
        attn = cm.calls[0][1]
        assert attn["seq_lens"] == [3, 1]
        assert attn["label"] == "main"
        assert attn["is_causal"] and attn["write_store"]
        assert attn["dense_gen"] is False
        rope = cm.calls[1][1]
        assert rope["seq_lens"] == [3, 1]
        assert rope["label"] == "main" and rope["pos_ids"] is None

    def test_combined_plan_orders_forks_first(self):
        cm = _RecordingManager(["r0"])
        decl = StepDeclaration(
            plans=(
                PlanSpec(
                    labels=("main", "uncond"),
                    spans={"main": (5,), "uncond": (7,)},
                    is_causal=False, write_store=False,
                    dense_gen=True, rope=True,
                    rope_pos_ids={"main": []},
                    combined=True,
                ),
            ),
            pre_forks=(("main", "cfg_text"),),
        )
        StepRunner().drive(decl, cm)
        assert [c[0] for c in cm.calls] == [
            "snapshot_all", "plan_attention_batched_cfg", "plan_rope_batched_cfg",
        ]
        assert cm.calls[0][1:] == ("main", "cfg_text")
        attn = cm.calls[1][1]
        assert attn["labels"] == ["main", "uncond"]
        assert attn["seq_lens"] == {"main": [5], "uncond": [7]}
        assert not attn["is_causal"] and not attn["write_store"]
        assert attn["dense_gen"] is True
        assert attn["combined_label"] == "_cfg_batched"
        rope = cm.calls[2][1]
        assert rope["per_label_pos_ids"] == {"main": []}

    def test_plans_run_in_declaration_order(self):
        cm = _RecordingManager(["r0"])
        decl = StepDeclaration(plans=(
            PlanSpec(labels=("main",), spans={"main": (1,)}, rope=True),
            PlanSpec(labels=("cfg_img",), spans={"cfg_img": (1,)}, rope=True),
        ))
        StepRunner().drive(decl, cm)
        labels = [
            c[1]["label"] for c in cm.calls
            if c[0] in ("plan_attention", "plan_rope")
        ]
        assert labels == ["main", "main", "cfg_img", "cfg_img"]

    def test_no_rope_plans_attention_only(self):
        cm = _RecordingManager(["r0"])
        decl = StepDeclaration(plans=(
            PlanSpec(labels=("main",), spans={"main": (2,)}),
        ))
        StepRunner().drive(decl, cm)
        assert [c[0] for c in cm.calls] == ["plan_attention"]

    def test_single_label_combined_plan_batches(self):
        cm = _RecordingManager(["r0"])
        decl = StepDeclaration(plans=(
            PlanSpec(
                labels=("main",), spans={"main": (6,)}, combined=True,
                is_causal=False, write_store=False,
            ),
        ))
        StepRunner().drive(decl, cm)
        assert [c[0] for c in cm.calls] == ["plan_attention_batched_cfg"]
        assert cm.calls[0][1]["labels"] == ["main"]


class TestCommit:
    def _pool_manager(self, request_ids, labels):
        pool, manager = _make_pool()
        for rid in request_ids:
            pool.add_request(rid, labels)
        return _PoolManager(pool, request_ids), manager

    def test_commits_declared_spans(self):
        cm, manager = self._pool_manager(["r0", "r1"], ["main"])
        decl = StepDeclaration(plans=(
            PlanSpec(labels=("main",), spans={"main": (3, 1)}),
        ))
        StepRunner().commit(decl, cm)
        assert manager.get_state("r0", "main").seq_len == 3
        assert manager.get_state("r0", "main").position_id_start == 3
        assert manager.get_state("r1", "main").seq_len == 1
        assert manager.get_state("r1", "main").position_id_start == 1

    def test_pos_advance_overrides_span(self):
        cm, manager = self._pool_manager(["r0"], ["main"])
        decl = StepDeclaration(plans=(
            PlanSpec(
                labels=("main",), spans={"main": (10,)}, pos_advance=(14,),
            ),
        ))
        StepRunner().commit(decl, cm)
        state = manager.get_state("r0", "main")
        assert state.seq_len == 10
        assert state.position_id_start == 14

    def test_non_committing_plan_is_skipped(self):
        cm, manager = self._pool_manager(["r0"], ["main", "uncond"])
        decl = StepDeclaration(plans=(
            PlanSpec(
                labels=("main", "uncond"),
                spans={"main": (6,), "uncond": (6,)},
                commit=False,
            ),
        ))
        StepRunner().commit(decl, cm)
        assert manager.get_state("r0", "main").seq_len == 0
        assert manager.get_state("r0", "uncond").seq_len == 0

    def test_combined_plan_commits_every_label(self):
        cm, manager = self._pool_manager(["r0", "r1"], ["main", "uncond"])
        decl = StepDeclaration(plans=(
            PlanSpec(
                labels=("main", "uncond"),
                spans={"main": (4, 2), "uncond": (5, 3)},
            ),
        ))
        StepRunner().commit(decl, cm)
        assert manager.get_state("r0", "main").seq_len == 4
        assert manager.get_state("r1", "main").seq_len == 2
        assert manager.get_state("r0", "uncond").seq_len == 5
        assert manager.get_state("r1", "uncond").seq_len == 3

    def test_zero_span_is_a_no_op(self):
        cm, manager = self._pool_manager(["r0"], ["main"])
        decl = StepDeclaration(plans=(
            PlanSpec(labels=("main",), spans={"main": (0,)}),
        ))
        StepRunner().commit(decl, cm)
        state = manager.get_state("r0", "main")
        assert state.seq_len == 0 and state.position_id_start == 0

    def test_post_forks_apply_after_commits(self):
        cm, manager = self._pool_manager(["r0"], ["main", "cfg_text"])
        committed_at_fork = []
        original = cm.snapshot_all

        def snapshot_all(from_label, to_label):
            committed_at_fork.append(manager.get_state("r0", "main").seq_len)
            original(from_label, to_label)

        cm.snapshot_all = snapshot_all
        decl = StepDeclaration(
            plans=(PlanSpec(labels=("main",), spans={"main": (8,)}),),
            post_forks=(("main", "cfg_text"),),
        )
        StepRunner().commit(decl, cm)
        assert cm.forks == [("main", "cfg_text")]
        assert committed_at_fork == [8]


class TestDeclareStepDefault:
    def test_base_submodule_declares_nothing(self):
        from mstar.model.submodule_base import NodeSubmodule

        class _Stub(NodeSubmodule):
            __abstractmethods__ = frozenset()

        assert _Stub().declare_step("walk", None, []) is None
