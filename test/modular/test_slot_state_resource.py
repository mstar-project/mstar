"""The slot-state resource: fixed-size per-request state as an engine
resource, driven the way the runner drives it.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources import (
    SINK_SLOT,
    SlotStateConfig,
    SlotStateSpec,
    SlotStateStep,
    SlotTensorSpec,
    StepRunner,
    SubmoduleStep,
)
from mstar.engine.resources.base import CGSlotSpec, EngineResourceInfo
from mstar.engine.resources.slot_state.manager import SlotStateManager
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext

L, H, D, C, W = 2, 3, 4, 6, 3


def _config(max_slots: int = 3, shard: bool = False) -> SlotStateConfig:
    return SlotStateConfig(
        tensors={
            "recurrent": SlotTensorSpec(
                shape=(L, H, D, D), dtype=torch.float32, slot_dim=1,
                shard_dim=1 if shard else None,
            ),
            "conv": SlotTensorSpec(
                shape=(L, C, W), dtype=torch.float32, slot_dim=1,
                shard_dim=1 if shard else None,
            ),
        },
        max_slots=max_slots,
    )


def _manager(max_slots: int = 3) -> SlotStateManager:
    return SlotStateManager(_config(max_slots), "kda", torch.device("cpu"))


def _ctx(rids, walk="decode", padded=None, lease=None, capture=False) -> StepContext:
    ctx = StepContext(
        request_ids=tuple(rids), graph_walk=walk, slot=0, capture=capture,
        slot_lease=lease,
    )
    if padded is not None:
        ctx.set_padded_rids(tuple(padded))
    return ctx


def _step(rids, spans, mode) -> SlotStateStep:
    return SlotStateStep(
        segments=tuple(
            Segment(request_id=rid, label="main", span=span)
            for rid, span in zip(rids, spans, strict=True)
        ),
        mode=mode,
    )


def _drive(m: SlotStateManager, rids, spans, mode, ctx=None):
    ctx = ctx or _ctx(rids)
    step = _step(ctx.padded_request_ids if ctx.padded_request_ids else rids, spans, mode)
    outcome = m.admit(step, ctx)
    assert outcome.ok, outcome.reason
    plan = m.plan(step, ctx)
    m.commit(step, ctx)
    return plan


class TestPoolShape:
    def test_pool_holds_max_slots_plus_sink(self):
        m = _manager(3)
        assert tuple(m.pool("recurrent").shape) == (L, 4, H, D, D)
        assert tuple(m.pool("conv").shape) == (L, 4, C, W)
        assert tuple(m.slot_view("recurrent", 2).shape) == (L, H, D, D)
        assert m.num_free == 3

    def test_build_shards_a_declared_head_axis(self):
        class _Group:
            world_size = 3
            rank = 0

        spec = SlotStateSpec(resource_key="kda", nodes={"llm"}, config=_config(shard=True))
        m = SlotStateManager.build(
            spec, EngineResourceInfo(device=torch.device("cpu"), joint_comm_group=_Group()),
        )
        assert tuple(m.pool("recurrent").shape) == (L, 4, 1, D, D)
        assert tuple(m.pool("conv").shape) == (L, 4, 2, W)
        # idempotent
        m.config.shard(3)
        assert m.config.tensors["conv"].shape == (L, 2, W)

    def test_yaml_override_sets_max_slots(self):
        spec = SlotStateSpec(resource_key="kda", nodes={"llm"}, config=_config())
        spec.apply_yaml_overrides(max_slots=7)
        assert spec.config.max_slots == 7
        with pytest.raises(TypeError):
            spec.apply_yaml_overrides(page_size=1)


class TestLifecycle:
    def test_ingest_leases_nothing_first_chunk_leases_one(self):
        m = _manager()
        m.ingest_request("a")
        assert m.slot_of("a") is None and m.num_free == 3
        plan = _drive(m, ["a"], [5], "chunk")
        slot = m.slot_of("a")
        assert slot is not None and slot != SINK_SLOT
        assert plan.mode == "chunk"
        assert plan.slot_index.tolist() == [slot]
        assert [(s.request_id, s.slot, s.q_start, s.q_len, s.ctx_start, s.real) for s in plan.spans] == [
            ("a", slot, 0, 5, 0, True)
        ]
        assert m.committed("a") == 5

    def test_commit_after_forward_not_at_plan(self):
        m = _manager()
        m.ingest_request("a")
        step = _step(["a"], [4], "chunk")
        ctx = _ctx(["a"])
        assert m.admit(step, ctx).ok
        plan = m.plan(step, ctx)
        assert plan.spans[0].ctx_start == 0
        assert m.committed("a") == 0, "nothing lands until commit"
        m.commit(step, ctx)
        assert m.committed("a") == 4
        # a second chunk starts where the first committed
        plan2 = _drive(m, ["a"], [2], "chunk")
        assert plan2.spans[0].ctx_start == 4
        assert m.committed("a") == 6

    def test_uncommitted_step_leaves_the_count(self):
        m = _manager()
        m.ingest_request("a")
        _drive(m, ["a"], [3], "chunk")
        step = SlotStateStep(segments=(Segment("a", "main", 1),), mode="step", commit=False)
        ctx = _ctx(["a"])
        assert m.admit(step, ctx).ok
        m.plan(step, ctx)
        m.commit(step, ctx)
        assert m.committed("a") == 3

    def test_decode_batch_rows_follow_segment_order(self):
        m = _manager()
        for rid in ("a", "b", "c"):
            m.ingest_request(rid)
            _drive(m, [rid], [2], "chunk")
        slots = {rid: m.slot_of(rid) for rid in ("a", "b", "c")}
        assert len(set(slots.values())) == 3 and SINK_SLOT not in slots.values()
        plan = _drive(m, ["c", "a"], [1, 1], "step")
        assert plan.slot_index.tolist() == [slots["c"], slots["a"]]
        assert m.committed("c") == 3 and m.committed("a") == 3 and m.committed("b") == 2

    def test_step_before_first_chunk_is_refused(self):
        m = _manager()
        m.ingest_request("a")
        step = _step(["a"], [1], "step")
        ctx = _ctx(["a"])
        assert m.admit(step, ctx).ok
        with pytest.raises(RuntimeError, match="no committed tokens"):
            m.plan(step, ctx)

    def test_remove_frees_the_slot_and_zeroes_on_reuse(self):
        m = _manager(1)
        m.ingest_request("a")
        _drive(m, ["a"], [3], "chunk")
        slot = m.slot_of("a")
        m.slot_view("recurrent", slot).fill_(5.0)
        m.remove_request("a")
        assert m.num_free == 1 and m.slot_of("a") is None
        m.remove_request("a")  # idempotent
        m.ingest_request("b")
        _drive(m, ["b"], [1], "chunk")
        assert m.slot_of("b") == slot
        assert m.slot_view("recurrent", slot).abs().sum() == 0

    def test_exhaustion_is_a_retryable_admit_failure(self):
        m = _manager(1)
        for rid in ("a", "b"):
            m.ingest_request(rid)
        _drive(m, ["a"], [1], "chunk")
        step = _step(["b"], [1], "chunk")
        outcome = m.admit(step, _ctx(["b"]))
        assert not outcome.ok
        assert outcome.ready
        assert "exhausted" in outcome.reason.message
        # AllocationFailed is what the worker's hold/backoff path keys on
        assert type(outcome.reason).__name__ == "AllocationFailed"
        assert outcome.reason.request_id == "b"
        m.remove_request("a")
        assert m.admit(step, _ctx(["b"])).ok

    def test_zero_span_rows_lease_nothing(self):
        m = _manager()
        m.ingest_request("a")
        step = _step(["a"], [0], "chunk")
        ctx = _ctx(["a"])
        assert m.admit(step, ctx).ok
        plan = m.plan(step, ctx)
        assert m.slot_of("a") is None
        assert plan.slot_index.tolist() == [SINK_SLOT]
        assert plan.spans[0].real is False


class TestCaptureAndPadding:
    def test_capture_dummies_lease_and_any_reset_releases(self):
        """Capture drives dummy rids as the step's own request_ids: they
        lease like real ones, and every reset (free=False between captures,
        free=True at release_all) hands the slot back — the runner holds
        dummies per (config, cg slot), so keeping slots across resets would
        drain the pool on the second slot's capture.
        """
        m = _manager(2)
        dummies = ["__cg_x_0__", "__cg_x_1__"]
        for rid in dummies:
            m.ingest_request(rid)
        ctx = _ctx(dummies, lease=SlotLease(slot=0, bucket=BucketKey("decode", 2, 2)), capture=True)
        m.build_cuda_graph_buffers(
            [CGSlotSpec(bucket=ctx.slot_lease.bucket, slot=0, config=None)], max_bs=2, max_seq_len=2,
        )
        # a decode capture: single-token rows that never ran a chunk are
        # legal under ctx.capture (their zero state is what gets recorded)
        step = _step(dummies, [1, 1], "step")
        assert m.admit(step, ctx).ok
        plan = m.plan(step, ctx)
        assert plan.mode == "step"
        assert m.num_free == 0
        assert plan.slot_index.data_ptr() == m._cg_index[0].data_ptr()
        for rid in dummies:
            m.reset_request(rid, free=False)
        assert m.num_free == 2 and all(m.committed(r) == 0 for r in dummies)
        assert all(m.slot_of(r) is None for r in dummies)

    def test_capture_sequence_over_two_cg_slots_fits_the_pool(self):
        """The runner's capture order on a pool of exactly max_bs slots:
        (bs, slot 0) then (bs, slot 1) with distinct dummy rows, a reset
        after each; every capture must admit."""
        m = _manager(2)
        b = BucketKey("decode", 2, 2)
        m.build_cuda_graph_buffers(
            [CGSlotSpec(bucket=b, slot=s, config=None) for s in (0, 1)], max_bs=2, max_seq_len=2,
        )
        for slot in (0, 1):
            rids = [f"__cg_{slot}_{i}__" for i in range(2)]
            for rid in rids:
                m.ingest_request(rid)
            ctx = _ctx(rids, lease=SlotLease(slot=slot, bucket=b), capture=True)
            for _ in range(3):  # NUM_WARMUP forwards + the capture, each re-prepared
                step = _step(rids, [1, 1], "step")
                assert m.admit(step, ctx).ok, slot
                plan = m.plan(step, ctx)
                assert plan.slot_index.data_ptr() == m._cg_index[slot].data_ptr()
                for rid in rids:
                    m.reset_request(rid, free=False)
        for slot in (0, 1):
            for rid in [f"__cg_{slot}_{i}__" for i in range(2)]:
                m.reset_request(rid, free=True)
        assert m.num_free == 2

    def test_padding_rows_read_the_sink_and_never_commit(self):
        m = _manager(3)
        m.ingest_request("a")
        _drive(m, ["a"], [3], "chunk")
        slot_a = m.slot_of("a")
        pad = "__cg_x_1__"
        m.ingest_request(pad)  # ingested by the dummy pool, no slot after release_all
        bucket = BucketKey("decode", 2, 2)
        m.build_cuda_graph_buffers([CGSlotSpec(bucket=bucket, slot=1, config=None)], max_bs=2, max_seq_len=2)
        ctx = _ctx(["a"], padded=["a", pad], lease=SlotLease(slot=1, bucket=bucket))
        step = _step(["a", pad], [1, 1], "step")
        assert m.admit(step, ctx).ok
        plan = m.plan(step, ctx)
        m.commit(step, ctx)
        assert plan.slot_index.tolist() == [slot_a, SINK_SLOT]
        assert plan.slot_index.data_ptr() == m._cg_index[1].data_ptr()
        assert [s.real for s in plan.spans] == [True, False]
        assert m.committed("a") == 4
        assert m.committed(pad) == 0 and m.slot_of(pad) is None
        assert m.num_free == 2, "a padding row must not lease a slot"

    def test_replay_overwrites_the_same_static_buffer(self):
        m = _manager(3)
        bucket = BucketKey("decode", 2, 2)
        m.build_cuda_graph_buffers([CGSlotSpec(bucket=bucket, slot=0, config=None)], max_bs=2, max_seq_len=2)
        for rid in ("a", "b"):
            m.ingest_request(rid)
            _drive(m, [rid], [1], "chunk")
        lease = SlotLease(slot=0, bucket=bucket)
        p1 = _drive(m, ["a", "b"], [1, 1], "step", ctx=_ctx(["a", "b"], padded=["a", "b"], lease=lease))
        first = p1.slot_index.tolist()
        p2 = _drive(m, ["b", "a"], [1, 1], "step", ctx=_ctx(["b", "a"], padded=["b", "a"], lease=lease))
        assert p2.slot_index.data_ptr() == p1.slot_index.data_ptr()
        assert p2.slot_index.tolist() == first[::-1]

    def test_cg_buffers_grow_only(self):
        m = _manager(4)
        bucket = BucketKey("decode", 4, 4)
        m.build_cuda_graph_buffers([CGSlotSpec(bucket=bucket, slot=0, config=None)], max_bs=4, max_seq_len=4)
        ptr = m._cg_index[0].data_ptr()
        m.build_cuda_graph_buffers(
            [CGSlotSpec(bucket=BucketKey("decode", 2, 2), slot=0, config=None)], max_bs=2, max_seq_len=2,
        )
        assert m._cg_index[0].data_ptr() == ptr and m._cg_index[0].shape[0] == 4

    def test_leased_plan_without_buffers_is_loud(self):
        m = _manager(2)
        m.ingest_request("a")
        _drive(m, ["a"], [1], "chunk")
        ctx = _ctx(["a"], lease=SlotLease(slot=0, bucket=BucketKey("decode", 1, 1)))
        step = _step(["a"], [1], "step")
        assert m.admit(step, ctx).ok
        with pytest.raises(RuntimeError, match="static index buffer"):
            m.plan(step, ctx)


class TestThroughTheRunner:
    def test_runner_sweeps_admit_plan_commit_remove(self):
        m = _manager(2)
        runner = StepRunner({"kda": m}, node_resources={"llm": ["kda"]})
        runner.ingest_request("a")
        ctx = _ctx(["a"], walk="prefill")
        step = SubmoduleStep(
            segments=[Segment("a", "main", 6)],
            steps={"kda": SlotStateStep(mode="chunk")},
        )
        step.set_ctx(ctx)
        assert runner.admit(step).ok
        runner.plan(step)
        assert ctx.plan_results["kda"] is m.current_plan()
        runner.commit(step)
        assert m.committed("a") == 6
        runner.remove_request("a")
        assert m.num_free == 2

    def test_spec_declares_no_preplan(self):
        assert _manager().supports_preplan is False
