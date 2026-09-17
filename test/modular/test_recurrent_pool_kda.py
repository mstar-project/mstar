"""Kimi Delta Attention planned against the recurrent state pool, on CPU: the pool hands out
slots and addressing, the ``KDAManager`` adds the token layout (``cu_seqlens``, the decode /
prefill walk), both stage a step ahead, and the torch reference kernels run the paged forward
against the pool's blocks with the same numbers as the explicit-state reference.

CPU-only: the reference kernels loop over rows, so no device is needed."""
from __future__ import annotations

import torch

from mstar.engine.resources import (
    AllocationFailed,
    BucketKey,
    DeltaNetGeometry,
    LinearAttnConfig,
    LinearAttnSpec,
    LinearAttnStep,
    LinearAttnVariant,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
    Segment,
    SlotLease,
    StepContext,
    StepRunner,
    SubmoduleStep,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.linear_attn.kda import KDAManager, KDAPlan
from mstar.engine.resources.recurrent.pool import SINK_SLOT, RecurrentStatePool

POOL, ATTN = "kda_state", "kda_attn"
H, D, W, LAYERS = 2, 8, 4, 2
DEV = torch.device("cpu")


def specs(max_slots=4):
    geometry = DeltaNetGeometry(num_k_heads=H, num_v_heads=H, head_k_dim=D, head_v_dim=D, conv_kernel_size=W)
    return [
        RecurrentStateSpec(POOL, {"LLM"}, RecurrentStateConfig(
            num_layers=LAYERS, blocks=geometry.to_blocks(state_dtype=torch.float32, conv_dtype=torch.float32),
            max_slots=max_slots)),
        LinearAttnSpec(ATTN, {"LLM"}, LinearAttnConfig(
            recurrent_state=POOL, variant=LinearAttnVariant.KDA, gate_lower_bound=-5.0)),
    ]


def build(max_slots=4):
    sp = specs(max_slots)
    info = EngineResourceInfo(device=DEV, dependencies={POOL: sp[0]})
    pool = build_resource(sp[0], info)
    attn = build_resource(sp[1], info)
    assert isinstance(pool, RecurrentStatePool) and isinstance(attn, KDAManager)
    runner = StepRunner({POOL: pool, ATTN: attn})
    return pool, attn, runner


def step(segs, walk="prefill", slot_lease=None, preplan=False):
    s = SubmoduleStep(segments=segs, steps={POOL: RecurrentStep(), ATTN: LinearAttnStep()})
    ctx = StepContext(request_ids=[x.request_id for x in segs], graph_walk=walk, slot=0, capture=False,
                      is_preplan=preplan, slot_lease=slot_lease)
    s.set_ctx(ctx)
    return s, ctx


def test_pool_and_manager_plan_a_step():
    pool, attn, runner = build()
    for rid in ("a", "b"):
        runner.ingest_request(rid)
    s, ctx = step([Segment("a", "main", 5), Segment("b", "main", 3)])
    assert runner.admit(s).ok
    runner.plan(s)
    plan = attn.current_plan("main")
    assert isinstance(plan, KDAPlan) and plan.num_rows == 2 and plan.num_tokens == 8 and not plan.is_decode
    assert plan.cu_seqlens_cpu == [0, 5, 8] and plan.cu_seqlens[:3].tolist() == [0, 5, 8]
    slots = plan.slot_ids_cpu
    assert len(set(slots)) == 2 and SINK_SLOT not in slots and plan.has_state_cpu == [False, False]
    runner.commit(s)
    # the next step of the same rows: the same slots, now holding state, one token each
    s2, _ = step([Segment("b", "main", 1), Segment("a", "main", 1)], walk="decode")
    assert runner.admit(s2).ok
    runner.plan(s2)
    plan2 = attn.current_plan()
    assert plan2.is_decode and plan2.slot_ids_cpu == [slots[1], slots[0]] and plan2.has_state_cpu == [True, True]
    assert plan2.cu_seqlens_cpu == [0, 1, 2]
    # a third request finds the pool full (4 slots: the sink + a + b + one more)
    runner.ingest_request("c")
    runner.ingest_request("d")
    s3, _ = step([Segment("c", "main", 2), Segment("d", "main", 2)])
    out = runner.admit(s3)
    assert not out.ok and isinstance(out.reason, AllocationFailed)
    # c's slot from the failed admit stays reserved for c (like KV pages); freeing a and c returns both
    runner.remove_request("a")
    assert pool.num_free_slots == 1
    runner.remove_request("c")
    assert pool.num_free_slots == 2


def test_padding_rows_address_the_sink():
    pool, attn, runner = build()
    runner.ingest_request("a")
    s, _ = step([Segment("a", "main", 1), Segment("pad", "main", 0)], walk="decode")
    assert runner.admit(s).ok
    runner.plan(s)
    plan = attn.current_plan()
    assert plan.num_rows == 2 and plan.slot_ids_cpu[1] == SINK_SLOT and plan.has_state_cpu[1] is False
    assert plan.cu_seqlens_cpu == [0, 1, 1] and not plan.is_decode  # a zero-span row is not a decode row


def test_static_buffers_under_a_lease_and_preplan():
    pool, attn, runner = build(max_slots=8)
    for rid in "abc":
        runner.ingest_request(rid)
    lease = SlotLease(bucket=BucketKey(bs=4, num_tokens=4, graph_walk="decode"), slot=0)
    attn.build_cuda_graph_buffers([], max_bs=4, max_seq_len=16)
    pool.build_cuda_graph_buffers([], max_bs=4, max_seq_len=16)
    s, _ = step([Segment("a", "main", 1), Segment("b", "main", 1)], walk="decode", slot_lease=lease)
    assert runner.admit(s).ok
    runner.plan(s)
    p1 = attn.current_plan()
    cu_ptr, slot_ptr = p1.cu_seqlens.data_ptr(), p1.slot_ids.data_ptr()
    assert p1.cu_seqlens.numel() >= 5  # sized to the bucket, not to this plan's two rows
    p1_slots = p1.slot_ids_cpu  # read now: the next plan under this lease rewrites the same buffer
    runner.commit(s)
    # the same lease again with other rows: the same device buffers, new contents
    s2, _ = step([Segment("c", "main", 1), Segment("a", "main", 1), Segment("b", "main", 1)], walk="decode",
                 slot_lease=lease)
    assert runner.admit(s2).ok
    runner.plan(s2)
    p2 = attn.current_plan()
    assert p2.cu_seqlens.data_ptr() == cu_ptr and p2.slot_ids.data_ptr() == slot_ptr
    assert p2.cu_seqlens_cpu == [0, 1, 2, 3] and p2.slot_ids_cpu[1:] == p1_slots
    runner.commit(s2)
    # staging a step ahead: the promoted plan is the staged one
    assert pool.supports_preplan and attn.supports_preplan
    s3, ctx3 = step([Segment("a", "main", 1)], walk="decode", slot_lease=lease, preplan=True)
    assert runner.admit(s3).ok
    runner.plan(s3)
    staged = attn.current_plan()
    s3b, _ = step([Segment("a", "main", 1)], walk="decode", slot_lease=lease)
    runner.plan(s3b)
    assert attn.current_plan() is staged
    runner.commit(s3b)


def test_reference_kernels_run_the_paged_forward_against_the_pool():
    """The torch reference kernels on the pool's blocks (W - 1 conv columns) match the explicit-state
    reference layer step by step, prefill then decode."""
    from mstar.model.kimi_k3.components.kda import ParallelKDAAttention, TorchKDAKernels

    torch.manual_seed(0)
    hidden = 16
    layer = ParallelKDAAttention(hidden_size=hidden, num_heads=H, head_dim=D, conv_kernel_size=W, gate_lower_bound=-5.0)
    for prm in layer.parameters():
        with torch.no_grad():
            prm.normal_(std=0.2)
    pool, attn, runner = build()
    layer.bind_resources({POOL: pool, ATTN: attn})
    assert isinstance(attn.kernels, TorchKDAKernels) and not attn.cuda_graph_safe  # off-GPU: the reference
    for rid in ("a", "b"):
        runner.ingest_request(rid)
    xa, xb = torch.randn(5, hidden), torch.randn(3, hidden)
    with torch.no_grad():
        ref_a, st_a = layer.forward_dense(xa)
        ref_b, st_b = layer.forward_dense(xb)
        s, _ = step([Segment("a", "main", 5), Segment("b", "main", 3)])
        assert runner.admit(s).ok
        runner.plan(s)
        attn.set_default_layer_idx(1)
        out = layer(torch.cat([xa, xb]))
        runner.commit(s)
        torch.testing.assert_close(out[:5], ref_a, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out[5:], ref_b, rtol=1e-4, atol=1e-4)
        # the pool holds each request's state in layer 1: the reference's V-first recurrent state and
        # the last W - 1 conv inputs
        plan = attn.current_plan()
        from mstar.model.kimi_k3.reference.kda import to_v_first

        sa, sb = plan.slot_ids_cpu
        torch.testing.assert_close(pool.block("state", 1)[sa], to_v_first(st_a.recurrent), rtol=1e-5, atol=1e-6)
        conv_b = torch.cat([st_b.conv_q, st_b.conv_k, st_b.conv_v])[:, 1:]
        torch.testing.assert_close(pool.block("conv", 1)[sb], conv_b.to(pool.block("conv", 1).dtype))
        # one decode token each, rows in the other order
        ya, yb = torch.randn(1, hidden), torch.randn(1, hidden)
        d_a, _ = layer.forward_dense(ya, st_a)
        d_b, _ = layer.forward_dense(yb, st_b)
        s2, _ = step([Segment("b", "main", 1), Segment("a", "main", 1)], walk="decode")
        assert runner.admit(s2).ok
        runner.plan(s2)
        attn.set_default_layer_idx(1)  # a plan resets the cursors; the model loop sets them per layer
        out2 = layer(torch.cat([yb, ya]))
        runner.commit(s2)
        torch.testing.assert_close(out2[0:1], d_b, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(out2[1:2], d_a, rtol=1e-4, atol=1e-4)


def test_replay_padding_rows_take_no_slots():
    """A captured replay pads the batch with the runner's dummy rows: ingested once, span 1 like the
    template row, named per capture slot and row. They must reserve nothing (two slots times the widest
    bucket of such names would drain a pool sized for the real concurrency), address the sink, and
    never be committed; the real rows keep their own slots across the rotation of padding names."""
    pool, attn, runner = build(max_slots=6)  # 5 usable slots for 2 real requests
    for rid in ("a", "b"):
        runner.ingest_request(rid)
    dummies = [f"__cg_LLM_{slot}_{i}__" for slot in (0, 1) for i in range(4)]
    for rid in dummies:
        runner.ingest_request(rid)  # DummyRowPool.ensure ingests them once and keeps them
    for _ in range(3):
        for slot in (0, 1):
            names = [f"__cg_LLM_{slot}_{i}__" for i in range(4)]
            real = ["a", "b"]
            padded = [*real, *names[2:]]
            s = SubmoduleStep(
                segments=[Segment(rid, "main", 1) for rid in padded],
                steps={POOL: RecurrentStep(), ATTN: LinearAttnStep()})
            ctx = StepContext(request_ids=real, graph_walk="decode", slot=slot, capture=False)
            ctx.set_padded_rids(padded)
            s.set_ctx(ctx)
            assert runner.admit(s).ok
            assert pool.num_free_slots == 3  # a and b hold one slot each, the padding rows none
            runner.plan(s)
            plan = attn.current_plan()
            assert plan.num_rows == 4
            assert plan.slot_ids_cpu[2] == plan.slot_ids_cpu[3] == SINK_SLOT
            assert plan.has_state_cpu[2] is False and plan.has_state_cpu[3] is False
            assert plan.slot_ids_cpu[0] != plan.slot_ids_cpu[1] and SINK_SLOT not in plan.slot_ids_cpu[:2]
            runner.commit(s)
    for rid in dummies:
        assert not pool._slots.get(rid), rid  # no slot ever handed to a padding row
    assert pool._slots["a"]["main"].has_state and pool._slots["b"]["main"].has_state


def test_capture_rows_take_no_slots():
    """The CUDA-graph capture's dummy rows (``ctx.capture``) reserve nothing and address the sink:
    two capture slots of the widest bucket would otherwise exhaust a pool sized for the deployment."""
    pool, attn, runner = build(max_slots=4)
    rids = [f"__cg_LLM_0_slot1_{i}__" for i in range(8)]  # more rows than the pool has slots
    for rid in rids:
        runner.ingest_request(rid)
    s = SubmoduleStep(segments=[Segment(rid, "main", 1) for rid in rids], steps={POOL: RecurrentStep(), ATTN: LinearAttnStep()})
    ctx = StepContext(request_ids=rids, graph_walk="decode", slot=1, capture=True,
                      slot_lease=SlotLease(bucket=BucketKey(bs=8, num_tokens=8, graph_walk="decode"), slot=1))
    s.set_ctx(ctx)
    assert runner.admit(s).ok and pool.num_free_slots == 3
    runner.plan(s)
    plan = attn.current_plan()
    assert plan.is_decode and plan.slot_ids_cpu == [SINK_SLOT] * 8 and plan.has_state_cpu == [False] * 8
    runner.commit(s)
    assert pool.num_free_slots == 3
    # a real request afterwards still gets a slot of its own
    runner.ingest_request("a")
    s2, _ = step([Segment("a", "main", 3)])
    assert runner.admit(s2).ok
    runner.plan(s2)
    assert attn.current_plan().slot_ids_cpu[0] != SINK_SLOT
