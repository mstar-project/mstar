"""CPU lifecycle tests for the recurrent-state resource (slot allocation, addressing,
CUDA-graph static buffers, preplan, offload/reload)."""
import torch

from mstar.engine.resources import (
    AllocationFailed,
    BucketKey,
    CommitMode,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStateStep,
    Segment,
    SlotLease,
    StatePart,
    StepContext,
    SubmoduleStep,
)
from mstar.engine.resources.base import EngineResourceInfo, build_resource
from mstar.engine.resources.recurrent.manager import SCRATCH_SLOT

KEY = "kda_state"


def make_manager(max_slots=3, offload=0, num_layers=2):
    cfg = RecurrentStateConfig(
        num_layers=num_layers,
        parts={
            "recurrent": StatePart((4, 8, 8), torch.float32, shard_dim=0),
            "conv": StatePart((4 * 8 * 3, 3), torch.bfloat16, shard_dim=0),
        },
        max_num_slots=max_slots, cpu_offload_slots=offload,
    )
    spec = RecurrentStateSpec(resource_key=KEY, nodes={"LLM"}, config=cfg)
    return build_resource(spec, EngineResourceInfo(device=torch.device("cpu")))


def step(segs, mode=CommitMode.IN_PLACE, walk="prefill", slot_lease=None, preplan=False):
    s = SubmoduleStep(segments=segs, steps={KEY: RecurrentStateStep(commit_mode=mode)})
    ctx = StepContext(request_ids=[x.request_id for x in segs], graph_walk=walk, slot=0, capture=False,
                      is_preplan=preplan, slot_lease=slot_lease)
    s.set_ctx(ctx)
    return s.get(KEY), ctx


def test_allocation_addressing_and_commit():
    m = make_manager(max_slots=3)
    assert m.part("recurrent").shape == (2, 4, 4, 8, 8) and m.part("conv").shape == (2, 4, 96, 3)
    for rid in ("a", "b", "c"):
        m.ingest_request(rid)
    # prefill: three requests, distinct slots, no resident state yet
    st, ctx = step([Segment("a", "main", 5), Segment("b", "main", 3), Segment("c", "main", 7)])
    assert m.admit(st, ctx).ok
    out = m.plan(st, ctx)
    assert sorted(out.slot_ids_cpu) == [1, 2, 3] and out.has_state_cpu == [False] * 3
    assert not out.is_decode and out.num_rows == 3
    assert m.plan_output is out
    m.commit(st, ctx)
    assert all(m.has_state(r) for r in "abc")
    # decode: same slots, state present, is_decode
    st, ctx = step([Segment("b", "main", 1), Segment("a", "main", 1)], walk="decode")
    assert m.admit(st, ctx).ok
    out = m.plan(st, ctx)
    assert out.slot_ids_cpu == [m.slot_of("b"), m.slot_of("a")] and out.has_state_cpu == [True, True]
    assert out.is_decode
    # a zero-span (padding) segment addresses the scratch slot and reserves nothing
    st, ctx = step([Segment("a", "main", 1), Segment("__pad__", "main", 0)], walk="decode")
    m.ingest_request("__pad__")
    assert m.admit(st, ctx).ok
    out = m.plan(st, ctx)
    assert out.slot_ids_cpu[1] == SCRATCH_SLOT and out.has_state_cpu[1] is False
    assert m.slot_of("__pad__") is None
    # exhaustion -> AllocationFailed naming the request
    m.ingest_request("d")
    st, ctx = step([Segment("d", "main", 2)])
    res = m.admit(st, ctx)
    assert not res.ok and isinstance(res.reason, AllocationFailed) and res.reason.request_id == "d"
    # removing a request frees its slot for the next one
    m.remove_request("c")
    assert m.admit(st, ctx).ok and m.slot_of("d") == 3
    # reset with free (dummy rows after capture) releases and clears residency
    m.reset_request("d", free=True)
    assert m.slot_of("d") is None and not m.has_state("d") and m.num_free_slots == 1


def test_static_buffers_under_lease_and_preplan():
    m = make_manager(max_slots=4)
    m.build_cuda_graph_buffers([], max_bs=4, max_seq_len=16)
    for rid in ("a", "b"):
        m.ingest_request(rid)
    lease = SlotLease(slot=0, bucket=BucketKey(graph_walk="decode", bs=4, num_tokens=4))
    st, ctx = step([Segment("a", "main", 1), Segment("b", "main", 1)], walk="decode", slot_lease=lease)
    assert m.admit(st, ctx).ok
    out = m.plan(st, ctx)
    assert out.slot_ids.shape == (4,) and out.slot_ids.dtype == torch.int32
    assert out.slot_ids.tolist() == [1, 2, SCRATCH_SLOT, SCRATCH_SLOT]
    assert out.has_state.tolist() == [False] * 4
    m.commit(st, ctx)
    # the same static buffers are rewritten for the next step on this slot
    st2, ctx2 = step([Segment("b", "main", 1)], walk="decode", slot_lease=lease)
    assert m.admit(st2, ctx2).ok
    out2 = m.plan(st2, ctx2)
    assert out2.slot_ids.data_ptr() == out.slot_ids.data_ptr()
    assert out2.slot_ids.tolist() == [2, SCRATCH_SLOT, SCRATCH_SLOT, SCRATCH_SLOT]
    assert out2.has_state.tolist() == [True, False, False, False]
    # preplan a step ahead for a new request, then abandon it: its slot is released
    m.ingest_request("c")
    st3, ctx3 = step([Segment("c", "main", 3)], preplan=True)
    assert m.admit(st3, ctx3).ok and m.slot_of("c") == 3
    m.plan(st3, ctx3)
    assert m.supports_preplan
    m.clear_preplan()
    assert m.slot_of("c") is None and m.num_free_slots == 2
    # preplan then promote: the promoted plan is the pending one and the slot stays
    assert m.admit(st3, ctx3).ok
    slot_c = m.slot_of("c")
    assert slot_c is not None
    pre = m.plan(st3, ctx3)
    st4, ctx4 = step([Segment("c", "main", 3)])
    assert m.admit(st4, ctx4).ok  # no-op while preplanned
    assert m.plan(st4, ctx4) is pre and m.slot_of("c") == slot_c


def test_offload_and_reload_roundtrip():
    m = make_manager(max_slots=2, offload=2)
    m.ingest_request("a")
    st, ctx = step([Segment("a", "main", 4)])
    assert m.admit(st, ctx).ok
    m.plan(st, ctx)
    slot = m.slot_of("a")
    m.part("recurrent")[:, slot].normal_()
    m.part("conv")[:, slot].normal_()
    saved = {k: m.part(k)[:, slot].clone() for k in ("recurrent", "conv")}
    assert m.reclaimable("a") == 0  # in flight until commit
    m.commit(st, ctx)
    assert m.supports_eviction and m.reclaimable("a") == 1
    assert m.offload("a") == 1 and m.is_offloaded("a") and m.slot_of("a") is None
    assert m.num_free_slots == 2
    # admitting while offloaded asks the worker to reload first
    st2, ctx2 = step([Segment("a", "main", 1)], walk="decode")
    res = m.admit(st2, ctx2)
    assert not res.ok and type(res.reason).__name__ == "RequestOffloading"
    assert m.reload("a") and not m.is_offloaded("a") and m.has_state("a")
    new_slot = m.slot_of("a")
    for k, v in saved.items():
        assert torch.equal(m.part(k)[:, new_slot], v)
    # the scratch slot is untouched by everything above
    assert torch.count_nonzero(m.part("recurrent")[:, SCRATCH_SLOT]) == 0


def test_padded_replay_rows_use_the_scratch_slot():
    """A captured replay pads the batch with the runner's dummy rids: they must not take
    state slots (54 MiB each on the real model) and must address the scratch slot."""
    m = make_manager(max_slots=4)
    dummies = ["__cg_x_0__", "__cg_x_1__", "__cg_x_2__"]
    for rid in ["a", *dummies]:
        m.ingest_request(rid)
    segs = [Segment("a", "main", 1)] + [Segment(rid, "main", 1) for rid in dummies]
    s = SubmoduleStep(segments=segs, steps={KEY: RecurrentStateStep()})
    ctx = StepContext(request_ids=["a"], graph_walk="decode", slot=0, capture=False)
    ctx.set_padded_rids([x.request_id for x in segs])
    s.set_ctx(ctx)
    st = s.get(KEY)
    free_before = m.num_free_slots
    assert m.admit(st, ctx).ok
    out = m.plan(st, ctx)
    assert out.slot_ids_cpu[1:] == [SCRATCH_SLOT] * 3 and out.has_state_cpu[1:] == [False] * 3
    assert out.slot_ids_cpu[0] != SCRATCH_SLOT
    assert m.num_free_slots == free_before - 1  # only the real request took a slot
    m.commit(st, ctx)
    for rid in dummies:
        m.reset_request(rid, free=False)
    assert m.num_free_slots == free_before - 1 and m.has_state("a")
