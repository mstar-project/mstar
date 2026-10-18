"""Rows that stand for no request must not cost the recurrent pool a slot.

A captured replay pads its batch with the runner's dummy rows: ingested once,
carrying the template span (1 for decode), named per capture slot and row. If
each of them took a slot, the step would have to hand it back afterwards
(and the KV cache's pages with it, which is what turned a near-full arena into
a deadlock), or the dummy names would hold two capture slots times the widest
bucket of slots for good. Capture rows are in the same position. Both address
the sink instead.

CPU-only: nothing here launches a kernel.
"""

from __future__ import annotations

import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import NO_SLOT, SINK_SLOT, RecurrentStatePool
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext


def build_pool(max_slots: int, disable_sink_slot: bool = False) -> RecurrentStatePool:
    spec = RecurrentStateSpec(
        "gdn_state", {"llm"},
        RecurrentStateConfig(
            num_layers=2,
            blocks={
                "state": RecurrentBlockConfig(shape=(4, 16, 16), dtype=torch.float32),
                "conv": RecurrentBlockConfig(shape=(96, 3), dtype=torch.float32),
            },
            max_slots=max_slots,
            disable_sink_slot=disable_sink_slot,
        ),
    )
    return RecurrentStatePool.build(spec, EngineResourceInfo(device=torch.device("cpu")))


def padded_ctx(real: list[str], padded: list[str], bs: int, capture: bool = False):
    ctx = StepContext(
        request_ids=real, graph_walk="decode", slot=0, capture=capture,
        slot_lease=SlotLease(slot=0, bucket=BucketKey("decode", bs, bs)),
    )
    ctx.set_padded_rids(padded)
    return ctx


def decode_step(rids: list[str]) -> RecurrentStep:
    return RecurrentStep(segments=tuple(Segment(rid, "main", 1) for rid in rids))


def test_replay_padding_rows_take_no_slots():
    """Two real requests in a bucket of 8 on a 5-slot pool, over three steps
    and both capture slots: the padding rows reserve nothing, address the sink
    and are never committed; the real rows keep their own slots throughout."""
    pool = build_pool(max_slots=6)  # 5 usable
    for rid in ("a", "b"):
        pool.ingest_request(rid)
    dummies = [f"__cg_LLM_{slot}_{i}__" for slot in (0, 1) for i in range(8)]
    for rid in dummies:
        pool.ingest_request(rid)  # the runner ingests its dummy rows once and keeps them
    for _ in range(3):
        for slot in (0, 1):
            names = [f"__cg_LLM_{slot}_{i}__" for i in range(8)]
            real = ["a", "b"]
            padded = [*real, *names[2:]]
            step = decode_step(padded)
            ctx = padded_ctx(real, padded, bs=8)
            assert pool.admit(step, ctx).ok
            assert pool.num_free_slots == 3, "a and b hold one slot each, the padding rows none"
            addr = pool.plan(step, ctx)["main"]
            rows = addr.slot_indices[:8].tolist()
            assert rows[2:] == [SINK_SLOT] * 6
            assert SINK_SLOT not in rows[:2] and rows[0] != rows[1]
            assert not addr.has_state[2:8].any()
            pool.commit(step, ctx)
    for rid in dummies:
        assert not pool._slots.get(rid), f"{rid} was handed a slot"
    assert pool._slots["a"]["main"].has_state and pool._slots["b"]["main"].has_state


def test_padding_rows_do_not_need_a_pool_wider_than_the_bucket():
    """One real request in a bucket of 32 on a 3-slot pool admits fine."""
    pool = build_pool(max_slots=3)
    names = [f"__cg_LLM_0_{i}__" for i in range(32)]
    for rid in ["r", *names]:
        pool.ingest_request(rid)
    padded = ["r", *names[1:]]
    ctx = padded_ctx(["r"], padded, bs=32)
    assert pool.admit(decode_step(padded), ctx).ok
    assert pool.num_free_slots == 1


def test_capture_rows_take_no_slots():
    """Capture runs the dummy rows as the step's own requests; they still
    reserve nothing, so the pool need not be as wide as the widest bucket."""
    pool = build_pool(max_slots=2)  # one usable slot, bucket of 16
    names = [f"__cg_LLM_0_{i}__" for i in range(16)]
    for rid in names:
        pool.ingest_request(rid)
    step = decode_step(names)
    ctx = StepContext(request_ids=names, graph_walk="decode", slot=0, capture=True)
    assert pool.admit(step, ctx).ok
    assert pool.num_free_slots == 1
    addr = pool.plan(step, ctx)["main"]
    assert addr.slot_indices[:16].tolist() == [SINK_SLOT] * 16
    pool.commit(step, ctx)
    for rid in names:
        assert not pool._slots.get(rid)


def test_padding_rows_carry_the_sentinel_when_the_sink_is_off():
    pool = build_pool(max_slots=4, disable_sink_slot=True)
    names = [f"__cg_LLM_0_{i}__" for i in range(4)]
    for rid in ["r", *names]:
        pool.ingest_request(rid)
    padded = ["r", *names[1:]]
    ctx = padded_ctx(["r"], padded, bs=4)
    assert pool.admit(decode_step(padded), ctx).ok
    rows = pool.plan(decode_step(padded), ctx)["main"].slot_indices[:4].tolist()
    assert rows[1:] == [NO_SLOT] * 3 and rows[0] >= 0


def test_a_real_request_padding_and_release_across_the_rotation():
    """The runner resets padding rows after each step with `free=False`; with
    no slot behind them that is a no-op, and a real request that finished
    hands its slot back through `remove_request` as before."""
    pool = build_pool(max_slots=3)
    names = [f"__cg_LLM_0_{i}__" for i in range(4)]
    for rid in ["r", *names]:
        pool.ingest_request(rid)
    padded = ["r", *names[1:]]
    ctx = padded_ctx(["r"], padded, bs=4)
    assert pool.admit(decode_step(padded), ctx).ok
    pool.plan(decode_step(padded), ctx)
    pool.commit(decode_step(padded), ctx)
    for rid in names[1:]:
        pool.reset_request(rid, free=False)
    assert pool.num_free_slots == 1
    pool.remove_request("r")
    assert pool.num_free_slots == 2


def test_gather_scatter_follow_the_step_addressing():
    """``gather`` / ``scatter_`` address the step's rows: a replay's padding rows
    read and write the sink and nothing else, and a ``select`` of the step
    addresses the same slots as the step."""
    pool = build_pool(max_slots=6)
    step = decode_step(["a", "b"])
    eager = StepContext(request_ids=["a", "b"], graph_walk="decode", slot=0, capture=False)
    pool.admit(step, eager)
    rows = pool.plan(step, eager)["main"]
    value = torch.arange(2 * 96 * 3, dtype=torch.float32).view(2, 96, 3)
    pool.scatter_("conv", rows, value, layer_idx=1)
    torch.testing.assert_close(pool.gather("conv", rows, layer_idx=1), value)
    one = StepContext(request_ids=["b"], graph_walk="decode", slot=0, capture=False)
    torch.testing.assert_close(pool.gather("conv", pool.plan(decode_step(["b"]), one)["main"], layer_idx=1), value[1:])
    untouched = pool.block("conv", 0).clone()

    padded = decode_step(["a", "b", "pad0", "pad1"])
    ctx = padded_ctx(["a", "b"], ["a", "b", "pad0", "pad1"], bs=4)
    pool.admit(padded, ctx)
    rows = pool.plan(padded, ctx)["main"]
    assert rows.num_rows == 4
    got = pool.gather("conv", rows, layer_idx=1)
    torch.testing.assert_close(got[:2], value)
    pool.scatter_("conv", rows, torch.full((4, 96, 3), -1.0), layer_idx=1)
    conv = pool.block("conv", 1)
    live = {pool._slots[r]["main"].index for r in ("a", "b")}
    assert all(bool((conv[i] == -1).all()) for i in live | {SINK_SLOT})
    assert all(bool((conv[i] == 0).all()) for i in range(6) if i not in live | {SINK_SLOT})
    torch.testing.assert_close(pool.block("conv", 0), untouched)
