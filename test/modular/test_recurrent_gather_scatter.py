"""``RecurrentStatePool.gather`` / ``scatter_``: a forward reads and writes its
rows' slots through the step's addressing rather than indexing ``block()``.

The CPU cases drive admit / plan / commit by hand, eagerly and through a
replay's per-bucket buffers with padding rows. The CUDA case captures a graph
over the replay addressing and replays it for different batches.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import pytest
import torch

from mstar.engine.resources.base import EngineResourceInfo
from mstar.engine.resources.recurrent.config import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import NO_SLOT, SINK_SLOT, RecurrentStatePool
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext, SubmoduleStep
from mstar.engine.state import ArrayStep, Entry, TensorType, Value
from mstar.engine.state.adapters import build_handle, build_specs

MAX_SLOTS = 6  # 5 usable, slot 0 is the sink
SHAPE = (2, 8)


def build_pool(device: str = "cpu") -> RecurrentStatePool:
    spec = RecurrentStateSpec(
        "state", {"node"},
        RecurrentStateConfig(
            num_layers=2,
            blocks={
                "h": RecurrentBlockConfig(shape=SHAPE, dtype=torch.float32),
                "cursor": RecurrentBlockConfig(shape=(1,), dtype=torch.int64),
            },
            max_slots=MAX_SLOTS,
        ),
    )
    pool = RecurrentStatePool.build(spec, EngineResourceInfo(device=torch.device(device)))
    for rid in "abcd":
        pool.ingest_request(rid)
    return pool


def eager(real: list[str]) -> tuple[RecurrentStep, StepContext]:
    step = RecurrentStep(segments=tuple(Segment(rid, "main", 1) for rid in real))
    return step, StepContext(request_ids=real, graph_walk="decode", slot=0, capture=False)


def replay(real: list[str], bs: int, slot: int = 0, capture: bool = False):
    """A step padded to a capture bucket of ``bs`` rows, as the runner declares it."""
    declared = [*real, *(f"__cg_node_{slot}_{i}__" for i in range(len(real), bs))]
    step = RecurrentStep(segments=tuple(Segment(rid, "main", 1) for rid in declared))
    ctx = StepContext(
        request_ids=real, graph_walk="decode", slot=slot, capture=capture,
        slot_lease=SlotLease(slot=slot, bucket=BucketKey("decode", bs, bs)),
    )
    ctx.set_padded_rids(declared)
    return step, ctx


def plan(pool: RecurrentStatePool, step: RecurrentStep, ctx: StepContext):
    assert pool.admit(step, ctx).ok
    return pool.plan(step, ctx)["main"]


def slot_of(pool: RecurrentStatePool, rid: str) -> int:
    return pool._slots[rid]["main"].index


def rows_of(n: int, base: float) -> torch.Tensor:
    return base + torch.arange(n * SHAPE[0] * SHAPE[1], dtype=torch.float32).view(n, *SHAPE)


def advance(pool: RecurrentStatePool, inc: torch.Tensor) -> None:
    """A stand-in forward: read every row's state, update it, write it back."""
    pool.scatter_("h", pool.gather("h", layer_idx=1) * 2 + inc, layer_idx=1)
    pool.scatter_("cursor", pool.gather("cursor") + 1)


def test_round_trip_and_a_shorter_step_reads_only_its_rows():
    pool = build_pool()
    step, ctx = eager(["a", "b", "c"])
    plan(pool, step, ctx)
    value = rows_of(3, 1.0)
    pool.scatter_("h", value, layer_idx=1)
    # has_state comes from the plan, so this step's own writes read as no saved state yet
    torch.testing.assert_close(pool.gather("h", layer_idx=1), torch.zeros_like(value))
    pool.commit(step, ctx)
    assert torch.equal(pool.block("h", 0), torch.zeros(MAX_SLOTS, *SHAPE)), "layer 0 untouched"

    # the eager buffers still hold three rows; this step addresses two, in another order
    step, ctx = eager(["c", "a"])
    rows = plan(pool, step, ctx)
    assert rows.num_rows == 2 and rows.slot_indices.numel() == 3
    torch.testing.assert_close(pool.gather("h", layer_idx=1), value[[2, 0]])
    assert rows.has_state[: rows.num_rows].tolist() == [True, True]


def test_each_step_scatters_to_its_own_rows():
    """Consecutive steps on the same buffers (eager, and one replay bucket) with
    different rows each write their own slots, not the previous step's."""
    for steps in (
        [eager(["a", "b"]), eager(["c"]), eager(["b", "d", "a"])],
        [replay(["a", "b"], bs=4), replay(["c"], bs=4), replay(["b", "d", "a"], bs=4)],
    ):
        pool = build_pool()
        for n, (step, ctx) in enumerate(steps):
            plan(pool, step, ctx)
            rows = pool.addressing("main").num_rows
            value = torch.full((rows, *SHAPE), float(n + 1))
            pool.scatter_("h", value, layer_idx=1)
            for rid in (seg.request_id for seg in step.segments if seg.request_id in "abcd"):
                assert bool((pool.block("h", 1)[slot_of(pool, rid)] == n + 1).all()), (n, rid)
            pool.commit(step, ctx)


def test_replay_addressing_matches_eager():
    """The same forward on the same requests, once eagerly and once through a
    bucket of 4 with padding rows, leaves the real slots identical; the replay
    writes nothing else but the sink."""
    pools = {"eager": build_pool(), "replay": build_pool()}
    real = ["b", "a", "c"]
    for pool in pools.values():
        # same history in both: a, b and c take slots 1, 2 and 3 and hold state
        step, ctx = eager(["a", "b", "c"])
        plan(pool, step, ctx)
        pool.scatter_("h", rows_of(3, 1.0), layer_idx=1)
        pool.commit(step, ctx)

    step, ctx = eager(real)
    plan(pools["eager"], step, ctx)
    advance(pools["eager"], rows_of(3, 100.0))
    pools["eager"].commit(step, ctx)

    pool = pools["replay"]
    pool.block("h", 1)[SINK_SLOT].fill_(7.0)
    step, ctx = replay(real, bs=4)
    rows = plan(pool, step, ctx)
    assert rows.num_rows == 4, "num_rows counts the padding rows the step declared"
    assert rows.slot_indices.tolist()[3] == SINK_SLOT
    assert bool((pool.gather("h", layer_idx=1)[3] == 0).all()), "a padding row reads zeros, not the sink"
    advance(pool, torch.cat([rows_of(3, 100.0), rows_of(1, 100.0)]))
    pool.commit(step, ctx)

    for rid in real:
        i, j = slot_of(pools["eager"], rid), slot_of(pool, rid)
        assert i == j
        torch.testing.assert_close(pool.block("h", 1)[j], pools["eager"].block("h", 1)[i])
        assert pool.block("cursor", 0)[j].item() == pools["eager"].block("cursor", 0)[i].item() == 1
    written = {SINK_SLOT} | {slot_of(pool, rid) for rid in real}
    untouched = [s for s in range(MAX_SLOTS) if s not in written]
    assert all(bool((pool.block("h", 1)[s] == 0).all()) for s in untouched)


def test_replay_buffers_are_static_per_bucket_and_capture_slot():
    """What a captured graph records: one buffer per (bucket, capture slot),
    the same storage on every step, and a capture addresses only the sink."""
    pool = build_pool()
    step, ctx = replay([], bs=4, capture=True)
    captured = plan(pool, step, ctx)
    assert captured.slot_indices.tolist() == [SINK_SLOT] * 4

    ptrs = set()
    for real in (["a"], ["b", "c"], ["c", "a", "b", "d"]):
        step, ctx = replay(real, bs=4)
        rows = plan(pool, step, ctx)
        assert rows.slot_indices.data_ptr() == captured.slot_indices.data_ptr()
        assert rows.slot_indices.tolist()[len(real):] == [SINK_SLOT] * (4 - len(real))
        pool.commit(step, ctx)
    for slot in (0, 1):
        step, ctx = replay(["a"], bs=4, slot=slot)
        ptrs.add(plan(pool, step, ctx).slot_indices.data_ptr())
    step, ctx = eager(["a"])
    ptrs.add(plan(pool, step, ctx).slot_indices.data_ptr())
    assert len(ptrs) == 3, "capture slots 0 and 1 and the eager path each have their own buffers"


def test_padding_rows_read_and_write_only_the_sink():
    pool = build_pool()
    pool.block("h", 1)[SINK_SLOT].fill_(7.0)
    step, ctx = replay(["a"], bs=4)
    plan(pool, step, ctx)
    got = pool.gather("h", layer_idx=1)
    assert bool((got == 0).all()), "a fresh slot and the padding rows read zeros, not the sink's leftovers"
    start = torch.full(SHAPE, 2.0)
    assert bool((pool.gather("h", layer_idx=1, start=start) == 2.0).all()), "or the start state given"

    value = torch.cat([torch.full((1, *SHAPE), 5.0), torch.full((3, *SHAPE), -1.0)])
    with pytest.raises(ValueError, match="1 rows for a step that declared 4"):
        pool.scatter_("h", value[:1], layer_idx=1)
    pool.scatter_("h", value, layer_idx=1)
    h = pool.block("h", 1)
    assert bool((h[slot_of(pool, "a")] == 5.0).all())
    assert bool((h[SINK_SLOT] == -1.0).all())
    others = [s for s in range(MAX_SLOTS) if s not in (SINK_SLOT, slot_of(pool, "a"))]
    assert all(bool((h[s] == 0).all()) for s in others)
    pool.commit(step, ctx)
    assert pool.num_free_slots == MAX_SLOTS - 2, "only a holds a slot"


def test_a_step_whose_rows_all_have_state_reads_them_with_no_select(monkeypatch):
    """The select of a fresh row's start is three ops per field, and a step whose rows all have saved state skips it:
    it reads the rows the select would, and a step with a fresh row still reads its start."""
    pool = build_pool()
    step, ctx = eager(["a", "b"])
    plan(pool, step, ctx)
    pool.scatter_("h", rows_of(2, 1.0), layer_idx=1)
    pool.commit(step, ctx)
    start = torch.full(SHAPE, 9.0)

    step, ctx = eager(["b", "a"])
    rows = plan(pool, step, ctx)
    with monkeypatch.context() as patched:
        patched.setattr(torch, "where", Mock(side_effect=AssertionError("the read ran the select")))
        skipped = pool.gather("h", layer_idx=1, start=start)
    pool._current["main"] = replace(rows, every_row_has_state=False)
    assert torch.equal(skipped, pool.gather("h", layer_idx=1, start=start)), "the skip reads other rows than the select"
    torch.testing.assert_close(skipped, rows_of(2, 1.0)[[1, 0]])
    pool.commit(step, ctx)

    step, ctx = eager(["c", "a"])
    plan(pool, step, ctx)
    read = pool.gather("h", layer_idx=1, start=start)
    assert torch.equal(read[0], start), "a fresh row reads its start"
    torch.testing.assert_close(read[1], rows_of(2, 1.0)[0])


def test_a_freed_slot_reads_as_zeros():
    pool = build_pool()
    step, ctx = eager(["a"])
    plan(pool, step, ctx)
    pool.scatter_("h", rows_of(1, 3.0), layer_idx=1)
    pool.commit(step, ctx)
    freed = slot_of(pool, "a")
    pool.remove_request("a")
    # teardown zeroes the slot's storage in every layer, for backends that read it directly
    assert all(bool((pool.block("h", layer)[freed] == 0).all()) for layer in (0, 1))

    step, ctx = eager(["d"])
    rows = plan(pool, step, ctx)
    assert slot_of(pool, "d") == freed
    assert not rows.has_state[0].item()
    assert bool((pool.gather("h", layer_idx=1) == 0).all())
    # a resumed slot reads its saved state whatever start is given
    pool.scatter_("h", rows_of(1, 4.0), layer_idx=1)
    pool.commit(step, ctx)
    step, ctx = eager(["d"])
    plan(pool, step, ctx)
    torch.testing.assert_close(pool.gather("h", layer_idx=1, start=torch.full(SHAPE, 2.0)), rows_of(1, 4.0))


def test_a_pool_without_a_sink_refuses_gather_and_scatter():
    """Its padding rows address NO_SLOT, which no index op takes: a clear
    error naming the pool rather than an IndexError, or on CUDA a
    device-side assert."""
    spec = RecurrentStateSpec(
        "no_sink_state", {"node"},
        RecurrentStateConfig(
            num_layers=1, blocks={"h": RecurrentBlockConfig(shape=SHAPE, dtype=torch.float32)},
            max_slots=MAX_SLOTS, disable_sink_slot=True,
        ),
    )
    pool = RecurrentStatePool.build(spec, EngineResourceInfo(device=torch.device("cpu")))
    pool.ingest_request("a")
    step, ctx = replay(["a"], bs=2)
    rows = plan(pool, step, ctx)
    assert rows.slot_indices.tolist()[1] == NO_SLOT
    with pytest.raises(ValueError, match="'no_sink_state' has no sink slot"):
        pool.gather("h")
    with pytest.raises(ValueError, match="'no_sink_state' has no sink slot"):
        pool.scatter_("h", rows_of(2, 0.0))


def test_a_block_in_the_engine_dtype_takes_each_engines_own():
    """A ``dtype=None`` block is resolved by the pool, not written back into
    the model's config, which its other specs and engines share."""
    config = RecurrentStateConfig(
        num_layers=1, blocks={"h": RecurrentBlockConfig(shape=SHAPE, dtype=None)}, max_slots=MAX_SLOTS,
    )
    pools = [
        RecurrentStatePool.build(
            RecurrentStateSpec("state", {"node"}, config),
            EngineResourceInfo(device=torch.device("cpu"), kv_dtype=kv_dtype),
        )
        for kv_dtype in (torch.bfloat16, torch.float16)
    ]
    assert [pool.block("h", 0).dtype for pool in pools] == [torch.bfloat16, torch.float16]
    assert config.blocks["h"].dtype is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="captures a CUDA graph")
def test_a_captured_graph_addresses_the_rows_each_replay_plans():
    """Capture once over a bucket's addressing, then replay for batches of
    different requests: each replay updates exactly the slots its plan named,
    as the eager forward does."""
    pool, ref = build_pool("cuda"), build_pool("cuda")
    bs = 4
    inc = torch.zeros(bs, *SHAPE, device="cuda")

    step, ctx = replay([], bs=bs, capture=True)
    plan(pool, step, ctx)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            advance(pool, inc)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        advance(pool, inc)

    for n, real in enumerate((["a", "b"], ["c"], ["b", "d", "a", "c"], ["d", "a"])):
        values = rows_of(len(real), 10.0 * (n + 1)).cuda()
        step, ctx = replay(real, bs=bs)
        plan(pool, step, ctx)
        inc.zero_()
        inc[: len(real)] = values
        graph.replay()
        pool.commit(step, ctx)

        step, ctx = eager(real)
        plan(ref, step, ctx)
        advance(ref, values)
        ref.commit(step, ctx)

        for rid in "abcd":
            if rid not in ref._slots or "main" not in ref._slots[rid]:
                continue
            i, j = slot_of(ref, rid), slot_of(pool, rid)
            torch.testing.assert_close(pool.block("h", 1)[j], ref.block("h", 1)[i])
            assert pool.block("cursor", 0)[j].item() == ref.block("cursor", 0)[i].item()
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="captures a CUDA graph")
def test_a_values_read_and_write_captured_address_the_rows_each_replay_plans():
    """A Value's handle calls the pool behind a graph break, so a capture records the gather, the fresh-row select and
    the scatter: a replay reads its own rows, a padding or fresh row reads the start, and a written row its entry."""
    array = Value("state", nodes={"node"}, layers={0}, entry=Entry({
        "h": TensorType(SHAPE, dtype=torch.float32), "n": TensorType((), dtype=torch.int64),
    }))
    (spec,) = build_specs([array], kernel_settings={"state": {"max_slots": MAX_SLOTS}})
    pool = RecurrentStatePool.build(spec, EngineResourceInfo(device=torch.device("cuda")))
    for rid in "abcd":
        pool.ingest_request(rid)
    handle = build_handle(array, {"state": pool})
    handle.set_start({"h": torch.full(SHAPE, 100.0, device="cuda")})
    inc = torch.zeros(4, *SHAPE, device="cuda")

    def forward():
        fields = handle.read(0).fields
        handle.write(0, h=fields["h"] + inc, n=fields["n"] + 1)

    def declared(step, ctx) -> SubmoduleStep:
        contract_step = SubmoduleStep(segments=list(step.segments), state={"state": ArrayStep()})
        contract_step.set_ctx(ctx)
        return contract_step

    def one_forward(contract_step) -> None:
        handle.begin_step(contract_step)
        forward()
        handle.check_step(contract_step)
        handle.end_step(contract_step)

    step, ctx = replay([], bs=4, capture=True)
    plan(pool, step, ctx)
    contract_step = declared(step, ctx)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        one_forward(contract_step)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        one_forward(contract_step)

    for real in (["a", "b"], ["b", "c", "a"]):
        step, ctx = replay(real, bs=4)
        plan(pool, step, ctx)
        inc.zero_()
        inc[: len(real)] = 1.0
        graph.replay()
        pool.commit(step, ctx)
    torch.cuda.synchronize()
    h, n = pool.block("h", 0), pool.block("n", 0)
    assert h[slot_of(pool, "a")].eq(102.0).all() and n[slot_of(pool, "a")] == 2, "a fresh row started from the start"
    assert h[slot_of(pool, "b")].eq(102.0).all() and n[slot_of(pool, "b")] == 2, "a written row read its own entry"
    assert h[slot_of(pool, "c")].eq(101.0).all() and n[slot_of(pool, "c")] == 1, "a row's first replay read the start"
