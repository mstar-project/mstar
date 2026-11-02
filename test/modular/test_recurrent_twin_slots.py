"""A pool with twin slots: a decode step reads one and writes the other, and they swap at commit. Every other path
keeps addressing the slot the state is in, so it never learns the twin exists. CPU only; the kernels are in
test/rocm/test_gdn_aiter.py."""

from types import SimpleNamespace

import pytest
import torch

from mstar.engine.resources.linear_attn.gdn import GDNManager
from mstar.engine.resources.recurrent import (
    RecurrentBlockConfig,
    RecurrentStateConfig,
    RecurrentStatePool,
    RecurrentStateSpec,
    RecurrentStep,
)
from mstar.engine.resources.recurrent.pool import SINK_SLOT
from mstar.engine.resources.step import BucketKey, Segment, SlotLease, StepContext


def _pool(max_slots=4, twin=True, **config) -> RecurrentStatePool:
    blocks = {"state": RecurrentBlockConfig(shape=(2,), dtype=torch.float32)}
    return RecurrentStatePool(torch.device("cpu"), RecurrentStateConfig(
        num_layers=1, blocks=blocks, max_slots=max_slots, twin_slots=twin, **config,
    ))


def _step(pool, rows, walk="decode", capture=False, preplan=False, force_prefill=False, lease=None, padding=()):
    """Admit and plan ``rows`` of (rid, span), and return what commit needs and the addressing."""
    real = [rid for rid, _ in rows]
    ctx = StepContext(request_ids=real, graph_walk=walk, slot=0, capture=capture, is_preplan=preplan,
                      force_prefill=force_prefill, slot_lease=lease)
    if padding:
        ctx.set_padded_rids([*real, *padding])
    segments = (*(Segment(r, "main", s) for r, s in rows), *(Segment(r, "main", 1) for r in padding))
    step = RecurrentStep(segments=segments)
    for rid in real:
        pool.ingest_request(rid)
    assert pool.admit(step, ctx).ok
    return step, ctx, pool.plan(step, ctx)["main"]


def _reads_writes(addressing):
    n = addressing.num_rows
    return addressing.slot_indices[:n].tolist(), addressing.write_slot_indices[:n].tolist()


def test_a_request_holds_two_slots_and_max_slots_still_counts_requests():
    pool = _pool(max_slots=4)
    assert pool.config.usable_slots == 3
    assert pool.config.allocated_slots == 7
    assert pool.block("state", 0).shape[0] == 7
    assert pool.num_free_slots == 6
    for rid in "abc":
        _step(pool, [(rid, 1)])
    assert pool.num_free_slots == 0
    # the fourth request is refused like any request past max_slots
    pool.ingest_request("d")
    ctx = StepContext(request_ids=["d"], graph_walk="decode", slot=0, capture=False)
    assert not pool.admit(RecurrentStep(segments=(Segment("d", "main", 1),)), ctx).ok


def test_decode_reads_one_writes_the_other_and_commit_swaps():
    pool = _pool()
    step, ctx, addressing = _step(pool, [("a", 1), ("b", 1)])
    reads, writes = _reads_writes(addressing)
    slots = pool._slots
    assert reads == [slots["a"]["main"].index, slots["b"]["main"].index]
    assert writes == [slots["a"]["main"].twin[0], slots["b"]["main"].twin[0]]
    assert len({*reads, *writes}) == 4
    pool.commit(step, ctx)
    assert (slots["a"]["main"].index, slots["a"]["main"].twin) == (writes[0], (reads[0],))
    step, ctx, addressing = _step(pool, [("a", 1), ("b", 1)])
    assert _reads_writes(addressing) == (writes, reads)
    pool.commit(step, ctx)
    assert slots["a"]["main"].index == reads[0]
    assert slots["a"]["main"].tokens == 2


@pytest.mark.parametrize(
    ("rows", "kw"),
    [
        ([("a", 5)], dict(walk="prefill")),
        ([("a", 1)], dict(force_prefill=True)),
        ([("a", 1), ("b", 3)], dict(walk="prefill")),
    ],
    ids=["a prompt", "a packed capture's one-token row", "a longer row beside it"],
)
def test_only_a_decode_step_writes_the_twin(rows, kw):
    pool = _pool()
    step, ctx, addressing = _step(pool, rows, **kw)
    assert addressing.write_slot_indices is None
    index = pool._slots["a"]["main"].index
    pool.commit(step, ctx)
    assert pool._slots["a"]["main"].index == index


def _mixed_pool():
    pool = _pool(max_slots=6)
    pool.split_walks(frozenset({"mixed"}))
    return pool


def test_a_split_step_swaps_each_decode_row_once_and_leaves_its_prompt_rows():
    pool = _mixed_pool()
    # "a" and "b" hold a state; "c" and "d" start in this step
    pool.commit(*_step(pool, [("a", 5), ("b", 3)], walk="prefill")[:2])
    slots = pool._slots
    a, b = slots["a"]["main"], slots["b"]["main"]
    a0, a1, b0, b1 = a.index, a.twin[0], b.index, b.twin[0]

    step, ctx, addressing = _step(pool, [("a", 1), ("c", 4), ("b", 1), ("d", 1)], walk="mixed")
    c, d = slots["c"]["main"], slots["d"]["main"]
    n = addressing.num_rows
    assert addressing.slot_indices[:n].tolist() == [a0, c.index, b0, d.index]
    # a row writes its twin only if it takes the decode kernel: "d" is a one-token prompt
    assert addressing.host_write_slots == [a1, SINK_SLOT, b1, SINK_SLOT]
    assert addressing.write_slot_indices is None
    c0, d0 = c.index, d.index
    pool.commit(step, ctx)
    assert (a.index, a.twin, a.swapped) == (a1, (a0,), True)
    assert (b.index, b.twin, b.swapped) == (b1, (b0,), True)
    assert (c.index, c.swapped, d.index, d.swapped) == (c0, False, d0, False)

    # the next step finds every row where the last one left it, "c" and "d" now decode rows
    step, ctx, addressing = _step(pool, [("a", 1), ("c", 1), ("d", 1)], walk="mixed")
    assert addressing.slot_indices[:3].tolist() == [a1, c0, d0]
    assert addressing.host_write_slots == [a0, c.twin[0], d.twin[0]]
    pool.commit(step, ctx)
    assert (a.index, c.swapped, d.swapped) == (a0, True, True)


def test_a_split_step_keeps_padding_and_capture_rows_on_the_sink():
    pool = _mixed_pool()
    pool.commit(*_step(pool, [("a", 5)], walk="prefill")[:2])
    lease = SlotLease(slot=0, bucket=BucketKey("mixed", bs=2, num_tokens=8))
    step, ctx, addressing = _step(pool, [("a", 1)], walk="mixed", lease=lease, padding=("pad",))
    assert addressing.host_write_slots == [pool._slots["a"]["main"].twin[0], SINK_SLOT]
    pool.commit(step, ctx)

    capture = _mixed_pool()
    step, ctx, addressing = _step(capture, [("a", 1)], walk="mixed", capture=True, lease=lease, padding=("pad",))
    assert addressing.host_write_slots == [SINK_SLOT, SINK_SLOT]
    capture.commit(step, ctx)


def test_a_split_step_over_a_pool_without_twins_writes_in_place():
    pool = _pool(twin=False)
    pool.split_walks(frozenset({"mixed"}))
    pool.commit(*_step(pool, [("a", 5)], walk="prefill")[:2])
    index = pool._slots["a"]["main"].index
    step, ctx, addressing = _step(pool, [("a", 1)], walk="mixed")
    assert addressing.host_write_slots is None
    pool.commit(step, ctx)
    assert pool._slots["a"]["main"].index == index


class _NoKernels:
    def without_qk_l2norm(self):
        return self

    def captured_prefill(self, bs, num_tokens):
        return SimpleNamespace(plan=lambda plan: None)

    chunked_prefill = decode = lambda self, *args: SimpleNamespace(plan=lambda plan: None)


def test_a_mixed_wrapper_hands_the_decode_kernel_each_rows_read_and_write():
    from mstar.engine.resources.linear_attn.wrappers import GDNMixedWrapper

    wrapper = GDNMixedWrapper(torch.device("cpu"), pad_slot_id=-1, null_slot_id=SINK_SLOT, kernels=_NoKernels())
    # a decode row, a prompt, a decode row, a one-token prompt
    spans, slots, has = [1, 3, 1, 1], (5, 7, 9, 11), (True, False, True, False)
    wrapper.plan(spans, slots, has, host_write_slots=[6, SINK_SLOT, 10, SINK_SLOT])
    plan = wrapper._decode._plan_state
    assert plan.slots.tolist() == [5, SINK_SLOT, 9, SINK_SLOT]
    assert plan.write_slots.tolist() == [6, SINK_SLOT, 10, SINK_SLOT]
    # the prompt rows update in place, in their own slots
    assert wrapper._prefill._plan_state.slots.tolist() == [7, 11]

    wrapper.plan(spans, slots, has)
    assert wrapper._decode._plan_state.write_slots is None


def test_a_pool_without_twins_is_todays():
    pool = _pool(twin=False)
    assert pool.config.allocated_slots == 4
    step, ctx, addressing = _step(pool, [("a", 1)])
    assert addressing.write_slot_indices is None
    index = pool._slots["a"]["main"].index
    assert pool._slots["a"]["main"].twin == ()
    pool.commit(step, ctx)
    assert pool._slots["a"]["main"].index == index


def test_padding_and_capture_rows_stay_on_the_sink():
    pool = _pool()
    lease = SlotLease(slot=0, bucket=BucketKey("decode", bs=2, num_tokens=2))
    step, ctx, addressing = _step(pool, [("a", 1)], lease=lease, padding=("pad",))
    assert addressing.write_slot_indices.tolist()[1] == SINK_SLOT
    assert addressing.slot_indices.tolist()[1] == SINK_SLOT
    pool.commit(step, ctx)
    # a capture's rows reserve nothing, and its commit swaps nothing
    capture = _pool()
    step, ctx, addressing = _step(capture, [("a", 1)], capture=True, lease=lease, padding=("pad",))
    assert addressing.write_slot_indices.tolist() == [SINK_SLOT, SINK_SLOT]
    capture.commit(step, ctx)


def test_a_pre_plan_after_the_commit_reads_what_the_step_wrote():
    pool = _pool()
    step, ctx, addressing = _step(pool, [("a", 1)])
    reads, writes = _reads_writes(addressing)
    pool.commit(step, ctx)
    _, _, staged = _step(pool, [("a", 1)], preplan=True)
    assert _reads_writes(staged) == (writes, reads)
    # dropping the stage leaves the swap alone, and a fresh plan reads the same
    pool.clear_preplan()
    assert pool._slots["a"]["main"].index == writes[0]
    _, _, again = _step(pool, [("a", 1)])
    assert _reads_writes(again) == (writes, reads)


def test_rollback_and_release_hand_back_both_slots_zeroed():
    pool = _pool()
    free = pool.num_free_slots
    step, ctx, _ = _step(pool, [("a", 1)])
    slot = pool._slots["a"]["main"]
    pool.block("state", 0)[[slot.index, slot.twin[0]]] = 1.0
    pool.rollback_admit(step, ctx)
    assert pool.num_free_slots == free
    assert pool.block("state", 0).abs().sum() == 0

    step, ctx, _ = _step(pool, [("a", 1)])
    pool.commit(step, ctx)
    slot = pool._slots["a"]["main"]
    pool.block("state", 0)[[slot.index, slot.twin[0]]] = 1.0
    pool.remove_request("a")
    assert pool.num_free_slots == free
    assert pool.block("state", 0).abs().sum() == 0


def test_a_decode_snapshot_copies_the_state_the_step_just_wrote():
    pool = _pool(prefix_slots=2)
    pool._kv = SimpleNamespace(config=SimpleNamespace(page_size=2), chain_key=lambda *a: None)
    pool._keyed.add(("a", "main"))
    pool.commit(*_step(pool, [("a", 1)])[:2])
    # the second token ends a page, so this commit saves the state
    step, ctx, addressing = _step(pool, [("a", 1)])
    _, writes = _reads_writes(addressing)
    pool.block("state", 0)[writes[0]] = 7.0
    pool.commit(step, ctx)
    decode = pool._decode[("a", "main")]
    assert pool.block("state", 0)[decode.slot].tolist() == [7.0, 7.0]
    assert decode.slot >= pool.config.allocated_slots


def test_a_fork_copies_from_the_slot_the_state_is_in():
    pool = _pool(labels=("main", "side"))
    step, ctx, _ = _step(pool, [("a", 1)])
    pool.commit(step, ctx)
    main = pool._slots["a"]["main"]
    pool.block("state", 0)[main.index] = 3.0
    pool._slots["a"]["side"] = pool._alloc("a", "side")
    pool._apply_fork("a", "main", "side")
    assert pool.block("state", 0)[pool._slots["a"]["side"].index].tolist() == [3.0, 3.0]


def test_twins_refuse_kernels_that_update_in_place():
    manager = GDNManager.__new__(GDNManager)
    pool = _pool()
    manager._pool_key, manager._spec_key, manager._splits = "gdn_state", None, {}
    manager._kernels = SimpleNamespace(writes_snapshots=True, writes_twin_slots=False)
    with pytest.raises(ValueError, match="twin_slots"):
        manager.bind_dependencies({"gdn_state": pool})
    manager._kernels = SimpleNamespace(writes_snapshots=True, writes_twin_slots=True)
    manager._splits = {"mixed": object()}
    manager.bind_dependencies({"gdn_state": pool})
    assert pool._split_walks == {"mixed"}


def _rank_pool():
    blocks = {"state": RecurrentBlockConfig(shape=(2,), dtype=torch.float32)}
    return RecurrentStatePool(torch.device("cpu"), RecurrentStateConfig(
        num_layers=1, blocks=blocks, max_slots=8, twin_slots=True,
    ), world_size=2)


def _parity_of(pool, rows):
    step, ctx, _ = _step(pool, [(r, 1) for r in rows], preplan=True)
    pool.clear_preplan()
    return step, ctx


def test_ranks_report_the_same_parity_for_a_step_with_one_of_them_ahead():
    """"b" joins two steps late and "a" leaves, so the rows' parities differ and move. A rank's teardown lands at its
    own time, so "c" takes a different slot on each rank, which is fine; which twin it reads is not free to differ."""
    script = [["a"], ["a"], ["a", "b"], ["a", "b"], ["b", "c"], ["b", "c"]]

    def run(pool, rows, seen):
        step, ctx = _parity_of(pool, rows)
        seen.append(ctx.slot_parity)
        pool.commit(step, ctx)

    ahead, behind = _rank_pool(), _rank_pool()
    seen_ahead, seen_behind = [], []
    for i, rows in enumerate(script):
        if rows == ["b", "c"] and "a" in ahead._slots:
            ahead.remove_request("a")
        run(ahead, rows, seen_ahead)
        if i:
            run(behind, script[i - 1], seen_behind)
    behind.remove_request("a")
    run(behind, script[-1], seen_behind)
    assert seen_ahead == seen_behind
    assert len(set(seen_ahead)) > 1
    assert ahead._slots["c"]["main"].group != behind._slots["c"]["main"].group


def test_a_rank_that_plans_before_its_commit_reports_another_parity():
    good, bad = _rank_pool(), _rank_pool()
    for pool in (good, bad):
        step, ctx = _parity_of(pool, ["a"])
        pool.commit(step, ctx)
        step, ctx = _parity_of(pool, ["a"])
        if pool is good:
            pool.commit(step, ctx)
    # `bad` plans the next step before committing the one in flight, so it reads the twin that step is still filling
    assert _parity_of(good, ["a"])[1].slot_parity != _parity_of(bad, ["a"])[1].slot_parity


@pytest.mark.parametrize(("twin", "world"), [(True, 1), (False, 2)])
def test_only_twins_under_lockstep_report_a_parity(twin, world):
    blocks = {"state": RecurrentBlockConfig(shape=(2,), dtype=torch.float32)}
    pool = RecurrentStatePool(torch.device("cpu"), RecurrentStateConfig(
        num_layers=1, blocks=blocks, max_slots=4, twin_slots=twin,
    ), world_size=world)
    step, ctx, _ = _step(pool, [("a", 1)])
    pool.commit(step, ctx)
    _, ctx, _ = _step(pool, [("a", 1)])
    assert ctx.slot_parity == 0


def test_the_yaml_override_turns_them_on():
    blocks = {"state": RecurrentBlockConfig(shape=(2,), dtype=torch.float32)}
    spec = RecurrentStateSpec("gdn_state", {"llm"}, RecurrentStateConfig(num_layers=1, blocks=blocks))
    spec.apply_yaml_overrides(max_slots=9, twin_slots=True)
    assert spec.config.twin_slots and spec.config.allocated_slots == 17
