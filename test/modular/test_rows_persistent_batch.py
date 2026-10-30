"""Slot assignment of a persistent batch: resident slots that stay put, steps
that name their slots in step order, diffs that take one committed state to
the next."""
import random

import numpy as np
import pytest

from mstar.engine.rows import batch as batch_mod

IMPLS = [batch_mod.PyPersistentBatch] + (
    [batch_mod.RustPersistentBatch] if batch_mod._RustRowSlots is not None else []
)
PersistentBatch = batch_mod.PersistentBatch


@pytest.fixture(autouse=True, params=IMPLS, ids=lambda cls: cls.__name__)
def _impl(request, monkeypatch):
    """Every test on each slot assignment backend."""
    monkeypatch.setitem(globals(), "PersistentBatch", request.param)


def _check(pb: PersistentBatch, rids: list[int]):
    """Propose ``rids`` and check the diff against the committed state."""
    held = {rid: pb.slot_of(rid) for rid in list(_held(pb))}
    free = set(range(pb.capacity)) - set(held.values())
    diff = pb.propose(rids)
    assert diff.order == list(rids) and diff.n_active == len(rids)
    assert len(set(diff.idx.tolist())) == len(rids)
    joined = dict(zip(diff.join_rids, diff.join_slots.tolist(), strict=True))
    assert set(joined) == {rid for rid in rids if rid not in held}
    for rid, slot in zip(rids, diff.idx.tolist(), strict=True):
        # a held row runs where it is; a joining row where it was given
        assert slot == (joined[rid] if rid in joined else held[rid])
    left = dict(zip(diff.leave_rids, diff.leave_slots.tolist(), strict=True))
    assert all(held[rid] == slot for rid, slot in left.items())
    assert not set(left) & set(rids), "no row the step runs leaves"
    # rows leave only to make room, and joins take free slots before those
    assert len(left) == max(0, len(joined) - len(free))
    used_free = [slot for slot in joined.values() if slot in free]
    assert used_free == sorted(free)[:len(used_free)]
    if left:
        idle = [rid for rid in held if rid not in rids and rid not in left]
        last = pb.last_run
        assert all(last(left_slot) <= last(held[rid]) for left_slot in left.values() for rid in idle)
    return diff


def _held(pb: PersistentBatch):
    return pb.held()


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("capacity", [8, 12, 40])
def test_random_steps_keep_slots_stable(seed, capacity):
    rng = random.Random(seed)
    pb = PersistentBatch(max_rows=8, capacity=capacity)
    live, next_rid = [], 0
    for _ in range(300):
        # requests finish, new ones arrive, and each step runs some of the live ones
        live = [rid for rid in live if rng.random() > 0.05]
        while len(live) < 20 and rng.random() < 0.5:
            live.append(next_rid)
            next_rid += 1
        step = rng.sample(live, min(len(live), rng.randint(1, 8)))
        diff = _check(pb, step)
        pb.commit(diff)
        assert list(pb.order) == step
        assert all(pb.slot_of(rid) == slot for rid, slot in zip(step, diff.idx.tolist(), strict=True))
        assert all(not pb.is_checked_out(rid) for rid in diff.leave_rids)
        assert pb.n_held <= capacity


def test_with_room_for_every_live_request_a_rotation_moves_no_rows():
    """The scheduler runs two halves of the live requests in turn: once each
    has joined, no step joins or leaves anything."""
    pb = PersistentBatch(max_rows=4, capacity=8)
    halves = [[0, 1, 2, 3], [4, 5, 6, 7]]
    for half in halves:
        pb.commit(pb.propose(half))
    for k in range(10):
        diff = pb.propose(halves[k % 2])
        assert not diff.join_rids and not diff.leave_rids
        assert not diff.empty(), "the step's slots changed: its header uploads"
        pb.commit(diff)


def test_without_room_a_rotation_evicts_the_rows_that_ran_least_recently():
    pb = PersistentBatch(max_rows=2, capacity=3)
    pb.commit(pb.propose([1, 2]))
    pb.commit(pb.propose([3]))
    diff = pb.propose([4, 3])
    # 1 and 2 ran together, before 3; the lower slot breaks the tie
    assert diff.leave_rids == [1] and diff.join_rids == [4]
    assert diff.join_slots.tolist() == [pb.slot_of(1)]


def test_the_same_rows_in_the_same_order_are_an_empty_diff():
    pb = PersistentBatch(8)
    pb.commit(pb.propose([5, 6, 7]))
    assert pb.propose([5, 6, 7]).empty()
    # another order runs the same slots in that order
    diff = pb.propose([7, 5, 6])
    assert diff.order == [7, 5, 6] and not diff.join_rids and not diff.leave_rids
    assert not diff.empty()
    assert diff.idx.tolist() == [pb.slot_of(7), pb.slot_of(5), pb.slot_of(6)]


def test_a_row_out_of_a_step_keeps_its_slot():
    pb = PersistentBatch(4, capacity=8)
    pb.commit(pb.propose([1, 2, 3, 4]))
    slot = pb.slot_of(2)
    diff = pb.propose([1, 3, 4])
    assert not diff.leave_rids
    pb.commit(diff)
    assert pb.slot_of(2) == slot and pb.is_checked_out(2)


def test_without_checks_rows_in():
    pb = PersistentBatch(8)
    pb.commit(pb.propose([1, 2, 3]))
    diff = pb.without([1])
    assert diff.leave_rids == [1] and diff.order == [2, 3] and not diff.join_rids
    pb.commit(diff)
    assert pb.slot_of(1) is None and pb.n_held == 2


def test_a_diff_commits_only_onto_the_state_it_was_taken_from():
    pb = PersistentBatch(8)
    stale = pb.propose([1, 2])
    pb.commit(pb.propose([3]))
    with pytest.raises(RuntimeError, match="version"):
        pb.commit(stale)


def test_refuses_too_many_rows_repeated_rows_and_too_little_capacity():
    pb = PersistentBatch(2)
    with pytest.raises(ValueError):
        pb.propose([1, 2, 3])
    with pytest.raises(ValueError):
        pb.propose([1, 1])
    with pytest.raises(ValueError, match="capacity"):
        PersistentBatch(4, capacity=3)


def test_diff_columns_are_int64():
    pb = PersistentBatch(4)
    diff = pb.propose([1, 2])
    assert diff.join_slots.dtype == np.int64 and diff.idx.dtype == np.int64


def test_a_forgotten_row_leaves_with_nothing_read_out_and_frees_its_slot():
    pb = PersistentBatch(8)
    pb.commit(pb.propose([1, 2, 3]))
    assert pb.forget(2) and not pb.forget(9)
    assert not pb.is_checked_out(2)
    diff = pb.propose([1, 3])
    assert diff.leave_rids == [2] and diff.dropped == {2}
    assert pb.commit(diff).tolist() == [1]
    assert pb.slot_of(2) is None


def test_a_row_forgotten_after_a_diff_was_taken_leaves_at_the_next():
    pb = PersistentBatch(8)
    pb.commit(pb.propose([1, 2, 3]))
    diff = pb.propose([1, 3])
    pb.forget(2)
    assert diff.dropped == frozenset()
    assert pb.commit(diff).tolist() == []
    later = pb.propose([1, 3])
    assert later.leave_rids == [2] and later.dropped == {2}
    assert pb.commit(later).tolist() == [1]


def test_a_recycled_handle_joins_afresh_while_its_old_row_leaves():
    pb = PersistentBatch(8)
    pb.commit(pb.propose([1, 2, 3]))
    pb.forget(2)
    diff = pb.propose([1, 2, 3])
    assert diff.leave_rids == [2] and diff.dropped == {2}
    assert diff.join_rids == [2]
    assert diff.order == [1, 2, 3]
    assert pb.commit(diff).tolist() == [1]
    assert pb.is_checked_out(2)


def test_a_step_that_keeps_a_forgotten_row_is_refused():
    pb = PersistentBatch(8)
    pb.commit(pb.propose([1, 2]))
    diff = pb.propose([1, 2, 3])
    pb.forget(2)
    with pytest.raises(RuntimeError, match="removed requests"):
        pb.commit(diff)


def _cols(d):
    return (
        d.base_version, d.n_active, list(d.order), d.idx.tolist(), d.join_slots.tolist(), list(d.join_rids),
        d.leave_slots.tolist(), list(d.leave_rids), d.same_step, set(d.dropped),
    )


@pytest.mark.skipif(len(IMPLS) < 2, reason="the Rust extension predates RowSlots")
@pytest.mark.parametrize("seed", range(10))
def test_both_backends_assign_the_same_slots(seed, request):
    """Fed the same steps, removals and check-ins, the Python and Rust slot
    assignments give the same diffs and refuse the same steps."""
    if "RustPersistentBatch" in request.node.callspec.id:
        pytest.skip("drives both backends itself")
    rng = random.Random(seed)
    capacity = rng.choice([6, 9, 20])
    py, rs = (cls(max_rows=6, capacity=capacity) for cls in IMPLS)
    live = []
    for _ in range(400):
        live = [rid for rid in live if rng.random() > 0.05]
        while len(live) < 14 and rng.random() < 0.5:
            live.append(rng.randrange(30))  # handles come back
        live = list(dict.fromkeys(live))
        if live and rng.random() < 0.1:
            rid = rng.choice(live)
            assert py.forget(rid) == rs.forget(rid)
        if rng.random() < 0.1:
            gone = rng.sample(live, min(len(live), 2))
            diffs = [b.without(gone) for b in (py, rs)]
        else:
            step = rng.sample(live, min(len(live), rng.randint(0, 6)))
            if rng.random() < 0.05:
                step = list(py.order)  # the same step again
            diffs = [b.propose(step) for b in (py, rs)]
        assert _cols(diffs[0]) == _cols(diffs[1])
        outcomes = []
        for b, d in zip((py, rs), diffs, strict=True):
            try:
                outcomes.append(b.commit(d).tolist())
            except RuntimeError as e:
                outcomes.append(type(e))
        assert outcomes[0] == outcomes[1]
        assert py.held() == rs.held() and py.version == rs.version and list(py.order) == list(rs.order)
