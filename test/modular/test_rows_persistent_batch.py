"""Slot assignment of a persistent batch: dense rows, stable slots, diffs that
take one committed state to the next."""
import random

import numpy as np
import pytest

from mstar.engine.rows import PersistentBatch


def _check(pb: PersistentBatch, before: list[int], rids: list[int]):
    diff = pb.propose(rids)
    slot_before = {rid: slot for slot, rid in enumerate(before)}
    n = len(rids)
    assert sorted(diff.order) == sorted(rids)
    assert diff.n_before == len(before) and diff.n_active == n
    assert set(diff.leave_rids) == set(before) - set(rids)
    assert [slot_before[r] for r in diff.leave_rids] == diff.leave_slots.tolist()
    assert set(diff.join_rids) == set(rids) - set(before)
    for slot, rid in enumerate(diff.order):
        old = slot_before.get(rid)
        if old is not None and old < n:
            assert old == slot, "a row that stays inside the step keeps its slot"
    moved = dict(zip(diff.move_dst.tolist(), diff.move_src.tolist(), strict=True))
    for dst, src in moved.items():
        assert src >= n and dst < n and before[src] == diff.order[dst]
    for slot, rid in zip(diff.join_slots.tolist(), diff.join_rids, strict=True):
        assert diff.order[slot] == rid
    # the three phases touch disjoint slots where they must
    assert not set(diff.move_dst.tolist()) & set(diff.join_slots.tolist())
    assert all(src >= n for src in diff.move_src.tolist())
    return diff


@pytest.mark.parametrize("seed", range(10))
def test_random_steps_keep_rows_dense_and_slots_stable(seed):
    rng = random.Random(seed)
    pb = PersistentBatch(max_rows=16)
    live, next_rid = [], 0
    for _ in range(300):
        keep = [rid for rid in live if rng.random() > 0.15]
        rng.shuffle(keep)
        while len(keep) < 16 and rng.random() < 0.3:
            keep.append(next_rid)
            next_rid += 1
        before = list(pb.order)
        diff = _check(pb, before, keep)
        pb.commit(diff)
        assert list(pb.order) == diff.order
        assert all(pb.slot_of(rid) == slot for slot, rid in enumerate(pb.order))
        assert all(not pb.is_checked_out(rid) for rid in diff.leave_rids)
        live = list(pb.order)


def test_the_same_rows_in_slot_order_are_an_empty_diff():
    pb = PersistentBatch(8)
    pb.commit(pb.propose([5, 6, 7]))
    diff = pb.propose([5, 6, 7])
    assert diff.empty()
    # another order is the same rows: no moves, but the step runs in slot order
    diff = pb.propose([7, 5, 6])
    assert diff.order == [5, 6, 7] and not len(diff.move_src) and not diff.join_rids


def test_a_leave_swaps_the_last_row_into_its_hole():
    pb = PersistentBatch(8)
    pb.commit(pb.propose([1, 2, 3, 4]))
    diff = pb.propose([1, 3, 4])
    assert diff.order == [1, 4, 3]
    assert diff.move_src.tolist() == [3] and diff.move_dst.tolist() == [1]
    assert diff.leave_rids == [2] and diff.leave_slots.tolist() == [1]


def test_without_checks_rows_in():
    pb = PersistentBatch(8)
    pb.commit(pb.propose([1, 2, 3]))
    diff = pb.without([1])
    assert diff.leave_rids == [1] and diff.order == [3, 2] and not diff.join_rids


def test_a_diff_commits_only_onto_the_state_it_was_taken_from():
    pb = PersistentBatch(8)
    stale = pb.propose([1, 2])
    pb.commit(pb.propose([3]))
    with pytest.raises(RuntimeError, match="version"):
        pb.commit(stale)


def test_refuses_too_many_rows_and_repeated_rows():
    pb = PersistentBatch(2)
    with pytest.raises(ValueError):
        pb.propose([1, 2, 3])
    with pytest.raises(ValueError):
        pb.propose([1, 1])


def test_diff_columns_are_int64():
    pb = PersistentBatch(4)
    diff = pb.propose([1, 2])
    assert diff.join_slots.dtype == np.int64 and diff.move_src.dtype == np.int64


def test_a_forgotten_row_leaves_with_nothing_read_out_and_frees_its_slot():
    pb = PersistentBatch(8)
    pb.commit(pb.propose([1, 2, 3]))
    assert pb.forget(2) and not pb.forget(9)
    assert not pb.is_checked_out(2)
    diff = pb.propose([1, 3])
    assert diff.leave_rids == [2] and diff.dropped == {2}
    assert pb.commit(diff).tolist() == [1]
    assert pb.slot_of(2) is None


def test_a_diff_taken_before_the_forget_still_frees_the_slot():
    pb = PersistentBatch(8)
    pb.commit(pb.propose([1, 2, 3]))
    diff = pb.propose([1, 3])
    pb.forget(2)
    assert diff.dropped == frozenset()
    assert pb.commit(diff).tolist() == [1]


def test_a_recycled_handle_joins_afresh_while_its_old_row_leaves():
    pb = PersistentBatch(8)
    pb.commit(pb.propose([1, 2, 3]))
    pb.forget(2)
    diff = pb.propose([1, 2, 3])
    assert diff.leave_rids == [2] and diff.dropped == {2}
    assert diff.join_rids == [2]
    assert sorted(diff.order) == [1, 2, 3]
    assert pb.commit(diff).tolist() == [1]
    assert pb.is_checked_out(2)


def test_a_step_that_keeps_a_forgotten_row_is_refused():
    pb = PersistentBatch(8)
    pb.commit(pb.propose([1, 2]))
    diff = pb.propose([1, 2, 3])
    pb.forget(2)
    with pytest.raises(RuntimeError, match="removed requests"):
        pb.commit(diff)
