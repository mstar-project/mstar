"""Persistent rows over many steps: after each step's diff and replay, the
dense view holds the step's rows in its order and padding past them, every
resident row its request's state, and each mirror its device rows."""
import random

import numpy as np
import pytest
import torch

from mstar.engine.rows import RowSpace

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
MAX_ROWS, PAGES, STATE = 8, 4, 3
CAPACITY = 12


class _Sim:
    """A resource's view of the rows: a length and page table with host truth
    per request, and device-only state checked out of a per-request store."""

    def __init__(self, device, capacity=CAPACITY):
        self.space = RowSpace(MAX_ROWS, device, depth=2, block_bytes=256, capacity=capacity)
        # a length the forward appends one to, as KV's would be
        self.lens = self.space.buffer("lens", (), torch.int32, pad=0, step=1)
        self.table = self.space.buffer("table", (PAGES,), torch.int32, pad=-1)
        self.state = self.space.buffer("state", (STATE,), torch.float32, pad=0.0, mirrored=False)
        self.store = torch.zeros(64, STATE, dtype=torch.float32, device=device)
        self.truth_len: dict[int, int] = {}
        self.truth_table: dict[int, list[int]] = {}
        self.truth_state: dict[int, np.ndarray] = {}

    def forward(self):
        """A step's forward: the engine gathers its rows, it advances the
        device state, and the engine steps the rows and scatters them back."""
        self.space.gather(MAX_ROWS)
        self.state.device[:] += self._active_state()
        self.space.after_forward()

    def _active_state(self):
        return (torch.arange(MAX_ROWS, device=self.space.device) < self.space.n_active).unsqueeze(1)

    def step(self, rids, rng, commit=True, forward=None):
        rows = self.space.propose(rids)
        d = rows.diff
        for rid in d.join_rids:
            if rid not in self.truth_len:
                self.truth_len[rid] = rng.randrange(1, 5)
                self.truth_table[rid] = [rng.randrange(100) for _ in range(PAGES)]
                self.truth_state[rid] = np.zeros(STATE, dtype=np.float32)
        if d.leave_rids:
            rows.copy_out(self.state, d.leave_slots, self.store, np.array(d.leave_rids))
        if d.join_rids:
            rows.write(self.lens, d.join_slots, np.array([self.truth_len[r] for r in d.join_rids], np.int32))
            rows.write(self.table, d.join_slots, np.array([self.truth_table[r] for r in d.join_rids], np.int32))
            rows.copy_in(self.state, d.join_slots, self.store, np.array(d.join_rids))
        # a page appended to some staying rows, as KV would at a page crossing
        staying = [rid for rid in d.order if rid not in d.join_rids]
        appended = [rid for rid in staying if rng.random() < 0.3]
        if appended:
            cols = np.array([rng.randrange(PAGES) for _ in appended])
            vals = np.array([rng.randrange(100) for _ in appended], np.int32)
            rows.write_at(self.table, np.array([rows.slot_of(r) for r in appended]), (cols,), vals)
        rows.stage()
        if not commit:
            return rows
        if appended:
            for rid, c, v in zip(appended, cols, vals, strict=True):
                self.truth_table[rid][c] = int(v)
        rows.commit()
        rows.apply()
        (forward or self.forward)()
        for rid in d.order:
            self.truth_len[rid] += 1
            self.truth_state[rid] += 1
        return rows

    def check(self):
        space = self.space
        order = list(space.batch.order)
        n = len(order)
        assert int(space.n_active.item()) == n
        # the dense view: the last step's rows in order (stepped), padding past them
        lens = self.lens.device.cpu().numpy()
        table = self.table.device.cpu().numpy()
        state = self.state.device.cpu().numpy()
        for i, rid in enumerate(order):
            assert lens[i] == self.truth_len[rid]
            assert table[i].tolist() == self.truth_table[rid]
            np.testing.assert_array_equal(state[i], self.truth_state[rid])
        assert (lens[n:] == 0).all() and (table[n:] == -1).all() and (state[n:] == 0).all()
        # every held row, run by the last step or not, holds its request's state
        held = space.batch.held()
        lens, table = self.lens.resident.cpu().numpy(), self.table.resident.cpu().numpy()
        state = self.state.resident.cpu().numpy()
        for rid, slot in held.items():
            assert lens[slot] == self.truth_len[rid]
            assert table[slot].tolist() == self.truth_table[rid]
            np.testing.assert_array_equal(state[slot], self.truth_state[rid])
        # the pad slot, which padding rows gather, stays pad
        pad = space.pad_slot
        assert lens[pad] == 0 and (table[pad] == -1).all() and (state[pad] == 0).all()
        space.check_mirrors()
        # a request that left has its device state back in the store
        store = self.store.cpu().numpy()
        for rid, truth in self.truth_state.items():
            if rid not in held:
                np.testing.assert_array_equal(store[rid], truth)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("capacity", [MAX_ROWS, CAPACITY])
def test_random_steps_keep_rows_and_mirrors_exact(device, seed, capacity):
    """More live requests than a step runs, so steps rotate through them,
    and rows are evicted when the slots run out."""
    rng = random.Random(seed)
    sim = _Sim(device, capacity)
    live, next_rid = [], 0
    for _ in range(120):
        live = [rid for rid in live if rng.random() > 0.1]
        while len(live) < 16 and rng.random() < 0.5:
            # a returning request rejoins from its store and truth
            gone = [r for r in sim.truth_len if r not in live]
            if gone and rng.random() < 0.3:
                live.append(rng.choice(gone))
            elif next_rid < 64:
                live.append(next_rid)
                next_rid += 1
            else:
                break
        if not live:
            continue
        sim.step(rng.sample(live, min(len(live), rng.randint(1, MAX_ROWS))), rng)
        sim.check()


@pytest.mark.parametrize("device", DEVICES)
def test_a_rotation_with_room_uploads_only_its_step(device):
    """Once two alternating steps' rows have all joined, a step stages one
    segment, its header, and no row moves."""
    rng = random.Random(0)
    sim = _Sim(device)
    halves = [[1, 2, 3, 4], [5, 6, 7, 8]]
    for half in halves:
        sim.step(half, rng)
    for k in range(6):
        rows = sim.space.propose(halves[k % 2])
        assert not rows.diff.join_rids and not rows.diff.leave_rids
        rows.stage()
        assert [count for _, count, _ in rows._staged.phases] == [1]
        rows.commit()
        rows.apply()
        sim.forward()
        for rid in halves[k % 2]:
            sim.truth_len[rid] += 1
            sim.truth_state[rid] += 1
        sim.check()


@pytest.mark.parametrize("device", DEVICES)
def test_a_discarded_step_changes_nothing(device):
    rng = random.Random(0)
    sim = _Sim(device)
    sim.step([1, 2, 3], rng)
    lens, table = sim.lens.host.copy(), sim.table.host.copy()
    sim.step([2, 4], rng, commit=False)
    np.testing.assert_array_equal(sim.lens.host, lens)
    np.testing.assert_array_equal(sim.table.host, table)
    assert list(sim.space.batch.order) == [1, 2, 3]
    sim.step([2, 4], rng)
    sim.check()


@pytest.mark.parametrize("device", DEVICES)
def test_a_steady_step_stages_nothing(device):
    rng = random.Random(0)
    sim = _Sim(device)
    sim.step([1, 2], rng)
    for _ in range(3):
        # the forward steps the lengths on both sides: nothing to upload
        rows = sim.space.propose([1, 2])
        np.testing.assert_array_equal(rows.mirror(sim.lens)[:2], sim.lens.host[rows.diff.idx])
        rows.stage()
        assert rows._staged is None
        rows.commit()
        rows.apply()
        sim.forward()
    sim.space.check_mirrors()


@pytest.mark.parametrize("device", DEVICES)
def test_a_check_in_does_not_step(device):
    """Rows that change without a forward (a check-in) keep their lengths."""
    rng = random.Random(0)
    sim = _Sim(device)
    sim.step([1, 2, 3], rng)
    slot_of = sim.space.batch.slot_of
    lens = {rid: int(sim.lens.host[slot_of(rid)]) for rid in (1, 3)}
    rows = sim.space.without([2])
    rows.copy_out(sim.state, rows.diff.leave_slots, sim.store, np.array(rows.diff.leave_rids))
    rows.stage()
    rows.commit(forward=False)
    rows.apply()
    assert {rid: int(sim.lens.host[slot_of(rid)]) for rid in (1, 3)} == lens
    sim.space.check_mirrors()


@pytest.mark.parametrize("device", DEVICES)
def test_a_resync_puts_the_device_rows_back(device):
    """A step whose rows committed but whose forward never ran: the next
    upload brings the device rows back to the mirror."""
    rng = random.Random(0)
    sim = _Sim(device)
    sim.step([1, 2], rng)
    rows = sim.space.propose([1, 2])
    rows.stage()
    rows.commit()
    rows.apply()  # and no forward
    with pytest.raises(AssertionError, match="lens"):
        sim.space.check_mirrors()
    sim.space.resync()
    rows = sim.space.propose([1, 2])
    rows.stage()
    rows.commit(forward=False)
    rows.apply()
    sim.space.check_mirrors()


def test_a_join_left_unwritten_is_refused():
    space = RowSpace(4, "cpu")
    lens = space.buffer("lens", (), torch.int32)
    space.buffer("other", (), torch.int32)
    rows = space.propose([7])
    rows.write(lens, rows.diff.join_slots, np.array([3], np.int32))
    with pytest.raises(RuntimeError, match="other"):
        rows.stage()



@pytest.mark.skipif(not torch.cuda.is_available(), reason="captures a CUDA graph")
def test_a_captured_step_steps_the_rows_active_at_replay():
    """The gather, forward and ``after_forward`` captured once: the replay steps the rows
    active when it runs, not those active at capture."""
    rng = random.Random(0)
    sim = _Sim("cuda")
    sim.space.seal()  # as the engine does, before any capture records the rows
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph):
        sim.forward()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(40):
        sim.step(rng.sample(range(12), rng.randint(1, MAX_ROWS)), rng, forward=graph.replay)
        sim.check()


@pytest.mark.parametrize("device", DEVICES)
def test_a_proposal_copied_before_a_forward_is_stale(device):
    """A forward steps the rows without changing them, so a proposal whose
    mirrors were copied before it would lose that step: it is stale, and
    refused if committed anyway."""
    rng = random.Random(0)
    sim = _Sim(device)
    sim.step([1, 2], rng)
    ahead = sim.space.propose([1, 2, 3])
    ahead.mirror(sim.lens)  # copies the mirrors
    sim.step([1, 2], rng)
    assert ahead.stale
    with pytest.raises(RuntimeError, match="copied before a forward"):
        ahead.commit()
    sim.step([1, 2, 3], rng)
    sim.check()
    sim.space.check_mirrors()


def test_copy_in_refuses_a_mirrored_buffer():
    """Its mirror, the truth for the host, would not see the rows."""
    space = RowSpace(4, torch.device("cpu"))
    buf = space.buffer("x", (), torch.int32)
    rows = space.propose([5])
    with pytest.raises(ValueError, match="mirrored"):
        rows.copy_in(buf, rows.diff.join_slots, torch.zeros(8, dtype=torch.int32), np.array([5]))


def test_a_row_buffer_needs_a_dtype_numpy_has():
    space = RowSpace(4, torch.device("cpu"))
    for mirrored in (True, False):
        with pytest.raises(ValueError, match="numpy has no"):
            space.buffer(f"x{mirrored}", (), torch.bfloat16, mirrored=mirrored)
