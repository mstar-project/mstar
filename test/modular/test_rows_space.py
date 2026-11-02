"""Persistent rows over many steps: after each step's diff and replay, the
dense view holds the step's rows in its order and padding past them, every
other held row's slot its request's state, and each mirror its device rows."""
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

    def __init__(self, device, capacity=CAPACITY, staged_bytes=256 << 10):
        self.space = RowSpace(
            MAX_ROWS, device, depth=2, block_bytes=256, capacity=capacity, staged_bytes=staged_bytes,
        )
        # a length the forward appends one to, as KV's would be
        self.lens = self.space.buffer("lens", (), torch.int32, pad=0, step=1)
        self.table = self.space.buffer("table", (PAGES,), torch.int32, pad=-1)
        self.state = self.space.buffer("state", (STATE,), torch.float32, pad=0.0, mirrored=False)
        self.store = torch.zeros(64, STATE, dtype=torch.float32, device=device)
        self.truth_len: dict[int, int] = {}
        self.truth_table: dict[int, list[int]] = {}
        self.truth_state: dict[int, np.ndarray] = {}

    def forward(self):
        """A step's forward: it advances the device state in place, and the
        engine steps the rows."""
        self.state.device[:] += self._active_state()
        self.space.after_forward()

    def _active_state(self):
        return (torch.arange(MAX_ROWS, device=self.space.device) < self.space.n_active).unsqueeze(1)

    def step(self, rids, rng, commit=True, forward=None, in_graph=False, append=True):
        rows = self.space.propose(rids)
        d = rows.diff
        for rid in d.join_rids:
            if rid not in self.truth_len:
                self.truth_len[rid] = rng.randrange(1, 5)
                self.truth_table[rid] = [rng.randrange(100) for _ in range(PAGES)]
                self.truth_state[rid] = np.zeros(STATE, dtype=np.float32)
        self._read_back(rows)
        if d.join_rids:
            rows.write(self.lens, d.join_slots, np.array([self.truth_len[r] for r in d.join_rids], np.int32))
            rows.write(self.table, d.join_slots, np.array([self.truth_table[r] for r in d.join_rids], np.int32))
            rows.copy_in(self.state, d.join_slots, self.store, np.array(d.join_rids))
        # a page appended to some staying rows, as KV would at a page crossing
        staying = [rid for rid in d.order if rid not in d.join_rids]
        appended = [rid for rid in staying if append and rng.random() < 0.3]
        if appended:
            cols = np.array([rng.randrange(PAGES) for _ in appended])
            vals = np.array([rng.randrange(100) for _ in appended], np.int32)
            rows.write_at(self.table, np.array([rows.slot_of(r) for r in appended]), (cols,), vals)
        rows.stage(in_graph=in_graph)
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

    def _read_back(self, rows):
        d = rows.diff
        keep = [i for i, rid in enumerate(d.leave_rids) if rid not in d.dropped]
        if keep:
            rows.copy_out(self.state, d.leave_slots[keep], self.store, np.array(d.leave_rids)[keep])

    def lend(self, rids):
        """A step outside the walk runs ``rids``: their rows are lent, then
        it advances their truth and their store's state, as a mixed step's
        resources would."""
        rows = self.space.lend(rids)
        if rows is not None:
            self._read_back(rows)
            rows.copy_out(self.state, rows.lent_slots, self.store, np.array(rows.lent_rids))
            rows.stage()
            rows.commit(forward=False)
            rows.apply()
        for rid in rids:
            self.truth_len[rid] += 1
            self.truth_state[rid] += 1
            self.store[rid] += 1
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
            if rid in space._lent:
                continue  # stale until the walk runs it again
            assert lens[i] == self.truth_len[rid]
            assert table[i].tolist() == self.truth_table[rid]
            np.testing.assert_array_equal(state[i], self.truth_state[rid])
        assert (lens[n:] == 0).all() and (table[n:] == -1).all() and (state[n:] == 0).all()
        # every other held row's slot holds its request's state
        held = space.batch.held()
        lent = space._lent
        lens, table = self.lens.resident.cpu().numpy(), self.table.resident.cpu().numpy()
        state = self.state.resident.cpu().numpy()
        for rid, slot in held.items():
            if rid in order or rid in lent:
                continue
            assert lens[slot] == self.truth_len[rid]
            assert table[slot].tolist() == self.truth_table[rid]
            np.testing.assert_array_equal(state[slot], self.truth_state[rid])
        # the pad slot, which places past a step's rows come back to, stays pad
        pad = space.pad_slot
        assert lens[pad] == 0 and (table[pad] == -1).all() and (state[pad] == 0).all()
        space.check_mirrors()
        # a request that left, or whose row is lent, has its device state back in the store
        store = self.store.cpu().numpy()
        for rid, truth in self.truth_state.items():
            if rid not in held or rid in lent:
                np.testing.assert_array_equal(store[rid], truth)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("capacity", [MAX_ROWS, CAPACITY])
@pytest.mark.parametrize("lends", [False, True])
def test_random_steps_keep_rows_and_mirrors_exact(device, seed, capacity, lends):
    """More live requests than a step runs, so steps rotate through them,
    and rows are evicted when the slots run out; with ``lends``, steps
    outside the walk run some of them between."""
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
        ran = [rid for rid in live if rid in sim.truth_len]
        if lends and ran and rng.random() < 0.3:
            sim.lend(rng.sample(ran, rng.randint(1, len(ran))))
            sim.check()
        sim.step(rng.sample(live, min(len(live), rng.randint(1, MAX_ROWS))), rng)
        sim.check()


@pytest.mark.parametrize("device", DEVICES)
def test_a_rotation_moves_only_the_rows_that_change_places(device):
    """Once two alternating steps' rows have all joined, a step that keeps two
    places and changes two moves only those two out to their slots and the
    other two in, per buffer, besides its header."""
    rng = random.Random(0)
    sim = _Sim(device)
    steps = [[1, 2, 3, 4], [1, 2, 5, 6]]
    for rids in steps:
        sim.step(rids, rng)
    for k in range(6):
        rows = sim.space.propose(steps[k % 2])
        assert not rows.diff.join_rids and not rows.diff.leave_rids
        rows.stage()
        moved = 2 * len(sim.space.buffers)
        assert [count for _, count, _ in rows._staged.phases] == [moved, moved, 1]
        rows.commit()
        rows.apply()
        sim.forward()
        for rid in steps[k % 2]:
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
    """The forward and ``after_forward`` captured once: the replay steps the rows
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


def _captured_prologue(sim):
    """A graph of what a captured step runs: the rows' prologue, the forward, ``after_forward``."""
    sim.space.seal()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph):
        sim.space.prologue()
        sim.state.device[:] += sim._active_state()
        sim.space.after_forward()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


@pytest.mark.skipif(not torch.cuda.is_available(), reason="captures a CUDA graph")
@pytest.mark.parametrize("staged_bytes", [256 << 10, 64])
def test_a_replay_applies_the_upload_staged_for_it(staged_bytes):
    """A step staged for its graph uploads into the block the graph's prologue
    applies before its forward; one too large for it is applied before the
    replay instead. A repeated step stages nothing, and its replay applies
    nothing again."""
    rng = random.Random(1)
    sim = _Sim("cuda", staged_bytes=staged_bytes)
    graph = _captured_prologue(sim)
    rids = None
    for _ in range(80):
        repeat = rids is not None and rng.random() < 0.4
        if not repeat:
            rids = rng.sample(range(12), rng.randint(1, MAX_ROWS))
        sim.step(rids, rng, forward=graph.replay, in_graph=True, append=not repeat)
        sim.check()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="captures a CUDA graph")
def test_an_upload_whose_replay_never_ran_is_dropped():
    rng = random.Random(2)
    sim = _Sim("cuda")
    graph = _captured_prologue(sim)
    rids = [1, 2, 3]
    sim.step(rids, rng, forward=graph.replay, in_graph=True)
    for _ in range(5):
        # pages staged for the replay, then the step fails before it commits
        while not sim.step(rids, rng, commit=False, in_graph=True)._staged:
            pass
        sim.space.drop_staged()
        # nothing to stage: a replay applying the failed step's pages would show them
        sim.step(rids, rng, forward=graph.replay, in_graph=True, append=False)
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


@pytest.mark.parametrize("device", DEVICES)
def test_an_unmoved_buffer_is_addressed_by_slot(device):
    """Wide rows a step changes in part stay resident: no step moves them, and
    its slots say which resident row each dense row is."""
    space = RowSpace(4, device)
    lens = space.buffer("lens", (), torch.int32, step=1)
    wide = space.buffer("wide", (6,), torch.bool, pad=False, mirrored=False, moved=False)
    with pytest.raises(ValueError, match="unmoved"):
        space.buffer("mirrored", (), torch.int32, moved=False)
    rows = space.propose([5, 6])
    d = rows.diff
    rows.write(lens, d.join_slots, np.array([1, 2], np.int32))
    rows.write(wide, d.join_slots, np.eye(2, 6, dtype=np.bool_))
    rows.stage()
    rows.commit()
    rows.apply()
    # the forward sets element 5 of every row it runs, padding rows into the trash slot
    wide.resident.view(-1).scatter_(0, space.scatter_slots.long() * 6 + 5, True)
    space.after_forward()
    with pytest.raises(RuntimeError, match="resident"):
        _ = wide.device
    held = space.batch.held()
    resident = wide.resident.cpu()
    gather = space.gather_slots.cpu().tolist()
    for i, rid in enumerate(space.batch.order):
        assert gather[i] == held[rid]
    for j, rid in enumerate(d.join_rids):
        assert resident[held[rid]].tolist() == [k in (j, 5) for k in range(6)]
    assert not resident[space.pad_slot].any()
    # a moved buffer's rows of the step are its dense view's
    assert lens.device.cpu()[:2].tolist() == [2, 3]


@pytest.mark.parametrize("device", DEVICES)
def test_a_lent_row_keeps_its_slot_and_place_and_resyncs_when_run_again(device):
    """A step outside the walk runs two of the rows: they are read back and
    lent, keeping their slots and places, and the next step of the walk
    writes them whole in place, moving nothing."""
    rng = random.Random(0)
    sim = _Sim(device)
    sim.step([1, 2, 3, 4], rng)
    slots = dict(sim.space.batch.held())
    sim.lend([2, 4])
    assert sim.space._lent == {2, 4} and dict(sim.space.batch.held()) == slots
    assert not sim.space.holds_live(2) and sim.space.holds(2)
    assert sim.space.committed(sim.lens, 2) is None
    sim.check()
    rows = sim.step([1, 2, 3, 4], rng)
    assert rows.diff.join_rids == [2, 4] and rows.diff.join_slots.tolist() == [slots[2], slots[4]]
    assert not len(rows._batch_diff.join_rids) and dict(sim.space.batch.held()) == slots
    assert [count for _, count, _ in rows._staged.phases] == [2 * len(sim.space.buffers)]
    assert not sim.space._lent
    sim.check()


@pytest.mark.parametrize("device", DEVICES)
def test_a_lent_row_that_leaves_reads_nothing_back(device):
    """The store, not the stale row, holds a lent request's state: an
    eviction or a later step without it leaves the store as the lend and
    the step outside left it."""
    rng = random.Random(0)
    sim = _Sim(device, capacity=MAX_ROWS)
    sim.step([1, 2, 3], rng)
    sim.lend([2])
    rows = sim.space.without([2])
    assert rows.diff.dropped == {2} and not len(rows._batch_diff.dropped)
    sim._read_back(rows)
    rows.stage()
    rows.commit(forward=False)
    rows.apply()
    assert not sim.space._lent and not sim.space.holds(2)
    sim.check()
    sim.step([2, 3], rng)
    sim.check()


@pytest.mark.parametrize("device", DEVICES)
def test_a_proposal_from_before_a_lend_is_stale(device):
    rng = random.Random(0)
    sim = _Sim(device)
    sim.step([1, 2], rng)
    ahead = sim.space.propose([1, 2])
    sim.lend([2])
    assert ahead.stale
    with pytest.raises(RuntimeError, match="lent"):
        ahead.commit()
    sim.step([1, 2], rng)
    sim.check()


@pytest.mark.parametrize("device", DEVICES)
def test_a_lend_of_lent_or_unheld_rows_is_nothing(device):
    rng = random.Random(0)
    sim = _Sim(device)
    sim.step([1, 2], rng)
    assert sim.lend([2]) is not None
    assert sim.space.lend([2, 9]) is None
