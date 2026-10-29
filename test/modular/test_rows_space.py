"""Persistent rows over many steps: after each step's diff and replay, every
active row holds its request's state, every padding row its pad value, and
each mirror equals its device rows."""
import random

import numpy as np
import pytest
import torch

from mstar.engine.rows import RowSpace

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
MAX_ROWS, PAGES, STATE = 8, 4, 3


class _Sim:
    """A resource's view of the rows: a length and page table with host truth
    per request, and device-only state checked out of a per-request store."""

    def __init__(self, device):
        self.space = RowSpace(MAX_ROWS, device, depth=2, block_bytes=256)
        self.lens = self.space.buffer("lens", (), torch.int32, pad=0)
        self.table = self.space.buffer("table", (PAGES,), torch.int32, pad=-1)
        self.state = self.space.buffer("state", (STATE,), torch.float32, pad=0.0, mirrored=False)
        self.store = torch.zeros(64, STATE, dtype=torch.float32, device=device)
        self.truth_len: dict[int, int] = {}
        self.truth_table: dict[int, list[int]] = {}
        self.truth_state: dict[int, np.ndarray] = {}

    def step(self, rids, rng, commit=True):
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
        staying = [s for s, rid in enumerate(d.order) if rid not in d.join_rids]
        appended = [s for s in staying if rng.random() < 0.3]
        if appended:
            cols = np.array([rng.randrange(PAGES) for _ in appended])
            vals = np.array([rng.randrange(100) for _ in appended], np.int32)
            rows.write_at(self.table, np.array(appended), (cols,), vals)
        rows.stage()
        if not commit:
            return rows
        if appended:
            for s, c, v in zip(appended, cols, vals, strict=True):
                self.truth_table[d.order[s]][c] = int(v)
        rows.apply()
        n = d.n_active
        # the replay: rows advance, padding does not
        active = torch.arange(MAX_ROWS, device=self.lens.device.device) < self.space.n_active
        self.lens.device.add_(active.to(torch.int32))
        self.state.device[:n] += 1
        rows.advance(self.lens, 1)
        rows.commit()
        for rid in d.order:
            self.truth_len[rid] += 1
            self.truth_state[rid] += 1
        return rows

    def check(self):
        order = list(self.space.batch.order)
        n = len(order)
        lens = self.lens.device.cpu().numpy()
        table = self.table.device.cpu().numpy()
        state = self.state.device.cpu().numpy()
        assert int(self.space.n_active.item()) == n
        for slot, rid in enumerate(order):
            assert lens[slot] == self.truth_len[rid]
            assert table[slot].tolist() == self.truth_table[rid]
            np.testing.assert_array_equal(state[slot], self.truth_state[rid])
        assert (lens[n:] == 0).all() and (table[n:] == -1).all() and (state[n:] == 0).all()
        np.testing.assert_array_equal(self.lens.host, lens)
        np.testing.assert_array_equal(self.table.host, table)
        # a request that left has its device state back in the store
        store = self.store.cpu().numpy()
        for rid, truth in self.truth_state.items():
            if rid not in order:
                np.testing.assert_array_equal(store[rid], truth)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("seed", range(6))
def test_random_steps_keep_rows_and_mirrors_exact(device, seed):
    rng = random.Random(seed)
    sim = _Sim(device)
    live, next_rid = [], 0
    for _ in range(120):
        keep = [rid for rid in live if rng.random() > 0.2]
        rng.shuffle(keep)
        while len(keep) < MAX_ROWS and rng.random() < 0.35:
            # a returning request rejoins from its store and truth
            gone = [r for r in sim.truth_len if r not in keep and r not in live]
            if gone and rng.random() < 0.3:
                keep.append(rng.choice(gone))
            elif next_rid < 64:
                keep.append(next_rid)
                next_rid += 1
            else:
                break
        sim.step(keep, rng)
        sim.check()
        live = list(sim.space.batch.order)


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
    rows = sim.space.propose([1, 2])
    assert rows.mirror(sim.lens) is sim.lens.host
    rows.stage()
    assert rows._staged is None


def test_a_join_left_unwritten_is_refused():
    space = RowSpace(4, "cpu")
    lens = space.buffer("lens", (), torch.int32)
    space.buffer("other", (), torch.int32)
    rows = space.propose([7])
    rows.write(lens, rows.diff.join_slots, np.array([3], np.int32))
    with pytest.raises(RuntimeError, match="other"):
        rows.stage()
