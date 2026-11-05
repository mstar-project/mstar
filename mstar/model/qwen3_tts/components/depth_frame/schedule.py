"""Static task lists for the frame kernel.

A phase is one op of one layer at one depth position; its tasks are tiles (or
rows, for the embedding step). Tasks are dealt round-robin over the CTAs in
phase order with a running counter, so the CTAs a narrow phase leaves idle
differ from phase to phase. Each CTA then places its weight tiles in a FIFO
ring of 4 KB shared-memory pages, each TMA copy in contiguous pages, and
records, per task, how many later tasks to issue before its wait: a tile is
issued as soon as the pages it needs are free (every earlier task that held
them has finished) and at most TB weight tiles are in flight, one mbarrier
each.

Counters: 8 per phase. QKV tiles signal the counter of their kv head (q heads
2g and 2g+1, k head g and v head g are 16 tiles), so an ATTEND task waits only
on its own 16; every other phase signals its first counter.

O and DOWN are split phases: two tasks per tile, dealt to the two CTAs of a
cluster. Task 2t + r is rank r of tile t and streams the r-th half of K. Rank 0
adds rank 1's partial and signals the phase counter; rank 1 signals a counter
nobody waits on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NPAGES, TB = 32, 8  # the weight ring's 4 KB pages, and the copies in flight (one mbarrier each)
DESC = 12  # type, layer, tile, wait_ctr, wait_target, signal_ctr, page, n_issue, pos, n_part, wslot, -
T_QKV, T_O, T_GATEUP, T_DOWN, T_ATTEND, T_INPROJ, T_HEAD, T_EMBED = range(8)


@dataclass(frozen=True)
class Op:
    name: str
    type: int
    tasks: int        # tasks in the phase
    pages: int        # weight pages per task (0: none)
    align: int        # page alignment: one TMA copy's pages, so no copy straddles the ring's end
    split: bool = False

    @property
    def signals(self) -> int:
        return self.tasks // 2 if self.split else self.tasks


QKV = Op("qkv", T_QKV, 128, 16, 16)
ATTEND = Op("attend", T_ATTEND, 128, 0, 1)  # one q head, 8 rows
GATEUP = Op("gateup", T_GATEUP, 96, 32, 16)
INPROJ = Op("inproj", T_INPROJ, 64, 16, 8)
HEAD = Op("head", T_HEAD, 128, 8, 8)
EMBED = Op("embed", T_EMBED, 64, 0, 1)
O = Op("o", T_O, 128, 8, 8, True)           # 8 of the tile's 16 chunks, one copy
DOWN = Op("down", T_DOWN, 128, 12, 4, True)  # 12 of 24 chunks, three copies of 4
MIN_GRID = max(O.tasks, DOWN.tasks)


def qkv_group(t: int) -> int:
    """The kv head whose attention needs 32-column qkv tile t (q, then k, then v columns)."""
    return t // 8 if t < 64 else (t - 64) // 4 if t < 96 else (t - 96) // 4


@dataclass(frozen=True)
class Phase:
    op: Op
    layer: int
    pos: int
    n_part: int


def frame_phases(layers: int = 5, positions: int = 16):
    ph = []
    for p in range(positions):
        if p == 0:
            ph.append(Phase(INPROJ, 0, 0, 0))
        for l in range(layers):
            ph.append(Phase(QKV, l, p, 64 if (l > 0 or p == 0) else 4))
            ph += [Phase(ATTEND, l, p, 0), Phase(O, l, p, 0), Phase(GATEUP, l, p, 64), Phase(DOWN, l, p, 0)]
        if p > 0:
            ph.append(Phase(HEAD, 0, p, 64))
        ph.append(Phase(EMBED, 0, p, 0))
    return ph


def build(phases, grid: int, npages: int, tb: int, lookahead: int = 1 << 30):
    """Returns (tasks[grid, maxt, DESC] int32, ntasks[grid] int32, n_counters)."""
    zero = len(phases) * 8  # a counter nobody signals: the first phase waits on it with target 0
    sink = zero + 1         # signalled by split phases' rank 1 tasks, never waited on
    assert grid % 2 == 0
    assert all(ph.op.tasks <= grid for ph in phases if ph.op.split), (
        f"on {grid} CTAs a cluster gets two tiles of a split phase, and rank 1's second partial can overwrite the "
        f"first in rank 0's combine buffer before rank 0 reads it. The frame needs {MIN_GRID} CTAs"
    )
    lists = [[] for _ in range(grid)]
    gt = 0
    for i, ph in enumerate(phases):
        prev = phases[i - 1] if i > 0 else None
        if ph.op.split:
            gt += gt % 2  # task 2t + r on CTA 2c + r: a cluster's two CTAs
        for t in range(ph.op.tasks):
            if prev is None:
                wc, wt = zero, 0
            elif ph.op.type == T_ATTEND:
                assert prev.op.type == T_QKV
                wc, wt = (i - 1) * 8 + (t // 8) // 2, 16
            else:
                wc, wt = (i - 1) * 8, prev.op.signals
            nxt = phases[i + 1] if i + 1 < len(phases) else None
            grouped = ph.op.type == T_QKV and nxt is not None and nxt.op.type == T_ATTEND
            sc = i * 8 + (qkv_group(t) if grouped else 0)
            if ph.op.split and t % 2:
                sc = sink
            d = [ph.op.type, ph.layer, t, wc, wt, sc, 0, 0, ph.pos, ph.n_part, 0, 0]
            lists[gt % grid].append((d, ph.op.pages, ph.op.align))
            gt += 1
    maxt = max(len(l) for l in lists)
    tasks = np.zeros((grid, maxt, DESC), np.int32)
    ntasks = np.zeros(grid, np.int32)
    for c, lst in enumerate(lists):
        place(lst, npages, tb, lookahead)
        ntasks[c] = len(lst)
        for i, (d, _, _) in enumerate(lst):
            tasks[c, i] = d
    return tasks, ntasks, zero + 8


def place(lst, npages: int, tb: int, lookahead: int = 1 << 30):
    """Assign each weight tile its pages and mbarrier slot (d[6], d[10]: its
    ordinal among the CTA's weight tiles), and each task the number of later
    tasks issued just before its wait (d[7]; issue order = list order)."""
    assert npages % 2 == 0
    owner = [-1] * npages      # task holding each page
    head = 0
    issue_at = []
    wtasks = []                # list indices of weight-carrying tasks
    for j, (d, need, align) in enumerate(lst):
        floor = issue_at[-1] if issue_at else 0
        if need == 0:
            issue_at.append(min(max(floor, 0), j))
            continue
        assert need <= npages
        assert npages % align == 0 and need % align == 0
        head += (-head) % align
        pages = [(head + k) % npages for k in range(need)]
        d[6] = head % npages
        head = (head + need) % npages
        d[10] = len(wtasks)
        busy = [owner[p] for p in pages if owner[p] >= 0]
        at = max(busy) + 1 if busy else 0
        if len(wtasks) >= tb:
            at = max(at, wtasks[-tb] + 1)   # its barrier's previous user has been waited on
        at = max(at, floor, j - lookahead)
        while issue_at.count(at) >= 2:  # a packed row carries two issue entries
            at += 1
        assert at <= j, (j, at)
        issue_at.append(at)
        wtasks.append(j)
        for p in pages:
            owner[p] = j
    counts = [0] * len(lst)
    for at in issue_at:
        counts[at] += 1
    for i, (d, _, _) in enumerate(lst):
        d[7] = counts[i]


def weight_row(ty: int, layer: int, tile: int, pos: int) -> int:
    """First row of a task's weight tile in its stacked weight matrix."""
    return {T_QKV: layer * 4096 + tile * 32, T_O: layer * 1024 + tile * 16, T_GATEUP: (layer * 96 + tile) * 64,
            T_DOWN: layer * 1024 + tile * 16, T_INPROJ: tile * 16, T_HEAD: (pos - 1) * 2048 + tile * 16}[ty]


ROW = 8


def pack(tasks, ntasks, tb: int):
    """Packed task rows, 8 int32 each, for the kernel's task loop:
    0: type | layer << 4 | tile << 8 | pos << 16 | weight tiles issued here << 20 | n_part << 24
    1: wait counter | wait target << 16
    2: signal counter | first page << 16 | (wslot mod 2 TB) << 24
    3, 4: the weight tiles issued before this task's wait, type | page << 4 | barrier << 10 | rank << 13 | row << 16.
    An O or DOWN task's tile field is 2 * tile + rank."""
    assert tb == 8
    grid, maxt, _ = tasks.shape
    out = np.zeros((grid, maxt, ROW), np.int64)
    for c in range(grid):
        n = int(ntasks[c])
        at = []
        for i in range(n):
            at += [i] * int(tasks[c, i, 7])
        issued = [[] for _ in range(n)]
        for j in range(n):
            d = tasks[c, j]
            if d[0] in (T_ATTEND, T_EMBED):
                continue
            halves = d[0] in (T_O, T_DOWN)
            tile, rank = (int(d[2]) // 2, int(d[2]) % 2) if halves else (int(d[2]), 0)
            row = weight_row(int(d[0]), int(d[1]), tile, int(d[8]))
            assert row < 1 << 15 and d[6] < 64, row
            issued[at[j]].append(int(d[0]) | int(d[6]) << 4 | (int(d[10]) % tb) << 10 | rank << 13 | row << 16)
        for i in range(n):
            d = tasks[c, i]
            assert len(issued[i]) <= 2 and d[1] < 16 and d[2] < 256 and d[8] < 16 and d[9] < 128
            assert d[3] < 1 << 16 and d[4] < 1 << 15 and d[5] < 1 << 16 and d[6] < 256
            out[c, i, 0] = (int(d[0]) | int(d[1]) << 4 | int(d[2]) << 8 | int(d[8]) << 16 | len(issued[i]) << 20
                            | int(d[9]) << 24)
            out[c, i, 1] = int(d[3]) | int(d[4]) << 16
            out[c, i, 2] = int(d[5]) | int(d[6]) << 16 | (int(d[10]) % (2 * tb)) << 24
            for k, e in enumerate(issued[i]):
                out[c, i, 3 + k] = e
    assert out.max() < 1 << 31
    return out.astype(np.int32)
