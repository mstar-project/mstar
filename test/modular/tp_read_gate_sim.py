"""Explicit-state model of a lockstep instance whose follower input read can fail.

Two requests, r0 and r1, each need one input before their step. Rank 0 (the
leader) reads both successfully; the follower reads r0, and its read of r1
either lands or fails. Every interleaving of {follower read outcome, follower
report, leader scheduling, follower execution, failure drain} is explored.

Modes:
  NONE  -- rank 0 schedules from its own reads alone (main and #419).
  ZERO  -- as NONE, but a follower whose read failed runs the step with zeros
           of the advertised shape (#628).
  GATE  -- rank 0 schedules a row only once the follower reported its input
           landed (``mstar/worker/read_gate.py``).

Input kinds:
  data  -- the values do not change the step's shape (``vision_embeds``).
  shape -- the step's shape is derived from the values (``image_grid_thw``):
           zeros give the follower a different step than the leader's.

Checked at every terminal state:
  live     -- no reachable state is stuck short of done (the instance hang).
  lockstep -- every step ran with the same shape on both ranks.
  isolated -- r0 completes; only r1 can fail.
"""
from __future__ import annotations

from dataclasses import dataclass, field

RIDS = ("r0", "r1")


@dataclass
class Result:
    states: int = 0
    violations: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.violations


def _initial():
    return (
        # follower read state per rid: pending | landed | failed
        (("r0", "pending"), ("r1", "pending")),
        frozenset(),          # rids the follower reported landed (to rank 0)
        frozenset(),          # rids rank 0 has scheduled
        (),                   # follower FIFO: tuple of batches (tuples of rids)
        frozenset(),          # rids whose step ran on both ranks
        False,                # conductor drained r1 (follower reported failure)
        False,                # a step ran with mismatched shapes
    )


def _successors(state, mode, kind):
    reads, reported, scheduled, fifo, done, drained, mismatch = state
    rd = dict(reads)
    # the follower's reads resolve: r0 always lands, r1 may land or fail
    for rid in RIDS:
        if rd[rid] == "pending":
            outcomes = ("landed",) if rid == "r0" else ("landed", "failed")
            for out in outcomes:
                nr = dict(rd)
                nr[rid] = out
                yield (tuple(sorted(nr.items())), reported, scheduled, fifo, done,
                       drained or out == "failed", mismatch)
    # the follower reports a landed read to rank 0
    for rid in RIDS:
        if rd[rid] == "landed" and rid not in reported:
            yield (reads, reported | {rid}, scheduled, fifo, done, drained, mismatch)
    # rank 0 schedules any non-empty set of its ready rows as one step
    ready = [
        r for r in RIDS
        if r not in scheduled and not (r == "r1" and drained)
        and (mode != "GATE" or r in reported)
    ]
    for mask in range(1, 1 << len(ready)):
        batch = tuple(r for i, r in enumerate(ready) if mask >> i & 1)
        yield (reads, reported, scheduled | set(batch), fifo + (batch,), done,
               drained, mismatch)
    # the follower runs its FIFO head together with rank 0 (the collective)
    if fifo:
        head = fifo[0]
        if all(rd[r] == "landed" for r in head):
            yield (reads, reported, scheduled, fifo[1:], done | set(head),
                   drained, mismatch)
        elif mode == "ZERO" and all(rd[r] != "pending" for r in head):
            # zeros for the failed input; r1's output is dropped
            yield (reads, reported, scheduled, fifo[1:],
                   done | {r for r in head if rd[r] == "landed"},
                   drained, mismatch or kind == "shape")


def _terminal_ok(state):
    reads, _reported, _scheduled, fifo, done, drained, _mismatch = state
    rd = dict(reads)
    if fifo:
        return False
    if "r0" not in done:
        return False
    # r1 either ran or failed and was drained
    return "r1" in done or (rd["r1"] == "failed" and drained)


def explore(mode: str, kind: str) -> Result:
    res = Result()
    seen = set()
    stack = [_initial()]
    while stack:
        state = stack.pop()
        if state in seen:
            continue
        seen.add(state)
        succ = list(_successors(state, mode, kind))
        if state[6]:
            res.violations.append(("lockstep", repr(state)))
        if not succ and not _terminal_ok(state):
            res.violations.append(("live", repr(state)))
        stack.extend(succ)
    res.states = len(seen)
    return res
