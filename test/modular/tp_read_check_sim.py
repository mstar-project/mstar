"""Explicit-state model of a lockstep instance whose follower input read can fail.

Two requests, r0 and r1, each need one input before their step. Rank 0 (the
leader) reads both successfully; the follower reads r0, and its read of r1
either lands or fails. Every interleaving of {follower read outcome, leader
scheduling, follower execution, failure drain} is explored.

Modes:
  NONE    -- rank 0 schedules from its own reads alone (main).
  ZERO    -- as NONE, but a follower whose read failed runs the row's
             ``prepare_inputs`` on zeros of the advertised shape (#628).
  CAPTURE -- as NONE, but a follower whose read failed puts the walk's
             capture input at rank 0's prepared length in the row's place,
             without running ``prepare_inputs`` on it; the row is dropped
             after the forward.
  CHECK   -- a follower builds rank 0's step once every row's read landed or
             failed, leaving out the rows it could not read; before preparing
             the step every rank agrees on those rows (one all-reduce) and
             takes them out (``Worker._drop_rows_whose_read_failed``).

Input kinds:
  data    -- the values do not change the step (``vision_embeds``).
  shape   -- the step's length is derived from the values (``image_grid_thw``):
             zeros give the follower a different step than the leader's.
  prepare -- ``prepare_inputs`` issues a collective for the row (a text part's
             vocab-parallel embedding in Qwen3.5's ``prefill_vision``), so a
             rank that puts another row in its place skips that collective.

Checked at every terminal state:
  live     -- no reachable state is stuck short of done (the instance hang).
  lockstep -- every step ran the same collectives on both ranks.
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
        frozenset(),          # rids rank 0 has scheduled
        (),                   # follower FIFO: tuple of batches (tuples of rids)
        frozenset(),          # rids whose step ran on both ranks
        False,                # conductor drained r1 (follower reported failure)
        False,                # a step ran different collectives on the two ranks
    )


def _step_mismatch(mode, kind, head, rd):
    """Whether the follower's step differs from rank 0's for ``head``, given
    which of its rows the follower could not read."""
    failed = [r for r in head if rd[r] == "failed"]
    if not failed:
        return False
    if mode == "ZERO":
        # prepare runs on zeros: its collectives match, its length may not
        return kind == "shape"
    if mode == "CAPTURE":
        # rank 0's length, but rank 0's prepare ran collectives the follower's didn't
        return kind == "prepare"
    # CHECK: every rank takes the row out before preparing
    return False


def _successors(state, mode, kind):
    reads, scheduled, fifo, done, drained, mismatch = state
    rd = dict(reads)
    # the follower's reads resolve: r0 always lands, r1 may land or fail
    for rid in RIDS:
        if rd[rid] == "pending":
            outcomes = ("landed",) if rid == "r0" else ("landed", "failed")
            for out in outcomes:
                nr = dict(rd)
                nr[rid] = out
                yield (tuple(sorted(nr.items())), scheduled, fifo, done,
                       drained or out == "failed", mismatch)
    # rank 0 schedules any non-empty set of its ready rows as one step
    ready = [r for r in RIDS if r not in scheduled and not (r == "r1" and drained)]
    for mask in range(1, 1 << len(ready)):
        batch = tuple(r for i, r in enumerate(ready) if mask >> i & 1)
        yield (reads, scheduled | set(batch), fifo + (batch,), done, drained, mismatch)
    # the follower runs its FIFO head together with rank 0 (the collective)
    if fifo:
        head = fifo[0]
        if all(rd[r] == "landed" for r in head):
            yield (reads, scheduled, fifo[1:], done | set(head), drained, mismatch)
        elif mode != "NONE" and all(rd[r] != "pending" for r in head):
            # the failed row is dropped (before the forward for CHECK, after it
            # for the others); the rest of the step runs
            yield (reads, scheduled, fifo[1:],
                   done | {r for r in head if rd[r] == "landed"},
                   drained, mismatch or _step_mismatch(mode, kind, head, rd))


def _terminal_ok(state):
    reads, _scheduled, fifo, done, drained, _mismatch = state
    rd = dict(reads)
    if fifo or "r0" not in done:
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
        if state[5]:
            res.violations.append(("lockstep", repr(state)))
        if not succ and not _terminal_ok(state):
            res.violations.append(("live", repr(state)))
        stack.extend(succ)
    res.states = len(seen)
    return res
