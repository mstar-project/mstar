"""The peak planner and the safe-state test, against brute force.

``PeakPlanner`` answers "what is the most pages this set can hold at any round"
from a cached structure; here every answer is checked against a simulation that
adds the footprints round by round. ``banker_safe`` is checked against trying
every order. The EASY rule is checked for the one thing it promises: a request
it lets past the head never moves the head's start later.
"""

from __future__ import annotations

import itertools
import random
import sys

sys.path.insert(0, ".")

import pytest

from mstar.engine.resources.kv.peak_plan import (
    PeakPlanner,
    PlanEntry,
    banker_safe,
    easy_allows,
    footprint,
    shadow_sum,
)

SEED = 20261006
TRIALS = 400


def _entries(rng: random.Random, max_n: int = 7) -> tuple[list[PlanEntry], int]:
    page_size = rng.choice([1, 2, 4, 8, 16])
    entries = []
    for _ in range(rng.randrange(0, max_n + 1)):
        held = rng.randrange(0, 7)
        entries.append(PlanEntry(
            held=held,
            claim=held + rng.randrange(0, 9) - (1 if rng.random() < 0.05 else 0),
            now=rng.randrange(0, 5),
            growth=rng.randrange(0, 4),
            rounds=0 if rng.random() < 0.15 else rng.randrange(1, 60),
        ))
    return entries, page_size


def _candidate(rng: random.Random) -> PlanEntry:
    held = rng.randrange(0, 3)
    return PlanEntry(
        held=held, claim=held + rng.randrange(0, 12), now=rng.randrange(0, 6),
        growth=rng.randrange(0, 4), rounds=rng.randrange(1, 70),
    )


def _horizon(entries, *more: int) -> int:
    return max([e.rounds for e in entries] + list(more) + [1]) + 4


def _brute_peak(entries: list[PlanEntry], page_size: int) -> int:
    return max(
        sum(footprint(e, t, page_size) for e in entries)
        for t in range(_horizon(entries))
    )


def _brute_peak_from(
    entries: list[PlanEntry], entry: PlanEntry, start: int, page_size: int,
) -> int:
    rounds = max(1, entry.rounds)
    entry = entry._replace(rounds=rounds)
    return max(
        sum(footprint(e, u, page_size) for e in entries)
        + footprint(entry, u - start, page_size)
        for u in range(start, _horizon(entries, start + rounds))
    )


def _brute_shadow(entries, head, capacity, page_size):
    starts = [0] + sorted({e.rounds for e in entries if e.rounds > 0})
    for start in starts:
        peak = _brute_peak_from(entries, head, start, page_size)
        if peak <= capacity:
            return start, capacity - peak
    return None


# ── the planner ─────────────────────────────────────────────────────────


def test_the_peak_is_what_adding_the_footprints_round_by_round_gives():
    rng = random.Random(SEED)
    for trial in range(TRIALS):
        entries, page_size = _entries(rng)
        planner = PeakPlanner(entries, capacity=0, page_size=page_size)
        assert planner.peak() == _brute_peak(entries, page_size), (
            f"trial {trial}: {entries} page_size={page_size}"
        )


def test_a_request_added_to_the_set_raises_the_peak_as_a_simulation_would():
    rng = random.Random(SEED + 1)
    for trial in range(TRIALS):
        entries, page_size = _entries(rng)
        cand = _candidate(rng)
        planner = PeakPlanner(entries, capacity=0, page_size=page_size)
        assert planner.peak_from(cand) == _brute_peak_from(entries, cand, 0, page_size), (
            f"trial {trial}: {entries} + {cand} page_size={page_size}"
        )


def test_a_request_started_later_ages_the_set_by_that_much():
    rng = random.Random(SEED + 2)
    for trial in range(TRIALS):
        entries, page_size = _entries(rng)
        cand = _candidate(rng)
        start = rng.randrange(0, 70)
        planner = PeakPlanner(entries, capacity=0, page_size=page_size)
        assert planner.peak_from(cand, start) == _brute_peak_from(
            entries, cand, start, page_size
        ), f"trial {trial}: start {start}: {entries} + {cand} page_size={page_size}"


def test_fits_is_the_peak_against_the_capacity():
    entries = [PlanEntry(held=2, claim=8, now=0, growth=1, rounds=96)]
    cand = PlanEntry(held=0, claim=4, now=2, growth=1, rounds=32)
    planner = PeakPlanner(entries, capacity=10, page_size=16)

    assert planner.peak_from(cand) == 8
    assert planner.fits(cand)
    assert not planner.fits(cand, capacity=7)


def test_a_set_that_sums_past_the_pool_can_peak_inside_it():
    """A holds 2, grows to 8 by round 6; B holds 2, grows to 4 and ends at round 2.
    The reservations sum to 12 on a pool of 10, and the peak is 8."""
    page_size = 1
    a = PlanEntry(held=2, claim=8, now=0, growth=1, rounds=7)
    b = PlanEntry(held=2, claim=4, now=0, growth=1, rounds=3)

    planner = PeakPlanner([a], capacity=10, page_size=page_size)

    assert a.claim + b.claim == 12
    assert planner.peak_from(b) == 8
    assert planner.fits(b)


def test_a_finished_request_holds_its_pages_until_it_is_removed():
    done = PlanEntry(held=5, claim=9, now=0, growth=2, rounds=0)
    cand = PlanEntry(held=0, claim=6, now=1, growth=1, rounds=3)

    planner = PeakPlanner([done], capacity=10, page_size=4)

    assert planner.peak() == 5, "a finished request grew"
    assert planner.peak_from(cand) == 5 + 2, "a finished request was counted as gone"


def test_an_empty_set_peaks_at_the_candidate_alone():
    planner = PeakPlanner([], capacity=10, page_size=16)
    cand = PlanEntry(held=0, claim=6, now=2, growth=1, rounds=100)

    assert planner.peak() == 0
    assert planner.peak_from(cand) == footprint(cand, 99, 16) == 6


# ── the safe-state test ─────────────────────────────────────────────────


def _brute_safe(free: int, held: list[int], need: list[int]) -> bool:
    if free < 0:
        return False
    for order in itertools.permutations(range(len(need))):
        avail = free
        for i in order:
            if need[i] > avail:
                break
            avail += held[i]
        else:
            return True
    return not need


def test_the_safe_state_test_agrees_with_trying_every_order():
    rng = random.Random(SEED + 3)
    verdicts = set()
    for trial in range(2000):
        n = rng.randrange(0, 7)
        held = [rng.randrange(0, 6) for _ in range(n)]
        need = [rng.randrange(0, 9) for _ in range(n)]
        free = rng.randrange(-1, 8)
        expect = _brute_safe(free, held, need)
        verdicts.add(expect)
        assert banker_safe(free, held, need) == expect, (
            f"trial {trial}: free={free} held={held} need={need}"
        )
    assert verdicts == {True, False}, "the cases never covered both answers"


def test_the_deadlock_counterexample_is_unsafe_once_a_and_b_have_grown():
    # pool of 20: A and B hold 8 each and need 2 more, C holds 4 and needs 2
    assert not banker_safe(0, held=[8, 8, 4], need=[2, 2, 2])
    # the same state with C advanced first is safe: C finishes and frees 6
    assert banker_safe(2, held=[8, 8, 2], need=[2, 2, 0])


# ── EASY ────────────────────────────────────────────────────────────────


def test_the_shadow_is_the_first_start_that_passes_the_peak_test():
    rng = random.Random(SEED + 4)
    answers = set()
    for trial in range(TRIALS):
        entries, page_size = _entries(rng)
        head = _candidate(rng)
        capacity = rng.randrange(6, 40)
        planner = PeakPlanner(entries, capacity, page_size)
        expect = _brute_shadow(entries, head, capacity, page_size)
        answers.add(
            None if expect is None else "now" if expect[0] == 0 else "later"
        )
        assert planner.shadow(head) == expect, (
            f"trial {trial}: capacity {capacity}: {entries} head {head} page_size={page_size}"
        )
    assert answers == {None, "now", "later"}, f"the fixtures only ever gave {answers}"


def test_a_request_the_easy_rule_lets_pass_never_moves_the_head_later():
    rng = random.Random(SEED + 5)
    allowed = refused = 0
    for trial in range(2 * TRIALS):
        entries, page_size = _entries(rng)
        head = _candidate(rng)
        capacity = rng.randrange(8, 40)
        planner = PeakPlanner(entries, capacity, page_size)
        shadow = planner.shadow(head)
        if shadow is None:
            continue
        cand = _candidate(rng)
        if not planner.fits(cand):
            continue
        rounds = max(1, cand.rounds)
        cost = footprint(cand, rounds - 1, page_size)
        if not easy_allows(rounds, cost, shadow):
            refused += 1
            continue
        allowed += 1
        after = _brute_shadow(entries + [cand], head, capacity, page_size)
        assert after is not None and after[0] <= shadow[0], (
            f"trial {trial}: {cand} moved the head from {shadow[0]} to {after}: "
            f"{entries} head {head} capacity {capacity} page_size={page_size}"
        )
    assert allowed > 30 and refused > 30, (allowed, refused)


def test_a_request_the_rule_refuses_can_delay_the_head():
    # one pool of 10: A holds 6 for 50 rounds. The head needs 8, so it starts when A is gone.
    page_size = 16
    a = PlanEntry(held=6, claim=6, now=0, growth=0, rounds=50)
    head = PlanEntry(held=0, claim=8, now=8, growth=0, rounds=20)
    planner = PeakPlanner([a], capacity=10, page_size=page_size)
    shadow = planner.shadow(head)
    assert shadow == (50, 2)

    # C takes 3 pages for 80 rounds: it is still there at 50 and leaves the head 7
    long = PlanEntry(held=0, claim=3, now=3, growth=0, rounds=80)
    short = PlanEntry(held=0, claim=3, now=3, growth=0, rounds=40)
    small = PlanEntry(held=0, claim=2, now=2, growth=0, rounds=80)

    assert planner.fits(long) and planner.fits(short)
    assert not easy_allows(80, 3, shadow), "a request still running at the head's start passed"
    assert _brute_shadow([a, long], head, 10, page_size)[0] > 50
    assert easy_allows(40, 3, shadow), "a request gone before the head's start was refused"
    assert _brute_shadow([a, short], head, 10, page_size)[0] == 50
    assert easy_allows(80, 2, shadow), "a request that fits beside the head was refused"
    assert _brute_shadow([a, small], head, 10, page_size)[0] == 50


def test_with_no_start_in_sight_there_is_nothing_to_protect():
    assert easy_allows(1000, 1000, None)


# ── the summed test's shadow ────────────────────────────────────────────


def _brute_shadow_sum(entries, supply, head_need):
    for start in [0] + sorted({e.rounds for e in entries if e.rounds > 0}):
        owed = sum(e.need for e in entries if not 0 < e.rounds <= start)
        free = supply + sum(e.held for e in entries if 0 < e.rounds <= start)
        if owed + head_need <= free:
            return start, free - owed - head_need
    return None


def test_the_summed_shadow_waits_for_the_releases_that_make_room():
    rng = random.Random(SEED + 6)
    seen = set()
    for trial in range(TRIALS):
        entries, _ = _entries(rng)
        supply = rng.randrange(0, 30)
        head_need = rng.randrange(0, 20)
        expect = _brute_shadow_sum(entries, supply, head_need)
        seen.add(None if expect is None else expect[0] > 0)
        assert shadow_sum(entries, supply, head_need) == expect, (
            f"trial {trial}: {entries} supply={supply} head_need={head_need}"
        )
    assert seen == {None, True, False}


@pytest.mark.parametrize("rounds", [0, 1, 5])
def test_a_planner_for_nothing_but_a_candidate_is_its_own_footprint(rounds):
    cand = PlanEntry(held=0, claim=5, now=3, growth=1, rounds=rounds)
    planner = PeakPlanner([], capacity=5, page_size=2)

    # a candidate always runs a round, so rounds=0 is read as 1
    assert planner.peak_from(cand) == footprint(cand._replace(rounds=max(1, rounds)), max(1, rounds) - 1, 2)
