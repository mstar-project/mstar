"""Scheduling policies: the order of ready (node, walk) keys, and when a speculation chain yields."""

import pytest
from test_scheduler_rebuild import DECODE, MIXED, NODE, PREFILL, _Engine, _Runtime, _State

from mstar.engine.resources.step import FULL_ADMIT_NOT_READY, FULL_ADMIT_OK
from mstar.testing_utils.worker import make_scheduler
from mstar.worker.rebuild import Rebuilds
from mstar.worker.scheduling_policy import Priority, RoundRobin
from mstar.worker.services import walk_priorities_of

P, D, VAE = (NODE, PREFILL), (NODE, DECODE), ("VAEDecoder", "vae_chunk")


class _Peek:
    """Answers each peek with ``answer(exclude, among)`` and records it."""

    def __init__(self, answer=lambda exclude, among: True):
        self.answer, self.asked = answer, []

    def has_ready_excluding(self, exclude_target, among=None):
        self.asked.append((exclude_target, None if among is None else frozenset(among)))
        return self.answer(exclude_target, among)


def test_round_robin_orders_least_recently_scheduled_first_ties_in_scan_order():
    ready = dict.fromkeys([D, P, VAE], [])
    assert RoundRobin().order(ready, {D: 7, P: 3}) == [VAE, P, D]
    assert RoundRobin().order(ready, {}) == [D, P, VAE]


def test_round_robin_keeps_a_chains_first_step_and_asks_nothing_for_it():
    peek = _Peek()
    assert RoundRobin().chain_yields(D, 0, False, peek) is False and peek.asked == []
    assert RoundRobin().chain_yields(D, 1, False, peek) is True and peek.asked == [(D, None)]
    assert RoundRobin().chain_yields(P, 0, True, _Peek()) is True  # a prompt cut into chunks


def test_priority_orders_higher_first_and_round_robin_among_equals():
    ready = dict.fromkeys([D, VAE, P], [])
    assert Priority({P: 1}).order(ready, {P: 9, D: 5, VAE: 2}) == [P, VAE, D]


def test_priority_asks_only_about_the_walks_above_a_chain_at_its_first_step():
    policy = Priority({P: 1})
    peek = _Peek()
    assert policy.chain_yields(D, 0, False, peek) is True and peek.asked == [(D, frozenset({P}))]
    nothing_above = _Peek(lambda exclude, among: False)
    assert policy.chain_yields(D, 0, False, nothing_above) is False
    # the most urgent chain has nothing above it: round robin, which keeps its first step unasked
    peek = _Peek()
    assert policy.chain_yields(P, 0, False, peek) is False and peek.asked == []


def test_priority_makes_one_peek_a_pass_after_the_first_step():
    peek = _Peek(lambda exclude, among: False)
    assert Priority({P: 1}).chain_yields(D, 1, False, peek) is False
    assert peek.asked == [(D, None)], "a pass would scan twice"


def test_the_scheduler_takes_its_policy_from_the_models_priorities():
    assert type(make_scheduler().policy) is RoundRobin
    assert type(make_scheduler(walk_priorities={D: 0}).policy) is RoundRobin
    policy = make_scheduler(walk_priorities={P: 1}).policy
    assert isinstance(policy, Priority) and (policy.priority(P), policy.priority(D)) == (1, 0)


def test_a_combined_walks_key_takes_its_most_urgent_walks_priority():
    combined = {P: MIXED, D: MIXED}
    policy = make_scheduler(combined_walk_of=combined, walk_priorities={P: 2}).policy
    assert (policy.priority((NODE, MIXED)), policy.priority(P), policy.priority(D)) == (2, 2, 0)


class _ExcludingRuntime(_Runtime):
    """Leaves out ``exclude_target`` as the graph runtime does: a real walk, or a
    combined one covering each of its walks."""

    combined_walk_of: dict = {}

    def get_ready_nodes(self, exclude_rids, target=None, exclude_target=None):
        def excluded(walk):
            return exclude_target in ((NODE, walk), (NODE, self.combined_walk_of.get((NODE, walk))))

        return [spec for spec in super().get_ready_nodes(exclude_rids, target) if not excluded(spec.graph_walk)]


def _priority_scheduler(walks, *, combined=False, budget=None, refuse=(), mixes=None):
    engine = _Engine(budget=budget)
    engine.check_ready = lambda node, rid, info, allow_reload=True: (
        FULL_ADMIT_NOT_READY if rid in refuse else FULL_ADMIT_OK
    )
    runtime = _ExcludingRuntime(walks)
    runtime.combined_walk_of = {P: MIXED, D: MIXED} if combined else {}
    runtime.has_ready_excluding = lambda exclude, exclude_target: True
    sched = make_scheduler(
        type("Engines", (), {"get_engine": staticmethod(lambda name: engine)})(),
        parallel_leader_nodes={NODE},
        combined_walk_of={P: MIXED, D: MIXED} if combined else None,
        graph_runtime=runtime, request_state=_State(runtime), rebuilds=Rebuilds({}, {}),
        walk_priorities={P: 1}, speculation_mixes_walks=None if mixes is None else {NODE: mixes},
    )
    return sched, runtime


def test_a_decode_chain_yields_at_once_to_a_prompt_the_engine_would_take():
    sched, runtime = _priority_scheduler({"p0": PREFILL, "d0": DECODE})
    runtime.ready = {"p0", "d0"}
    assert sched.chain_yields(D, 0, False)


def test_a_prompt_the_engine_cannot_take_yet_does_not_stop_decode_speculating():
    """A prompt waiting for capacity that decode rows free as they finish: yielding
    to it would run every decode step serially for as long as it waits."""
    sched, runtime = _priority_scheduler({"p0": PREFILL, "d0": DECODE}, refuse={"p0"})
    runtime.ready = {"p0", "d0"}
    assert not sched.chain_yields(D, 0, False)


@pytest.mark.parametrize("mixes", [True, False])
def test_a_mixing_chain_keeps_the_prompts_it_merges(mixes):
    """Under a combined walk whose chain mixes, the decode chain's own next step
    takes the prompt; yielding would only lose that step's overlap."""
    sched, runtime = _priority_scheduler({"p0": PREFILL, "d0": DECODE}, combined=True, budget=64, mixes=mixes)
    runtime.ready = {"p0", "d0"}
    assert sched.chain_yields(D, 0, False) is not mixes


class _Section:
    def __init__(self, nodes):
        self.nodes = nodes

    def get_nodes(self):
        return self.nodes


def test_a_priority_for_a_pair_no_walk_runs_refuses_the_start():
    walks = {PREFILL: _Section({NODE}), DECODE: _Section({NODE})}

    class _Model:
        def __init__(self, priorities):
            self.priorities = priorities

        def get_walk_priorities(self):
            return self.priorities

    assert walk_priorities_of(_Model({NODE: {PREFILL: 1}}), walks) == {P: 1}
    for typo in ({NODE: {"prefil": 1}}, {"Talkr": {PREFILL: 1}}):
        with pytest.raises(ValueError, match="does not run"):
            walk_priorities_of(_Model(typo), walks)


def test_at_its_first_step_a_chain_keeps_its_step_from_walks_not_above_it():
    """Round robin's rule for walks of its own priority or below: the chain has
    its first step before they take a turn."""
    sched, runtime = _priority_scheduler({"o0": "other", "d0": DECODE})
    runtime.ready = {"o0", "d0"}
    assert not sched.chain_yields(D, 0, False)
    assert sched.chain_yields(D, 1, False)
