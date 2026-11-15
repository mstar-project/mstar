"""Scheduling policies: the order of ready (node, walk) keys, when a speculation chain
yields, how a model's or a deployment's entries resolve, and what ``next_batch`` hands out."""

from types import SimpleNamespace

import pytest
from test_scheduler_rebuild import DECODE, MIXED, NODE, PREFILL, _Engine, _Runtime, _State

from mstar.engine.resources.step import FULL_ADMIT_NOT_READY, FULL_ADMIT_OK
from mstar.testing_utils.worker import make_scheduler
from mstar.worker.rebuild import Rebuilds
from mstar.worker.scheduling_policy import Priority, RoundRobin, WalkPriority, resolve_walk_priorities
from mstar.worker.services import SchedulingSettings, walk_priorities_of

OTHER = "other"
P, D, O, VAE = (NODE, PREFILL), (NODE, DECODE), (NODE, OTHER), ("VAEDecoder", "vae_chunk")


class _Peek:
    """The scheduler's state a policy reads; answers each peek with ``answer(exclude, among)``."""

    def __init__(self, answer=lambda exclude, among: True, now=0, last=None):
        self.answer, self.asked = answer, []
        self.batch_number, self.node_and_walk_to_last_batch_num = now, dict(last or {})
        self.chain_steps, self.chain_step_of = now, dict(last or {})

    def has_ready_excluding(self, exclude_target, among=None):
        self.asked.append((exclude_target, None if among is None else frozenset(among)))
        return self.answer(exclude_target, among)


# -- the policies -------------------------------------------------------------------------


def test_round_robin_orders_least_recently_scheduled_first_ties_in_scan_order():
    ready = dict.fromkeys([D, P, VAE], [])
    assert RoundRobin().order(ready, _Peek(last={D: 7, P: 3})) == [VAE, P, D]
    assert RoundRobin().order(ready, _Peek()) == [D, P, VAE]


def test_round_robin_keeps_a_chains_first_step_and_asks_nothing_for_it():
    peek = _Peek()
    assert RoundRobin().chain_yields(D, 0, False, peek) is False and peek.asked == []
    assert RoundRobin().chain_yields(D, 1, False, peek) is True and peek.asked == [(D, None)]
    assert RoundRobin().chain_yields(P, 0, True, _Peek()) is True  # a prompt cut into chunks


def test_priority_orders_higher_first_and_round_robin_among_equals():
    ready = dict.fromkeys([D, VAE, P], [])
    assert Priority({P: 1}, {}).order(ready, _Peek(last={P: 9, D: 5, VAE: 2})) == [P, VAE, D]


def test_priority_asks_only_about_the_walks_above_a_chain_at_its_first_step():
    policy = Priority({P: 1}, {})
    peek = _Peek()
    assert policy.chain_yields(D, 0, False, peek) is True and peek.asked == [(D, frozenset({P}))]
    assert policy.chain_yields(D, 0, False, _Peek(lambda exclude, among: False)) is False
    # the most urgent chain has nothing above it: round robin, which keeps its first step unasked
    peek = _Peek()
    assert policy.chain_yields(P, 0, False, peek) is False and peek.asked == []


def test_priority_makes_one_peek_a_pass_after_the_first_step():
    peek = _Peek(lambda exclude, among: False)
    assert Priority({P: 1}, {D: 2}).chain_yields(D, 1, False, peek) is False
    assert peek.asked == [(D, None)], "a pass would scan twice"


def test_a_key_past_its_max_wait_goes_first_longest_waiting_first():
    policy = Priority({P: 2, O: 1}, {D: 3, O: 3})
    ready = dict.fromkeys([P, D, O], [])
    assert policy.order(ready, _Peek(now=10, last={P: 10, D: 9, O: 8})) == [P, O, D]  # nobody overdue
    assert policy.order(ready, _Peek(now=10, last={P: 10, D: 7, O: 8})) == [D, P, O]
    assert policy.order(ready, _Peek(now=10, last={P: 10, D: 6, O: 5})) == [O, D, P]
    # overdue but not ready: nothing to run
    assert policy.order({P: [], O: []}, _Peek(now=10, last={P: 10, D: 1, O: 9})) == [P, O]


def test_at_its_first_step_a_chain_yields_to_a_key_past_its_bound_too():
    policy = Priority({P: 1}, {D: 3})
    peek = _Peek(now=10, last={P: 10, D: 5})
    assert policy.chain_yields(P, 0, False, peek) is True and peek.asked == [(P, frozenset({D}))]
    peek = _Peek(now=10, last={P: 10, D: 9})
    assert policy.chain_yields(P, 0, False, peek) is False and peek.asked == []


# -- resolving entries --------------------------------------------------------------------

WALKS = {PREFILL: {NODE, "Encoder"}, DECODE: {NODE}, OTHER: {NODE}}


def test_the_most_specific_entry_decides_a_pair():
    entries = [
        WalkPriority(node=NODE, priority=1, max_wait=9),
        WalkPriority(walk=PREFILL, priority=2),
        WalkPriority(node=NODE, walk=DECODE, priority=0, max_wait=4),
    ]
    got = resolve_walk_priorities(entries, WALKS, "test")
    assert {pair: (e.priority, e.max_wait) for pair, e in got.items()} == {
        P: (2, None), ("Encoder", PREFILL): (2, None), D: (0, 4), O: (1, 9),
    }


@pytest.mark.parametrize("entry, match", [
    (WalkPriority(priority=1), "neither"),
    (WalkPriority(walk="prefil", priority=1), "no walk"),
    (WalkPriority(node="Encoder", walk=DECODE, priority=1), "not a node of walk"),
    (WalkPriority(node="Talkr", priority=1), "not a node of any walk"),
    (WalkPriority(walk=PREFILL, priority=-1), "non-negative"),
    (WalkPriority(walk=PREFILL, priority=1, max_wait=0), "max_wait"),
])
def test_an_entry_that_names_nothing_the_model_runs_or_a_bad_value_refuses_the_start(entry, match):
    with pytest.raises(ValueError, match=match):
        resolve_walk_priorities([entry], WALKS, "test")
    with pytest.raises(ValueError, match="two entries"):
        resolve_walk_priorities([WalkPriority(walk=DECODE), WalkPriority(walk=DECODE, priority=1)], WALKS, "test")


class _Section:
    def __init__(self, nodes):
        self.nodes = nodes

    def get_nodes(self):
        return self.nodes


class _Model:
    def __init__(self, entries):
        self.entries = entries

    def get_walk_priorities(self):
        return self.entries


def test_a_deployments_entries_replace_the_models():
    walks = {walk: _Section(nodes) for walk, nodes in WALKS.items()}
    model = _Model([WalkPriority(walk=PREFILL, priority=1)])
    assert set(walk_priorities_of(model, walks)) == {P, ("Encoder", PREFILL)}
    settings = SchedulingSettings.from_config(
        {"policy": "priority", "max_wait": 8, "walk_priorities": [{"node": NODE, "walk": DECODE, "priority": 3}]}
    )
    assert (settings.policy, settings.max_wait) == ("priority", 8)
    assert {pair: e.priority for pair, e in walk_priorities_of(model, walks, settings.entries).items()} == {D: 3}
    assert SchedulingSettings.from_config(None) == SchedulingSettings()
    for bad in ({"policy": "fifo"}, {"max_wait": 0}, {"walk_priorities": [{"walk": DECODE, "prio": 1}]}, {"x": 1}):
        with pytest.raises(ValueError, match="scheduling"):
            SchedulingSettings.from_config(bad)


# -- the scheduler's policy ---------------------------------------------------------------


def _entries(*entries):
    return {(e.node, e.walk): e for e in entries}


def test_the_scheduler_takes_its_policy_from_the_entries_and_the_deployment():
    assert type(make_scheduler().policy) is RoundRobin
    assert type(make_scheduler(walk_priorities=_entries(WalkPriority(NODE, DECODE))).policy) is RoundRobin
    policy = make_scheduler(walk_priorities=_entries(WalkPriority(NODE, PREFILL, 1))).policy
    assert isinstance(policy, Priority) and (policy.priority(P), policy.priority(D)) == (1, 0)
    off = make_scheduler(walk_priorities=_entries(WalkPriority(NODE, PREFILL, 1)),
                         layout_fields={"scheduling_policy": "round_robin"})
    assert type(off.policy) is RoundRobin
    # a bound alone makes a policy: round robin holds no key back past it
    bounded = make_scheduler(layout_fields={"default_max_wait": 4, "my_walks": frozenset({P, D})}).policy
    assert isinstance(bounded, Priority) and bounded.max_waits == {P: 4, D: 4}


def test_a_combined_walks_key_takes_its_most_urgent_walks_priority_and_bound():
    combined = {P: MIXED, D: MIXED}
    policy = make_scheduler(
        combined_walk_of=combined,
        walk_priorities=_entries(WalkPriority(NODE, PREFILL, 2, 6), WalkPriority(NODE, DECODE, 0, 3)),
    ).policy
    assert (policy.priority((NODE, MIXED)), policy.priority(P), policy.priority(D)) == (2, 2, 0)
    # a bound sits on the key steps are recorded under, the combined one
    assert policy.max_waits == {(NODE, MIXED): 3}


# -- through the real scheduler -----------------------------------------------------------


class _ExcludingRuntime(_Runtime):
    """Leaves out ``exclude_target`` as the graph runtime does: a real walk, or a
    combined one covering each of its walks."""

    combined_walk_of: dict = {}

    def get_ready_nodes(self, exclude_rids, target=None, exclude_target=None):
        def excluded(walk):
            return exclude_target in ((NODE, walk), (NODE, self.combined_walk_of.get((NODE, walk))))

        return [spec for spec in super().get_ready_nodes(exclude_rids, target) if not excluded(spec.graph_walk)]


def _priority_scheduler(walks, *, entries=(WalkPriority(NODE, PREFILL, 1),), combined=False, budget=None,
                        refuse=(), mixes=None):
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
        walk_priorities=_entries(*entries), speculation_mixes_walks=None if mixes is None else {NODE: mixes},
        layout_fields={"my_walks": frozenset({(NODE, walk) for walk in set(walks.values())})},
    )
    return sched, runtime


def _walk(batch):
    return None if batch is None else batch.graph_walk


def test_a_decode_chain_yields_at_once_to_a_prompt_the_engine_would_take():
    sched, runtime = _priority_scheduler({"p0": PREFILL, "d0": DECODE})
    runtime.ready = {"p0", "d0"}
    assert sched.chain_yields(D, 0, False)
    # and the yield-away hands out the prompt's step
    assert _walk(sched.next_batch(exclude_target=D)) == PREFILL


def test_the_next_step_goes_to_a_prompt_over_a_less_recent_decode():
    sched, runtime = _priority_scheduler({"p0": PREFILL, "d0": DECODE})
    sched.node_and_walk_to_last_batch_num.update({P: 5, D: 1})
    sched.batch_number = 5
    runtime.ready = {"p0", "d0"}
    assert _walk(sched.next_batch()) == PREFILL


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


def test_at_its_first_step_a_chain_keeps_its_step_from_walks_not_above_it():
    """Round robin's rule for walks of its own priority or below: the chain has
    its first step before they take a turn."""
    sched, runtime = _priority_scheduler({"o0": OTHER, "d0": DECODE})
    runtime.ready = {"o0", "d0"}
    assert not sched.chain_yields(D, 0, False)
    assert sched.chain_yields(D, 1, False)


def _steps(sched, runtime, keep_ready, n):
    """The walks of ``n`` steps the plain ``next_batch`` hands out, with
    ``keep_ready``'s rows back on the queues after each."""
    walks = []
    for _ in range(n):
        runtime.ready |= set(keep_ready)
        batch = sched.next_batch()
        walks.append(_walk(batch))
        sched.note_ran(batch)
    return walks


def test_a_higher_walk_that_stays_ready_holds_a_lower_one_back_only_to_its_bound():
    walks = {"p0": PREFILL, "d0": DECODE}
    sched, runtime = _priority_scheduler(walks)
    assert _steps(sched, runtime, walks, 6).count(DECODE) == 0, "unbounded, decode never runs"
    sched, runtime = _priority_scheduler(walks, entries=(WalkPriority(NODE, PREFILL, 1),
                                                         WalkPriority(NODE, DECODE, 0, max_wait=3)))
    steps = _steps(sched, runtime, walks, 9)
    assert steps.count(DECODE) >= 2 and DECODE in steps[:4], steps


def test_two_higher_walks_taking_turns_do_not_strand_a_third_past_its_bound():
    walks = {"p0": PREFILL, "o0": OTHER, "d0": DECODE}
    entries = (WalkPriority(NODE, PREFILL, 1), WalkPriority(NODE, OTHER, 1), WalkPriority(NODE, DECODE, 0, 4))
    sched, runtime = _priority_scheduler(walks, entries=entries)
    steps = _steps(sched, runtime, walks, 10)
    assert DECODE in steps[:5] and steps.count(DECODE) >= 2, steps


def _ran(sched, node, walk):
    sched.note_ran(SimpleNamespace(node_name=node, graph_walk=walk, request_to_worker_graph={"r": "wg0"},
                                   turn_order=None))


def test_a_bounded_walk_under_a_combined_walk_waits_on_the_combined_key():
    """The combined key ran every step, so nothing is overdue: a bound on the real
    walk, which no step records, would read as overdue for the rest of the run."""
    sched, runtime = _priority_scheduler({"p0": PREFILL, "d0": DECODE}, entries=(), combined=True, mixes=False)
    sched.policy = sched._policy(SimpleNamespace(
        scheduling_policy="priority", default_max_wait=2, walk_priorities={}, my_walks=frozenset({P, D}),
    ))
    for _ in range(5):
        _ran(sched, NODE, DECODE)
    runtime.ready = {"p0", "d0"}
    assert not sched.chain_yields(D, 0, False)


def test_every_chain_step_counts_toward_a_wait_and_side_lane_steps_do_not():
    sched = make_scheduler(side_keys={VAE})
    for _ in range(3):
        _ran(sched, NODE, DECODE)  # a chain continuing its own rows is a step too
    _ran(sched, *VAE)
    assert sched.chain_steps == 3 and sched.chain_step_of == {D: 3}


def test_a_side_lane_key_gets_no_bound():
    sched = make_scheduler(side_keys={VAE}, layout_fields={"default_max_wait": 4, "my_walks": frozenset({D, VAE})})
    assert sched.policy.max_waits == {D: 4}


def test_a_wait_counts_the_chain_steps_note_ran_records():
    """Steps the scheduler didn't build (a chain continuing its own rows) still make
    a bounded key wait, in the order and at a chain's first step."""
    sched, runtime = _priority_scheduler({"p0": PREFILL, "d0": DECODE},
                                         entries=(WalkPriority(NODE, PREFILL, 1), WalkPriority(NODE, DECODE, 0, 3)))
    _ran(sched, NODE, DECODE)
    for _ in range(3):
        _ran(sched, NODE, PREFILL)
    assert sched.batch_number == 0, "no step was built"
    assert sched.policy.order({P: [], D: []}, sched) == [D, P]
    runtime.ready = {"p0", "d0"}
    assert sched.chain_yields(P, 0, False)
