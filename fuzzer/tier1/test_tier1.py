"""Tier 1 under pytest: the corpus, a short search, and the oracle self-tests.

Run these tests with ``pytest fuzzer/``. The search here has the size that CI
needs. It uses no GPU, no weights and no kernel. For a full search, use
``python -m fuzzer.tier1 run`` with a larger budget.
"""

from __future__ import annotations

import importlib
import inspect
import re

import pytest

from fuzzer.common.case import Case, InvariantError, Op
from fuzzer.common.driver import generate, load_corpus, run_case
from fuzzer.tier1.machines import MACHINES, TIER

CI_SEEDS = 300
CI_OPS = 40


def _corpus() -> list[tuple[str, Case]]:
    """Load every saved case of every machine, for the parameters of a test."""
    entries = []
    for machine_name in MACHINES:
        for path, case in load_corpus(TIER, machine_name):
            entries.append((f"{machine_name}/{path.stem}", case))
    return entries


CORPUS = _corpus()
# Signatures of the open corpus failures. A search may rediscover these and
# nothing else.
KNOWN_OPEN = {
    ("invariant", case.notes["invariant"])
    for _, case in CORPUS
    if case.notes.get("status") == "open"
}


def test_corpus_is_not_empty():
    assert CORPUS, "no corpus cases; tier 1 has no regression coverage"


@pytest.mark.parametrize("name,case", CORPUS, ids=[name for name, _ in CORPUS])
def test_corpus_case(name, case):
    """Replay one saved case.

    A case with the status ``fixed`` must pass. It is a regression guard.

    A case with the status ``open`` must fail, and it must fail with the same
    signature. If such a case passes, somebody corrected the bug. This test
    then fails and asks you to change the status to ``fixed``.
    """
    del name
    failure = run_case(MACHINES[case.machine], case)
    if case.notes.get("status") == "open":
        if failure is None:
            pytest.fail(
                "this known-open case now passes; the underlying bug looks "
                "fixed, so flip its corpus status to 'fixed'"
            )
        assert failure.signature == tuple(case.notes["signature"]), (
            f"the case still fails, but differently: {failure.signature} "
            f"instead of {tuple(case.notes['signature'])}"
        )
        return
    assert failure is None, f"regression: {failure}\n\n{case.pretty()}"


@pytest.mark.parametrize("machine_name", list(MACHINES))
def test_bounded_search(machine_name):
    """A short search must turn up nothing outside KNOWN_OPEN."""
    machine = MACHINES[machine_name]
    for seed in range(CI_SEEDS):
        case, failure = generate(machine, seed, CI_OPS)
        if failure is None or failure.signature in KNOWN_OPEN:
            continue
        pytest.fail(
            f"new failure {failure.signature}: {failure.message}\n\n"
            f"{case.pretty()}\n\n"
            "Reproduce and shrink with:\n"
            f"  python -m fuzzer.tier1 run --machine {machine_name} "
            f"--seeds {CI_SEEDS} --ops {CI_OPS}"
        )


@pytest.mark.parametrize("machine_name", list(MACHINES))
def test_docstring_lists_every_invariant(machine_name):
    """The header of a machine must name the invariants that it checks.

    A machine that builds on another one takes the invariants of that other
    machine. Its header must name those too. A reader of one header then sees
    every invariant that the machine holds.
    """
    machine = MACHINES[machine_name]
    module = importlib.import_module(machine.__module__)
    in_code: set[str] = set()
    for klass in machine.__mro__:
        if not klass.__module__.startswith("fuzzer."):
            continue
        source = inspect.getsource(importlib.import_module(klass.__module__))
        in_code |= set(re.findall(r'require\(\s*\n?\s*"([a-z_]+\.[a-z_]+)"', source))
    listed = set(re.findall(r"^([a-z_]+\.[a-z_]+)\s{2,}", module.__doc__ or "", re.M))
    assert in_code == listed, (
        f"{machine_name}: the docstring omits {sorted(in_code - listed)} and "
        f"names {sorted(listed - in_code)}, which the code does not check"
    )


# ---------------------------------------------------------------------------
# The shape of a generated model
# ---------------------------------------------------------------------------

def test_the_generator_makes_no_node_without_an_output():
    """A node that sends nothing is not an ancestor of the tail of its loop.

    The tail of the loop then completes while that node is still pending.
    That shape meets the two open failures of ``tier0/graph_io``. One frequent
    failure hides every other failure of the same oracle, so the generator
    does not make this shape. ``corpus/model_run/`` holds it.
    """
    import random

    from fuzzer.tier1.spec import gen_spec, plan

    for seed in range(200):
        graph_plan = plan(gen_spec(random.Random(seed)))
        for node in graph_plan.all_nodes():
            assert node.outputs, f"seed {seed}: {node.name} sends nothing"


def test_a_case_replays_in_another_process():
    """A case must give the same values in a second process.

    Python gives a new hash seed to each process. The order of a set of
    strings therefore changes between two runs. The driver sorts
    ``ready_node_names`` for this reason. Without that sort, a replay takes a
    different schedule from the schedule that the case recorded.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    program = (
        "import hashlib, json;"
        "from fuzzer.common.driver import generate;"
        "from fuzzer.tier1.machines import MACHINES;"
        "M = MACHINES['model_run'];"
        "d = hashlib.blake2b(digest_size=8);"
        "[d.update(json.dumps(generate(M, s, 40)[0].to_json(), sort_keys=True).encode())"
        " for s in range(40)];"
        "print(d.hexdigest())"
    )
    digests = set()
    for hash_seed in ("0", "1", "12345"):
        environment = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(root)}
        result = subprocess.run(
            [sys.executable, "-c", program],
            cwd=root, env=environment, capture_output=True, text=True, check=True,
        )
        digests.add(result.stdout.strip())
    assert len(digests) == 1, f"the same seeds gave {digests}"


def test_a_case_replays_identically():
    """The same case must give the same result every time.

    ``ready_node_names`` is a set of strings, and the order of such a set
    changes between processes. The driver sorts it. Without that sort a case
    does not replay, and a shrink step can move the search to a different
    failure.
    """
    machine_cls = MACHINES["model_run"]
    case, _failure = generate(machine_cls, 7, CI_OPS)
    verdicts = set()
    for _ in range(5):
        failure = run_case(machine_cls, case)
        verdicts.add(None if failure is None else failure.signature)
    assert len(verdicts) == 1, f"the same case gave {verdicts}"


# ---------------------------------------------------------------------------
# The self-tests of the oracles
#
# An invariant that cannot fail is worse than no invariant. It looks like
# coverage, but it gives none. Each test below breaks the run on purpose. The
# machine must then report the damage.
# ---------------------------------------------------------------------------

LOOP_SPEC = {
    "stages": [{
        "kind": "loop", "max_iters": 2, "ext": 0,
        "num_outputs": 1, "num_accum": 0,
        "body": [{"kind": "nodes", "width": 1, "fanin": 1}],
    }],
    "num_requests": 1,
    "stops": [],
}

CHAIN_SPEC = {
    "stages": [
        {"kind": "nodes", "width": 1, "fanin": 1},
        {"kind": "nodes", "width": 1, "fanin": 1},
    ],
    "num_requests": 1,
    "stops": [],
}


def _started(spec: dict):
    machine = MACHINES["model_run"](spec)
    machine.execute(Op("start", (0,)))
    return machine, machine.runs[0]


def test_oracle_catches_a_corrupted_buffer():
    """A step reads a buffer that holds the wrong bytes.

    Every checksum after that step then changes.
    """
    machine, run = _started(CHAIN_SPEC)
    edge = next(iter(run.io.nodes["s0n0"].ready_signals.ready_inputs.values()))
    edge.tensor_info[0].uuid = "0" * 16
    with pytest.raises(InvariantError) as excinfo:
        machine.execute(Op("drain", (0,)))
    assert excinfo.value.invariant == "emit.matches_reference"


def test_oracle_catches_a_loop_that_runs_one_iteration_too_many():
    """A loop that ends at the wrong iteration is the shape of a request that
    emits the wrong number of tokens."""
    machine, run = _started(LOOP_SPEC)
    run.io.loops["s0L"].max_iters += 1
    with pytest.raises(InvariantError) as excinfo:
        machine.execute(Op("drain", (0,)))
    assert excinfo.value.invariant == "emit.runs_match_reference"


def test_oracle_catches_a_misrouted_edge():
    """An edge with a name that no node takes is lost.

    The request then waits for an input that never arrives.
    """
    machine, run = _started(CHAIN_SPEC)
    run.io.nodes["s0n0"].outputs[0].name = "no_such_name"
    machine.execute(Op("step", (0, 0)))
    with pytest.raises(InvariantError) as excinfo:
        machine.check()
    assert excinfo.value.invariant == "route.every_edge_lands"


def test_oracle_catches_a_loop_that_does_not_end():
    """A loop that never reports itself done is a request that hangs."""
    machine, run = _started(LOOP_SPEC)
    run.io.loops["s0L"].max_iters = 10_000
    with pytest.raises(InvariantError) as excinfo:
        machine.execute(Op("drain", (0,)))
    assert excinfo.value.invariant == "graph.pass_terminates"


def test_oracle_catches_state_that_survives_a_pass():
    """State that one pass leaves behind reaches the next pass. That is the
    shape of data that crosses between two requests."""
    machine, run = _started(CHAIN_SPEC)
    run.io.clear = lambda: None
    with pytest.raises(InvariantError) as excinfo:
        machine.execute(Op("drain", (0,)))
    assert excinfo.value.invariant == "state.clear_empties_the_graph"


# ---------------------------------------------------------------------------
# The self-tests of the cache oracles
#
# These break the resource on purpose, not the harness. Tier 0 cannot reach
# any of them: it drives stub resources, so it covers the order of the calls
# and no behavior.
# ---------------------------------------------------------------------------

def _kv_spec(stages: list[dict], num_requests: int = 1, **overrides) -> dict:
    """A spec that declares a real KV cache, with a fixed shape."""
    import random

    from fuzzer.tier1.reference import Reference
    from fuzzer.tier1.spec import gen_kv, plan

    spec = {"stages": stages, "num_requests": num_requests, "stops": []}
    graph_plan = plan(spec)
    runs = Reference(spec, graph_plan).run_pass("r0", 0).runs
    spec["kv"] = gen_kv(random.Random(0), graph_plan, runs, num_requests)
    defaults = {
        "page_size": 2, "num_layers": 1, "num_kv_heads": 1, "head_dim": 1,
        "capture": None, "preplan": False, "forks": [],
    }
    spec["kv"].update({**defaults, **overrides})
    spec["kv"]["spans"] = dict.fromkeys(spec["kv"]["spans"], 2)
    return spec


# Two stages that share one cache label: the second node reads what the first
# one wrote. ``_node_labels`` gives both the label ``kv0``.
SHARED_LABEL_STAGES = [
    {"kind": "nodes", "width": 1, "fanin": 1},
    {"kind": "nodes", "width": 1, "fanin": 1},
]


def _kv_machine(spec: dict):
    machine = MACHINES["kv_run"](spec)
    machine.execute(Op("start", (0,)))
    return machine


def test_the_cache_oracle_catches_a_corrupted_page():
    """A page holds bytes that the step which owns it did not write.

    Tier 0 cannot see this fault. The value of a step covers the tokens that
    its stream already holds. The next step to read that page therefore gives
    a different number, and the client sees it.
    """
    machine = _kv_machine(_kv_spec(SHARED_LABEL_STAGES))
    machine.execute(Op("step", (0, 0)))  # the first node writes its tokens
    assert machine.rig.stream_lengths()[("r0", "kv0")] == 2
    # A write of another step lands in the page that this stream holds.
    page = machine.rig.held_pages()[("r0", "kv0")][0]
    machine.rig.kv.kv_cache.tensor[0, page, 0, 0, 0, 0] += 1
    with pytest.raises(InvariantError) as excinfo:
        machine.execute(Op("drain", (0,)))
    assert excinfo.value.invariant == "emit.matches_reference"


def test_the_cache_oracle_catches_a_page_handed_to_two_streams():
    """Two streams that hold one page write over each other's tokens.

    ``tier0/page_allocator`` looks for this fault in the allocator alone.
    Here the manager gives the page out.
    """
    machine = _kv_machine(_kv_spec(SHARED_LABEL_STAGES))
    machine.execute(Op("step", (0, 0)))
    streams = machine.rig.kv._streams["r0"]
    page = streams["kv0"].page_indices[0]
    streams["other"] = type(streams["kv0"])(page_indices=[page])
    with pytest.raises(InvariantError) as excinfo:
        machine.check()
    assert excinfo.value.invariant == "kv.pages_are_not_shared"


def test_the_cache_oracle_catches_a_page_that_never_goes_back():
    """A page that ``remove_request`` does not free is lost.

    A cache that loses a page with each request has no free page left.
    """
    machine = _kv_machine(_kv_spec(SHARED_LABEL_STAGES))
    machine.execute(Op("step", (0, 0)))
    # The arena keeps what it is given, and gives nothing back.
    machine.rig.kv._arena.release = lambda pages: None
    with pytest.raises(InvariantError) as excinfo:
        machine.execute(Op("drain", (0,)))
        machine.final_check()
    assert excinfo.value.invariant in {
        "kv.pages_are_conserved", "kv.remove_frees_the_pages",
    }


def test_the_cache_oracle_catches_a_write_that_lands_elsewhere():
    """A write and a read that do not agree about an address.

    The step reads its own slots again. It therefore sees this fault before
    any value leaves the node.
    """
    machine = _kv_machine(_kv_spec(SHARED_LABEL_STAGES))
    cache = machine.rig.kv.kv_cache
    write_tokens = cache.write_tokens

    def drop_the_write(layer_idx, k, v, page_idx, cache_idx, return_tensor=False):
        return write_tokens(
            layer_idx=layer_idx, k=k * 0, v=v, page_idx=page_idx,
            cache_idx=cache_idx, return_tensor=return_tensor,
        )

    cache.write_tokens = drop_the_write
    with pytest.raises(InvariantError) as excinfo:
        machine.execute(Op("step", (0, 0)))
    assert excinfo.value.invariant == "kv.readback_matches_the_write"


def test_a_step_really_reads_the_cache():
    """The value of a step must depend on the cache that the step reads.

    Every cache oracle needs this property. Without it, a step covers its
    identity alone, and no cache oracle can fail.

    The test runs the same step two times from the same state. It changes one
    token of the stream between the two runs. The value of the node must then
    differ.
    """
    from fuzzer.tier1.values import values_of

    def value_after_change(change: bool) -> str:
        machine = _kv_machine(_kv_spec(SHARED_LABEL_STAGES))
        machine.execute(Op("step", (0, 0)))
        if change:
            page = machine.rig.held_pages()[("r0", "kv0")][0]
            machine.rig.kv.kv_cache.tensor[0, page, 0, 0, 0, 0] += 1
        run = machine.runs[0]
        machine._run_node(0, "s1n0")
        return values_of(run.io.nodes["s1n0"].outputs[0])[0]

    assert value_after_change(False) != value_after_change(True), (
        "the value of a step does not depend on the cache it reads"
    )


def test_the_position_oracle_catches_a_counter_that_drifted():
    """The position resource holds its own counter for each stream.

    That counter must follow the length of the KV stream. A counter that
    moves alone gives a step the wrong position ids. The shape of the ids
    stays correct, so only their values report the fault.
    """
    spec = _kv_spec(SHARED_LABEL_STAGES, positions=True)
    machine = _kv_machine(spec)
    machine.execute(Op("step", (0, 0)))
    # Advance the counter without the cache.
    positions = machine.rig.runner.resources["pos"]
    positions._counters["r0"]["kv0"] += 1
    with pytest.raises(InvariantError) as excinfo:
        machine.execute(Op("step", (0, 0)))
    assert excinfo.value.invariant == "pos.ids_match_the_stream"


def test_the_generated_models_declare_real_resources():
    """Every ``kv_run`` case builds real resources and drives them.

    The resources come from ``mstar/engine/resources``, and the real runner
    drives them. Without this test the machine can become ``model_run`` again
    without a sign. The graph still runs, and the interpreter still agrees.
    Every cache oracle then passes over a cache that no step touched.
    """
    import random

    from mstar.engine.resources.kv.manager import KVManager
    from mstar.engine.resources.position.manager import PositionManager
    from mstar.engine.resources.runner import StepRunner

    machine_cls = MACHINES["kv_run"]
    seen_positions = 0
    for seed in range(20):
        rng = random.Random(seed)
        machine = machine_cls(machine_cls.gen_config(rng))
        rig = machine.rig
        assert isinstance(rig.runner, StepRunner)
        assert isinstance(rig.runner.resources["kv"], KVManager)
        if rig.with_positions:
            seen_positions += 1
            assert isinstance(rig.runner.resources["pos"], PositionManager)
            # The runner resolved the dependency and put the cache first.
            assert rig.runner.order == ("kv", "pos")

        # A case moves real pages.
        for _ in range(30):
            machine.execute(machine.gen_op(rng))
        used = rig.config.max_num_pages - rig.free_pages()
        if sum(rig.stream_lengths().values()):
            assert used > 1, "streams hold tokens but no page left the arena"
    assert seen_positions, "no generated model declared a position resource"


# ---------------------------------------------------------------------------
# The self-tests of the pre-plan, capture and fork oracles
#
# None of these tests can run against tier 0. It drives stub resources, so it
# has no cache to fork, no buffer to capture into, and no staged state to give
# back.
# ---------------------------------------------------------------------------

CAPTURE = {"bs": 2, "num_tokens": 6, "slots": 2}


def test_the_fork_oracle_catches_a_copy_that_did_not_happen():
    """A fork copies one stream onto another label.

    A later step reads the target label. A copy that does not happen
    therefore changes what the client sees.
    """
    spec = _kv_spec(SHARED_LABEL_STAGES)
    spec["kv"]["labels"]["s1n0"] = "kv1"
    spec["kv"]["forks"] = [{"node": "s0n0", "to": "kv1", "when": "post"}]
    machine = _kv_machine(spec)
    # The resource declares no fork, so the copy never runs. The interpreter
    # still waits for it.
    machine.rig.forks = {}
    with pytest.raises(InvariantError) as excinfo:
        machine.execute(Op("drain", (0,)))
    assert excinfo.value.invariant in {
        "emit.matches_reference", "kv.stream_matches_reference",
    }


def test_the_capture_oracle_catches_padding_over_a_live_page():
    """The rows of a bucket that carry no token must address the sink page.

    With any other page, a replay writes into the cache of another request.
    """
    spec = _kv_spec(SHARED_LABEL_STAGES, capture=CAPTURE)
    machine = _kv_machine(spec)
    machine.execute(Op("step", (0, 0)))  # the stream now holds a page
    page = machine.rig.held_pages()[("r0", "kv0")][0]
    replay_padding = machine.rig._replay_padding

    def point_at_a_live_page(label, lease):
        state = machine.rig.kv._current_plan_states[label]
        state.token_to_page[state.total_tokens:lease.bucket.num_tokens] = page
        return replay_padding(label, lease)

    machine.rig._replay_padding = point_at_a_live_page
    machine.execute(Op("drain", (0,)))
    with pytest.raises(InvariantError) as excinfo:
        machine.check()
    assert excinfo.value.invariant == "capture.padding_stays_off_live_pages"


def test_the_preplan_oracle_catches_a_stage_that_raises():
    """``admit`` is the gate for a staged step and for a live step."""
    spec = _kv_spec(SHARED_LABEL_STAGES, capture=CAPTURE, preplan=True)
    machine = _kv_machine(spec)
    machine.execute(Op("step", (0, 0)))

    def raise_from_the_plan_thread(step):
        raise RuntimeError("the plan thread failed")

    machine.rig.runner.pre_plan = raise_from_the_plan_thread
    with pytest.raises(InvariantError) as excinfo:
        machine.execute(Op("preplan", (0, 0)))
    assert excinfo.value.invariant == "preplan.staging_does_not_raise"


def test_the_preplan_oracle_catches_a_rollback_that_did_not_happen():
    """A staged step that never runs must give its state back.

    Without that, the next step plans against a length that no step wrote.
    """
    spec = _kv_spec(SHARED_LABEL_STAGES, capture=CAPTURE, preplan=True)
    machine = _kv_machine(spec)
    machine.execute(Op("step", (0, 0)))
    machine.execute(Op("preplan", (0, 0)))
    assert machine.rig.staged is not None, "the step did not stage"

    # The staged step moved a length, and nothing puts it back.
    for stream in machine.rig.kv._streams["r0"].values():
        stream.stored_len += 1
    with pytest.raises(InvariantError) as excinfo:
        machine.execute(Op("abandon", ()))
    assert excinfo.value.invariant == "preplan.abandon_restores_state"


def test_a_staged_step_gives_the_same_values_as_an_unstaged_one():
    """A client must not see that a step was planned one step ahead.

    This is the metamorphic property of the tier. The interpreter knows
    nothing about a staged step. A promoted step must therefore give the
    values that the same step gives without a stage.
    """
    def run(stage_first: bool) -> dict:
        spec = _kv_spec(SHARED_LABEL_STAGES, capture=CAPTURE, preplan=True)
        machine = _kv_machine(spec)
        machine.execute(Op("step", (0, 0)))
        if stage_first:
            machine.execute(Op("preplan", (0, 0)))
            assert machine.rig.staged is not None, "the step did not stage"
        machine.execute(Op("drain", (0,)))
        return dict(machine.runs[0].emitted)

    staged, inline = run(True), run(False)
    assert staged == inline, (
        f"a promoted step gave {staged}, an inline one gave {inline}"
    )


# ---------------------------------------------------------------------------
# The self-tests of the race oracles
#
# A schedule finds a fault only when it reaches the interleaving that shows
# that fault. Each test below therefore runs many seeds, and requires the
# machine to report the damage on a minimum of one of them. The first test
# needs the one interleaving in which a teardown runs between two lines of
# ``publish``.
# ---------------------------------------------------------------------------

def _race_sweep(break_it, seeds: int = 120) -> dict[str, int]:
    """Run the race machine with something broken, and count what it says."""
    import random

    machine_cls = MACHINES["kv_race"]
    caught: dict[str, int] = {}
    for seed in range(seeds):
        rng = random.Random(seed)
        machine = machine_cls(machine_cls.gen_config(rng))
        break_it(machine)
        try:
            for _ in range(40):
                machine.execute(machine.gen_op(rng))
                machine.check()
            machine.final_check()
        except InvariantError as exc:
            caught[exc.invariant] = caught.get(exc.invariant, 0) + 1
    return caught


def test_the_race_machine_interleaves_and_replays():
    """A case must interleave the threads, and it must replay.

    Without the interleaving, the oracles never see a mixed state. Without
    the replay, a failure has no stable signature. It then cannot shrink, and
    it cannot become a corpus case.
    """
    import random

    machine_cls = MACHINES["kv_race"]
    config = machine_cls.gen_config(random.Random(3))

    traces = []
    for _ in range(3):
        machine = machine_cls(config)
        rng = random.Random(99)
        for _ in range(30):
            machine.execute(machine.gen_op(rng))
        traces.append(tuple(machine.driver.trace))
    assert len(set(traces)) == 1, "the same case gave different interleavings"

    switches = sum(1 for a, b in zip(traces[0], traces[0][1:], strict=False) if a != b)
    assert switches > 0, "the case never switched threads"


def test_the_race_oracle_catches_a_publish_that_lost_its_guard():
    """``publish`` reads ``_streams`` outside the lock and says why:
    ``remove_request`` can pop them from another thread. Take the guard away
    and the interleaving that pops them is a crash."""
    def unguard(machine):
        kv = machine.kv

        def publish(request_id):
            streams = kv._streams[request_id]      # was .get() plus a guard
            with kv._lock:
                return {label: s.stored_len for label, s in streams.items()}

        kv.publish = publish

    assert _race_sweep(unguard).get("race.thread_completes"), (
        "no interleaving put a teardown inside publish"
    )


def test_the_race_oracle_catches_pages_that_never_come_back():
    def leak(machine):
        kv = machine.kv

        def remove_request(rid):
            kv._streams.pop(rid, None)

        kv.remove_request = remove_request

    assert _race_sweep(leak, seeds=20).get("race.quiesce_frees_every_page")


def test_the_race_oracle_catches_a_page_issued_twice():
    """The fault every page allocator exists to avoid, seen from above: two
    streams holding one page write the same block."""
    def double_issue(machine):
        arena = machine.kv._arena
        acquire = arena.acquire
        state = {"calls": 0}

        def acquire_(count):
            state["calls"] += 1
            if state["calls"] == 3:
                held = [
                    page for streams in machine.kv._streams.values()
                    for stream in streams.values() for page in stream.page_indices
                ]
                if held:
                    return [held[0]] * count
            return acquire(count)

        arena.acquire = acquire_

    assert _race_sweep(double_issue).get("race.pages_are_not_shared")


def test_the_race_oracle_catches_a_span_counted_twice():
    """The data oracle: what a stream holds is what the steps that committed
    to it wrote."""
    def double_count(machine):
        kv = machine.kv
        commit = kv.commit

        def commit_(step, ctx):
            commit(step, ctx)
            for segment in step.segments:
                kv._streams[segment.request_id][segment.label].stored_len += segment.span

        kv.commit = commit_

    assert _race_sweep(double_count).get("race.stream_holds_its_writes")
