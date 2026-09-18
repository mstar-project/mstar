"""Tier 0 under pytest: the corpus, a short search, and the oracle self-tests.

Run these tests with ``pytest fuzzer/``. The size of the search here suits CI.
For a full search, use ``python -m fuzzer.tier0 run`` with a larger budget.
"""

from __future__ import annotations

import importlib
import inspect
import re

import pytest

from fuzzer.common.case import Case, Failure, InvariantError, Op
from fuzzer.common.driver import generate, load_corpus, run_case
from fuzzer.tier0.machines import MACHINES, TIER

# The number of cases for each machine in the CI search. Tier 0 runs
# approximately 1000 cases each second, so this costs a few seconds for each
# machine. The nightly job runs many more cases.
CI_SEEDS = 400
CI_OPS = 60

# A stand-in signature for "this case passed", so a replay test can compare
# verdicts without special-casing None.
_PASSED = Failure(("passed", ""), "", "", -1, "")


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
    (("invariant", case.notes["invariant"]))
    for _, case in CORPUS
    if case.notes.get("status") == "open"
}


def test_corpus_is_not_empty():
    assert CORPUS, "no corpus cases; tier 0 has no regression coverage"


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
    """A short search must turn up nothing outside KNOWN_OPEN.

    Each case stops at its first failed op. Thus a frequent failure hides the
    invariants that come later in the same case.
    """
    machine = MACHINES[machine_name]
    surprises = []
    for seed in range(CI_SEEDS):
        case, failure = generate(machine, seed, CI_OPS)
        if failure is None or failure.signature in KNOWN_OPEN:
            continue
        surprises.append((failure, case))
        break
    if surprises:
        failure, case = surprises[0]
        pytest.fail(
            f"new failure {failure.signature}: {failure.message}\n\n"
            f"{case.pretty()}\n\n"
            "Reproduce and shrink with:\n"
            f"  python -m fuzzer.tier0 run --machine {machine_name} "
            f"--seeds {CI_SEEDS} --ops {CI_OPS}"
        )


@pytest.mark.parametrize("machine_name", list(MACHINES))
def test_docstring_lists_every_invariant(machine_name):
    """The header of a machine must name exactly the invariants it checks."""
    module = importlib.import_module(MACHINES[machine_name].__module__)
    source = inspect.getsource(module)
    in_code = set(re.findall(r'require\(\s*\n?\s*"([a-z_]+\.[a-z_]+)"', source))
    listed = set(re.findall(r"^([a-z_]+\.[a-z_]+)\s{2,}", module.__doc__ or "", re.M))
    assert in_code == listed, (
        f"{machine_name}: the docstring omits {sorted(in_code - listed)} and "
        f"names {sorted(listed - in_code)}, which the code does not check"
    )


# ---------------------------------------------------------------------------
# The self-tests of the oracles
#
# An invariant that cannot fail is worse than no invariant. It looks like
# coverage, but it gives none. Each test below breaks the system under test on
# purpose. The machine must then report the damage.
# ---------------------------------------------------------------------------

def test_page_allocator_oracle_catches_a_duplicated_page():
    machine = MACHINES["page_allocator"]({"max_pages": 4})
    machine.execute(Op("alloc", (2,)))
    machine.check()
    # This machine exists to find one fault above all: the allocator gives the
    # same page to two holders.
    machine.alloc.free_pages.put(next(iter(machine.live.values()))[0])
    with pytest.raises(InvariantError) as excinfo:
        machine.check()
    assert excinfo.value.invariant == "pages.free_and_held_disjoint"


def test_alloc_concurrent_oracle_catches_a_page_held_by_two_threads():
    machine = MACHINES["alloc_concurrent"]({
        "max_pages": 2,
        "programs": [[["alloc", 1]], [["alloc", 1]]],
    })
    machine.check()
    # The fault this machine exists for: the same page reaches two threads.
    # Both would write the same KV block, and nothing would report it.
    machine.free_list.outstanding["t0"] = [0]
    machine.free_list.outstanding["t1"] = [0]
    with pytest.raises(InvariantError) as excinfo:
        machine.check()
    assert excinfo.value.invariant == "alloc.no_double_issue_across_threads"


def test_alloc_concurrent_replays_identically():
    """The schedule is part of the case, so an interleaving must not drift."""
    machine_cls = MACHINES["alloc_concurrent"]
    case, _failure = generate(machine_cls, 10, 40)

    def verdict():
        failure = run_case(machine_cls, case)
        return _PASSED.signature if failure is None else failure.signature

    verdicts = {verdict() for _ in range(8)}
    assert len(verdicts) == 1, f"the same case gave {verdicts}"


def test_tensor_store_oracle_catches_a_leaked_request_entry():
    machine = MACHINES["tensor_store"]({"num_requests": 1, "num_uuids": 1})
    machine.execute(Op("put", (0, 0)))
    machine.check()
    # An empty request that keeps its key in the store is the shape of
    # unlimited growth.
    machine.store.per_req_tensors["r0"].clear()
    with pytest.raises(InvariantError) as excinfo:
        machine.check()
    assert excinfo.value.invariant in {
        "store.mirrors_model", "store.no_empty_request_entries",
    }


def test_scheduler_oracle_catches_lost_work():
    machine = MACHINES["micro_scheduler"]({
        "num_requests": 1, "num_nodes": 1, "num_walks": 1,
        "num_worker_graphs": 1, "caps": [2], "max_consec_tp_follower_batches": 1,
    })
    machine.execute(Op("make_ready", (0, 0)))
    machine.check()
    # Drop the ready node, but do not schedule it. The request can then never
    # run again.
    machine.queues["wg0"].per_request_queues["r0"].ready_node_names.clear()
    with pytest.raises(InvariantError) as excinfo:
        machine.check()
    assert excinfo.value.invariant == "sched.no_ready_work_is_lost"


def test_step_runner_oracle_catches_a_leaked_request():
    machine = MACHINES["step_runner"]({
        "resources": [{
            "key": "R0", "deps": [], "preplan": False,
            "publishes": False, "retrieves": False,
        }],
        "nodes": {"node0": ["R0"]},
        "scope_nodes": True,
    })
    machine.execute(Op("ingest", (0,)))
    # A resource that ignores remove_request has the shape of bug #146.
    machine.stubs["R0"].remove_request = lambda rid: None
    with pytest.raises(InvariantError) as excinfo:
        machine.execute(Op("remove", (0,)))
    assert excinfo.value.invariant == "runner.remove_reaches_every_resource"


def test_step_runner_oracle_catches_a_dropped_live_request():
    machine = MACHINES["step_runner"]({
        "resources": [{
            "key": "R0", "deps": [], "preplan": False,
            "publishes": False, "retrieves": False,
        }],
        "nodes": {"node0": ["R0"]},
        "scope_nodes": True,
    })
    machine.execute(Op("ingest", (0,)))
    machine.check()
    # A resource that forgets a live request is the shape of a request that
    # runs against state it no longer has.
    machine.stubs["R0"].rids.clear()
    with pytest.raises(InvariantError) as excinfo:
        machine.check()
    assert excinfo.value.invariant == "runner.live_request_keeps_its_state"


def test_graph_io_oracle_catches_a_partially_fed_ready_node():
    config = {"widths": [2, 1], "loop": None}
    machine = MACHINES["graph_io"](config)
    # Put a node into the ready queue, but give it no input at all.
    machine.io.wg_state_registry.ready_names.add("n0_0")
    with pytest.raises(InvariantError) as excinfo:
        machine.check()
    assert excinfo.value.invariant == "graph.ready_implies_all_inputs"
