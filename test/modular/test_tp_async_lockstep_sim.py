"""Invariant checks for the TP async-scheduling protocol model.

See tp_async_sim.py for the model. These tests pin the two results that gate
the implementation:

1. B1 (gated commit/cancel) and B2 with void-only Cancel hold I1–I4 across
   every interleaving of the failure matrix, at 2 and 3 ranks.
2. The naive Cancel semantics (retract the spec batch from any rank that has
   not executed it yet — which is what "drop if unbuilt / unwind if built"
   means without a gate) is REFUTED: it desyncs the collective order.
   Kept as a negative control so the constraint survives refactors.
3. Each rank applying its own verdict (B2_LOCAL) is REFUTED as soon as a
   failure is on one rank only, and the voted verdict (B2_AGREED, what the
   worker runs) holds there and everywhere B2_LOCAL held.
"""

try:
    from .tp_async_sim import PER_RANK_SCENARIOS, SCENARIOS, explore
except ImportError:  # collected without package context
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tp_async_sim import PER_RANK_SCENARIOS, SCENARIOS, explore


def _run(mode, scenario, nranks=2):
    comp0, script, must = {**SCENARIOS, **PER_RANK_SCENARIOS}[scenario]
    return explore(mode, nranks, comp0, script, scenario, must)


def test_serial_baseline_holds():
    for name in SCENARIOS:
        assert _run("SERIAL", name).ok


def test_b1_holds_all_scenarios_2_and_3_ranks():
    for nranks in (2, 3):
        for name in SCENARIOS:
            res = _run("B1", name, nranks)
            assert res.ok, (name, nranks, res.violations[:1])


def test_b2_void_holds_all_scenarios_2_and_3_ranks():
    for nranks in (2, 3):
        for name in SCENARIOS:
            res = _run("B2_VOID", name, nranks)
            assert res.ok, (name, nranks, res.violations[:1])


def test_b2_local_holds_all_scenarios_2_and_3_ranks():
    """The implemented protocol (MSTAR_TP_ASYNC_SCHED=1): no cancel message,
    each rank derives the void when it completes the parent step, and a spec
    can only launch after that on its own rank. Must hold everywhere the
    signalled-retract variant fails."""
    for nranks in (2, 3):
        for name in SCENARIOS:
            res = _run("B2_LOCAL", name, nranks)
            assert res.ok, (name, nranks, res.violations[:1])


def test_b2_local_pays_no_wasted_forward_on_structural_cancel():
    """Where VOID must run S everywhere and discard it, LOCAL never launches
    it: the structural scenario costs LOCAL only the stop-waste (2), while
    VOID pays the voided forward on top (3)."""
    assert _run("B2_LOCAL", "structural-cancel").max_waste == 2
    assert _run("B2_VOID", "structural-cancel").max_waste == 3


def test_b2_agreed_holds_all_scenarios_2_and_3_ranks():
    """The protocol with the rank vote: B2_LOCAL, with each rank applying N's
    verdict once every rank has finished N, and the instance's verdict."""
    for nranks in (2, 3):
        for name in (*SCENARIOS, *PER_RANK_SCENARIOS):
            res = _run("B2_AGREED", name, nranks)
            assert res.ok, (name, nranks, res.violations[:1])


def test_b2_local_is_refuted_by_a_failure_on_one_rank():
    """A rank that drops the head the others run never joins their collective;
    the instance diverges whichever rank the failure is on."""
    for nranks in (2, 3):
        for name in PER_RANK_SCENARIOS:
            res = _run("B2_LOCAL", name, nranks)
            assert res.violations, (name, nranks, "per-rank verdicts unexpectedly held")
            assert any(kind.startswith(("I1", "I4")) for kind, _ in res.violations), res.violations[:1]


def test_b2_agreed_costs_what_b2_local_does_where_both_hold():
    for name in SCENARIOS:
        agreed, local = _run("B2_AGREED", name), _run("B2_LOCAL", name)
        assert agreed.ok and local.ok, name
        assert agreed.max_waste == local.max_waste, name


def test_b2_retract_is_refuted_on_structural_cancel():
    res = _run("B2_RETRACT", "structural-cancel")
    assert res.violations, "retract semantics unexpectedly held — model changed?"
    assert any("I1" in kind for kind, _ in res.violations)


def test_b2_waste_matches_design_prediction():
    # expected run-ahead cost: one wasted forward per request end at B=1
    assert _run("B2_VOID", "clean-3step").max_waste == 1


# --- a follower's input read failing after rank 0 scheduled the row ----------
# Model: tp_read_check_sim.py. It explores every order of the follower's read
# outcome, rank 0's scheduling and the lockstep step.

try:
    from .tp_read_check_sim import explore as explore_read_check
except ImportError:  # collected without package context
    from tp_read_check_sim import explore as explore_read_check


def test_the_read_check_holds_for_every_input_kind():
    """Every rank takes out the rows whose read failed before preparing the
    step: no interleaving hangs, every step runs the same collectives on both
    ranks, and only the request whose read failed fails."""
    for kind in ("data", "shape", "prepare"):
        res = explore_read_check("CHECK", kind)
        assert res.ok, (kind, res.violations[:1])


def test_scheduling_from_rank_0s_reads_alone_hangs_the_instance():
    """A step rank 0 broadcast can need a read that then fails on the
    follower, which can never run it."""
    for kind in ("data", "shape", "prepare"):
        res = explore_read_check("NONE", kind)
        assert any(v == "live" for v, _ in res.violations), kind


def test_zero_fill_desyncs_the_step_when_the_input_sets_its_length():
    """Zeros of the advertised shape keep a data input's step whole, but an
    input the step derives its length from (image_grid_thw) gives the follower
    a different step than rank 0's."""
    assert explore_read_check("ZERO", "data").ok
    assert any(v == "lockstep" for v, _ in explore_read_check("ZERO", "shape").violations)


def test_a_capture_input_row_desyncs_the_step_when_prepare_runs_a_collective():
    """The walk's capture input at rank 0's length fixes the length, but rank 0
    prepared the real row, and a prepare that runs a collective for it (a text
    part's vocab-parallel embedding) has no partner on the follower."""
    assert explore_read_check("CAPTURE", "shape").ok
    assert any(v == "lockstep" for v, _ in explore_read_check("CAPTURE", "prepare").violations)
