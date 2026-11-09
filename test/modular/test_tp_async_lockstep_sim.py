"""Invariant checks for the TP async-scheduling protocol model.

See tp_async_sim.py for the model. These tests pin the two results that gate
the implementation:

1. B1 (gated commit/cancel) and B2 with void-only Cancel hold I1–I4 across
   every interleaving of the failure matrix, at 2 and 3 ranks.
2. The naive Cancel semantics (retract the spec batch from any rank that has
   not executed it yet — which is what "drop if unbuilt / unwind if built"
   means without a gate) is REFUTED: it desyncs the collective order.
   Kept as a negative control so the constraint survives refactors.
"""

try:
    from .tp_async_sim import SCENARIOS, explore
except ImportError:  # collected without package context
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tp_async_sim import SCENARIOS, explore


def _run(mode, scenario, nranks=2):
    comp0, script, must = SCENARIOS[scenario]
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
    input the step derives its length from (a patch grid) gives the follower
    a different step than rank 0's."""
    assert explore_read_check("ZERO", "data").ok
    assert any(v == "lockstep" for v, _ in explore_read_check("ZERO", "shape").violations)


def test_a_capture_input_row_desyncs_the_step_when_prepare_runs_a_collective():
    """The walk's capture input at rank 0's length fixes the length, but rank 0
    prepared the real row, and a prepare that runs a collective for it (a text
    part's vocab-parallel embedding) has no partner on the follower."""
    assert explore_read_check("CAPTURE", "shape").ok
    assert any(v == "lockstep" for v, _ in explore_read_check("CAPTURE", "prepare").violations)


def test_verify_results_apply_alike_on_every_rank():
    """Speculative decoding's take-back: each rank's own interleaving of commit,
    postprocess and admit, and every rank applies the same results at every
    admit, with every due result recorded, in both modes."""
    from tp_async_sim import explore_verify

    for synchronous in (False, True):
        for nranks in (2, 3):
            res = explore_verify("ledger", synchronous, nranks, steps=4)
            assert res.ok, (synchronous, nranks, res.violations[:2])


def test_applying_whatever_arrived_is_refuted():
    """Negative control: applying every recorded result lets ranks admit from different lengths."""
    from tp_async_sim import explore_verify

    res = explore_verify("arrived", synchronous=False, nranks=2, steps=4)
    assert any(kind == "diverged" for kind, *_ in res.violations)


def test_the_ledger_rests_on_postprocess_preceding_the_admit_two_steps_on():
    """Without that stage-order guarantee a due result can be unrecorded at its admit,
    which the ledger raises on rather than letting ranks diverge."""
    from tp_async_sim import explore_verify

    res = explore_verify("ledger", synchronous=False, nranks=2, steps=4, stage_order=False)
    assert any(kind == "unrecorded-due" for kind, *_ in res.violations)


def test_a_follower_never_replays_an_offload_of_a_row_it_hasnt_recorded():
    """Rank 0 offloads a verifying row it has fully recorded and
    ships the move with its next step; a follower settles the row when it replays
    that move, which raises on an unrecorded result. Over every interleaving the
    follower has recorded every step of the row by then, in both modes."""
    from tp_async_sim import explore_offload

    for synchronous in (False, True):
        for nranks in (2, 3):
            res = explore_offload(synchronous, nranks, steps=5)
            assert res.ok and res.states > 100, (synchronous, nranks, res.violations[:2])
            # the case at issue is reached: an offload riding a speculative head (V skipped the step in flight)
            assert synchronous or res.speculative_offloads > 0


def test_a_follower_can_replay_the_settle_of_a_victim_that_stays_in_the_retry():
    """An offload that moved nothing still ships its settle, and V keeps running: the
    follower has recorded every step of V before it applies that settle, in both modes."""
    from tp_async_sim import explore_offload

    for synchronous in (False, True):
        res = explore_offload(synchronous, 2, steps=5, victim_stays=True)
        assert res.ok and res.states > 100, (synchronous, res.violations[:2])
    # the same serial-path gate is what this rests on
    res = explore_offload(False, 2, steps=5, serial_gate=False, victim_stays=True)
    assert any(kind == "unrecorded-at-replay" for kind, *_ in res.violations)


def test_the_offload_replay_rests_on_the_serial_path_draining_its_steps():
    """Negative control: a follower taking a serial head as early as a speculative
    one can replay the offload of a row from the step it hasn't post-processed."""
    from tp_async_sim import explore_offload

    res = explore_offload(False, 2, steps=5, serial_gate=False)
    assert any(kind == "unrecorded-at-replay" for kind, *_ in res.violations)
