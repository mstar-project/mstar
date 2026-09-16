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
