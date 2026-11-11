"""A verifying row's results are applied only at an admit, on every rank at the same step. An offload that frees
pages marks its victim, whose results are all due at its first admit back (`VerifyLedger.mark_offloaded`); a
follower marks it while replaying rank 0's OFFLOAD, in every branch of the replay, so the ranks apply them at the
same admit. An offload that frees nothing ships nothing and marks nothing. These tests check what rank 0 ships,
the follower's mark in each replay branch, and that the ranks' lengths agree at the row's next admit.

They run the toy's synchronous mode; the asynchronous case the mark exists for is in test_spec_resident.py, on the
real resident planner.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spec_toy import NODE, Driver, always_wrong, engine  # noqa: E402

from mstar.engine.resources.base import MoveKind  # noqa: E402
from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.utils.ipc_format import OffloadDelta  # noqa: E402

PROMPTS = {1: [5, 9, 2], 2: [17]}


class _StubTransfer:
    def __init__(self, *args, **kwargs):
        pass

    def get_kv_transfer_info(self, **kwargs):
        pass

    def remove_request(self, request_id):
        pass

    def cleanup(self):
        pass


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    # the toy declares no host pool, so a real `KVManager.offload` answers 0, as a full one does
    monkeypatch.setattr(KVManager, "supports_eviction", property(lambda self: True))


def _rank():
    eng = engine(3, always_wrong)
    # wire ids equal handles here: the two ranks' handles happen to match
    eng._graph_runtime = SimpleNamespace(get_rid_string=str, get_rid_handle=int)
    driver = Driver(eng)
    driver.add(1, PROMPTS[1], 30)
    driver.add(2, PROMPTS[2], 30)
    # row 2's last result is recorded and not yet applied: rank 0 may pick it as a victim
    driver.step([1, 2])
    driver.step([1])
    assert eng._drafts[NODE].ledger.pending(2) and not eng.verify_unrecorded(NODE, 2)
    return eng, driver


def _stored(eng, rid):
    return eng._resources["kv"]._streams[rid]["main"].stored_len


def _moves_to_host(monkeypatch) -> None:
    """A KV offload that frees a page and leaves the row on the host, as a real one with a host pool does."""
    from mstar.engine.engine import Engine
    from mstar.engine.resources.base import Residency

    moved, residency = set(), Engine.residency
    monkeypatch.setattr(KVManager, "offload", lambda self, rid: moved.add(rid) or 1)
    monkeypatch.setattr(Engine, "residency", lambda self, node, rid: (
        Residency.ON_HOST if rid in moved else residency(self, node, rid)))


def _offload_delta(rid: str) -> OffloadDelta:
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, rid, node=NODE)
    return delta


def test_an_offload_that_moves_nothing_ships_nothing_and_leaves_the_results_to_the_next_admit():
    leader, leader_driver = _rank()
    follower, follower_driver = _rank()
    leader._tp_leader_nodes, leader._resident_delta = {NODE}, OffloadDelta.new()
    follower._tp_follower_nodes = {NODE}
    assert leader.offload_request(NODE, 2) == 0  # nothing moved, so row 2 stays and nothing is journaled
    assert not leader.take_resident_delta()
    assert leader._drafts[NODE].ledger.pending(2) and not leader._drafts[NODE].ledger._returning
    assert follower.offload_request(NODE, 2) == 0  # a follower never evicts on its own
    # both apply row 2's results at its next admit
    leader_driver.step([1, 2])
    follower_driver.step([1, 2])
    assert _stored(leader, 2) == _stored(follower, 2)


@pytest.mark.parametrize("branch", ["holds_nothing", "already_off", "offloaded"])
def test_a_followers_replay_of_an_offload_marks_the_row_in_every_branch(monkeypatch, branch):
    """Rank 0 marked its victim; whichever state the follower's copy is in, its results come due at the same
    admit here, or the ranks admit that step from different lengths."""
    from mstar.engine.resources.base import Residency

    follower, _ = _rank()
    follower._tp_follower_nodes = {NODE}
    eng_type = type(follower)
    if branch == "holds_nothing":
        monkeypatch.setattr(eng_type, "reclaimable", lambda self, node, rid, affected_resources=None: 0)
    elif branch == "already_off":
        monkeypatch.setattr(eng_type, "residency", lambda self, node, rid: Residency.ON_HOST)
    else:
        _moves_to_host(monkeypatch)
    delta = _offload_delta("2")
    assert follower.apply_resident_delta(delta, SimpleNamespace()) and not delta
    ledger = follower._drafts[NODE].ledger
    assert 2 in ledger._returning and ledger.pending(2)
    assert ledger.take_due(ledger.next_step, {2}) and 2 not in ledger._returning


def test_a_victim_offloaded_after_its_last_step_comes_due_at_the_same_admit_on_both_ranks(monkeypatch):
    """Offloaded straight after the step that holds it, the row's last result is not due under the
    asynchronous two-step rule at the next commit: only the offload's mark makes it due, rank 0's from
    its own offload and the follower's from the replay, and the two take back the same."""
    _moves_to_host(monkeypatch)

    def rank():
        eng = engine(3, always_wrong)
        eng._graph_runtime = SimpleNamespace(get_rid_string=str, get_rid_handle=int)
        driver = Driver(eng)
        driver.add(1, PROMPTS[1], 30)
        driver.add(2, PROMPTS[2], 30)
        driver.step([1, 2])
        return eng

    leader, follower = rank(), rank()
    leader._tp_leader_nodes, leader._resident_delta = {NODE}, OffloadDelta.new()
    follower._tp_follower_nodes = {NODE}
    assert leader.offload_request(NODE, 2) == 1
    assert follower.apply_resident_delta(leader.take_resident_delta(), SimpleNamespace())
    takes = []
    for eng in (leader, follower):
        ledger = eng._drafts[NODE].ledger
        ledger.synchronous = False
        last = max(ledger._steps_of[2])
        assert last == ledger.next_step - 1, "the row's last step is the node's last commit"
        due = ledger.take_due(ledger.next_step, {2})
        takes.append({key: (take.rids, take.counts, take.label) for key, take in due.items()})
        assert not ledger.pending(2)
    assert takes[0] and takes[0] == takes[1]
