"""Rank 0 settles a victim's verify results before offloading it, and ships the settle when the offload moves
nothing. These tests check that journal: what rank 0 ships, a follower's replay and its wait on a held row, the
merge into Lockstep's pending delta, the wire format, and ``stored_len`` right after replay.

They run the toy's synchronous mode, where a row's lengths re-converge at its next admit anyway, so they don't
guard the divergence at an asynchronous retry's admit: the toy has no resident capture on the CPU.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from spec_toy import NODE, Driver, always_wrong, engine  # noqa: E402

from mstar.communication import wire  # noqa: E402
from mstar.engine.resources.kv import manager as manager_mod  # noqa: E402
from mstar.engine.resources.kv.manager import KVManager  # noqa: E402
from mstar.testing_utils.worker import make_worker  # noqa: E402
from mstar.utils.ipc_format import OffloadDelta, ScheduleTPNode  # noqa: E402

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
    assert eng._drafts[NODE].ledger.pending(2) and not eng.verify_unrecorded(2)
    return eng, driver


def _stored(eng, rid):
    return eng._resources["kv"]._streams[rid]["main"].stored_len


def test_ranks_keep_the_same_length_when_the_victims_offload_moves_nothing():
    leader, _ = _rank()
    follower, _ = _rank()
    leader._resident_delta[NODE] = OffloadDelta.new()
    follower._tp_follower_nodes = {NODE}
    assert _stored(leader, 2) == _stored(follower, 2)

    assert leader.offload_request(NODE, 2) == 0  # nothing moved, so `_evict` falls to hold and row 2 can stay
    assert follower.apply_resident_delta(NODE, leader.take_resident_delta(NODE))
    assert _stored(leader, 2) == _stored(follower, 2)


def test_an_offload_with_nothing_to_settle_ships_nothing():
    leader, _ = _rank()
    leader._resident_delta[NODE] = OffloadDelta.new()
    leader.offload_request(NODE, 2)
    assert len(leader.take_resident_delta(NODE)) == 1
    # applied now, so a second try has nothing to tell the followers
    assert leader.offload_request(NODE, 2) == 0
    assert not leader.take_resident_delta(NODE)


def test_the_follower_waits_for_a_step_that_holds_the_row(monkeypatch):
    """Applied once that step commits, as a refused offload is retried; never dropped."""
    leader, _ = _rank()
    follower, _ = _rank()
    leader._resident_delta[NODE] = OffloadDelta.new()
    leader.offload_request(NODE, 2)
    delta = leader.take_resident_delta(NODE)
    with monkeypatch.context() as held:
        held.setattr(KVManager, "holds_step", lambda self, rid: True)
        assert not follower.apply_resident_delta(NODE, delta) and len(delta) == 1
    assert follower.apply_resident_delta(NODE, delta) and not delta
    assert _stored(leader, 2) == _stored(follower, 2)


def test_a_settled_entry_reaches_a_follower_through_its_pending_delta(monkeypatch):
    """Lockstep merges each head's delta into what the follower still owes, so the entries' kinds have to
    survive the merge, including one arriving while an earlier one waits."""
    leader, _ = _rank()
    follower, _ = _rank()
    leader._resident_delta[NODE] = OffloadDelta.new()
    follower._tp_follower_nodes = {NODE}
    lockstep = make_worker(engine_manager=SimpleNamespace(get_engine=lambda node: follower)).lockstep
    leader.offload_request(NODE, 2)
    first = leader.take_resident_delta(NODE)
    leader.offload_request(NODE, 1)
    second = leader.take_resident_delta(NODE)
    with monkeypatch.context() as held:
        held.setattr(KVManager, "holds_step", lambda self, rid: True)
        assert not lockstep._apply_resident_delta(NODE, first)
        assert not lockstep._apply_resident_delta(NODE, second)
    owed = lockstep._pending_deltas[NODE]
    assert owed.rids == ["2", "1"] and owed.settled_only == [True, True]
    assert lockstep._apply_resident_delta(NODE)
    assert not owed
    for rid in (1, 2):
        assert _stored(leader, rid) == _stored(follower, rid)


def test_the_delta_stays_on_the_typed_wire_path():
    """Sent with every step, so it must not fall back to pickle (AGENTS.md, invariant 6)."""
    delta = OffloadDelta.new()
    delta.add_settled("a")
    delta.add_offloaded("b")
    assert wire._encoder(OffloadDelta).__name__ == "_encode_dataclass"
    head = ScheduleTPNode(node_name=NODE, graph_walk="decode", request_ids=["a"], resident_delta=delta)
    assert wire.decode(wire.encode(head)).resident_delta == delta


def test_a_settled_entry_keeps_its_place_in_the_journal():
    delta = OffloadDelta.new()
    delta.add_offloaded("a")
    delta.add_settled("b")
    delta.add_reloaded("c")
    copied = delta.copy()
    assert copied.settled_only == [False, True, False]
    assert delta.pop_left() == ("a", True)
    assert delta.settled_only == [True, False]
    assert delta.take().rids == ["b", "c"] and not delta
