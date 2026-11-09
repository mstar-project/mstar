"""The resident-set delta that keeps a TP instance's page state symmetric.

Rank 0 owns eviction for the whole instance, so every offload and reload it
makes has to be replayed by the other ranks. Two properties carry that:

* the delta is ORDERED. A set of offloads plus a set of reloads cannot say
  whether a request ended up resident — offload A, reload A, offload A replays
  from sets as "resident" while the rank that recorded it has A on the host.
* applying it is RESUMABLE. An offload is refused while a step still holds the
  pages and a reload waits for room, so a follower stops at the entry that
  refused and picks up there, rather than replaying from the top.

A rank that gets either wrong ends up holding different pages from rank 0, and
the next step they share deadlocks on a collective one of them cannot join.
"""

from __future__ import annotations

import logging
import sys
import threading
from concurrent.futures import Future
from functools import partial
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, ".")

import test_kv_host_index as host
from test_kv_preemption import _step
from test_kv_prefix_lease import NODE, PAGE_SIZE, _ctx, _StubTransfer

from mstar.engine import engine as engine_mod
from mstar.engine.chunking import ChunkingManagement
from mstar.engine.engine import Engine
from mstar.engine.resources import Rebuild, Residency
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import (
    KVReqConfig,
    KVStep,
    PagedKVConfig,
    PrefixSpan,
    RetentionPolicy,
    StreamLayout,
)
from mstar.engine.resources.kv.manager import KVManager
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import FULL_ADMIT_OK, AllocationFailed, Segment, StepContext, SubmoduleStep
from mstar.testing_utils.engine import make_engine, make_mgmt
from mstar.testing_utils.worker import deliver, make_scheduler, make_worker
from mstar.utils.ipc_format import (
    MessageSource,
    Move,
    MoveKind,
    OffloadDelta,
    RemoveRequest,
    ScheduleTPNode,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.utils.profiler import PhaseLaps
from mstar.worker.micro_scheduler import ScheduledBatch
from mstar.worker.rebuild import Rebuilds
from mstar.worker.stream_buffers import StreamBufferManager
from mstar.worker.worker import Worker


class _KeepsIds:
    """A rebuild log that keeps every request's ids and runs no rebuild."""
    nodes = frozenset()

    def kept(self, rid, node):
        return True

    def rebuilding(self, rid, node):
        return False


_rebuilds_none = _KeepsIds()


def _follower_scheduler(engine, rebuilds=None):
    """A follower's real scheduler and ``Lockstep`` over ``engine``, whose requests' handles are their wire ids."""
    return make_scheduler(
        SimpleNamespace(get_engine=lambda node: engine), rebuilds=rebuilds,
        graph_runtime=SimpleNamespace(get_rid_handle=lambda wire_rid: wire_rid),
    )


def _entries(delta: OffloadDelta) -> list[tuple[str, MoveKind]]:
    return list(zip(delta.rids, delta.kinds, strict=True))


# --- the queue itself -------------------------------------------------------


def test_the_order_of_offloads_and_reloads_survives():
    """The reason this is a queue and not two sets."""
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "a")
    delta.add(MoveKind.RELOAD, "a")
    delta.add(MoveKind.OFFLOAD, "a")

    assert _entries(delta) == [("a", MoveKind.OFFLOAD), ("a", MoveKind.RELOAD), ("a", MoveKind.OFFLOAD)]


def test_pop_left_takes_the_oldest_entry():
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "a")
    delta.add(MoveKind.RELOAD, "b")

    assert delta.pop_left() == Move("a", MoveKind.OFFLOAD)
    assert len(delta) == 1, "pop_left returned an entry without removing it"
    assert delta.pop_left() == Move("b", MoveKind.RELOAD)
    assert len(delta) == 0


def test_pop_left_on_an_empty_queue_answers_none():
    """Callers drain until empty, so the empty case has to be an answer rather
    than an IndexError."""
    assert OffloadDelta.new().pop_left() is None


def test_peek_left_does_not_consume():
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "a")

    assert delta.peek_left() == Move("a", MoveKind.OFFLOAD)
    assert len(delta) == 1
    assert OffloadDelta.new().peek_left() is None


def test_take_hands_the_queue_over_and_leaves_it_empty():
    """What a leader does on every broadcast. The fields are a NamedTuple's, so
    this cannot work by reassigning them — only the deques can be mutated."""
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "a")
    delta.add(MoveKind.RELOAD, "b")

    taken = delta.take()

    assert _entries(taken) == [("a", MoveKind.OFFLOAD), ("b", MoveKind.RELOAD)]
    assert len(delta) == 0, "the source kept the entries it handed over"

    delta.add(MoveKind.OFFLOAD, "c")
    assert _entries(taken) == [("a", MoveKind.OFFLOAD), ("b", MoveKind.RELOAD)]


def test_extend_appends_in_order():
    """A follower folds each message's delta onto whatever it still owes."""
    owed = OffloadDelta.new()
    owed.add(MoveKind.OFFLOAD, "a")
    arriving = OffloadDelta.new()
    arriving.add(MoveKind.RELOAD, "b")

    owed.extend(arriving)

    assert _entries(owed) == [("a", MoveKind.OFFLOAD), ("b", MoveKind.RELOAD)]


def test_an_empty_delta_is_falsy():
    """``__len__`` is the queue depth, not the tuple's two fields — several
    callers branch on the delta directly."""
    assert not OffloadDelta.new()
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "a")
    assert delta


# --- the engine's journal and replay ----------------------------------------


class _Resource:
    supports_eviction = True
    supports_preplan = False
    force_double_buffer = False

    def __init__(self):
        self.offloaded: set[str] = set()
        self.holds_nothing: set[str] = set()
        self.refuse_offload = False
        self.refuse_reload = False

    def residency(self, rid):
        return Residency.ON_HOST if rid in self.offloaded else Residency.RESIDENT

    def reclaimable(self, rid):
        return 0 if rid in self.holds_nothing else 1

    def can_reload(self, rid):
        return True

    def offload(self, rid):
        if self.refuse_offload:
            return 0  # a step still holds the pages
        self.offloaded.add(rid)
        return 1

    def reload(self, rid):
        if self.refuse_reload:
            return False  # no room yet
        self.offloaded.discard(rid)
        return True


class _Rids:
    """Wire string <-> worker-local handle, deliberately NOT identity.

    The journal has to hold WIRE strings: a handle is minted per worker, so rank
    0's handle for a request is not rank 1's, and a journal of local handles
    would name the wrong requests on the follower. An identity stub would let
    exactly that through, so this one prefixes -- an assertion on a journal that
    is missing the prefix is a journal holding handles.

    ``get_rid_handle`` answers None for an unknown rid, as the real one does for
    a request this rank never admitted or has already removed.
    """

    WIRE = "wire-"

    def get_rid_string(self, handle) -> str:
        return f"{self.WIRE}{handle}"

    def __init__(self, live=("a", "b", "c")):
        self._live = set(live)

    def get_rid_handle(self, rid: str):
        # A well-formed wire id with no live request behind it is the real case:
        # the rank removed it already. Not a malformed id.
        if not rid.startswith(self.WIRE):
            return None
        handle = rid[len(self.WIRE):]
        return handle if handle in self._live else None


class _Engine(Engine):
    """``Engine``'s eviction surface over one stub resource."""

    def decide_cached_prefix(self, batch):
        return {}

    def page_trails(self, node_name):
        return {}

    def __init__(self, role: str = "leader"):
        """``role`` is this rank's part for ``"node"``: ``leader`` (rank 0 of a
        sharded node — decides and journals), ``follower`` (another rank of one —
        replays only), or ``solo`` (not sharded — owns its own eviction)."""
        assert role in ("leader", "follower", "solo"), role
        super().__init__(graph_runtime=_Rids())
        self.resource = _Resource()
        self._resources = {"kv": self.resource}
        self._submodules["node"] = make_mgmt(resources={"kv": self.resource})
        self._runner = SimpleNamespace(
            admit_retrieve=lambda **kw: FULL_ADMIT_OK, admit_request=lambda *args: None,
            readmit=lambda *args: None,
        )
        self._resident_delta = OffloadDelta.new() if role == "leader" else None
        self._tp_leader_nodes = {"node"} if role == "leader" else set()
        self._tp_follower_nodes = {"node"} if role == "follower" else set()


def _info():
    return SimpleNamespace(graph_walk="walk", resource_publish_info={})


def test_a_follower_may_not_start_an_offload_of_its_own():
    """Rank 0 decides what the instance evicts. A follower choosing for itself is
    the original deadlock: LRU orders on wall clock, so the ranks pick different
    victims and then a step one of them scheduled is one the other cannot join."""
    engine = _Engine("follower")

    assert engine.offload_request("node", "a") == 0
    assert engine.residency("node", "a") is not Residency.ON_HOST
    assert engine.resource.offloaded == set(), "the resource was touched anyway"


def test_a_follower_may_not_start_a_reload_of_its_own():
    """A reload spends the pages an eviction freed, so it is equally rank 0's
    call — and reloading a rid the delta is about to evict would undo it."""
    engine = _Engine("follower")
    engine.resource.offloaded = {"a"}

    assert engine.reload_request("node", "a") is False
    assert engine.residency("node", "a") is Residency.ON_HOST, "reloaded despite following"


def test_replay_is_exempt_from_both_guards():
    """The guards are on *initiating* a move. Replaying rank 0's is the one thing
    a follower must do, so it reaches the resources directly."""
    engine = _Engine("follower")
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "wire-a", node="node")

    assert engine.apply_resident_delta(delta, _rebuilds_none) is True
    assert engine.residency("node", "a") is Residency.ON_HOST

    back = OffloadDelta.new()
    back.add(MoveKind.RELOAD, "wire-a", node="node")
    assert engine.apply_resident_delta(back, _rebuilds_none) is True
    assert engine.residency("node", "a") is not Residency.ON_HOST


def test_an_unsharded_node_still_evicts_for_itself():
    """The guard is per node: a rank that follows one node still owns eviction on
    its own un-sharded ones."""
    engine = _Engine("solo")

    assert engine.offload_request("node", "a") == 1
    assert engine.residency("node", "a") is Residency.ON_HOST
    assert engine.reload_request("node", "a") is True


def test_reload_request_reports_whether_it_worked():
    """``check_ready`` gates admission on this, so a successful reload that
    answers falsy makes the request permanently unschedulable — the whole rank
    then sits idle with work it will not look at."""
    engine = _Engine()
    engine.resource.offloaded = {"a"}

    assert engine.reload_request("node", "a") is True
    assert engine.residency("node", "a") is not Residency.ON_HOST

    engine.resource.offloaded = {"b"}
    engine.resource.refuse_reload = True
    assert engine.reload_request("node", "b") is False


class _Pages(_Resource):
    def __init__(self, free: int, need: int):
        super().__init__()
        self.free = free
        self.need = need

    def offload(self, rid):
        if rid in self.offloaded:
            return 0
        self.offloaded.add(rid)
        self.free += self.need
        return self.need

    def reload(self, rid):
        if self.need > self.free:
            return False
        self.free -= self.need
        self.offloaded.discard(rid)
        return True


def test_a_partial_reload_gives_back_what_it_reloaded():
    """Resources reload in turn, so the first can fit while the second can't, as
    whisper's ``kv_cache`` does ahead of the tighter ``cross_kv_cache``. The delta
    carries only a full reload and a follower never reloads by itself, so pages
    the first kept would be pages no follower holds, and the next admit short of
    them refuses on rank 0 alone."""
    engine = _Engine()
    first, second = _Pages(free=8, need=4), _Pages(free=0, need=12)
    first.offloaded = {"a"}
    second.offloaded = {"a"}
    engine._submodules = {
        "node": make_mgmt(resources={"first": first, "second": second})
    }

    assert engine.reload_request("node", "a") is False
    assert len(engine.take_resident_delta()) == 0
    assert "a" in second.offloaded
    assert "a" in first.offloaded and first.free == 8, (
        "the first resource kept pages the delta never mentions"
    )


class _Unsure(_Resource):
    """Answers ``can_reload`` as ``KVManager`` does: no for a request it never offloaded."""

    def __init__(self, fits: bool = True):
        super().__init__()
        self.fits = fits

    def can_reload(self, rid):
        return rid in self.offloaded and self.fits


def test_a_resource_that_never_offloaded_the_request_does_not_block_its_reload():
    """An offload that found nothing to move on one resource leaves it out of the
    reload too; asking it anyway refuses every retry and strands the request."""
    engine = _Engine()
    holder, bystander = _Unsure(), _Unsure()
    holder.offloaded = {"a"}
    engine._submodules = {
        "node": make_mgmt(resources={"holder": holder, "bystander": bystander})
    }

    assert engine.reload_request("node", "a") is True
    assert engine.residency("node", "a") is not Residency.ON_HOST


def test_a_resource_that_cannot_fit_stops_the_reload_before_anything_moves():
    engine = _Engine()
    first, second = _Unsure(), _Unsure(fits=False)
    first.offloaded = {"a"}
    second.offloaded = {"a"}
    first.reload = second.reload = lambda rid: pytest.fail("reloaded past a refusal")
    engine._submodules = {
        "node": make_mgmt(resources={"first": first, "second": second})
    }

    assert engine.reload_request("node", "a") is False
    assert len(engine.take_resident_delta()) == 0


def test_a_reloaded_request_is_ready_again():
    """The same bug one level up, where it actually bites."""
    engine = _Engine()
    engine.resource.offloaded = {"a"}

    outcome = engine.check_ready("node", "a", _info(), allow_reload=True)

    assert outcome.ok and outcome.ready


def test_both_directions_are_journalled_in_order():
    engine = _Engine()
    engine.offload_request("node", "a")
    engine.reload_request("node", "a")
    engine.offload_request("node", "b")

    assert _entries(engine.take_resident_delta()) == [
        ("wire-a", MoveKind.OFFLOAD), ("wire-a", MoveKind.RELOAD), ("wire-b", MoveKind.OFFLOAD),
    ]
    assert len(engine.take_resident_delta()) == 0, "drained once only"


def test_a_rank_that_leads_nothing_journals_nothing():
    """Only a leader replicates. An un-drained journal grows forever, which is
    what happens if the leader set is built with ``world_size > 0`` — true of
    every node — rather than ``> 1``."""
    engine = _Engine("solo")
    engine.offload_request("node", "a")

    assert engine.residency("node", "a") is Residency.ON_HOST, "a solo rank still evicts"
    assert len(engine.take_resident_delta()) == 0


def test_replaying_a_delta_reproduces_the_leaders_state():
    engine = _Engine("follower")
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "wire-a", node="node")
    delta.add(MoveKind.RELOAD, "wire-a", node="node")
    delta.add(MoveKind.OFFLOAD, "wire-a", node="node")

    assert engine.apply_resident_delta(delta, _rebuilds_none) is True
    assert engine.residency("node", "a") is Residency.ON_HOST, (
        "replay ended with 'a' resident; the leader has it on the host"
    )
    assert len(delta) == 0


def test_a_refused_entry_stops_the_replay_where_it_is():
    """Resumable, not restartable: the entries already applied are gone, so the
    retry does not redo them."""
    engine = _Engine("follower")
    engine.resource.refuse_offload = True
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "wire-a", node="node")
    delta.add(MoveKind.OFFLOAD, "wire-b", node="node")

    assert engine.apply_resident_delta(delta, _rebuilds_none) is False
    assert len(delta) == 2, "a refused entry must stay queued"

    engine.resource.refuse_offload = False
    assert engine.apply_resident_delta(delta, _rebuilds_none) is True
    assert engine.residency("node", "a") is Residency.ON_HOST and engine.residency("node", "b") is Residency.ON_HOST


def test_a_move_for_a_request_this_rank_has_not_taken_in_waits_for_it():
    """Nothing behind the move jumps ahead while it waits."""
    engine = _Engine("follower")
    engine._graph_runtime._live.discard("a")
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "wire-a", node="node")
    delta.add(MoveKind.OFFLOAD, "wire-b", node="node")

    assert engine.apply_resident_delta(delta, _rebuilds_none) is False
    assert len(delta) == 2 and engine.residency("node", "b") is not Residency.ON_HOST, (
        "a move ran ahead of one still owed"
    )

    engine._graph_runtime._live.add("a")
    assert engine.apply_resident_delta(delta, _rebuilds_none) is True
    assert engine.residency("node", "a") is Residency.ON_HOST, "the move was lost while it waited"


@pytest.mark.parametrize("kind", [kind for kind in MoveKind if kind is not MoveKind.OFFLOAD])
def test_a_move_of_any_kind_waits_for_a_request_this_rank_has_not_taken_in(kind):
    engine = _Engine("follower")
    engine._graph_runtime._live.discard("a")
    delta = OffloadDelta.new()
    delta.add(kind, "wire-a", node="node")
    delta.add(MoveKind.OFFLOAD, "wire-b", node="node")

    assert engine.apply_resident_delta(delta, _rebuilds_none) is False
    assert len(delta) == 2 and engine.residency("node", "b") is not Residency.ON_HOST, (
        "a move ran ahead of one still owed"
    )


def test_a_request_holding_nothing_is_not_a_refusal():
    """``offload`` answers 0 both for "a step still holds these pages, retry" and
    for "there is nothing here to free". Only the first is retryable; treating
    the second as a refusal wedges the follower on a delta it can never finish."""
    engine = _Engine("follower")
    engine.resource.holds_nothing = {"a"}
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "wire-a", node="node")

    assert engine.apply_resident_delta(delta, _rebuilds_none) is True
    assert len(delta) == 0


def test_replaying_does_not_journal_it_onward():
    """A follower replaying rank 0 has nothing of its own to replicate; if it
    journalled, a rank that both leads and follows would echo."""
    engine = _Engine("leader")  # journalling is on
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "wire-a", node="node")

    engine.apply_resident_delta(delta, _rebuilds_none)

    assert len(engine.take_resident_delta()) == 0


# --- the replay says what it did --------------------------------------------


def test_each_replayed_move_is_logged(caplog):
    """So a run's two ranks can be lined up: the leader logs its eviction and its
    reload, and the follower logs replaying each of them.

    The WIRE id, not the local handle -- handles are minted per worker, so the
    two ranks' lines could not be matched up if they named handles."""
    engine = _Engine("follower")
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "wire-a", node="node")
    delta.add(MoveKind.RELOAD, "wire-a", node="node")

    with caplog.at_level(logging.INFO, logger=engine_mod.__name__):
        assert engine.apply_resident_delta(delta, _rebuilds_none) is True

    lines = [r.getMessage() for r in caplog.records]
    assert any("Replayed rank 0's offload of request wire-a from node" in ln for ln in lines), lines
    assert any("Replayed rank 0's reload of request wire-a on node" in ln for ln in lines), lines


def test_a_no_op_replay_is_not_logged_as_a_move(caplog):
    """An entry this rank already matches moved nothing, so it must not read as
    a move — that would make the two ranks' logs disagree."""
    engine = _Engine("follower")
    engine.resource.offloaded = {"a"}
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "wire-a", node="node")  # already off here

    with caplog.at_level(logging.INFO, logger=engine_mod.__name__):
        assert engine.apply_resident_delta(delta, _rebuilds_none) is True

    assert not [r for r in caplog.records if "Replayed" in r.getMessage()]


def test_a_leaders_reload_is_logged_too(caplog):
    """Its eviction is logged by the worker; without this its reload is the one
    resident-set move with no line anywhere, and reloads are what undid
    evictions in every hang so far."""
    engine = _Engine("leader")
    engine.resource.offloaded = {"a"}

    with caplog.at_level(logging.INFO, logger=engine_mod.__name__):
        assert engine.reload_request("node", "a") is True

    assert any(
        "Reloaded request wire-a on node" in r.getMessage()
        for r in caplog.records
    ), [r.getMessage() for r in caplog.records]


# --- which nodes this rank leads, follows, or owns alone --------------------


class _Groups:
    """``WorkerParallelGroups``' two questions, from a {node: (rank, size)} map."""

    def __init__(self, layout):
        self._layout = layout

    def get_instance_rank_for_node(self, node):
        return self._layout[node][0]

    def get_instance_world_size_for_node(self, node):
        return self._layout[node][1]


def _roles(layout):
    engine = make_engine()
    engine._classify_tp_roles(set(layout), _Groups(layout))
    return engine


def test_an_unsharded_node_is_neither_leader_nor_follower():
    """``world_size > 0`` is true of every node, so it would make this one a
    leader — and a leader journals resident-set moves that nothing drains,
    because the TP broadcast that drains them skips un-sharded nodes."""
    engine = _roles({"solo": (0, 1)})

    assert engine._tp_leader_nodes == set()
    assert engine._tp_follower_nodes == set()
    assert engine._resident_delta is None, "an un-drained journal grows forever"


def test_rank_zero_of_a_sharded_node_leads_and_journals():
    engine = _roles({"shared": (0, 2)})

    assert engine._tp_leader_nodes == {"shared"}
    assert engine._tp_follower_nodes == set()
    assert engine._resident_delta is not None


def test_another_rank_of_a_sharded_node_follows_and_does_not_journal():
    engine = _roles({"shared": (1, 2)})

    assert engine._tp_leader_nodes == set()
    assert engine._tp_follower_nodes == {"shared"}
    assert engine._resident_delta is None


def test_the_roles_are_per_node():
    """One rank can lead one node, follow another, and own a third alone."""
    engine = _roles({"lead": (0, 2), "follow": (1, 2), "solo": (0, 1)})

    assert engine._tp_leader_nodes == {"lead"}
    assert engine._tp_follower_nodes == {"follow"}
    assert engine._resident_delta is not None


# --- the wiring: the leader puts its moves on the wire ----------------------
#
# Both halves have to land together. The engine refuses a follower's own
# eviction, so if nothing supplies the replacement the follower simply never
# matches rank 0 — its admit keeps failing while rank 0 proceeds into the
# collective, and the run hangs with the follower idle.


def _leader(followers=("rank1", "rank2")) -> Worker:
    """Rank 0's real ``Lockstep``, over a stub engine and communicator."""
    engine = _Engine("leader")
    sent: list = []
    # the broadcast reads it to say what its page state should add up to
    request_state = SimpleNamespace(per_request_info={"r0": object()})
    leader = make_worker(
        engine_manager=SimpleNamespace(get_engine=lambda node: engine),
        parallel_nodes={"node"}, parallel_leader_nodes={"node"},
        # Its ``get_rid_string`` is the same prefixing stub ``_Engine`` uses, so a
        # broadcast that shipped local handles instead of wire ids is visible.
        graph_runtime=SimpleNamespace(get_rid_string=_Rids().get_rid_string),
        communicator=SimpleNamespace(send=lambda worker, msg: sent.append((worker, msg))),
        request_state=request_state, streams=StreamBufferManager(request_state, None, []),
    )
    leader.lockstep._followers = {"node": list(followers)}
    leader.engine, leader.sent = engine, sent
    return leader


def _node_batch():
    return SimpleNamespace(
        node_name="node", graph_walk="walk", request_ids=("r0",), rebuild_rids=frozenset(), chunk_ranges={},
        step_context=StepContext(
            request_ids=("r0",), graph_walk="walk", slot=0, capture=False,
        ),
    )


def test_the_broadcast_carries_the_leaders_moves():
    """Without this the delta is always empty and no follower ever replays."""
    leader = _leader(followers=("rank1",))
    leader.engine.offload_request("node", "victim")

    leader.lockstep.broadcast(_node_batch())

    assert len(leader.sent) == 1
    body = leader.sent[0][1].body
    # wire ids, not local handles: the follower mints its own, so a broadcast
    # naming rank 0's would replay against the wrong requests
    assert _entries(body.resident_delta) == [("wire-victim", MoveKind.OFFLOAD)]
    assert body.request_ids == ["wire-r0"]


def test_every_follower_gets_the_same_moves_in_its_own_queue():
    """Drained once, copied per follower. Draining inside the send loop would
    give the first follower everything and leave the rest behind forever."""
    leader = _leader(followers=("rank1", "rank2"))
    leader.engine.offload_request("node", "victim")

    leader.lockstep.broadcast(_node_batch())

    deltas = [msg.body.resident_delta for _, msg in leader.sent]
    assert len(deltas) == 2
    assert all(_entries(d) == [("wire-victim", MoveKind.OFFLOAD)] for d in deltas), (
        "a follower was sent an empty delta"
    )
    # and one replaying does not drain another's
    deltas[0].pop_left()
    assert len(deltas[1]) == 1


def _second_node(engine) -> _Resource:
    """Give ``engine`` a node ``other`` in the same role as ``node``, with a resource of its own."""
    resource = _Resource()
    engine._resources["other_kv"] = resource
    engine._submodules["other"] = SimpleNamespace(resources={"other_kv": resource}, resident=None)
    for nodes in (engine._tp_leader_nodes, engine._tp_follower_nodes):
        if "node" in nodes:
            nodes.add("other")
    return resource


def test_a_step_of_one_node_carries_rank_0s_moves_on_another():
    """A worker's lockstep nodes share one instance, so rank 0 journals their moves in one stream. A move on a node
    with no step to send reaches the followers with another node's step, and each follower replays it on the node
    it names."""
    leader, follower = _Engine("leader"), _Engine("follower")
    other = [_second_node(engine) for engine in (leader, follower)][1]
    leader.offload_request("other", "a")
    leader.offload_request("node", "b")

    assert follower.apply_resident_delta(leader.take_resident_delta(), _rebuilds_none)

    assert other.offloaded == {"a"} and follower.resource.offloaded == {"b"}, "a move landed on another node"


def test_the_journal_is_drained_by_the_broadcast():
    leader = _leader(followers=("rank1",))
    leader.engine.offload_request("node", "victim")

    leader.lockstep.broadcast(_node_batch())
    leader.sent.clear()
    leader.lockstep.broadcast(_node_batch())

    assert len(leader.sent[0][1].body.resident_delta) == 0, (
        "the same move went out twice"
    )


def test_a_pass_with_no_step_of_the_instance_sends_rank_0s_moves_in_a_step_with_no_rows():
    """The step goes through the worker's own batch builder and broadcast, so a flush that can't build its step
    fails here."""
    leader, follower = _leader(followers=("rank1",)), _Engine("follower")
    leader.lockstep.broadcast(_node_batch())
    leader.sent.clear()
    leader.engine.offload_request("node", "a")

    leader.lockstep.flush_moves()

    [(worker, msg)] = leader.sent
    assert (worker, msg.body.node_name, msg.body.request_ids) == ("rank1", "node", [])
    assert follower.apply_resident_delta(msg.body.resident_delta, _rebuilds_none)
    assert follower.resource.offloaded == {"a"}, "the follower never got rank 0's offload"


def test_rank_0_sends_its_moves_to_the_instance_it_leads_before_its_first_step():
    """A request can fail before rank 0 sends any step of the instance, and teardowns wait while the journal holds
    moves. Rank 0 knows its followers from the deployment, so the moves go out at once rather than with the next
    request's first step."""
    leader = _leader()
    leader.lockstep._worker_id = "rank0"
    instances = {0: [["rank0"]], 1: [["rank2", "rank3"], ["rank0", "rank1"]]}
    leader.lockstep._followers = leader.lockstep._instance_followers(
        instances, SimpleNamespace(all_wg_nodes={0: {"encoder"}, 1: {"node"}}),
    )
    leader.engine.offload_request("node", "a")

    leader.lockstep.flush_moves()

    assert [(worker, msg.body.request_ids) for worker, msg in leader.sent] == [("rank1", [])], (
        "rank 0 held its moves until a step of the instance went out"
    )


# --- the wiring: a follower does not run rank 0's OOM recovery --------------


class _Queue:
    """Stands in for the graph runtime, which owns the ready queues."""

    def __init__(self):
        self.pushed_back: list[str] = []

    def push_back_node(self, node_name, rids, wg_ids):
        del node_name, wg_ids
        self.pushed_back.extend(rids)


def _rank(*, leader: bool, parallel: bool = True) -> Worker:
    """One rank of ``"node"``, the real refusal path over a page-holding "victim"
    on a stub engine. Its teardown flushes are counted: a teardown releases
    pages, so spending one before evicting is the cheaper move and the order
    matters."""
    engine = SimpleNamespace(offload_attempts=[])
    engine.evictable = lambda node, resources=None: True
    engine.residency = lambda node, rid: Residency.RESIDENT
    engine.reclaimable = lambda node, rid, resources: True
    engine.can_offload = lambda node, rid: True
    engine.offload_request = lambda node, rid, affected_resources=None: engine.offload_attempts.append(node) or 1
    engine.owes_moves = lambda: False  # rank 0's journal holds no moves, so its teardowns go
    rank = make_worker(
        graph_runtime=_Queue(),
        engine_manager=SimpleNamespace(get_engine=lambda node: engine),
        parallel_nodes={"node"} if parallel else set(),
        parallel_leader_nodes={"node"} if leader else set(),
        is_tp_follower=parallel and not leader,
        scheduler=SimpleNamespace(
            last_in_line=lambda node, walk, rids: next(iter(rids)),
            could_refill=lambda node, walk, rids: False,
        ),
    )
    rank.engine = engine
    rank.flushed_removes = 0
    rank.flush_protected = None
    rank.lifecycle.add("victim", "victim")
    teardown = rank.pipeline._teardown

    def flush(in_flight_rids):
        rank.flushed_removes += 1
        assert engine.offload_attempts == [], "a teardown was spent after the eviction"
        rank.flush_protected = set(in_flight_rids)
        teardown(in_flight_rids)
    rank.pipeline._teardown = flush
    return rank


def _oom(rank, referenced_rids=frozenset()):
    batch = ScheduledBatch(
        node_name="node", graph_walk="walk",
        request_to_worker_graph={"r0": "wg", "r1": "wg"},
    )
    rank.pipeline._handle_admit_failure(batch, SimpleNamespace(
        node_name="node", failed_resource=None,
        admit_error=AllocationFailed("out of pages", pages_short=1, label="main", request_id="r0"),
        request_ids=["r0", "r1"], cached_rids=set(),
    ), referenced_rids)


def test_a_follow_rank_neither_evicts_nor_holds():
    """What the log showed it doing instead: "no offload possible, holding 15
    requests". The eviction is rank 0's and arrives as a delta, and a follow rank
    has no scheduling decision a backoff could improve — the hold only delays
    building the ScheduleTPNode that carries the fix."""
    follower = _rank(leader=False)

    _oom(follower)

    assert follower.engine.offload_attempts == [], "a follow rank picked its own victim"
    assert follower.admission.held_until == {}
    # the batch still has to go back, or rank 0's retry finds nothing ready
    assert sorted(follower._graph_runtime.pushed_back) == ["r0", "r1"]


def test_rank_zero_still_evicts_and_holds_its_victim():
    leader = _rank(leader=True)

    _oom(leader)

    assert leader.engine.offload_attempts == ["node"]
    assert leader.flushed_removes == 1, (
        "a pending teardown releases pages, so it has to be spent before an "
        "eviction is chosen — otherwise a page move happens that need not"
    )
    assert list(leader.admission.held_until) == ["victim"]
    assert sorted(leader._graph_runtime.pushed_back) == ["r0", "r1"]


def test_an_unsharded_node_still_runs_its_own_recovery():
    solo = _rank(leader=False, parallel=False)  # "node" is not sharded here

    _oom(solo)

    assert solo.engine.offload_attempts == ["node"]
    assert list(solo.admission.held_until) == ["victim"]


# --- the replay is verbatim, and says where it stopped ----------------------


def test_the_broadcast_ships_the_sequence_verbatim():
    """The property this protects: every move the follower replays is one the
    leader executed, so a move it cannot apply proves the states differ. A
    collapsed sequence was never executed by anyone, so a refusal could be an
    artifact of the rewrite instead."""
    engine = _Engine("leader")
    engine.offload_request("node", "a")
    engine.resource.offloaded = {"a"}
    engine.reload_request("node", "a")

    assert _entries(engine.take_resident_delta()) == [
        ("wire-a", MoveKind.OFFLOAD), ("wire-a", MoveKind.RELOAD),
    ]
def test_a_refused_replay_says_which_move_it_stopped_on(caplog):
    """A follower that stops mid-delta and never finishes leaves rank 0 on a
    collective it will never join — and until now that was silent."""
    engine = _Engine("follower")
    engine.resource.refuse_reload = True
    engine.resource.offloaded = {"a"}
    delta = OffloadDelta.new()
    delta.add(MoveKind.RELOAD, "wire-a", node="node")

    with caplog.at_level(logging.WARNING, logger=engine_mod.__name__):
        assert engine.apply_resident_delta(delta, _rebuilds_none) is False

    line = next(r.getMessage() for r in caplog.records if "does not fit" in r.getMessage())
    assert "reload of wire-a on node" in line
    assert "1 moves still owed" in line


# --- neither consumer of the follow FIFO builds mid-replay -------------------
#
# A step admitted against a half-replayed page state is one the other rank
# admits differently, which deadlocks the next collective. Both the serial
# path and the async follower go through ``Lockstep.settle_delta``.


def test_the_serial_follow_path_will_not_build_mid_replay():
    """A refused replay leaves moves owed, so the rids the head names do not mean
    here what they meant on the leader. Wait rather than build."""
    engine = SimpleNamespace(
        apply_resident_delta=lambda delta, rebuilding, caught_up: False,  # refuses
        residency=lambda node, rid: Residency.RESIDENT,
        check_ready=lambda *a, **kw: FULL_ADMIT_OK,
        get_max_batch_size=lambda node, walk: None,
    )
    sched = make_scheduler(
        engine_manager=SimpleNamespace(get_engine=lambda node: engine),
        parallel_leader_nodes=set(),
    )
    message = ScheduleTPNode(
        node_name="node", graph_walk="walk", request_ids=["r0"], spec_seq=5,
    )
    message.resident_delta.add(MoveKind.OFFLOAD, "wire-victim", node="node")
    sched.lockstep.enqueue_follow(message)

    assert sched._try_schedule_tp_follow() is None
    assert sched.lockstep.follow_head() is message, "the head was consumed mid-replay"


# --- a forwarded teardown is ordered against the step stream -----------------
#
# A teardown releases pages, and the resident-set delta cannot describe it: the
# request is not offloaded, it is gone. So it carries the step rank 0 released it
# after, and a follower holds it until it has consumed that step.


def _consume(rank, seq: int) -> None:
    """``rank`` takes leader step ``seq`` off its FIFO."""
    rank.lockstep.enqueue_follow(ScheduleTPNode(
        node_name="node", graph_walk="walk", request_ids=["step"], spec_seq=seq,
    ))
    rank.lockstep.pop_follow_head()


def _remove_rank(consumed: int, *, follower: bool = True) -> Worker:
    """A rank whose lifecycle orders a forwarded removal against the step stream."""
    rank = make_worker(engine_manager=SimpleNamespace(), is_tp_follower=follower)
    if consumed >= 0:
        _consume(rank, consumed)
    return rank


def _forwarded(rank: Worker, rid: int, after_tp_seq: int) -> None:
    rank.lifecycle.remove(rid, RemoveRequest(
        request_id=f"wire-{rid}", source=MessageSource.TP_RANK_0, after_tp_seq=after_tp_seq,
    ), rank.lockstep.last_consumed_seq)


def _due(rank: Worker) -> list[int]:
    due = rank.lifecycle.removes_ready(set(), rank.lockstep.last_consumed_seq)
    for rid in due:
        rank.lifecycle.removed(rid, f"wire-{rid}")
    return due


def test_a_removal_for_a_step_not_yet_reached_waits():
    rank = _remove_rank(consumed=5)
    _forwarded(rank, 1, after_tp_seq=7)

    assert _due(rank) == []
    assert rank.lifecycle._removes_awaiting_step == {7: [1]}


def test_a_removal_for_a_step_already_consumed_applies():
    rank = _remove_rank(consumed=7)
    _forwarded(rank, 1, after_tp_seq=7)

    assert _due(rank) == [1]


def test_an_unstamped_removal_is_not_ordered():
    """The conductor's own sends, which only a rank that leads takes."""
    rank = _remove_rank(consumed=-1, follower=False)
    rank.lifecycle.remove(1, RemoveRequest(request_id="r"), -1)

    assert _due(rank) == [1]


def test_a_held_removal_lands_once_its_step_does():
    rank = _remove_rank(consumed=5)
    _forwarded(rank, 1, after_tp_seq=7)
    _forwarded(rank, 2, after_tp_seq=4)

    assert _due(rank) == [2], "applied a teardown from a step not reached"
    assert rank.lifecycle._removes_awaiting_step == {7: [1]}

    _consume(rank, 7)

    assert _due(rank) == [1]
    assert rank.lifecycle._removes_awaiting_step == {}


def test_held_removals_land_in_step_order():
    """Two teardowns from different gaps have to be applied in the order rank 0
    made them, for the same reason the delta is a queue."""
    rank = _remove_rank(consumed=9)
    _forwarded(rank, 1, after_tp_seq=8)
    _forwarded(rank, 2, after_tp_seq=3)

    assert _due(rank) == [2, 1]


def _forwarding_leader(broadcast_seq: int) -> Worker:
    """Rank 0's real removal, over stubs, to pin what it forwards."""
    group = SimpleNamespace(tp_size=2, _tp_rank=0, _workers=["rank0", "rank1"])
    sent: list = []
    request_state = SimpleNamespace(remove_request=lambda rid: None)
    tensor_manager = SimpleNamespace(force_cleanup_request=lambda rid: None)
    leader = make_worker(
        # ``remove_request`` here is what frees the handle
        graph_runtime=SimpleNamespace(
            get_rid_handle=lambda rid: 0 if rid == "gone" else None,
            get_rid_string=lambda handle: "gone" if handle == 0 else None,
            get_sharding_config=lambda rid: SimpleNamespace(groups=[group]),
            remove_request=lambda rid: None,
        ),
        request_state=request_state,
        engine_manager=SimpleNamespace(remove_request=lambda rid: None),
        tensor_manager=tensor_manager,
        streams=StreamBufferManager(request_state, tensor_manager, []),
        profile_info=SimpleNamespace(pop_request=lambda rid: None),
        scheduler=SimpleNamespace(clear_rid=lambda rid: None),
        communicator=SimpleNamespace(send=lambda worker, msg: sent.append((worker, msg))),
    )
    leader.lockstep._broadcast_seq = broadcast_seq
    leader.sent = sent
    leader.lifecycle.add(0, "gone")
    return leader


def _remove(rank: Worker, body: RemoveRequest) -> None:
    deliver(rank, WorkerMessage(WorkerMessageType.REMOVE_REQUEST, body))


def test_the_forwarded_teardown_carries_the_step_it_follows():
    """Without the stamp a follower has nothing to order the teardown against,
    and applies it on whichever side of the step its loop happens to be on."""
    leader = _forwarding_leader(broadcast_seq=1229)

    _remove(leader, RemoveRequest(request_id="gone"))

    assert [w for w, _ in leader.sent] == ["rank1"]
    body = leader.sent[0][1].body
    assert body.request_id == "gone"
    assert body.after_tp_seq == 1228, (
        "the last step broadcast, which is the gap this rank released the pages in"
    )


def test_a_rank_that_leads_never_parks_a_teardown():
    """``Lockstep.last_consumed_seq`` only advances on a rank that consumes follow
    steps, so a leader that parked a stamped teardown would hold it for good —
    and hold its pages with it."""
    leader = _forwarding_leader(broadcast_seq=1229)

    # stamped, as if it had somehow arrived ordered
    _remove(leader, RemoveRequest(request_id="gone", after_tp_seq=9999))

    assert leader.lifecycle._removes_awaiting_step == {}, "a leader parked a teardown"
    assert [w for w, _ in leader.sent] == ["rank1"], "and it still forwarded"


def test_the_teardown_flush_spares_only_what_the_speculation_touches():
    """The failed step's own pages are fair game — its admit refused, so no
    forward ran. What is not is whatever the speculation touches: its pre-plan
    may have pre-admitted, marking those streams in flight, and the speculation
    is not dropped until after this returns."""
    leader = _rank(leader=True)

    _oom(leader, referenced_rids=frozenset({"spec_rid"}))

    assert leader.flush_protected == {"spec_rid"}, (
        "the flush protected the failed step's rids, whose pages are reclaimable"
    )


def test_the_teardown_flush_runs_after_the_push_back():
    """A rid torn down by the flush must not then be pushed onto queues that the
    teardown has just dismantled."""
    leader = _rank(leader=True)
    order: list[str] = []
    leader._graph_runtime.push_back_node = (
        lambda node_name, rids, wg_ids: order.append("push")
    )
    leader.pipeline._teardown = lambda in_flight: order.append("flush")

    _oom(leader)

    assert order.index("push") < order.index("flush"), order


def test_the_speculation_is_cleared_before_the_recovery_runs():
    """Ordering, and it buys two things. Dropping the pre-plan releases the pages
    it reserved, so the eviction that follows may be unnecessary; and it unmarks
    those streams, so the teardown flush has nothing left to spare. Only a
    yield-away batch survives the clear, and only that needs sparing."""
    rank = _rank(leader=True)
    pipeline = rank.pipeline
    order: list[str] = []

    def clear_speculation():
        order.append("clear")
        pipeline._speculation = None

    pipeline._clear_speculation = clear_speculation
    pipeline._handle_admit_failure = lambda *a, **kw: order.append("recover")
    pipeline._speculation = SimpleNamespace(is_yield_away=False)
    rank._graph_runtime.set_in_flight = lambda node, rids, wg_ids, value: None
    done = Future()
    done.set_result({})
    pipeline.pending = SimpleNamespace(
        node_name="node", graph_walk="walk", future=done, speculative_new_iter=False,
        batch=ScheduledBatch(node_name="node", graph_walk="walk", request_to_worker_graph={"r0": "wg"}),
        node_batch=SimpleNamespace(
            admit_error=AllocationFailed("out of pages", pages_short=1, label="main", request_id="r0"),
            failed_requests={},
        ),
    )

    pipeline._finish_chain_step(PhaseLaps("test."))

    assert order == ["clear", "recover"], (
        "recovery runs before the speculation is cleared, so it evicts to make "
        "room the pre-plan was about to give back"
    )


def test_a_leader_with_no_next_step_tears_down_what_waited_behind_the_last():
    """A remove for a row that only a side step held waits through the preamble.
    With nothing speculated after N, it lands before the next step is built,
    whose admit may need its pages; otherwise the admit can be refused and hold
    a row."""
    rank = _rank(leader=True)
    pipeline = rank.pipeline
    torn: list[str] = []
    pipeline._tear_down_request = torn.append
    pipeline._handle_admit_failure = lambda *a, **kw: None
    rank._graph_runtime.set_in_flight = lambda node, rids, wg_ids, value: None
    rank.lifecycle.add("side", "side")
    rank.lifecycle.remove("side", RemoveRequest(request_id="side", source=MessageSource.CONDUCTOR), -1)
    done = Future()
    done.set_result({})
    pipeline.pending = SimpleNamespace(
        node_name="node", graph_walk="walk", future=done, speculative_new_iter=False,
        batch=ScheduledBatch(node_name="node", graph_walk="walk", request_to_worker_graph={"r0": "wg"}),
        node_batch=SimpleNamespace(
            admit_error=AllocationFailed("out of pages", pages_short=1, label="main", request_id="r0"),
            failed_requests={},
        ),
    )

    pipeline._finish_chain_step(PhaseLaps("test."))

    assert torn == ["side"], "the next step would be admitted with the removed row's pages still held"


def _follow_spec(*, replay_refuses: bool) -> Worker:
    """The real ``Speculator.follow`` and ``MicroScheduler``, as far
    as the delta gate, so the gate is driven by a real refusal rather than a
    stubbed return: the bug was the async path never asking, and a stub that
    answers proves nothing about whether it asked.
    """
    engine = SimpleNamespace(
        apply_resident_delta=lambda delta, rebuilding, caught_up: not replay_refuses,
        residency=lambda node, rid: Residency.RESIDENT,
        reclaimable=lambda node, rid: 0,
        check_ready=lambda *a, **kw: FULL_ADMIT_OK,
        get_max_batch_size=lambda node, walk: None,
    )
    engine_manager = SimpleNamespace(get_engine=lambda node: engine)
    # the build asks for the spec target right after reading the parent's
    # batch; None bails it out there, just past the gate
    runtime = SimpleNamespace(get_rid_handle=lambda rid: rid, get_spec_target=lambda *a, **kw: None)
    request_state = SimpleNamespace(per_request_info={})
    sched = make_scheduler(
        engine_manager, parallel_leader_nodes=set(), graph_runtime=runtime, request_state=request_state,
    )
    worker = make_worker(
        worker_id="worker_1", engine_manager=engine_manager, scheduler=sched, lockstep=sched.lockstep,
        admission=sched.admission, graph_runtime=runtime, request_state=request_state,
    )
    worker.head = head = ScheduleTPNode(
        node_name="node", graph_walk="walk", request_ids=["r0"],
        speculative=True, spec_seq=6, spec_from_seq=5,
    )
    head.resident_delta.add(MoveKind.OFFLOAD, "victim")
    worker.lockstep.enqueue_follow(head)
    return worker


class _SpecParent:
    """Step N on the follower: matches the head, and records whether anything
    read its batch.

    The read is the observable. Everything past the gate needs the batch and
    nothing before it does, so "was the parent touched" says which side of the
    gate we stopped on -- without depending on which bail comes next, which is
    what went stale when the refactor replaced the outputs check.
    """

    node_name = "node"
    graph_walk = "walk"
    tp_seq = 5

    def __init__(self):
        self.reads = 0
        parent = self

        class _Batch:
            node_name = "node"
            incomplete_node_rids = frozenset()

            @property
            def request_to_worker_graph(self):
                parent.reads += 1
                return {"r0": 0}

        self.batch = _Batch()
        self.node_batch = SimpleNamespace(commit_done=threading.Event(), request_ids=["r0"])


def test_the_async_follow_path_will_not_build_mid_replay():
    """The serial path waits while moves are owed; this one used to skip the delta
    entirely, read the half-applied state as "rid not ready", and poll for ever."""
    worker = _follow_spec(replay_refuses=True)
    parent = _SpecParent()

    assert worker.speculator.follow(parent) is None
    assert parent.reads == 0, "the build started before the replay had landed"
    assert worker.scheduler.lockstep.follow_head() is worker.head, (
        "the head was consumed mid-replay"
    )


class _Commit:
    """N's ``commit_done``, which commits when the follower waits on it."""

    def __init__(self):
        self.done = self.waited = False

    def is_set(self):
        return self.done

    def wait(self):
        self.waited = self.done = True


def test_the_async_follow_path_waits_for_ns_commit_to_replay_a_reload():
    """A reload in the head waits for N's commit. Polling for it waited up to 20 ms past the commit, which rank 0
    spent in its next collective."""
    worker = _follow_spec(replay_refuses=False)
    worker.engine_manager.get_engine("node").apply_resident_delta = lambda delta, rebuilding, caught_up: caught_up
    worker.head.resident_delta.kinds[0] = MoveKind.RELOAD
    parent = _SpecParent()
    parent.node_batch.commit_done = _Commit()

    assert worker.speculator.follow(parent) is None
    assert parent.node_batch.commit_done.waited and parent.reads == 1, "the follower polled instead of waiting"


@pytest.mark.parametrize("kind", [MoveKind.DROP, MoveKind.OFFLOAD], ids=["drop", "offload"])
def test_the_async_follow_path_waits_for_ns_commit_to_move_a_request_in_n(kind):
    """N's commit clears the marks of its rows, so a drop or an offload of a request in N waits for it, as a reload
    does. Polling for it waited up to 20 ms past the commit."""
    worker = _follow_spec(replay_refuses=False)
    worker.engine_manager.get_engine("node").apply_resident_delta = lambda delta, rebuilding, caught_up: caught_up
    worker.head.resident_delta.kinds[0], worker.head.resident_delta.rids[0] = kind, "r0"
    parent = _SpecParent()
    parent.node_batch.commit_done = _Commit()

    assert worker.speculator.follow(parent) is None
    assert parent.node_batch.commit_done.waited and parent.reads == 1, "the follower polled instead of waiting"


def test_a_follower_whose_step_raised_stops_on_rank_0s_drop_of_a_request_in_that_step():
    """The follower's forward of N raised, so N never commits and its rows stay marked in it, while the exit still
    releases N's waiters. A drop of a request in N then waited on the mark behind a warning every 2 s, and rank 0
    waited alone in its next collective."""
    worker = _follow_spec(replay_refuses=True)
    worker.head.resident_delta.kinds[0], worker.head.resident_delta.rids[0] = MoveKind.DROP, "r0"
    parent = _SpecParent()
    parent.node_batch.commit_done.set()

    with pytest.raises(SystemExit, match="after it committed"):
        worker.speculator.follow(parent)


def test_the_async_follow_path_proceeds_once_the_replay_lands():
    """The other direction: the gate must not be a permanent stop. Reaching the
    parent is what says the delta settled and the gate passed."""
    worker = _follow_spec(replay_refuses=False)
    parent = _SpecParent()

    assert worker.speculator.follow(parent) is None
    assert parent.reads == 1, "the gate blocked a build it should have allowed"


# --- rank 0's pool moves ------------------------------------------------------
#
# Rank 0 allots, earmarks, drops and rebuilds, and records each move in its journal.
# A follower makes the same moves from the delta, so both ranks admit every step from
# the same counts and free pages in the same order. If a grant past the room passes on
# one rank and fails on the other, only the rank where it passed enters the next collective.


# each rank's pool -> its engine, whose runner admits for it
_ENGINES: dict[int, Engine] = {}


def _admits(kv, rid: str, walk: str = "prefill") -> bool:
    """A readiness check of ``rid`` through ``kv``'s runner, after it takes whatever has since been admitted."""
    runner = _ENGINES[id(kv)]._runner
    runner.admit_waiting(lambda other, node: None)
    return runner.admit_retrieve(rid, NODE, walk).ready


def _returns(kv, rid: str) -> bool:
    """Whether ``kv``'s runner has admitted ``rid`` again, once it takes whatever has since been admitted."""
    runner = _ENGINES[id(kv)]._runner
    runner.admit_waiting(lambda other, node: None)
    return runner.admit_request(rid, NODE, "decode") is None


def _pool_rank(monkeypatch, rank: int, cfg: PagedKVConfig | None = None) -> tuple[KVManager, Engine]:
    """Rank ``rank`` of two: a pool of 10 pages for requests, or of ``cfg``, under the `Engine` methods that journal
    (rank 0) and replay (rank 1) its moves."""
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    kv = KVManager(
        cfg=cfg or PagedKVConfig(num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096, max_num_pages=11,
                                 page_size=PAGE_SIZE),
        name="kv", joint_comm_group=SimpleNamespace(rank=rank, world_size=2), transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.local_loops = {"decode"}
    engine = Engine.__new__(Engine)
    engine._resources, engine._resource_nodes = {"kv": kv}, {"kv": frozenset({NODE})}
    engine._submodules = {NODE: SimpleNamespace(resources={"kv": kv}, resident=None, submodule=SimpleNamespace(
        cleanup_request=lambda rid: None))}
    engine._runner, engine._loopback_steps = StepRunner({"kv": kv}), {}
    engine._chunking = ChunkingManagement({})
    engine._graph_runtime = SimpleNamespace(
        get_rid_string=str, get_rid_handle=lambda rid: rid if rid in kv._streams else None,
    )
    engine._tp_follower_nodes = {NODE} if rank else set()
    engine._tp_leader_nodes = set() if rank else {NODE}
    engine._resident_delta = None if rank else OffloadDelta.new()
    if not rank:
        kv.journal = partial(engine._journal, pool="kv")
    _ENGINES[id(kv)] = engine
    return kv, engine


def _ingest(kvs, rid: str, pages: int, arrival: int, rebuildable: bool = True) -> None:
    """``rid``'s prompt fills ``pages`` pages, and it decodes one token per iteration."""
    tokens = pages * PAGE_SIZE
    layout = StreamLayout(spans=(PrefixSpan(tokens, tokens, "prefill"),), growth=(("decode", 1),))
    for kv in kvs:
        _ENGINES[id(kv)]._runner.ingest_request(rid, {"kv": KVReqConfig(
            kv_layout={"main": layout}, loop_caps={"decode": 4 * PAGE_SIZE}, rebuildable=rebuildable,
        )}, arrival)


def _counts(kv) -> tuple:
    """What the next admit reads, and the order the next grant takes free pages in."""
    return (
        kv._unused,
        {rid: (a.pages, a.remaining, a.asks_again, a.grown, a.grown_from) for rid, a in kv._allotments.items()},
        # a request the moves set aside waits holding only on rank 0, which alone moves it
        sorted(kv._stalled & kv._asks.keys()),
        {rid: (e.pages, sorted(e.victims), e.granted) for rid, e in kv._earmarks.items()},
        sorted(rid for rid, earmark in kv._victims.items() if earmark is not None), sorted(kv._dropped),
        kv._arena.free_pages(),
    )


def _preempt(kv, rid: str, droppable: bool):
    """Rank 0's choice of victims for ``rid``'s shortage of one page, newest first."""
    return kv.preempt(rid, 1, _ENGINES[id(kv)]._runner.newer_than(rid), lambda other: droppable)


def _waits(kv, rid: str) -> None:
    """``rid`` runs short on rank 0 with nothing newer to move. A step of its own is in flight on another node, so it
    can't move itself to the host, and it waits holding its pages."""
    kv._streams[rid]["main"].step_in_flight = True
    assert _preempt(kv, rid, droppable=False).waits
    kv._streams[rid]["main"].step_in_flight = False


# v's rebuild: its 5-page prompt and the id its next decode reads
_REBUILD = Rebuild("prefill", [0] * (5 * PAGE_SIZE + 1))


def _both_short_then_drop(monkeypatch):
    """r (arrival 1) grows past its allotment until its next token finds no room, on both ranks.
    Rank 0 then chooses v (arrival 2) and drops it, with the tokens its rebuild writes."""
    (leader, lead), (follower, follow) = _pool_rank(monkeypatch, 0), _pool_rank(monkeypatch, 1)
    _ingest((leader, follower), "r", 3, arrival=1)
    _ingest((leader, follower), "v", 5, arrival=2)
    assert _admits(leader, "r")
    assert _admits(leader, "v")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    for rid, tokens in (("r", 3 * PAGE_SIZE), ("v", 5 * PAGE_SIZE), ("r", 2 * PAGE_SIZE), ("r", 1)):
        assert _step(leader, rid, tokens).ok == _step(follower, rid, tokens).ok, f"one rank alone admitted {rid}"
        assert _counts(leader) == _counts(follower)
    assert _preempt(leader, "r", droppable=True).to_drop == ["v"]
    lead.drop_request(NODE, "v", len(_REBUILD.ids) - 1)
    return (leader, lead), (follower, follow)


def test_a_follower_replays_the_drop_of_a_request_that_keeps_no_ids(monkeypatch):
    """A request dropped mid-prefill can keep no ids, since its rebuild runs its prompt's chunks again. Rank 0
    journals the drop with the tokens those chunks wrote, and a follower that keeps no ids either replays it."""
    (leader, lead), (follower, follow) = _both_short_then_drop(monkeypatch)
    scheduler = _follower_scheduler(follow, Rebuilds({}, {}))
    # the input the engine keeps for v's prompt mid-prefill
    follow._chunking.prepared["v", NODE] = SimpleNamespace()

    assert scheduler.lockstep._apply_resident_delta(NODE, lead.take_resident_delta())
    assert follower.residency("v") is Residency.DROPPED and _counts(leader) == _counts(follower)


def test_a_follower_whose_kv_holds_other_than_rank_0s_drop_counted_refuses_the_drop(monkeypatch):
    """A follower drops with the tokens rank 0's rebuild writes, which its own KV refuses unless it holds as many.
    Dropped anyway, the follower would rebuild other KV than it held."""
    (leader, lead), (follower, follow) = _both_short_then_drop(monkeypatch)
    follower._streams["v"]["main"].stored_len -= 1

    with pytest.raises(RuntimeError, match="would write 80 tokens, but its label holds 79"):
        follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    assert follower.residency("v") is Residency.RESIDENT, "the follower dropped what rank 0's rebuild doesn't write"


def test_a_follower_that_keeps_no_ids_for_a_request_rank_0_dropped_stops(monkeypatch):
    """Rank 0 drops v, which it rebuilds from its kept ids, and the follower's rebuild pool had no room for v's
    prompt. Dropped anyway, the follower couldn't start v's rebuild, and tried again on every pass while rank 0
    waited for it."""
    (leader, lead), (follower, follow) = _both_short_then_drop(monkeypatch)
    scheduler = _follower_scheduler(follow, Rebuilds({}, {}))
    _send(scheduler, lead, 0)

    with pytest.raises(SystemExit, match="keeps neither its ids nor the input of its prompt"):
        scheduler.lockstep.settle_delta(caught_up=True)
    assert follower.residency("v") is Residency.RESIDENT, "the follower dropped a request it can't rebuild"


@pytest.mark.parametrize("dropped", [False, True], ids=["voided head", "row dropped by prepare_inputs"])
def test_a_pre_plan_rank_0_throws_away_gives_back_what_its_pre_admit_took(monkeypatch, dropped):
    """Rank 0 pre-plans the next decode step of r and s, whose tokens each need a fourth page, and throws the stage
    away: N's verdict voids the head, or `prepare_inputs` drops s from a yield-away step it planned. A follower never
    planned the step, so rank 0 has to give back both pages, or the two ranks admit later steps from different
    counts. The pages go back where they came from, or the next prompt takes other pages on each rank and the next
    cache hit names pages the follower doesn't hold."""
    (leader, lead), (follower, follow) = _pool_rank(monkeypatch, 0), _pool_rank(monkeypatch, 1)
    for kv in (leader, follower):
        # as under MSTAR_KV_DEBUG_ASSERTS with the prefix cache open
        kv._arena.trail = (0, 0)
    _ingest((leader, follower), "r", 3, arrival=1)
    _ingest((leader, follower), "s", 3, arrival=2)
    assert _admits(leader, "r") and _admits(leader, "s")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    for rid in ("r", "s"):
        assert _step(leader, rid, 3 * PAGE_SIZE).ok and _step(follower, rid, 3 * PAGE_SIZE).ok
    step = SubmoduleStep({"kv": KVStep()}, segments=[Segment("r", "main", 1), Segment("s", "main", 1)])
    step.set_ctx(StepContext(["r", "s"], "decode", 0, False, is_preplan=True))

    assert lead._runner.pre_admit(step).ok
    step.ctx.is_preplan = False
    if dropped:
        step.ctx.request_ids = ["r"]
    lead._runner.clear_preplan()

    def held(kv):
        return [len(kv._streams[rid]["main"].page_indices) for rid in "rs"]

    assert held(leader) == held(follower), (
        f"rank 0 holds {held(leader)} pages for r and s after throwing its pre-plan away, and a follower that never "
        f"planned the step holds {held(follower)}"
    )
    assert _counts(leader) == _counts(follower), "rank 0's free list is in another order than the follower's"
    assert leader._arena.trail == follower._arena.trail, "a follower checking rank 0's page trail would stop"


def test_under_debug_asserts_a_follower_that_admits_from_other_counts_than_rank_0_stops(monkeypatch):
    """Under MSTAR_KV_DEBUG_ASSERTS rank 0 sends its counts with each step's page trail. Here the step that carried
    s's reservation never reached the follower, which would then refuse a grant rank 0 makes and leave rank 0 alone
    in the step's collective. So the follower stops before it builds the next step."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    (leader, lead), (follower, follow) = _pool_rank(monkeypatch, 0), _pool_rank(monkeypatch, 1)
    _ingest((leader, follower), "r", 3, arrival=1)
    _ingest((leader, follower), "s", 3, arrival=2)
    scheduler = _follower_scheduler(follow)

    def send(seq):
        scheduler.lockstep.enqueue_follow(ScheduleTPNode(
            NODE, "decode", [], spec_seq=seq, resident_delta=lead.take_resident_delta(),
            page_trails=lead.page_trails(NODE),
        ))

    assert _admits(leader, "r")
    send(0)
    assert scheduler.lockstep.settle_delta(caught_up=True), "the follower stopped on counts that agree"
    scheduler.lockstep.pop_follow_head()
    assert _admits(leader, "s")
    lead.take_resident_delta()
    send(1)

    with pytest.raises(SystemExit, match="TP ranks diverged"):
        scheduler.lockstep.settle_delta(caught_up=True)


def test_under_debug_asserts_a_commit_moves_nothing_a_page_trail_carries(monkeypatch):
    """Rank 0 sends the next step's page trail before its step in flight commits, and a follower can check it after
    its own has. Here the commit releases the pages a window no longer reads, so r holds fewer pages than it was
    allotted. A trail that carried the allotted pages not yet used then stopped a follower whose counts agreed."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    leader, lead = _pool_rank(monkeypatch, 0)
    window = RetentionPolicy(PAGE_SIZE)
    layout = StreamLayout((PrefixSpan(3 * PAGE_SIZE, 3 * PAGE_SIZE, "prefill"),), (("decode", 1),), retention=window)
    config = KVReqConfig(kv_layout={"main": layout}, loop_caps={"decode": PAGE_SIZE})
    lead._runner.ingest_request("r", {"kv": config}, 1)
    assert _admits(leader, "r")
    step = KVStep(segments=(Segment("r", "main", 3 * PAGE_SIZE),), retention={("r", "main"): window})
    assert leader.admit(step, _ctx("r")).ok
    leader.plan(step, _ctx("r"))
    trail = leader.page_trail()

    leader.commit(step, _ctx("r"))

    assert len(leader._streams["r"]["main"].pages()) < 3, "the window released nothing, so the test checks nothing"
    assert leader.page_trail() == trail, "the counts a follower checks moved with rank 0's commit"


def test_a_follower_admits_each_step_from_the_counts_rank_0_decided(monkeypatch):
    (leader, lead), (follower, follow) = _both_short_then_drop(monkeypatch)

    delta = lead.take_resident_delta()
    assert delta.kinds == [MoveKind.EARMARK, MoveKind.DROP]
    assert follow.apply_resident_delta(delta, _rebuilds_none)

    assert _counts(leader) == _counts(follower), "the follower's earmark or drop left other counts"
    assert _step(leader, "r", 1).ok and _step(follower, "r", 1).ok, "the page v's drop freed went elsewhere on one rank"
    assert _counts(leader) == _counts(follower)


def test_a_follower_takes_a_dropped_request_back_and_lays_it_out_as_rank_0_did(monkeypatch):
    (leader, lead), (follower, follow) = _both_short_then_drop(monkeypatch)
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    for kv, engine in ((leader, lead), (follower, follow)):
        assert _step(kv, "r", 1).ok
        engine.remove_request("r")

    assert _returns(leader, "v")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    # as `Lockstep._start_follow_rebuilds` starts the follower's rebuild at its first row
    for engine in (lead, follow):
        engine.prepare_rebuild(NODE, "v", _REBUILD)

    assert _counts(leader) == _counts(follower), "the follower holds another allotment for the return"
    assert follower._overrides["v"].kv_layout == leader._overrides["v"].kv_layout, "the rebuild writes another span"


@pytest.mark.parametrize("asserts", [False, True], ids=["", "under MSTAR_KV_DEBUG_ASSERTS"])
def test_a_drop_whose_victim_is_in_a_step_here_waits_and_nothing_behind_it_jumps_ahead(monkeypatch, asserts):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", asserts)
    (leader, lead), (follower, follow) = _both_short_then_drop(monkeypatch)
    # rank 0 tears r down once the step carrying the drop has gone out, and then v's history and a page fit
    delta = lead.take_resident_delta()
    lead.remove_request("r")
    assert _returns(leader, "v")
    delta.extend(lead.take_resident_delta())
    step, ctx = KVStep(segments=(Segment("v", "main", 0),)), StepContext(("v",), "prefill", 0, False)
    assert follower.admit(step, ctx).ok

    assert not follow.apply_resident_delta(delta, _rebuilds_none)
    assert delta.kinds == [MoveKind.DROP, MoveKind.RESERVE], "the drop or the return behind it moved early"
    assert "v" not in follower._dropped and "v" in follower._allotments

    follower.rollback_admit(step, ctx)
    assert follow.apply_resident_delta(delta, _rebuilds_none)
    follow.remove_request("r")
    assert _counts(leader) == _counts(follower)


@pytest.mark.parametrize("in_step", [False, True], ids=["at once", "once its step there lands"])
def test_a_follower_moves_rank_0s_victim_to_the_host_and_takes_it_back_as_rank_0_did(monkeypatch, in_step):
    """Rank 0 journals a victim's move to the host when it chooses the victim, and again when the victim moves. The
    follower gives up the victim's allotment as rank 0 does. Rank 0 admits the victim again from the head of the
    queue, and the return reserves the same pages on both ranks."""
    (leader, lead), (follower, follow) = _pool_rank(monkeypatch, 0), _pool_rank(monkeypatch, 1)
    leader._cpu_pool, follower._cpu_pool = host._HostPool(64), host._HostPool(64)
    _ingest((leader, follower), "r", 3, arrival=1)
    _ingest((leader, follower), "v", 5, arrival=2)
    assert _admits(leader, "r")
    assert _admits(leader, "v")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    for rid, tokens in (("r", 3 * PAGE_SIZE), ("v", 5 * PAGE_SIZE), ("r", 2 * PAGE_SIZE), ("r", 1)):
        assert _step(leader, rid, tokens).ok == _step(follower, rid, tokens).ok
    assert _preempt(leader, "r", droppable=False).to_host == ["v"]
    lead.offload_request(NODE, "v")
    delta = lead.take_resident_delta()
    assert delta.kinds == [MoveKind.EARMARK, MoveKind.OFFLOAD, MoveKind.OFFLOAD], "a choice or a move went unjournaled"
    if in_step:
        follower._streams["v"]["main"].step_in_flight = True
        assert not follow.apply_resident_delta(delta, _rebuilds_none), (
            "the follower moved a victim its step holds"
        )
        assert "v" in follower._allotments, "a refused move gave up the victim's allotment"
        follower._streams["v"]["main"].step_in_flight = False

    assert follow.apply_resident_delta(delta, _rebuilds_none)
    assert follower.residency("v") is Residency.ON_HOST and _counts(leader) == _counts(follower), (
        "the follower's move left other counts"
    )
    assert follower._asks == leader._asks and list(leader._asks) == ["v"], "the follower asked for the victim elsewhere"
    for kv, engine in ((leader, lead), (follower, follow)):
        assert _step(kv, "r", 1).ok
        engine.remove_request("r")
    assert not follow.reload_request(NODE, "v"), "a follower would reload v ahead of rank 0's journal"
    assert _returns(leader, "v") and lead.reload_request(NODE, "v") and leader.residency("v") is Residency.RESIDENT
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    assert follower.residency("v") is Residency.RESIDENT, "the follower didn't replay rank 0's return"
    assert _counts(leader) == _counts(follower), "the follower holds another allotment for the return"


def test_a_follower_moves_a_request_with_nothing_newer_to_move_to_the_host_and_takes_it_back_as_rank_0_did(
    monkeypatch,
):
    """r's grant needs two pages past its allotment, and only one is free. With nothing newer to move, rank 0 moves r
    itself to the host, as any other victim, and r returns with both pages. Only rank 0 counts the grant's shortage,
    so a follower that didn't replay r's wait returned r with one page."""
    (leader, lead), (follower, follow) = _pool_rank(monkeypatch, 0), _pool_rank(monkeypatch, 1)
    leader._cpu_pool, follower._cpu_pool = host._HostPool(64), host._HostPool(64)
    _ingest((leader, follower), "a", 4, arrival=1)
    _ingest((leader, follower), "r", 5, arrival=2)
    assert _admits(leader, "a") and _admits(leader, "r")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    for rid, tokens in (("a", 4 * PAGE_SIZE), ("r", 5 * PAGE_SIZE)):
        assert _step(leader, rid, tokens).ok and _step(follower, rid, tokens).ok
    assert not _step(leader, "r", PAGE_SIZE + 1).ok and not _step(follower, "r", PAGE_SIZE + 1).ok
    assert _preempt(leader, "r", droppable=False).to_host == ["r"]
    lead.offload_request(NODE, "r")
    delta = lead.take_resident_delta()
    assert delta.kinds == [MoveKind.WAIT, MoveKind.OFFLOAD, MoveKind.OFFLOAD], "r's wait or its move went unjournaled"

    assert follow.apply_resident_delta(delta, _rebuilds_none)
    assert follower.residency("r") is Residency.ON_HOST and _counts(leader) == _counts(follower)
    for engine in (lead, follow):
        engine.remove_request("a")
    assert _returns(leader, "r") and lead.reload_request(NODE, "r")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    assert _counts(leader) == _counts(follower), "the follower returned r with another allotment"
    assert all(_step(kv, "r", PAGE_SIZE + 1).ok for kv in (leader, follower)), "r came back short of its grant"


def test_only_rank_0_moves_requests_for_the_ones_that_wait(monkeypatch):
    """A follower replays rank 0's waits, so it holds the same requests waiting with their allotments, but only rank 0
    chooses moves for them, and a follower's pass walks no refusals."""
    (leader, lead), (follower, follow) = _pool_rank(monkeypatch, 0), _pool_rank(monkeypatch, 1)
    _ingest((leader, follower), "r", 5, arrival=1)
    _ingest((leader, follower), "v", 5, arrival=3)
    assert _admits(leader, "r") and _admits(leader, "v")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    for rid in ("r", "v"):
        assert _step(leader, rid, 5 * PAGE_SIZE).ok and _step(follower, rid, 5 * PAGE_SIZE).ok
    assert not _step(leader, "v", 1).ok and not _step(follower, "v", 1).ok
    _waits(leader, "v")

    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)

    assert follower._stalled == leader._stalled == {"v"}, "the follower holds other requests waiting with allotments"
    assert leader.any_stalled() and not follower.any_stalled(), "a follower would walk the refusals for make_room"
    _ingest((leader, follower), "o", 2, arrival=2)
    assert not _admits(leader, "o")
    (refusal,) = lead._runner.refusals()
    newer = lead._runner.newer_than("o")
    assert follower.make_room(refusal.mark, newer, bool) is None, "a follower chose v to move for o"
    assert leader.make_room(refusal.mark, newer, bool).to_drop == ["v"]


def test_a_follower_takes_the_host_room_rank_0_set_aside_and_so_each_requests_class(monkeypatch):
    """Rank 0 sets host room aside for a request that can move only there as it admits the request, and guarantees
    the request when the host has none left. A follower sets the room aside as it replays the reservation, and replays
    the flip, which can come passes before the reservation. Here b takes the host's one page, and x, admitted next,
    finds none."""
    (leader, lead), (follower, follow) = _pool_rank(monkeypatch, 0), _pool_rank(monkeypatch, 1)
    for kv in (leader, follower):
        kv._cpu_pool = host._HostPool(1)
    _ingest((leader, follower), "b", 1, arrival=1, rebuildable=False)
    _ingest((leader, follower), "x", 1, arrival=2, rebuildable=False)
    assert _admits(leader, "b") and _admits(leader, "x")
    assert "x" in leader._guaranteed and leader._cpu_pool.reserved == {"b": 1}, "the host had room for both"

    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)

    assert follower._cpu_pool.reserved == leader._cpu_pool.reserved, "the follower set other host room aside"
    assert follower._guaranteed == leader._guaranteed, "a request rank 0 never preempts is preemptible here"
    assert _counts(follower) == _counts(leader), "x's allotment here isn't the footprint rank 0 allotted it"


def test_a_follower_holds_the_class_and_allotment_rank_0_gave_a_request_as_its_step_admitted(monkeypatch):
    """r sets the host's two pages aside at admission and grows past them, so rank 0 guarantees r inside the admit
    of r's next page. The follower's host holds the room rank 0's does when it admits that step, so it guarantees r
    there too. A flip rank 0 journaled from the admit reached the follower only after the step."""
    (leader, lead), (follower, follow) = _pool_rank(monkeypatch, 0), _pool_rank(monkeypatch, 1)
    for kv in (leader, follower):
        kv._cpu_pool = host._HostPool(2)
    _ingest((leader, follower), "r", 2, arrival=1, rebuildable=False)
    assert _admits(leader, "r")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)

    for tokens in (2 * PAGE_SIZE, 1):
        assert _step(leader, "r", tokens).ok and _step(follower, "r", tokens).ok

    assert "r" in leader._guaranteed and "r" not in leader._cpu_pool.reserved
    assert not lead.take_resident_delta(), "rank 0 journaled a move from inside the admit"
    assert follower._guaranteed == leader._guaranteed, "r is preemptible here and never preempted on rank 0"
    assert follower._cpu_pool.reserved == leader._cpu_pool.reserved, "the follower keeps host room r no longer needs"
    assert _counts(follower) == _counts(leader), "r grows into pages here that rank 0 never allotted it"


def test_a_follower_refuses_the_page_rank_0_refuses_when_the_host_cant_cover_it(monkeypatch):
    """r sets the host's two pages aside at admission, and its next page finds the host full, where guaranteeing r
    would take a page more than the room has. Rank 0 refuses the step. A follower that left the host to rank 0 took
    the page and ran the step alone."""
    (leader, lead), (follower, follow) = _pool_rank(monkeypatch, 0), _pool_rank(monkeypatch, 1)
    for kv in (leader, follower):
        kv._cpu_pool = host._HostPool(2)
    _ingest((leader, follower), "r", 2, arrival=1, rebuildable=False)
    _ingest((leader, follower), "v", 5, arrival=2)
    assert _admits(leader, "r") and _admits(leader, "v")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    for rid, tokens in (("r", 2 * PAGE_SIZE), ("v", 5 * PAGE_SIZE)):
        assert _step(leader, rid, tokens).ok and _step(follower, rid, tokens).ok
    assert not _step(leader, "r", 1).ok, "the guarantee fit the room, so the test checks nothing"

    assert not _step(follower, "r", 1).ok, "the follower ran a step rank 0 refused"


def test_a_pre_plan_rank_0_throws_away_sets_no_host_room_aside(monkeypatch):
    """r's next page needs host room past what r has set aside. Rank 0 can't give the room back for a stage it throws
    away, and a follower may never plan the step, so the pre-admit refuses the step and exec admits it on every
    rank."""
    leader, lead = _pool_rank(monkeypatch, 0)
    leader._cpu_pool = host._HostPool(8)
    _ingest((leader,), "r", 2, arrival=1, rebuildable=False)
    assert _admits(leader, "r") and _step(leader, "r", 2 * PAGE_SIZE).ok
    step = SubmoduleStep({"kv": KVStep()}, segments=[Segment("r", "main", 1)])
    step.set_ctx(StepContext(["r"], "decode", 0, False, is_preplan=True))

    lead._runner.pre_admit(step)
    lead._runner.clear_preplan()

    assert leader._cpu_pool.reserved == {"r": 2}, "rank 0 set host room aside for a stage it threw away"


def test_a_flip_to_guaranteed_is_counted_and_logged_on_rank_0_alone(monkeypatch, caplog):
    """A follower replays the flip, and counting it there too logged the same line once per rank."""
    (leader, lead), (follower, follow) = _pool_rank(monkeypatch, 0), _pool_rank(monkeypatch, 1)
    for kv in (leader, follower):
        kv._cpu_pool = host._HostPool(1)
    _ingest((leader, follower), "b", 1, arrival=1, rebuildable=False)
    _ingest((leader, follower), "x", 1, arrival=2, rebuildable=False)

    with caplog.at_level(logging.WARNING):
        assert _admits(leader, "b") and _admits(leader, "x")
        assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)

    assert "x" in follower._guaranteed and (leader._cpu_pool.guaranteed, follower._cpu_pool.guaranteed) == (1, 0)
    assert caplog.text.count("no room to set aside for x's move") == 1, "every rank logged rank 0's flip"


def test_a_follower_allots_the_room_rank_0_counted_for_a_short_grant(monkeypatch):
    """r's grant needs three pages past its allotment where the room has two, so rank 0 allots r those two and
    drops v for the third. The follower replayed the earmark and the drop, and r's allotment there was two pages
    short of rank 0's."""
    (leader, lead), (follower, follow) = _pool_rank(monkeypatch, 0), _pool_rank(monkeypatch, 1)
    _ingest((leader, follower), "r", 3, arrival=1)
    _ingest((leader, follower), "v", 5, arrival=2)
    assert _admits(leader, "r") and _admits(leader, "v")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    for rid, tokens in (("r", 3 * PAGE_SIZE), ("v", 5 * PAGE_SIZE)):
        assert _step(leader, rid, tokens).ok and _step(follower, rid, tokens).ok
    outcome = _step(leader, "r", 3 * PAGE_SIZE)
    assert not outcome.ok and not _step(follower, "r", 3 * PAGE_SIZE).ok
    victims = leader.preempt("r", outcome.reason.pages_short, _ENGINES[id(leader)]._runner.newer_than("r"), bool)
    assert victims.to_drop == ["v"]
    lead.drop_request(NODE, "v", 5 * PAGE_SIZE)

    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)

    assert _counts(follower) == _counts(leader), "r's allotment here lacks the room rank 0 counted for its grant"


def test_a_short_grant_with_no_room_to_claim_leaves_the_allotment_alike_on_every_rank(monkeypatch):
    """r ran short with no room left, so rank 0 claims nothing for r as it chooses v. Allotting r nothing still raised
    r's allotment to the pages r held past it, on rank 0 alone, since a claim of nothing goes unjournaled."""
    (leader, lead), (follower, follow) = _pool_rank(monkeypatch, 0), _pool_rank(monkeypatch, 1)
    _ingest((leader, follower), "r", 3, arrival=1)
    _ingest((leader, follower), "v", 5, arrival=2)
    assert _admits(leader, "r") and _admits(leader, "v")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    for rid, tokens in (("r", 3 * PAGE_SIZE), ("v", 5 * PAGE_SIZE), ("r", 2 * PAGE_SIZE), ("r", 1)):
        assert _step(leader, rid, tokens).ok == _step(follower, rid, tokens).ok
    assert _preempt(leader, "r", droppable=True).to_drop == ["v"]

    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)

    assert _counts(follower) == _counts(leader), "r's allotment here isn't the one rank 0 holds"


def test_an_allotment_undone_in_the_sweep_that_made_it_never_reaches_the_followers(monkeypatch):
    """A node's second pool can hold a request back on every readiness check, undoing the first pool's allotment."""
    leader, lead = _pool_rank(monkeypatch, 0)
    _ingest((leader,), "x", 1, arrival=1)
    assert _admits(leader, "x")
    leader.unreserve("x")

    assert not lead.take_resident_delta(), "two moves per pass would pile up for each follower to replay"


def test_a_follower_whose_counts_cant_take_rank_0s_move_stops(monkeypatch):
    follower, follow = _pool_rank(monkeypatch, 1)
    _ingest((follower,), "x", 1, arrival=1)
    head = ScheduleTPNode(NODE, "prefill", ["x"], spec_seq=0)
    for kind in (MoveKind.RESERVE, MoveKind.UNRESERVE, MoveKind.UNRESERVE):
        head.resident_delta.add(kind, "x", "kv", 1)
    scheduler = _follower_scheduler(follow)
    scheduler.lockstep.enqueue_follow(head)

    with pytest.raises(SystemExit, match="TP ranks diverged"):
        scheduler.lockstep.settle_delta(caught_up=True)


def _make_room(kv, droppable: bool = False):
    """Rank 0's choice of victims for the oldest waiting request its runner refused that moves can let in,
    droppable only if ``droppable``."""
    runner = _ENGINES[id(kv)]._runner
    for refusal in runner.refusals():
        victims = kv.make_room(refusal.mark, runner.newer_than(refusal.rid), lambda other: droppable)
        if victims is not None:
            return victims
    return None


@pytest.mark.parametrize("rebuildable", [True, False])
def test_a_follower_replays_rank_0s_admit_of_the_request_it_moved_a_newer_one_for_right_after_the_move(
    monkeypatch, rebuildable,
):
    """o, older, first asks once v holds the rest of the pool, and v, with nothing newer to move, waits holding its
    pages. Rank 0 moves v to the host for o, and o takes the room as v's move lands. Only rank 0 knows o waits, so a
    follower that replayed v's earmark as a short grant's had no allotment of o's to grow. An o no drop can free then
    finds the host full, and turns guaranteed on every rank."""
    (leader, lead), (follower, follow) = _pool_rank(monkeypatch, 0), _pool_rank(monkeypatch, 1)
    for kv in (leader, follower):
        kv._cpu_pool = host._HostPool(8)
    _ingest((leader, follower), "f", 3, arrival=1)
    _ingest((leader, follower), "v", 4, arrival=3)
    assert _admits(leader, "f") and _admits(leader, "v")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    for rid, tokens in (("f", 3 * PAGE_SIZE), ("v", 4 * PAGE_SIZE), ("v", 3 * PAGE_SIZE)):
        assert _step(leader, rid, tokens).ok and _step(follower, rid, tokens).ok
    assert not _step(leader, "v", 1).ok and not _step(follower, "v", 1).ok
    _waits(leader, "v")
    lead.readmit_request(NODE, "v")
    _ingest((leader, follower), "o", 2, arrival=2, rebuildable=rebuildable)
    assert not _admits(leader, "o")

    assert _make_room(leader).to_host == ["v"]
    lead.offload_request(NODE, "v")
    assert "o" in leader._allotments, "v's move didn't make room for o"
    assert ("o" in leader._guaranteed) is not rebuildable, "the host had room for o's prompt"
    delta = lead.take_resident_delta()
    admit = [MoveKind.RESERVE] if rebuildable else [MoveKind.GUARANTEE, MoveKind.RESERVE]
    assert delta.kinds == [MoveKind.WAIT, MoveKind.RESERVE, MoveKind.OFFLOAD, MoveKind.OFFLOAD, *admit], (
        "rank 0 journaled o's admit ahead of v's move, which a follower replays before v's pages come free"
    )
    assert follow.apply_resident_delta(delta, _rebuilds_none)

    assert _counts(follower) == _counts(leader), "o's allotment here isn't the one rank 0 admitted it with"
    assert follower._guaranteed == leader._guaranteed, "o is preemptible here and never preempted on rank 0"
    assert follower._cpu_pool.reserved == leader._cpu_pool.reserved, "the follower set other host room aside"


def _windowed(pages: int = 8) -> PagedKVConfig:
    """A pool of one full and one sliding layer, whose window keeps a page."""
    cfg = PagedKVConfig(
        num_layers=2, num_kv_heads=2, head_dim=8, max_seq_len=4096, max_num_pages=pages, page_size=PAGE_SIZE,
    )
    cfg.keep_window(PAGE_SIZE, (1,))
    return cfg


@pytest.mark.parametrize("asserts", [False, True], ids=["", "under MSTAR_KV_DEBUG_ASSERTS"])
def test_a_follower_replays_rank_0s_admit_of_the_request_it_moved_a_newer_one_for_whatever_room_it_holds(
    monkeypatch, asserts,
):
    """Both ranks admit w's last prompt chunk, and its commit gives the room the window no longer holds back. Rank 0
    commits it before it moves v to the host for o, and the follower replays the move before its own commit. o fits
    rank 0's room as v's move lands and not yet the follower's, so a follower that admitted o itself never did. Under
    MSTAR_KV_DEBUG_ASSERTS the follower checked its allotments against free pages its commit hadn't given back yet."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", asserts)
    (leader, lead), (follower, follow) = (_pool_rank(monkeypatch, rank, _windowed()) for rank in (0, 1))
    for kv in (leader, follower):
        kv._cpu_pool = host._HostPool(64)
    _ingest((leader, follower), "w", 3, arrival=0)
    _ingest((leader, follower), "v", 4, arrival=3)
    assert _admits(leader, "w") and _admits(leader, "v")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    for rid, tokens in (("w", 2 * PAGE_SIZE), ("v", 4 * PAGE_SIZE), *[("v", PAGE_SIZE)] * 3):
        assert _step(leader, rid, tokens).ok and _step(follower, rid, tokens).ok
    assert not _step(leader, "v", PAGE_SIZE).ok and not _step(follower, "v", PAGE_SIZE).ok
    _waits(leader, "v")
    lead.readmit_request(NODE, "v")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    step, ctx = KVStep(segments=(Segment("w", "main", PAGE_SIZE),)), _ctx("w")
    for kv in (leader, follower):
        assert kv.admit(step, ctx).ok
        kv.plan(step, ctx)
    leader.commit(step, ctx)
    _ingest((leader, follower), "o", 5, arrival=2)
    assert not lead._runner.admit_retrieve("o", NODE, "prefill").ready

    assert _make_room(leader).to_host == ["v"]
    lead.offload_request(NODE, "v")
    assert "o" in leader._allotments, "v's move didn't make room for o"
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    follower.commit(step, ctx)

    # the free lists differ, since w's window and v's move gave their pages back in another order on each rank
    assert _counts(follower)[:-1] == _counts(leader)[:-1], "the follower holds another allotment for o than rank 0"


def _reload_into_the_window(monkeypatch):
    """v moves itself to the host. Both ranks admit a step of w's prompt chunk and u's next token, and rank 0 commits
    it, which gives back the pages w's window no longer holds. Rank 0 then reloads v into them, while the follower's
    commit is still pending."""
    (leader, lead), (follower, follow) = (_pool_rank(monkeypatch, rank, _windowed(7)) for rank in (0, 1))
    for kv in (leader, follower):
        kv._cpu_pool = host._HostPool(64)
    _ingest((leader, follower), "w", 3, arrival=0)
    _ingest((leader, follower), "u", 1, arrival=1)
    _ingest((leader, follower), "v", 2, arrival=3)
    assert _admits(leader, "w") and _admits(leader, "u") and _admits(leader, "v")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    for rid, tokens in (("w", PAGE_SIZE), ("u", PAGE_SIZE), ("v", 2 * PAGE_SIZE), ("v", PAGE_SIZE)):
        assert _step(leader, rid, tokens).ok and _step(follower, rid, tokens).ok
    assert not _step(leader, "v", PAGE_SIZE).ok and not _step(follower, "v", PAGE_SIZE).ok
    assert _preempt(leader, "v", droppable=False).to_host == ["v"]
    lead.offload_request(NODE, "v")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    step = KVStep(segments=(Segment("w", "main", 2 * PAGE_SIZE), Segment("u", "main", 1)))
    ctx = StepContext(("w", "u"), "prefill", 0, False)
    for kv in (leader, follower):
        assert kv.admit(step, ctx).ok
        kv.plan(step, ctx)
    leader.commit(step, ctx)
    reload = leader._reload_pages("v")
    assert follower._arena.num_free < reload <= leader._arena.num_free, "this checks nothing"
    assert _returns(leader, "v") and lead.reload_request(NODE, "v")
    delta = lead.take_resident_delta()
    return (leader, lead), (follower, follow), (step, ctx), delta


@pytest.mark.parametrize("asserts", [False, True], ids=["", "under MSTAR_KV_DEBUG_ASSERTS"])
def test_a_follower_takes_rank_0s_reload_into_the_pages_its_window_gives_back_once_its_own_commit_lands(
    monkeypatch, asserts,
):
    """Rank 0's reload of v takes pages its commit gave back. The follower's commit hadn't given them back yet, and
    the follower stopped though its counts agreed with rank 0's. It waits for its commit instead."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", asserts)
    (leader, lead), (follower, follow), (step, ctx), delta = _reload_into_the_window(monkeypatch)
    scheduler = _follower_scheduler(follow)
    scheduler.lockstep.enqueue_follow(ScheduleTPNode(NODE, "decode", [], spec_seq=0, resident_delta=delta))

    assert not scheduler.lockstep.settle_delta(caught_up=False), "the follower reloaded v before its commit"
    assert scheduler.lockstep.awaits_commit(()) and follower.residency("v") is Residency.ON_HOST

    follower.commit(step, ctx)
    assert scheduler.lockstep.settle_delta(caught_up=True)
    assert follower.residency("v") is Residency.RESIDENT
    assert _counts(follower)[:-1] == _counts(leader)[:-1], "the follower took v back with other counts"


def test_a_follower_whose_step_raised_stops_on_rank_0s_reload_rather_than_wait_for_its_commit(monkeypatch):
    """The follower's forward of the step raised, so the step never commits and its rows stay marked in it until
    they are removed, which the follower does only after the step carrying the reload. Its exit still releases the
    step's waiters, so the follower is caught up, can't take the reload and stops. Waiting on the marks left rank 0
    alone in its next collective."""
    (leader, lead), (follower, follow), _, delta = _reload_into_the_window(monkeypatch)
    scheduler = _follower_scheduler(follow)
    scheduler.lockstep.enqueue_follow(ScheduleTPNode(NODE, "decode", [], spec_seq=0, resident_delta=delta))
    assert follower._streams["w"]["main"].step_in_flight

    with pytest.raises(SystemExit, match="could not take"):
        scheduler.lockstep.settle_delta(caught_up=True)


def test_a_follower_grows_a_request_waiting_for_its_shortage_once_as_rank_0s_moves_cover_it(monkeypatch):
    """o runs short while v has a step in flight, so o waits for its page holding its allotment, and then v runs short
    with nothing newer to move. Rank 0 moves v to the host for o, and o takes its page once v's move covers it. A
    follower that took v's earmark as a short grant's grew o page by page as v's pages came free, and again from rank
    0's reservation."""
    (leader, lead), (follower, follow) = _pool_rank(monkeypatch, 0), _pool_rank(monkeypatch, 1)
    for kv in (leader, follower):
        kv._cpu_pool = host._HostPool(64)
    _ingest((leader, follower), "o", 3, arrival=1)
    _ingest((leader, follower), "v", 3, arrival=2)
    assert _admits(leader, "o") and _admits(leader, "v")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    for rid, tokens in (("o", 3 * PAGE_SIZE), ("v", 3 * PAGE_SIZE), ("v", 3 * PAGE_SIZE), ("o", 1)):
        assert _step(leader, rid, tokens).ok and _step(follower, rid, tokens).ok
    assert not _step(leader, "o", PAGE_SIZE).ok and not _step(follower, "o", PAGE_SIZE).ok
    leader._streams["v"]["main"].step_in_flight = True
    _waits(leader, "o")
    leader._streams["v"]["main"].step_in_flight = False
    lead.readmit_request(NODE, "o")
    assert not _step(leader, "v", 1).ok and not _step(follower, "v", 1).ok
    _waits(leader, "v")
    lead.readmit_request(NODE, "v")
    assert not _admits(leader, "o")

    assert _make_room(leader).to_host == ["v"]
    lead.offload_request(NODE, "v")
    assert "o" not in leader._asks, "v's move didn't cover o's page"
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)

    assert _counts(follower) == _counts(leader), "o's allotment here isn't the one rank 0 grew it to"


def _v1_hosted_and_v2_dropped_for_o(monkeypatch, rebuildable: bool):
    """v1 and v2 hold the 4 pages f leaves. v1 runs short while v2 has a step in flight, and then v2 runs short, so
    both wait holding their pages. o, older, then asks for its 2 pages. The host has room only for v1, which can move
    only there, so rank 0 moves v1 to the host and drops v2 for o, and v1's 3 pages alone cover o."""
    (leader, lead), (follower, follow) = _pool_rank(monkeypatch, 0), _pool_rank(monkeypatch, 1)
    for kv in (leader, follower):
        kv._cpu_pool = host._HostPool(4)
    _ingest((leader, follower), "f", 6, arrival=0)
    _ingest((leader, follower), "v1", 3, arrival=2, rebuildable=False)
    _ingest((leader, follower), "v2", 1, arrival=3)
    assert _admits(leader, "f") and _admits(leader, "v1") and _admits(leader, "v2")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    for rid, pages in (("f", 6), ("v1", 3), ("v2", 1)):
        assert _step(leader, rid, pages * PAGE_SIZE).ok and _step(follower, rid, pages * PAGE_SIZE).ok
    assert not _step(leader, "v1", 1).ok and not _step(follower, "v1", 1).ok
    leader._streams["v2"]["main"].step_in_flight = True
    _waits(leader, "v1")
    leader._streams["v2"]["main"].step_in_flight = False
    lead.readmit_request(NODE, "v1")
    assert not _step(leader, "v2", 1).ok and not _step(follower, "v2", 1).ok
    _waits(leader, "v2")
    lead.readmit_request(NODE, "v2")
    _ingest((leader, follower), "o", 2, arrival=1, rebuildable=rebuildable)
    assert not _admits(leader, "o")
    victims = _make_room(leader, droppable=True)
    assert victims.to_host == ["v1"] and victims.to_drop == ["v2"]
    return (leader, lead), (follower, follow)


def test_a_waiting_request_admitted_before_its_last_victim_lands_takes_no_more_room_on_any_rank(monkeypatch):
    """A host move lands before a drop, so v1's move admits o before v2 is dropped. v2's drop then grew o's allotment
    past its 2 pages, on every rank."""
    (leader, lead), (follower, follow) = _v1_hosted_and_v2_dropped_for_o(monkeypatch, rebuildable=True)
    lead.offload_request(NODE, "v1")
    assert "o" in leader._allotments, "v1's move didn't admit o"
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    assert _counts(follower) == _counts(leader), "o's allotment here isn't the one rank 0 admitted it with"

    lead.drop_request(NODE, "v2", PAGE_SIZE)
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)

    assert leader._allotments["o"].pages == 2, "v2's drop grew the allotment of a request v1's move admitted"
    assert _counts(follower) == _counts(leader), "o's allotment here isn't the one rank 0 left it with"


def test_a_waiting_request_that_turns_guaranteed_as_one_victim_lands_waits_for_its_footprint_on_every_rank(
    monkeypatch,
):
    """v1's move covers o's prompt, but the host v1 filled has no room for it, so o, which no drop can free, turns
    guaranteed and asks for its 6-page footprint on every rank. v2's drop then covered the prompt again, and every
    rank admitted o, guaranteed, with 2 pages."""
    (leader, lead), (follower, follow) = _v1_hosted_and_v2_dropped_for_o(monkeypatch, rebuildable=False)
    lead.offload_request(NODE, "v1")
    assert "o" in leader._guaranteed and "o" not in leader._allotments
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    assert follower._guaranteed == leader._guaranteed, "o is preemptible here and guaranteed on rank 0"
    assert _counts(follower) == _counts(leader), "o's earmark here waits for other than its footprint"

    lead.drop_request(NODE, "v2", PAGE_SIZE)
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)

    for kv in (leader, follower):
        assert "o" not in kv._allotments, "o was admitted guaranteed with less than its footprint"
    assert _counts(follower) == _counts(leader), "the follower holds other room for o than rank 0"


def test_an_unreserve_cancels_no_reservation_that_moves_victims():
    """A reservation `make_room` journals names the victims it moves, and the followers have to move them even when
    the request's next pool refuses it in the same pass."""
    delta = OffloadDelta.new()
    delta.add(MoveKind.RESERVE, "o", "kv", 2, ["v"])

    assert not delta.cancel("o", MoveKind.RESERVE, "kv"), "the followers would never move v"


def test_an_unreserve_cancels_no_reservation_that_grew_an_allotment(monkeypatch):
    """o waits holding its allotment for a page, and once v is removed rank 0 grows o's allotment by the page. A second
    pool on the node then holds o back in the same sweep, which takes the page back on rank 0 and leaves o waiting for
    it. Cancelling the two moves left the follower holding the grown allotment."""
    (leader, lead), (follower, follow) = _pool_rank(monkeypatch, 0), _pool_rank(monkeypatch, 1)
    _ingest((leader, follower), "o", 5, arrival=1)
    _ingest((leader, follower), "v", 5, arrival=2)
    assert _admits(leader, "o") and _admits(leader, "v")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)
    for rid in ("o", "v"):
        assert _step(leader, rid, 5 * PAGE_SIZE).ok and _step(follower, rid, 5 * PAGE_SIZE).ok
    assert not _step(leader, "o", 1).ok and not _step(follower, "o", 1).ok
    _waits(leader, "o")
    lead.readmit_request(NODE, "o")
    for engine in (lead, follow):
        engine.remove_request("v")
    assert _returns(leader, "o")

    leader.unreserve("o")
    assert follow.apply_resident_delta(lead.take_resident_delta(), _rebuilds_none)

    assert _counts(follower) == _counts(leader), "the follower holds the page rank 0 took back"


class _Forgets(_KeepsIds):
    """Records the requests whose ids it gives back."""

    def __init__(self):
        self.forgot = []

    def forget(self, rid, node):
        self.forgot.append((rid, node))


def test_a_follower_gives_back_the_ids_of_a_request_rank_0_journals_as_keeping_none(monkeypatch):
    (leader, lead), (follower, follow) = _pool_rank(monkeypatch, 0), _pool_rank(monkeypatch, 1)
    _ingest((leader, follower), "x", 1, arrival=1)
    rebuilds = _Forgets()

    lead.keeps_no_ids(NODE, "x")
    follow.keeps_no_ids(NODE, "x")
    delta = lead.take_resident_delta()

    assert delta.kinds == [MoveKind.FORGET], "rank 0 journaled no move, or a follower journaled one"
    assert follow.apply_resident_delta(delta, rebuilds) and rebuilds.forgot == [("x", NODE)]


def _follower_of(follow):
    """A follower's scheduler over ``follow``, whose log keeps v's ids and runs no rebuild."""
    scheduler = _follower_scheduler(follow, _KeepsIds())
    return scheduler


def _send(scheduler, lead, seq: int) -> None:
    scheduler.lockstep.enqueue_follow(ScheduleTPNode(
        NODE, "decode", [], spec_seq=seq, resident_delta=lead.take_resident_delta(), page_trails=lead.page_trails(NODE),
    ))


def _follow_next(scheduler) -> None:
    assert scheduler.lockstep.settle_delta(caught_up=True)
    scheduler.lockstep.pop_follow_head()


def test_a_follower_lands_a_rebuild_between_the_steps_rank_0_landed_it_between(monkeypatch):
    """Rank 0 lands v's rebuild when post-processing the rebuild's step K, after sending the step it built during K
    (a yield-away, under async scheduling). A follower post-processes K before taking the step built during K, so a
    follower that landed v then would hold one dropped request fewer than rank 0."""
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    (leader, lead), (follower, follow) = _both_short_then_drop(monkeypatch)
    scheduler = _follower_of(follow)
    _send(scheduler, lead, 0)
    _follow_next(scheduler)

    _send(scheduler, lead, 1)
    for engine in (lead, follow):
        engine.land_rebuild(NODE, "v")
    _follow_next(scheduler)
    _send(scheduler, lead, 2)
    _follow_next(scheduler)

    assert follower.residency("v") is Residency.RESIDENT and _counts(leader) == _counts(follower)


def test_a_follower_whose_rebuild_still_runs_when_rank_0_lands_it_stops(monkeypatch):
    """Rank 0's land of v arrives while the follower's own rebuild of v still runs, so the follower didn't run v's
    last chunk as a rebuild row in the step rank 0 did. It stops on the replay."""
    (leader, lead), (follower, follow) = _both_short_then_drop(monkeypatch)
    scheduler = _follower_of(follow)
    _send(scheduler, lead, 0)
    _follow_next(scheduler)
    scheduler.rebuilds.rebuilding = lambda rid, node: rid == "v"

    lead.land_rebuild(NODE, "v")
    _send(scheduler, lead, 1)

    with pytest.raises(SystemExit, match="rank 0 landed v's rebuild on LLM, and this rank's is still running"):
        scheduler.lockstep.settle_delta(caught_up=True)
