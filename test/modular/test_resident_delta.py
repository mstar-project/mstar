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
from functools import partial
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, ".")

from test_kv_preemption import _step
from test_kv_prefix_lease import NODE, PAGE_SIZE, _StubTransfer

from mstar.engine import engine as engine_mod
from mstar.engine.engine import Engine
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PagedKVConfig, PrefixSpan, StreamLayout
from mstar.engine.resources.kv.manager import Allotment, KVManager
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.step import FULL_ADMIT_OK, Segment, StepContext
from mstar.utils.containers import RecentSet
from mstar.utils.ipc_format import (
    Move,
    MoveKind,
    OffloadDelta,
    RemoveRequest,
    ScheduleTPNode,
)
from mstar.worker import worker as worker_mod
from mstar.worker.micro_scheduler import MicroScheduler
from mstar.worker.worker import Worker


def _no_kept_ids(rid):
    return None


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

    def __init__(self):
        self.offloaded: set[str] = set()
        self.holds_nothing: set[str] = set()
        self.refuse_offload = False
        self.refuse_reload = False

    def is_offloaded(self, rid):
        return rid in self.offloaded

    def is_resident(self, rid):
        return rid not in self.offloaded

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


class _Engine:
    """``Engine``'s eviction surface over one stub resource."""

    _journal = Engine._journal
    apply_resident_delta = Engine.apply_resident_delta
    check_ready = Engine.check_ready
    is_offloaded = Engine.is_offloaded
    is_resident = Engine.is_resident
    offload_request = Engine.offload_request
    paged_kv_caches = Engine.paged_kv_caches
    reclaimable = Engine.reclaimable
    reload_request = Engine.reload_request
    take_resident_delta = Engine.take_resident_delta

    def decide_cached_prefix(self, batch):
        return {}

    def page_trails(self, node_name):
        return {}

    def __init__(self, role: str = "leader"):
        """``role`` is this rank's part for ``"node"``: ``leader`` (rank 0 of a
        sharded node — decides and journals), ``follower`` (another rank of one —
        replays only), or ``solo`` (not sharded — owns its own eviction)."""
        assert role in ("leader", "follower", "solo"), role
        self.resource = _Resource()
        self._resources = {"kv": self.resource}
        self._submodules = {
            "node": SimpleNamespace(resources={"kv": self.resource})
        }
        self._runner = SimpleNamespace(admit_retrieve=lambda **kw: FULL_ADMIT_OK)
        self._resident_delta = (
            {"node": OffloadDelta.new()} if role == "leader" else {}
        )
        self._tp_follower_nodes = {"node"} if role == "follower" else set()
        self._graph_runtime = _Rids()


def _info():
    return SimpleNamespace(graph_walk="walk", resource_publish_info={})


def test_a_follower_may_not_start_an_offload_of_its_own():
    """Rank 0 decides what the instance evicts. A follower choosing for itself is
    the original deadlock: LRU orders on wall clock, so the ranks pick different
    victims and then a step one of them scheduled is one the other cannot join."""
    engine = _Engine("follower")

    assert engine.offload_request("node", "a") == 0
    assert not engine.is_offloaded("node", "a")
    assert engine.resource.offloaded == set(), "the resource was touched anyway"


def test_a_follower_may_not_start_a_reload_of_its_own():
    """A reload spends the pages an eviction freed, so it is equally rank 0's
    call — and reloading a rid the delta is about to evict would undo it."""
    engine = _Engine("follower")
    engine.resource.offloaded = {"a"}

    assert engine.reload_request("node", "a") is False
    assert engine.is_offloaded("node", "a"), "reloaded despite following"


def test_replay_is_exempt_from_both_guards():
    """The guards are on *initiating* a move. Replaying rank 0's is the one thing
    a follower must do, so it reaches the resources directly."""
    engine = _Engine("follower")
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "wire-a")

    assert engine.apply_resident_delta("node", delta, _no_kept_ids) is True
    assert engine.is_offloaded("node", "a")

    back = OffloadDelta.new()
    back.add(MoveKind.RELOAD, "wire-a")
    assert engine.apply_resident_delta("node", back, _no_kept_ids) is True
    assert not engine.is_offloaded("node", "a")


def test_an_unsharded_node_still_evicts_for_itself():
    """The guard is per node: a rank that follows one node still owns eviction on
    its own un-sharded ones."""
    engine = _Engine("solo")

    assert engine.offload_request("node", "a") == 1
    assert engine.is_offloaded("node", "a")
    assert engine.reload_request("node", "a") is True


def test_reload_request_reports_whether_it_worked():
    """``check_ready`` gates admission on this, so a successful reload that
    answers falsy makes the request permanently unschedulable — the whole rank
    then sits idle with work it will not look at."""
    engine = _Engine()
    engine.resource.offloaded = {"a"}

    assert engine.reload_request("node", "a") is True
    assert not engine.is_offloaded("node", "a")

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
        "node": SimpleNamespace(resources={"first": first, "second": second})
    }

    assert engine.reload_request("node", "a") is False
    assert len(engine.take_resident_delta("node")) == 0
    assert second.is_offloaded("a")
    assert first.is_offloaded("a") and first.free == 8, (
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
        "node": SimpleNamespace(resources={"holder": holder, "bystander": bystander})
    }

    assert engine.reload_request("node", "a") is True
    assert not engine.is_offloaded("node", "a")


def test_a_resource_that_cannot_fit_stops_the_reload_before_anything_moves():
    engine = _Engine()
    first, second = _Unsure(), _Unsure(fits=False)
    first.offloaded = {"a"}
    second.offloaded = {"a"}
    first.reload = second.reload = lambda rid: pytest.fail("reloaded past a refusal")
    engine._submodules = {
        "node": SimpleNamespace(resources={"first": first, "second": second})
    }

    assert engine.reload_request("node", "a") is False
    assert len(engine.take_resident_delta("node")) == 0


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

    assert _entries(engine.take_resident_delta("node")) == [
        ("wire-a", MoveKind.OFFLOAD), ("wire-a", MoveKind.RELOAD), ("wire-b", MoveKind.OFFLOAD),
    ]
    assert len(engine.take_resident_delta("node")) == 0, "drained once only"


def test_a_rank_that_leads_nothing_journals_nothing():
    """Only a leader replicates. An un-drained journal grows forever, which is
    what happens if the leader set is built with ``world_size > 0`` — true of
    every node — rather than ``> 1``."""
    engine = _Engine("solo")
    engine.offload_request("node", "a")

    assert engine.is_offloaded("node", "a"), "a solo rank still evicts"
    assert len(engine.take_resident_delta("node")) == 0


def test_replaying_a_delta_reproduces_the_leaders_state():
    engine = _Engine("follower")
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "wire-a")
    delta.add(MoveKind.RELOAD, "wire-a")
    delta.add(MoveKind.OFFLOAD, "wire-a")

    assert engine.apply_resident_delta("node", delta, _no_kept_ids) is True
    assert engine.is_offloaded("node", "a"), (
        "replay ended with 'a' resident; the leader has it on the host"
    )
    assert len(delta) == 0


def test_a_refused_entry_stops_the_replay_where_it_is():
    """Resumable, not restartable: the entries already applied are gone, so the
    retry does not redo them."""
    engine = _Engine("follower")
    engine.resource.refuse_offload = True
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "wire-a")
    delta.add(MoveKind.OFFLOAD, "wire-b")

    assert engine.apply_resident_delta("node", delta, _no_kept_ids) is False
    assert len(delta) == 2, "a refused entry must stay queued"

    engine.resource.refuse_offload = False
    assert engine.apply_resident_delta("node", delta, _no_kept_ids) is True
    assert engine.is_offloaded("node", "a") and engine.is_offloaded("node", "b")


def test_a_move_for_a_request_this_rank_has_not_taken_in_waits_for_it():
    """Nothing behind the move jumps ahead while it waits."""
    engine = _Engine("follower")
    engine._graph_runtime._live.discard("a")
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "wire-a")
    delta.add(MoveKind.OFFLOAD, "wire-b")

    assert engine.apply_resident_delta("node", delta, _no_kept_ids) is False
    assert len(delta) == 2 and not engine.is_offloaded("node", "b"), "a move ran ahead of one still owed"

    engine._graph_runtime._live.add("a")
    assert engine.apply_resident_delta("node", delta, _no_kept_ids) is True
    assert engine.is_offloaded("node", "a"), "the move was lost while it waited"


@pytest.mark.parametrize("kind", [kind for kind in MoveKind if kind is not MoveKind.OFFLOAD])
def test_a_move_of_any_kind_waits_for_a_request_this_rank_has_not_taken_in(kind):
    engine = _Engine("follower")
    engine._graph_runtime._live.discard("a")
    delta = OffloadDelta.new()
    delta.add(kind, "wire-a")
    delta.add(MoveKind.OFFLOAD, "wire-b")

    assert engine.apply_resident_delta("node", delta, _no_kept_ids) is False
    assert len(delta) == 2 and not engine.is_offloaded("node", "b"), "a move ran ahead of one still owed"


def test_a_request_holding_nothing_is_not_a_refusal():
    """``offload`` answers 0 both for "a step still holds these pages, retry" and
    for "there is nothing here to free". Only the first is retryable; treating
    the second as a refusal wedges the follower on a delta it can never finish."""
    engine = _Engine("follower")
    engine.resource.holds_nothing = {"a"}
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "wire-a")

    assert engine.apply_resident_delta("node", delta, _no_kept_ids) is True
    assert len(delta) == 0


def test_replaying_does_not_journal_it_onward():
    """A follower replaying rank 0 has nothing of its own to replicate; if it
    journalled, a rank that both leads and follows would echo."""
    engine = _Engine("leader")  # journalling is on
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "wire-a")

    engine.apply_resident_delta("node", delta, _no_kept_ids)

    assert len(engine.take_resident_delta("node")) == 0


# --- the replay says what it did --------------------------------------------


def test_each_replayed_move_is_logged(caplog):
    """So a run's two ranks can be lined up: the leader logs its eviction and its
    reload, and the follower logs replaying each of them.

    The WIRE id, not the local handle -- handles are minted per worker, so the
    two ranks' lines could not be matched up if they named handles."""
    engine = _Engine("follower")
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "wire-a")
    delta.add(MoveKind.RELOAD, "wire-a")

    with caplog.at_level(logging.INFO, logger=engine_mod.__name__):
        assert engine.apply_resident_delta("node", delta, _no_kept_ids) is True

    lines = [r.getMessage() for r in caplog.records]
    assert any("Replayed rank 0's offload of request wire-a from node" in ln for ln in lines), lines
    assert any("Replayed rank 0's reload of request wire-a on node" in ln for ln in lines), lines


def test_a_no_op_replay_is_not_logged_as_a_move(caplog):
    """An entry this rank already matches moved nothing, so it must not read as
    a move — that would make the two ranks' logs disagree."""
    engine = _Engine("follower")
    engine.resource.offloaded = {"a"}
    delta = OffloadDelta.new()
    delta.add(MoveKind.OFFLOAD, "wire-a")  # already off here

    with caplog.at_level(logging.INFO, logger=engine_mod.__name__):
        assert engine.apply_resident_delta("node", delta, _no_kept_ids) is True

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
    engine = _Engine.__new__(_Engine)
    engine._resident_delta = {}
    Engine._classify_tp_roles(engine, set(layout), _Groups(layout))
    return engine


def test_an_unsharded_node_is_neither_leader_nor_follower():
    """``world_size > 0`` is true of every node, so it would make this one a
    leader — and a leader journals resident-set moves that nothing drains,
    because the TP broadcast that drains them skips un-sharded nodes."""
    engine = _roles({"solo": (0, 1)})

    assert engine._tp_leader_nodes == set()
    assert engine._tp_follower_nodes == set()
    assert engine._resident_delta == {}, "an un-drained journal grows forever"


def test_rank_zero_of_a_sharded_node_leads_and_journals():
    engine = _roles({"shared": (0, 2)})

    assert engine._tp_leader_nodes == {"shared"}
    assert engine._tp_follower_nodes == set()
    assert set(engine._resident_delta) == {"shared"}


def test_another_rank_of_a_sharded_node_follows_and_does_not_journal():
    engine = _roles({"shared": (1, 2)})

    assert engine._tp_leader_nodes == set()
    assert engine._tp_follower_nodes == {"shared"}
    assert engine._resident_delta == {}


def test_the_roles_are_per_node():
    """One rank can lead one node, follow another, and own a third alone."""
    engine = _roles({"lead": (0, 2), "follow": (1, 2), "solo": (0, 1)})

    assert engine._tp_leader_nodes == {"lead"}
    assert engine._tp_follower_nodes == {"follow"}
    assert set(engine._resident_delta) == {"lead"}


# --- the wiring: the leader puts its moves on the wire ----------------------
#
# Both halves have to land together. The engine refuses a follower's own
# eviction, so if nothing supplies the replacement the follower simply never
# matches rank 0 — its admit keeps failing while rank 0 proceeds into the
# collective, and the run hangs with the follower idle.


class _Leader:
    """``maybe_send_zmq_to_tp_followers`` over stubs."""

    maybe_send_zmq_to_tp_followers = Worker.maybe_send_zmq_to_tp_followers
    _rid_str = Worker._rid_str

    def __init__(self, followers=("rank1", "rank2")):
        self.engine = _Engine("leader")
        self.engine_manager = SimpleNamespace(get_engine=lambda node: self.engine)
        self.parallel_nodes = {"node"}
        self.parallel_leader_nodes = {"node"}
        self._tp_broadcast_seq = 0
        self._tp_followers = {}
        self._victims = {}
        group = SimpleNamespace(_workers=["rank0", *followers])
        # The runtime owns both the sharding config and the handle table now.
        # Its ``get_rid_string`` is the same prefixing stub ``_Engine`` uses, so a
        # broadcast that shipped local handles instead of wire ids is visible.
        self._graph_runtime = SimpleNamespace(
            get_sharding_config=lambda rid: SimpleNamespace(
                get_sharding_group=lambda n, w: group
            ),
            get_rid_string=_Rids().get_rid_string,
        )
        # Live requests moved off the worker-graph manager onto here; the
        # broadcast reads it to say what its page state should add up to.
        self.request_state = SimpleNamespace(per_request_info={"r0": object()})
        self.sent: list = []
        self.communicator = SimpleNamespace(
            send=lambda worker, msg: self.sent.append((worker, msg))
        )


def _node_batch():
    return SimpleNamespace(
        node_name="node", graph_walk="walk", request_ids=("r0",), rebuild_rids=frozenset(),
        step_context=StepContext(
            request_ids=("r0",), graph_walk="walk", slot=0, capture=False,
        ),
    )


def test_the_broadcast_carries_the_leaders_moves():
    """Without this the delta is always empty and no follower ever replays."""
    leader = _Leader(followers=("rank1",))
    leader.engine.offload_request("node", "victim")

    leader.maybe_send_zmq_to_tp_followers(_node_batch())

    assert len(leader.sent) == 1
    body = leader.sent[0][1].body
    # wire ids, not local handles: the follower mints its own, so a broadcast
    # naming rank 0's would replay against the wrong requests
    assert _entries(body.resident_delta) == [("wire-victim", MoveKind.OFFLOAD)]
    assert body.request_ids == ["wire-r0"]


def test_every_follower_gets_the_same_moves_in_its_own_queue():
    """Drained once, copied per follower. Draining inside the send loop would
    give the first follower everything and leave the rest behind forever."""
    leader = _Leader(followers=("rank1", "rank2"))
    leader.engine.offload_request("node", "victim")

    leader.maybe_send_zmq_to_tp_followers(_node_batch())

    deltas = [msg.body.resident_delta for _, msg in leader.sent]
    assert len(deltas) == 2
    assert all(_entries(d) == [("wire-victim", MoveKind.OFFLOAD)] for d in deltas), (
        "a follower was sent an empty delta"
    )
    # and one replaying does not drain another's
    deltas[0].pop_left()
    assert len(deltas[1]) == 1


def test_the_broadcast_sends_the_rebuild_rows_apart_from_the_rows_the_follower_pops():
    leader, node_batch = _Leader(), _node_batch()
    node_batch.request_ids, node_batch.rebuild_rids = ("rb", "r0"), frozenset({"rb"})

    leader.maybe_send_zmq_to_tp_followers(node_batch)

    body = leader.sent[0][1].body
    assert (body.request_ids, body.rebuild_ids) == (["wire-r0"], ["wire-rb"]), "the follower can't pop a rebuild row"


def test_rank_0_drops_its_victim_before_it_drains_the_journal_for_the_step():
    """A follower drops the victim before it builds the step, so rank 0's drop has to travel with that step."""
    leader = _Leader()
    leader._victims = {"v": None}
    leader._move_victims = lambda node: leader.engine._resident_delta[node].add(MoveKind.DROP, "wire-v", count=8)

    leader.maybe_send_zmq_to_tp_followers(_node_batch())

    assert leader.sent[0][1].body.resident_delta.kinds == [MoveKind.DROP], "rank 0 admits a step its followers don't"


def test_the_journal_is_drained_by_the_broadcast():
    leader = _Leader(followers=("rank1",))
    leader.engine.offload_request("node", "victim")

    leader.maybe_send_zmq_to_tp_followers(_node_batch())
    leader.sent.clear()
    leader.maybe_send_zmq_to_tp_followers(_node_batch())

    assert len(leader.sent[0][1].body.resident_delta) == 0, (
        "the same move went out twice"
    )


# --- the wiring: a follower does not run rank 0's OOM recovery --------------


class _Queue:
    """Stands in for the graph runtime, which owns the ready queues."""

    def __init__(self):
        self.pushed_back: list[str] = []

    def push_back_node(self, node_name, rids, wg_ids):
        del node_name, wg_ids
        self.pushed_back.extend(rids)


class _Rank:
    """``_handle_allocation_failure`` over stubs, for one rank of ``"node"``."""

    _handle_allocation_failure = Worker._handle_allocation_failure
    _is_tp_follower_node = Worker._is_tp_follower_node
    _push_back_batch = Worker._push_back_batch
    _preempt = Worker._preempt

    def __init__(self, *, leader: bool):
        self._kv_caches = {}
        self.queue = _Queue()
        self._graph_runtime = self.queue
        self.held: list[str] = []
        self.scheduler = SimpleNamespace(hold_requests=self.held.extend)
        self._hold_logged = {}
        self.parallel_nodes = {"node"}
        self.parallel_leader_nodes = {"node"} if leader else set()
        self.offload_attempts: list[str] = []
        self._in_flight_rids: set[str] = set()
        # recorded rather than stubbed out: a teardown releases pages, so
        # spending one before evicting is the cheaper move and the order matters
        self.flushed_removes = 0
        self.flush_protected: set[str] | None = None

    def _apply_removes_whose_step_landed(self):
        self.flushed_removes += 1

    def _apply_pending_removes_safe_to_drop(self, in_flight_rids):
        self.flushed_removes += 1
        self.flush_protected = set(in_flight_rids)

    def _try_offload_cold_request(self, node_name, batch_ids, affected_resources=None):
        del batch_ids, affected_resources
        self.offload_attempts.append(node_name)
        return "victim"


def _oom(rank, referenced_rids=frozenset()):
    batch = SimpleNamespace(
        node_name="node", graph_walk="walk",
        request_to_worker_graph={"r0": "wg", "r1": "wg"},
    )
    node_batch = SimpleNamespace(node_name="node", failed_resource=None, admit_error=None)
    rank._handle_allocation_failure(batch, node_batch, referenced_rids)


def test_a_follow_rank_neither_evicts_nor_holds():
    """What the log showed it doing instead: "no offload possible, holding 15
    requests". The eviction is rank 0's and arrives as a delta, and a follow rank
    has no scheduling decision a backoff could improve — the hold only delays
    building the ScheduleTPNode that carries the fix."""
    follower = _Rank(leader=False)

    _oom(follower)

    assert follower.offload_attempts == [], "a follow rank picked its own victim"
    assert follower.held == []
    # the batch still has to go back, or rank 0's retry finds nothing ready
    assert sorted(follower.queue.pushed_back) == ["r0", "r1"]


def test_rank_zero_still_evicts_and_holds_its_victim():
    leader = _Rank(leader=True)

    _oom(leader)

    assert leader.offload_attempts == ["node"]
    assert leader.flushed_removes == 2, (
        "a pending teardown releases pages, so it has to be spent before an "
        "eviction is chosen — otherwise a page move happens that need not"
    )
    assert leader.held == ["victim"]
    assert sorted(leader.queue.pushed_back) == ["r0", "r1"]


def test_an_unsharded_node_still_runs_its_own_recovery():
    solo = _Rank(leader=False)
    solo.parallel_nodes = set()  # "node" is not sharded here

    _oom(solo)

    assert solo.offload_attempts == ["node"]
    assert solo.held == ["victim"]


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

    assert _entries(engine.take_resident_delta("node")) == [
        ("wire-a", MoveKind.OFFLOAD), ("wire-a", MoveKind.RELOAD),
    ]
def test_a_refused_replay_says_which_move_it_stopped_on(caplog):
    """A follower that stops mid-delta and never finishes leaves rank 0 on a
    collective it will never join — and until now that was silent."""
    engine = _Engine("follower")
    engine.resource.refuse_reload = True
    engine.resource.offloaded = {"a"}
    delta = OffloadDelta.new()
    delta.add(MoveKind.RELOAD, "wire-a")

    with caplog.at_level(logging.WARNING, logger=engine_mod.__name__):
        assert engine.apply_resident_delta("node", delta, _no_kept_ids) is False

    line = next(r.getMessage() for r in caplog.records if "does not fit" in r.getMessage())
    assert "reload of wire-a on node" in line
    assert "1 moves still owed" in line


# --- neither consumer of the follow FIFO builds mid-replay -------------------
#
# A step admitted against a half-replayed page state is one the other rank
# admits differently, which deadlocks the next collective. Both the serial
# path and the async follower go through ``settle_tp_follow_delta``.


def test_the_serial_follow_path_will_not_build_mid_replay():
    """A refused replay leaves moves owed, so the rids the head names do not mean
    here what they meant on the leader. Wait rather than build."""
    engine = SimpleNamespace(
        apply_resident_delta=lambda node, delta, kept_ids: False,  # refuses
        is_offloaded=lambda node, rid: False,
        check_ready=lambda *a, **kw: FULL_ADMIT_OK,
        get_max_batch_size=lambda node, walk: None,
    )
    sched = MicroScheduler(
        engine_manager=SimpleNamespace(get_engine=lambda node: engine),
        parallel_leader_nodes=set(),
    )
    message = ScheduleTPNode(
        node_name="node", graph_walk="walk", request_ids=["r0"], spec_seq=5,
    )
    message.resident_delta.add(MoveKind.OFFLOAD, "wire-victim")
    sched.register_tp_follow(message)

    assert sched._try_schedule_tp_follow(
        SimpleNamespace(per_request_info={"r0": object()})
    ) is None
    assert sched.peek_tp_follow() is message, "the head was consumed mid-replay"


# --- a forwarded teardown is ordered against the step stream -----------------
#
# A teardown releases pages, and the resident-set delta cannot describe it: the
# request is not offloaded, it is gone. So it carries the step rank 0 released it
# after, and a follower holds it until it has consumed that step.


class _RemoveRank:
    """``_remove_request``'s ordering, over stubs."""

    _apply_removes_whose_step_landed = Worker._apply_removes_whose_step_landed
    _removal_step_reached = Worker._removal_step_reached

    def __init__(self, consumed: int):
        self.scheduler = SimpleNamespace(last_consumed_tp_seq=consumed)
        self._removes_awaiting_step: dict[int, list[str]] = {}
        self.applied: list[str] = []

    def _remove_request(self, body):
        self.applied.append(body.request_id)


def test_a_removal_for_a_step_not_yet_reached_waits():
    rank = _RemoveRank(consumed=5)

    assert rank._removal_step_reached(
        RemoveRequest(request_id="r", after_tp_seq=7)
    ) is False


def test_a_removal_for_a_step_already_consumed_applies():
    rank = _RemoveRank(consumed=7)

    assert rank._removal_step_reached(
        RemoveRequest(request_id="r", after_tp_seq=7)
    ) is True


def test_an_unstamped_removal_is_not_ordered():
    """The conductor's own sends, and a rank re-applying its own deferral."""
    rank = _RemoveRank(consumed=-1)

    assert rank._removal_step_reached(RemoveRequest(request_id="r")) is True


def test_a_held_removal_lands_once_its_step_does():
    rank = _RemoveRank(consumed=5)
    rank._removes_awaiting_step = {7: ["late"], 4: ["early"]}

    rank._apply_removes_whose_step_landed()

    assert rank.applied == ["early"], "applied a teardown from a step not reached"
    assert rank._removes_awaiting_step == {7: ["late"]}

    rank.scheduler.last_consumed_tp_seq = 7
    rank._apply_removes_whose_step_landed()

    assert rank.applied == ["early", "late"]
    assert rank._removes_awaiting_step == {}


def test_held_removals_land_in_step_order():
    """Two teardowns from different gaps have to be applied in the order rank 0
    made them, for the same reason the delta is a queue."""
    rank = _RemoveRank(consumed=9)
    rank._removes_awaiting_step = {8: ["second"], 3: ["first"]}

    rank._apply_removes_whose_step_landed()

    assert rank.applied == ["first", "second"]


class _ForwardingLeader:
    """Rank 0's ``_remove_request``, over stubs, to pin what it forwards."""

    _remove_request = Worker._remove_request
    _removal_step_reached = Worker._removal_step_reached
    _rid = Worker._rid
    _mark_removed = Worker._mark_removed

    def __init__(self, broadcast_seq: int):
        self.is_tp_follower = False
        self._tp_broadcast_seq = broadcast_seq
        # handle-keyed state, per rid_table's list
        self._in_flight_rids: set[int] = set()
        self._pending_removes: dict[int, None] = {}
        self._last_active: dict = {}
        self._kv_keys, self._kept_ids, self._rebuild_rows = {}, {}, {}
        self._dropped, self._victims, self._preempted = {}, {}, {}
        self.streaming_buffers: dict = {}
        self._stream_buffer_rids: set[int] = set()
        # wire-string-keyed: parked teardowns and the drain bookkeeping, because
        # a handle is recycled and would reattach to the next request to get it
        self._removes_awaiting_step: dict[int, list[str]] = {}
        self._draining_rids: set[str] = set()
        self._pending_drains: set[str] = set()
        self._reads_done_sent: set[str] = set()
        self._removed_rids: RecentSet[str] = RecentSet(16)
        self._unprocessed_messages: dict = {}
        group = SimpleNamespace(tp_size=2, _tp_rank=0, _workers=["rank0", "rank1"])
        # The runtime owns the handle table and the sharding config now, and
        # ``remove_request`` there is what frees the handle.
        self._graph_runtime = SimpleNamespace(
            get_rid_handle=lambda rid: 0 if rid == "gone" else None,
            get_rid_string=lambda handle: "gone" if handle == 0 else None,
            get_sharding_config=lambda rid: SimpleNamespace(groups=[group]),
            remove_request=lambda rid: None,
        )
        self.request_state = SimpleNamespace(remove_request=lambda rid: None)
        self.engine_manager = SimpleNamespace(
            remove_request=lambda rid: None, evictable_nodes=lambda: (),
        )
        self.tensor_manager = SimpleNamespace(force_cleanup_request=lambda rid: None)
        self.profile_info = SimpleNamespace(pop_request=lambda rid: None)
        self.scheduler = SimpleNamespace(
            clear_rid=lambda rid, wire_rid: None,
            clear_wire_rid=lambda rid: None,
            last_consumed_tp_seq=-1,
        )
        self.sent: list = []
        self.communicator = SimpleNamespace(
            send=lambda worker, msg: self.sent.append((worker, msg))
        )


def test_the_forwarded_teardown_carries_the_step_it_follows():
    """Without the stamp a follower has nothing to order the teardown against,
    and applies it on whichever side of the step its loop happens to be on."""
    leader = _ForwardingLeader(broadcast_seq=1229)

    leader._remove_request(RemoveRequest(request_id="gone"))

    assert [w for w, _ in leader.sent] == ["rank1"]
    body = leader.sent[0][1].body
    assert body.request_id == "gone"
    assert body.after_tp_seq == 1228, (
        "the last step broadcast, which is the gap this rank released the pages in"
    )


def test_a_rank_that_leads_never_parks_a_teardown():
    """``last_consumed_tp_seq`` only advances on a rank that consumes follow
    steps, so a leader that parked a stamped teardown would hold it for good —
    and hold its pages with it."""
    leader = _ForwardingLeader(broadcast_seq=1229)

    # stamped, as if it had somehow arrived ordered
    leader._remove_request(
        RemoveRequest(request_id="gone", after_tp_seq=9999)
    )

    assert leader._removes_awaiting_step == {}, "a leader parked a teardown"
    assert [w for w, _ in leader.sent] == ["rank1"], "and it still forwarded"


def test_the_teardown_flush_spares_only_what_the_speculation_touches():
    """The failed step's own pages are fair game — its admit refused, so no
    forward ran. What is not is whatever the speculation touches: its pre-plan
    may have pre-admitted, marking those streams in flight, and the speculation
    is not dropped until after this returns."""
    leader = _Rank(leader=True)

    _oom(leader, referenced_rids=frozenset({"spec_rid"}))

    assert leader.flush_protected == {"spec_rid"}, (
        "the flush protected the failed step's rids, whose pages are reclaimable"
    )


def test_the_teardown_flush_runs_after_the_push_back():
    """A rid torn down by the flush must not then be pushed onto queues that the
    teardown has just dismantled."""
    leader = _Rank(leader=True)
    order: list[str] = []
    leader.queue.push_back_node = (
        lambda node_name, rids, wg_ids: order.append("push")
    )
    leader._apply_pending_removes_safe_to_drop = (
        lambda in_flight: order.append("flush")
    )

    _oom(leader)

    assert order.index("push") < order.index("flush"), order


def test_the_speculation_is_cleared_before_the_recovery_runs():
    """Ordering, and it buys two things. Dropping the pre-plan releases the pages
    it reserved, so the eviction that follows may be unnecessary; and it unmarks
    those streams, so the teardown flush has nothing left to spare. Only a
    yield-away batch survives the clear, and only that needs sparing."""
    import inspect

    src = inspect.getsource(worker_mod.Worker.run)
    block = src[src.index("if pending.node_batch.admit_error is not None:"):]
    clear = block.index("_maybe_clear_spec()")
    recover = block.index("_handle_admit_failure(")
    assert clear < recover, (
        "recovery runs before the speculation is cleared, so it evicts to make "
        "room the pre-plan was about to give back"
    )


class _FollowSpec:
    """``Worker._try_follow_speculation`` over stubs, as far as the delta gate.

    A real ``MicroScheduler`` so the gate is driven by a real refusal rather than
    a stubbed return: the bug was the async path never asking, and a stub that
    answers proves nothing about whether it asked.
    """

    _try_follow_speculation = Worker._try_follow_speculation

    def __init__(self, *, replay_refuses: bool):
        self.worker_id = "worker_1"
        self.engine = SimpleNamespace(
            apply_resident_delta=lambda node, delta, kept_ids: not replay_refuses,
            is_offloaded=lambda node, rid: False,
            reclaimable=lambda node, rid: 0,
            check_ready=lambda *a, **kw: FULL_ADMIT_OK,
            get_max_batch_size=lambda node, walk: None,
        )
        self.engine_manager = SimpleNamespace(get_engine=lambda node: self.engine)
        self.scheduler = MicroScheduler(
            engine_manager=self.engine_manager, parallel_leader_nodes=set(),
        )
        # the worker installs this; ``_check_resident_set_matches`` needs it to
        # compare page state in wire ids (and goes away with that check)
        # the build asks for the spec target right after reading the parent's
        # batch; None bails it out there, just past the gate
        self._graph_runtime = SimpleNamespace(
            get_spec_target=lambda *a, **kw: None,
        )
        # live requests live on the request-state manager now
        self.request_state = SimpleNamespace(per_request_info={})
        self.head = head = ScheduleTPNode(
            node_name="node", graph_walk="walk", request_ids=["r0"],
            speculative=True, spec_seq=6, spec_from_seq=5,
        )
        head.resident_delta.add(MoveKind.OFFLOAD, "victim")
        self.scheduler.register_tp_follow(head)


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

            @property
            def request_to_worker_graph(self):
                parent.reads += 1
                return {"r0": 0}

        self.batch = _Batch()
        self.node_batch = SimpleNamespace(commit_done=threading.Event())


def test_the_async_follow_path_will_not_build_mid_replay():
    """The serial path waits while moves are owed; this one used to skip the delta
    entirely, read the half-applied state as "rid not ready", and poll for ever."""
    worker = _FollowSpec(replay_refuses=True)
    parent = _SpecParent()

    assert worker._try_follow_speculation(parent) is None
    assert parent.reads == 0, "the build started before the replay had landed"
    assert worker.scheduler.peek_tp_follow() is worker.head, (
        "the head was consumed mid-replay"
    )


def test_the_async_follow_path_proceeds_once_the_replay_lands():
    """The other direction: the gate must not be a permanent stop. Reaching the
    parent is what says the delta settled and the gate passed."""
    worker = _FollowSpec(replay_refuses=False)
    parent = _SpecParent()

    assert worker._try_follow_speculation(parent) is None
    assert parent.reads == 1, "the gate blocked a build it should have allowed"


# --- rank 0's pool moves ------------------------------------------------------
#
# Rank 0 allots, earmarks, drops and rebuilds, and records each move in its journal.
# A follower makes the same moves from the delta, so both ranks admit every step from
# the same counts and free pages in the same order. If a grant past the room passes on
# one rank and fails on the other, only the rank where it passed enters the next collective.


def _pool_rank(monkeypatch, rank: int) -> tuple[KVManager, Engine]:
    """Rank ``rank`` of two: a pool of 10 pages for requests, under the `Engine` methods that journal
    (rank 0) and replay (rank 1) its moves."""
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)
    kv = KVManager(
        cfg=PagedKVConfig(num_layers=1, num_kv_heads=2, head_dim=8, max_seq_len=4096, max_num_pages=11,
                          page_size=PAGE_SIZE),
        name="kv", joint_comm_group=SimpleNamespace(rank=rank, world_size=2), transfer_engine_info=None,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    kv.local_loops = {"decode"}
    engine = Engine.__new__(Engine)
    engine._resources, engine._resource_nodes = {"kv": kv}, {"kv": frozenset({NODE})}
    engine._submodules = {NODE: SimpleNamespace(resources={"kv": kv}, submodule=SimpleNamespace(
        cleanup_request=lambda rid: None))}
    engine._runner, engine._loopback_steps = StepRunner({"kv": kv}), {}
    engine._graph_runtime = SimpleNamespace(
        get_rid_string=str, get_rid_handle=lambda rid: rid if rid in kv._streams else None,
    )
    engine._tp_follower_nodes = {NODE} if rank else set()
    engine._resident_delta = {} if rank else {NODE: OffloadDelta.new()}
    if not rank:
        kv.journal = partial(engine._journal, NODE, "kv")
    return kv, engine


def _ingest(kvs, rid: str, pages: int, arrival: int) -> None:
    """``rid``'s prompt fills ``pages`` pages, and it decodes one token per iteration."""
    tokens = pages * PAGE_SIZE
    layout = StreamLayout(spans=(PrefixSpan(tokens, tokens, "prefill"),), growth=(("decode", 1),))
    for kv in kvs:
        kv.ingest_request(rid, KVReqConfig(
            kv_layout={"main": layout}, loop_caps={"decode": 4 * PAGE_SIZE}, arrival=arrival,
            prompt_ids={"main": bytes(4 * tokens)},
        ))


def _counts(kv) -> tuple:
    """What the next admit reads, and the order the next grant takes free pages in."""
    return (
        kv._unused, {rid: (a.pages, a.remaining, a.prompt) for rid, a in kv._allotments.items()},
        {rid: (e.pages, sorted(e.victims), e.freed) for rid, e in kv._earmarks.items()},
        sorted(rid for rid, earmark in kv._victims.items() if earmark is not None), sorted(kv._dropped),
        list(kv._arena.allocator.free_pages.queue),
    )


def _kept(rid):
    return [0] * 8


def _both_short_then_drop(monkeypatch):
    """r (arrival 1) grows past its allotment until its next token finds no room, on both ranks.
    Rank 0 then chooses v (arrival 2) and drops it."""
    (leader, lead), (follower, follow) = _pool_rank(monkeypatch, 0), _pool_rank(monkeypatch, 1)
    _ingest((leader, follower), "r", 3, arrival=1)
    _ingest((leader, follower), "v", 5, arrival=2)
    assert leader.admit_retrieve("r", NODE, "prefill", None).ready
    assert leader.admit_retrieve("v", NODE, "prefill", None).ready
    assert follow.apply_resident_delta(NODE, lead.take_resident_delta(NODE), _kept)
    for rid, tokens in (("r", 3 * PAGE_SIZE), ("v", 5 * PAGE_SIZE), ("r", 2 * PAGE_SIZE), ("r", 1)):
        assert _step(leader, rid, tokens).ok == _step(follower, rid, tokens).ok, f"one rank alone admitted {rid}"
        assert _counts(leader) == _counts(follower)
    assert leader.preempt("r", 1, droppable=lambda rid: True) == ["v"]
    lead.drop_request(NODE, "v", len(_kept("v")))
    return (leader, lead), (follower, follow)


def test_a_follower_admits_each_step_from_the_counts_rank_0_decided(monkeypatch):
    (leader, lead), (follower, follow) = _both_short_then_drop(monkeypatch)

    delta = lead.take_resident_delta(NODE)
    assert delta.kinds == [MoveKind.EARMARK, MoveKind.DROP]
    assert follow.apply_resident_delta(NODE, delta, _kept)

    assert _counts(leader) == _counts(follower), "the follower's earmark or drop left other counts"
    assert _step(leader, "r", 1).ok and _step(follower, "r", 1).ok, "the page v's drop freed went elsewhere on one rank"
    assert _counts(leader) == _counts(follower)


def test_a_follower_takes_a_dropped_request_back_and_lays_it_out_as_rank_0_did(monkeypatch):
    (leader, lead), (follower, follow) = _both_short_then_drop(monkeypatch)
    assert follow.apply_resident_delta(NODE, lead.take_resident_delta(NODE), _kept)
    for kv, engine in ((leader, lead), (follower, follow)):
        assert _step(kv, "r", 1).ok
        engine.remove_request("r")

    assert leader.readmit("v", len(_kept("v")))
    lead.lay_out_rebuild(NODE, "v", _kept("v"))
    assert follow.apply_resident_delta(NODE, lead.take_resident_delta(NODE), _kept)

    assert _counts(leader) == _counts(follower), "the follower holds another allotment for the return"
    assert follower._overrides["v"].kv_layout == leader._overrides["v"].kv_layout, "the rebuild writes another span"


def test_a_follower_takes_a_returning_request_as_rank_0s_victim_again(monkeypatch):
    """A dropped request reserved again for its return holds pages a grant can take, so rank 0 may choose it."""
    (leader, lead), (follower, follow) = _both_short_then_drop(monkeypatch)
    assert leader.readmit("v", len(_kept("v")))
    lead.lay_out_rebuild(NODE, "v", _kept("v"))
    assert follow.apply_resident_delta(NODE, lead.take_resident_delta(NODE), _kept)
    _ingest((leader, follower), "x", 1, arrival=0)
    leader._allotments["x"] = follower._allotments["x"] = Allotment(pages=1)

    assert leader.preempt("x", 1, droppable=lambda rid: True) == ["v"]
    assert follow.apply_resident_delta(NODE, lead.take_resident_delta(NODE), _kept)
    assert _counts(leader) == _counts(follower)


def test_a_drop_whose_victim_is_in_a_step_here_waits_and_nothing_behind_it_jumps_ahead(monkeypatch):
    (leader, lead), (follower, follow) = _both_short_then_drop(monkeypatch)
    assert leader.readmit("v", len(_kept("v")))
    delta = lead.take_resident_delta(NODE)
    step, ctx = KVStep(segments=(Segment("v", "main", 0),)), StepContext(("v",), "prefill", 0, False)
    assert follower.admit(step, ctx).ok

    assert not follow.apply_resident_delta(NODE, delta, _kept)
    assert delta.kinds == [MoveKind.DROP, MoveKind.RESERVE], "the drop or the return behind it moved early"
    assert "v" not in follower._dropped and "v" in follower._allotments

    follower.rollback_admit(step, ctx)
    assert follow.apply_resident_delta(NODE, delta, _kept)
    assert _counts(leader) == _counts(follower)


def test_an_allotment_undone_in_the_sweep_that_made_it_never_reaches_the_followers(monkeypatch):
    """A node's second pool can hold a request back on every readiness check, undoing the first pool's allotment."""
    leader, lead = _pool_rank(monkeypatch, 0)
    _ingest((leader,), "x", 1, arrival=1)
    assert leader.admit_retrieve("x", NODE, "prefill", None).ready
    leader.requeue("x")

    assert not lead.take_resident_delta(NODE), "two moves per pass would pile up for each follower to replay"


def test_rank_0_strips_a_removed_requests_moves_from_its_journal(monkeypatch):
    leader, lead = _pool_rank(monkeypatch, 0)
    _ingest((leader,), "x", 1, arrival=1)
    _ingest((leader,), "y", 1, arrival=2)
    assert leader.admit_retrieve("x", NODE, "prefill", None).ready
    assert leader.admit_retrieve("y", NODE, "prefill", None).ready

    lead.remove_request("x")

    assert lead.take_resident_delta(NODE).rids == ["y"], "a follower would wait for a request it has torn down"


def test_a_follower_whose_counts_cant_take_rank_0s_move_stops(monkeypatch):
    follower, follow = _pool_rank(monkeypatch, 1)
    _ingest((follower,), "x", 1, arrival=1)
    head = ScheduleTPNode(NODE, "prefill", ["x"], spec_seq=0)
    for kind in (MoveKind.RESERVE, MoveKind.UNRESERVE, MoveKind.UNRESERVE):
        head.resident_delta.add(kind, "x", "kv", 1)
    scheduler = MicroScheduler(engine_manager=SimpleNamespace(get_engine=lambda node: follow))
    scheduler.register_tp_follow(head)

    with pytest.raises(SystemExit, match="TP ranks diverged"):
        scheduler.settle_tp_follow_delta(caught_up=True)


def test_a_follower_builds_rank_0s_rebuild_rows_and_runs_nothing_for_moves_alone():
    engine = SimpleNamespace(apply_resident_delta=lambda node, delta, kept_ids: True)
    scheduler = MicroScheduler(engine_manager=SimpleNamespace(get_engine=lambda node: engine))
    scheduler.rid_of = {"r0": 7}.get
    moves = ScheduleTPNode(NODE, "prefill", [], spec_seq=3)
    moves.resident_delta.add(MoveKind.DROP, "r0", count=8)
    rebuild = ScheduleTPNode(NODE, "prefill", [], spec_seq=4, rebuild_ids=["r0"])
    for message in (moves, rebuild):
        scheduler.register_tp_follow(message)

    assert scheduler._try_schedule_tp_follow(None) is None and scheduler.peek_tp_follow() is rebuild
    batch = scheduler._try_schedule_tp_follow(None)
    assert batch.rebuild_rids == (7,) and not batch.request_to_worker_graph, "the follower's step isn't rank 0's"


def test_rank_0_holds_a_victim_and_the_request_it_was_chosen_for_until_the_drop(monkeypatch):
    (leader, lead), (follower, _) = _pool_rank(monkeypatch, 0), _pool_rank(monkeypatch, 1)
    _ingest((leader, follower), "r", 3, arrival=1)
    _ingest((leader, follower), "v", 5, arrival=2)
    assert leader.admit_retrieve("r", NODE, "prefill", None).ready
    assert leader.admit_retrieve("v", NODE, "prefill", None).ready
    for rid, tokens in (("r", 3 * PAGE_SIZE), ("v", 5 * PAGE_SIZE), ("r", 2 * PAGE_SIZE), ("r", 1)):
        _step(leader, rid, tokens)
    assert leader.preempt("r", 1, droppable=lambda rid: True) == ["v"]

    for rid in ("r", "v"):
        assert not leader.admit_retrieve(rid, NODE, "decode", None).ready, f"a step before the drop could take {rid}"
    lead.drop_request(NODE, "v", len(_kept("v")))
    assert leader.admit_retrieve("r", NODE, "decode", None).ready, "r still waits after the drop went out"
