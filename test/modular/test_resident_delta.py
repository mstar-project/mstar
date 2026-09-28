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
from types import SimpleNamespace

sys.path.insert(0, ".")

from mstar.engine import engine as engine_mod
from mstar.engine.engine import Engine
from mstar.engine.resources.step import FULL_ADMIT_OK
from mstar.utils.ipc_format import OffloadDelta
from mstar.worker.worker import Worker


def _entries(delta: OffloadDelta) -> list[tuple[str, bool]]:
    return list(zip(delta.rids, delta.is_offload, strict=True))


# --- the queue itself -------------------------------------------------------


def test_the_order_of_offloads_and_reloads_survives():
    """The reason this is a queue and not two sets."""
    delta = OffloadDelta.new()
    delta.add_offloaded("a")
    delta.add_reloaded("a")
    delta.add_offloaded("a")

    assert _entries(delta) == [("a", True), ("a", False), ("a", True)]


def test_pop_left_takes_the_oldest_entry():
    delta = OffloadDelta.new()
    delta.add_offloaded("a")
    delta.add_reloaded("b")

    assert delta.pop_left() == ("a", True)
    assert len(delta) == 1, "pop_left returned an entry without removing it"
    assert delta.pop_left() == ("b", False)
    assert len(delta) == 0


def test_pop_left_on_an_empty_queue_answers_none():
    """Callers drain until empty, so the empty case has to be an answer rather
    than an IndexError."""
    assert OffloadDelta.new().pop_left() is None


def test_peek_left_does_not_consume():
    delta = OffloadDelta.new()
    delta.add_offloaded("a")

    assert delta.peek_left() == ("a", True)
    assert len(delta) == 1
    assert OffloadDelta.new().peek_left() is None


def test_take_hands_the_queue_over_and_leaves_it_empty():
    """What a leader does on every broadcast. The fields are a NamedTuple's, so
    this cannot work by reassigning them — only the deques can be mutated."""
    delta = OffloadDelta.new()
    delta.add_offloaded("a")
    delta.add_reloaded("b")

    taken = delta.take()

    assert _entries(taken) == [("a", True), ("b", False)]
    assert len(delta) == 0, "the source kept the entries it handed over"

    # and the two are independent afterwards
    delta.add_offloaded("c")
    assert _entries(taken) == [("a", True), ("b", False)]


def test_extend_appends_in_order():
    """A follower folds each message's delta onto whatever it still owes."""
    owed = OffloadDelta.new()
    owed.add_offloaded("a")
    arriving = OffloadDelta.new()
    arriving.add_reloaded("b")

    owed.extend(arriving)

    assert _entries(owed) == [("a", True), ("b", False)]


def test_an_empty_delta_is_falsy():
    """``__len__`` is the queue depth, not the tuple's two fields — several
    callers branch on the delta directly."""
    assert not OffloadDelta.new()
    delta = OffloadDelta.new()
    delta.add_offloaded("a")
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

    def reclaimable(self, rid):
        return 0 if rid in self.holds_nothing else 1

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

    apply_resident_delta = Engine.apply_resident_delta
    check_ready = Engine.check_ready
    is_offloaded = Engine.is_offloaded
    offload_request = Engine.offload_request
    reclaimable = Engine.reclaimable
    reload_request = Engine.reload_request
    take_resident_delta = Engine.take_resident_delta

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
    delta.add_offloaded("wire-a")

    assert engine.apply_resident_delta("node", delta) is True
    assert engine.is_offloaded("node", "a")

    back = OffloadDelta.new()
    back.add_reloaded("wire-a")
    assert engine.apply_resident_delta("node", back) is True
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
        ("wire-a", True), ("wire-a", False), ("wire-b", True),
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
    delta.add_offloaded("wire-a")
    delta.add_reloaded("wire-a")
    delta.add_offloaded("wire-a")

    assert engine.apply_resident_delta("node", delta) is True
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
    delta.add_offloaded("wire-a")
    delta.add_offloaded("wire-b")

    assert engine.apply_resident_delta("node", delta) is False
    assert len(delta) == 2, "a refused entry must stay queued"

    engine.resource.refuse_offload = False
    assert engine.apply_resident_delta("node", delta) is True
    assert engine.is_offloaded("node", "a") and engine.is_offloaded("node", "b")


def test_a_move_for_a_request_this_rank_no_longer_has_is_spent():
    """``get_rid_handle`` answers None for a rid this rank has already removed --
    a real race, since the conductor's REMOVE and rank 0's delta come from
    different peers. Its pages went with the teardown, so the move is moot.

    It must be POPPED, not retried, or the replay wedges on an entry that can
    never land; and it must not reach the resources, which is what an unguarded
    None did -- ``offload(None)`` logged a move of "request None"."""
    engine = _Engine("follower")
    delta = OffloadDelta.new()
    delta.add_offloaded("wire-gone")     # _Rids answers None for unknown ids
    delta.add_offloaded("wire-a")        # and the rest of the queue still lands

    assert engine.apply_resident_delta("node", delta) is True
    assert len(delta) == 0, "the moot entry was left to be retried for ever"
    assert engine.is_offloaded("node", "a"), "it stopped short of the next move"
    assert None not in engine.resource.offloaded, "None reached the resources"


def test_a_request_holding_nothing_is_not_a_refusal():
    """``offload`` answers 0 both for "a step still holds these pages, retry" and
    for "there is nothing here to free". Only the first is retryable; treating
    the second as a refusal wedges the follower on a delta it can never finish."""
    engine = _Engine("follower")
    engine.resource.holds_nothing = {"a"}
    delta = OffloadDelta.new()
    delta.add_offloaded("wire-a")

    assert engine.apply_resident_delta("node", delta) is True
    assert len(delta) == 0


def test_replaying_does_not_journal_it_onward():
    """A follower replaying rank 0 has nothing of its own to replicate; if it
    journalled, a rank that both leads and follows would echo."""
    engine = _Engine("leader")  # journalling is on
    delta = OffloadDelta.new()
    delta.add_offloaded("wire-a")

    engine.apply_resident_delta("node", delta)

    assert len(engine.take_resident_delta("node")) == 0


# --- the replay says what it did --------------------------------------------


def test_each_replayed_move_is_logged(caplog):
    """So a run's two ranks can be lined up: the leader logs its eviction and its
    reload, and the follower logs replaying each of them.

    The WIRE id, not the local handle -- handles are minted per worker, so the
    two ranks' lines could not be matched up if they named handles."""
    engine = _Engine("follower")
    delta = OffloadDelta.new()
    delta.add_offloaded("wire-a")
    delta.add_reloaded("wire-a")

    with caplog.at_level(logging.INFO, logger=engine_mod.__name__):
        assert engine.apply_resident_delta("node", delta) is True

    lines = [r.getMessage() for r in caplog.records]
    assert any("Replayed rank 0's offload of request wire-a from node" in ln for ln in lines), lines
    assert any("Replayed rank 0's reload of request wire-a on node" in ln for ln in lines), lines


def test_a_no_op_replay_is_not_logged_as_a_move(caplog):
    """An entry this rank already matches moved nothing, so it must not read as
    a move — that would make the two ranks' logs disagree."""
    engine = _Engine("follower")
    engine.resource.offloaded = {"a"}
    delta = OffloadDelta.new()
    delta.add_offloaded("wire-a")  # already off here

    with caplog.at_level(logging.INFO, logger=engine_mod.__name__):
        assert engine.apply_resident_delta("node", delta) is True

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
        self.sent: list = []
        self.communicator = SimpleNamespace(
            send=lambda worker, msg: self.sent.append((worker, msg))
        )


def _node_batch():
    return SimpleNamespace(
        node_name="node", graph_walk="walk", request_ids=("r0",),
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
    assert _entries(body.resident_delta) == [("wire-victim", True)]
    assert body.request_ids == ["wire-r0"]


def test_every_follower_gets_the_same_moves_in_its_own_queue():
    """Drained once, copied per follower. Draining inside the send loop would
    give the first follower everything and leave the rest behind forever."""
    leader = _Leader(followers=("rank1", "rank2"))
    leader.engine.offload_request("node", "victim")

    leader.maybe_send_zmq_to_tp_followers(_node_batch())

    deltas = [msg.body.resident_delta for _, msg in leader.sent]
    assert len(deltas) == 2
    assert all(_entries(d) == [("wire-victim", True)] for d in deltas), (
        "a follower was sent an empty delta"
    )
    # and one replaying does not drain another's
    deltas[0].pop_left()
    assert len(deltas[1]) == 1


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

    def __init__(self, *, leader: bool):
        self.queue = _Queue()
        self._graph_runtime = self.queue
        self.held: list[str] = []
        self.scheduler = SimpleNamespace(hold_requests=self.held.extend)
        self._hold_logged = {}
        self.parallel_nodes = {"node"}
        self.parallel_leader_nodes = {"node"} if leader else set()
        self.offload_attempts: list[str] = []

    def _try_offload_cold_request(self, node_name, batch_ids, affected_resources=None):
        del batch_ids, affected_resources
        self.offload_attempts.append(node_name)
        return "victim"


def _oom(rank):
    batch = SimpleNamespace(
        node_name="node", graph_walk="walk",
        node_objects={"r0": object(), "r1": object()},
        request_to_worker_graph={"r0": "wg", "r1": "wg"},
    )
    node_batch = SimpleNamespace(node_name="node", failed_resource=None)
    rank._handle_allocation_failure(batch, node_batch)


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
    assert leader.held == ["victim"]
    assert sorted(leader.queue.pushed_back) == ["r0", "r1"]


def test_an_unsharded_node_still_runs_its_own_recovery():
    solo = _Rank(leader=False)
    solo.parallel_nodes = set()  # "node" is not sharded here

    _oom(solo)

    assert solo.offload_attempts == ["node"]
    assert solo.held == ["victim"]
