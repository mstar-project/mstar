"""A request whose grant runs short moves newer requests off the device and takes the pages they free.

A new prompt waits in the runner's queue for room, since a pool admits only its oldest waiting request.
A request that grows past what it was allotted can't wait that way: every request beside it
may be growing too, and nothing would ever finish. The newest requests go to the host when the host holds
them whole, and are otherwise dropped if they can be rebuilt from their token ids. They wait in the queue at their
arrival position. A request on the host returns once its pages and one more page fit, and a dropped request
once its whole history and one more page fit. One that came back and runs short again waits for its shortage
rather than moving itself again, unless an older request waits in the queue.
So the oldest request always gets its page and finishes.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import test_kv_host_index as host
from test_kv_admission import _remove, _runner
from test_kv_prefix_lease import NODE, PAGE_SIZE, WALK, _ctx, _keys, _manager, _StubTransfer
from test_kv_row_passes import backend  # noqa: F401, the Rust and the Python page arena

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.engine import Engine
from mstar.engine.resources import Residency
from mstar.engine.resources.base import CachedPrefix
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.cache import GroupPages
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PrefixSpan, StreamLayout
from mstar.engine.resources.kv.manager import Victims
from mstar.engine.resources.step import AdmitOutcome, Segment
from mstar.model.submodule_base import ARNodeInputs


@pytest.fixture(autouse=True)
def _stub_transfer(monkeypatch):
    monkeypatch.setattr(manager_mod, "KVTransferManager", _StubTransfer)


def _pool():
    """A pool where requests can hold 10 pages, past the sink page, on a worker that runs the decode loop."""
    kv = _manager(max_num_pages=11)
    kv.local_loops = {"decode"}
    return kv


def _request(
    kv, rid: str, pages: int, arrival: int | None, keyed: list[int] | None = None, guaranteed=False, rebuildable=True,
) -> None:
    """``rid``'s prompt fills ``pages`` pages, and it decodes one token per iteration."""
    tokens = pages * PAGE_SIZE
    layout = StreamLayout(
        spans=(PrefixSpan(tokens, tokens, WALK),), growth=(("decode", 1),), keyed_spans=int(bool(keyed)),
    )
    _runner(kv).ingest_request(rid, {"kv": KVReqConfig(
        prefix_keys={"main": _keys(keyed)} if keyed else None, kv_layout={"main": layout},
        loop_caps={"decode": 4 * PAGE_SIZE}, unpausable=guaranteed, rebuildable=rebuildable,
    )}, arrival)


def _admitted(kv, rid: str) -> bool:
    """Whether the runner has admitted ``rid``, once it takes whatever has since been admitted."""
    runner = _runner(kv)
    runner.admit_waiting(lambda other, node: None)
    return runner.admit_request(rid, NODE, WALK) is None


def _ready(kv, rid: str) -> bool:
    """A readiness check as the engine makes it: admission, a reload for a request on the host, then retrieve."""
    if not _admitted(kv, rid):
        return False
    if kv.is_offloaded(rid) and not (kv.can_reload(rid) and kv.reload(rid)):
        return False
    return _runner(kv).admit_retrieve(rid, NODE, WALK, None).ready


def _drop(kv, rid: str) -> int:
    freed = kv.drop(rid)
    _runner(kv).readmit(rid, NODE)
    return freed


def _offload(kv, rid: str) -> int:
    freed = kv.offload(rid)
    _runner(kv).readmit(rid, NODE)
    return freed


def _preempt(kv, rid: str, pages_short: int, droppable) -> Victims | None:
    """`KVManager.preempt` over the runner's arrival order, as the worker calls it."""
    return kv.preempt(rid, pages_short, _runner(kv).newer_than(rid), droppable)


def _queue(kv) -> list[str]:
    return _runner(kv)._admission._queues["kv"]


def _step(kv, rid: str, tokens: int) -> AdmitOutcome:
    """One step of ``tokens`` for ``rid``, committed when it is admitted."""
    step = KVStep(segments=(Segment(rid, "main", tokens),))
    outcome = kv.admit(step, _ctx(rid))
    if outcome.ok:
        kv.plan(step, _ctx(rid))
        kv.commit(step, _ctx(rid))
    else:
        kv.rollback_admit(step, _ctx(rid))
    return outcome


def _running(kv, rid: str, pages: int, arrival: int, **kwargs) -> None:
    """Admit ``rid`` and prefill its prompt of ``pages`` pages, ending on a full page."""
    _request(kv, rid, pages, arrival, **kwargs)
    assert _ready(kv, rid)
    assert _step(kv, rid, pages * PAGE_SIZE).ok


def test_a_dropped_request_rejoins_the_queue_at_its_arrival():
    kv = _pool()
    _running(kv, "x", 8, arrival=0)
    for rid, pages, arrival in (("c", 5, 3), ("a", 3, 1), ("b", 3, 2)):
        _request(kv, rid, pages, arrival)
        assert not _ready(kv, rid)
    assert _queue(kv) == ["a", "b", "c"], "the queue kept the order of the readiness checks"

    _remove(kv, "x")
    assert _ready(kv, "a") and _ready(kv, "b") and not _ready(kv, "c")
    assert _step(kv, "a", 3 * PAGE_SIZE).ok and _step(kv, "b", 2 * PAGE_SIZE).ok
    kv.drop("b")
    assert "b" not in kv._allotments and kv._unused == 0, "the drop kept part of b's allotment"
    _runner(kv).readmit("b", NODE)

    assert "b" in kv._allotments and not _ready(kv, "c"), "c passed the request dropped ahead of it"


def test_the_engine_orders_requests_by_the_arrival_their_request_info_carries():
    kv = _pool()
    engine = Engine.__new__(Engine)
    engine._runner, engine._submodules = _runner(kv), {}
    # a's second partition sends it again
    for rid, arrival in (("v", 2), ("r", 0), ("a", 1), ("x", None), ("a", 1)):
        info = CurrentForwardPassInfo(rid, WALK, 0, 0, 1, arrival=arrival)
        Engine.add_request(engine, rid, {"kv": KVReqConfig()}, info)

    assert engine.newer_than("x") is None, "a request the conductor gave no number was offered victims"
    assert engine.newer_than("r") == ["v", "a"], "the requests after r aren't listed newest first"


def _short(kv, rid: str) -> int:
    """Step ``rid``'s next token, which needs a page past its allotment, and return how many pages ``rid`` is short."""
    outcome = _step(kv, rid, 1)
    assert not outcome.ok
    return outcome.reason.pages_short


def test_a_newer_victim_waits_until_the_older_request_has_its_page():
    kv = _pool()
    _running(kv, "r", 3, arrival=1)
    _running(kv, "v", 5, arrival=2)
    assert _step(kv, "r", 2 * PAGE_SIZE).ok, "r grows past its allotment into the room"

    assert _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: True).to_drop == ["v"]
    _drop(kv, "v")

    assert kv._allotments["r"].remaining == 1, "the page v freed isn't reserved for r"
    assert _step(kv, "r", 1).ok, "the older request didn't get the page its victim freed"
    assert not _admitted(kv, "v"), "v came back into the room r needs to grow"
    _remove(kv, "r")
    # its history is the prompt's 5 pages and the sample its next step writes
    assert _admitted(kv, "v") and kv._allotments["v"].pages == 7, "v didn't reserve its history and a page"


def test_each_victim_sets_aside_what_it_frees_before_the_next_one_moves():
    kv = _pool()
    # two pages stay free, so the room v2's drop frees would hold v2's return
    _running(kv, "r", 3, arrival=1)
    _running(kv, "v1", 3, arrival=2)
    _running(kv, "v2", 2, arrival=3)
    short = _step(kv, "r", 6 * PAGE_SIZE).reason.pages_short
    assert _preempt(kv, "r", short, droppable=lambda rid: True).to_drop == ["v2", "v1"]

    _drop(kv, "v2")
    _admitted(kv, "v2")
    _drop(kv, "v1")

    assert kv._room() >= 0, f"the pool promises {kv._unused} pages, {-kv._room()} more than free and evictable"


def test_a_victim_still_to_move_is_chosen_for_no_other_request():
    kv = _pool()
    _running(kv, "r1", 3, arrival=1)
    _running(kv, "r2", 3, arrival=2)
    _running(kv, "v", 4, arrival=3)
    assert _preempt(kv, "r1", _short(kv, "r1"), droppable=lambda rid: True).to_drop == ["v"]

    waits = _preempt(kv, "r2", _short(kv, "r2"), droppable=lambda rid: True)
    assert waits == Victims([], [], [], waits=True), "r2 took r1's victim or moved itself"
    _runner(kv).readmit("r2", NODE)
    assert _preempt(kv, "r1", _short(kv, "r1"), droppable=lambda rid: True) == Victims([], [], []), "r1 chose again"
    _drop(kv, "v")

    assert not kv._earmarks and not kv._victims, "v's move didn't settle r1's earmark"
    assert _step(kv, "r1", 1).ok, "r1 didn't get the page its victim freed"
    assert _admitted(kv, "r2") and _step(kv, "r2", 1).ok, "r2 didn't get a page from the room v's move made"


def test_a_request_waiting_on_its_own_victims_isnt_chosen_as_a_victim():
    kv = _pool()
    _running(kv, "r", 3, arrival=1)
    _running(kv, "m", 3, arrival=2)
    _running(kv, "v", 4, arrival=3)
    assert _preempt(kv, "m", _short(kv, "m"), droppable=lambda rid: True).to_drop == ["v"]

    victims = _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: True)

    assert "m" not in victims.to_drop, "m, which waits on the room its own victim frees, was chosen as a victim"


@pytest.mark.parametrize("flight", ["step_in_flight", "read_pending"])
def test_a_request_with_a_step_or_read_in_flight_isnt_chosen(flight):
    kv = _pool()
    kv._cpu_pool = host._HostPool(16)
    _running(kv, "r", 5, arrival=1)
    _running(kv, "v", 5, arrival=2)
    setattr(kv._streams["v"]["main"], flight, True)

    victims = _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: False)

    assert "v" not in victims.to_host + victims.to_drop, f"v was chosen with its {flight}, so its move would be refused"


def test_a_victim_whose_move_fails_comes_off_the_earmark():
    kv = _pool()
    kv._cpu_pool = host._HostPool(16)
    _running(kv, "r", 5, arrival=1)
    _running(kv, "v", 5, arrival=2)
    assert _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: False).to_host == ["v"]
    # the host fills before v moves
    kv._cpu_pool.arena.acquire(kv._cpu_pool.arena.num_free)

    assert _offload(kv, "v") == 0
    assert "r" not in kv._earmarks and not kv._victims, "v's failed move left r's earmark open, so r can't choose again"


@pytest.mark.parametrize("then", ["drops itself", "waits", "waits, then is dropped"])
def test_the_oldest_request_drops_itself_or_waits_for_room(then):
    kv = _pool()
    _running(kv, "older", 5, arrival=0)
    _running(kv, "r", 5, arrival=1)

    victims = _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: then == "drops itself")

    drops = then == "drops itself"
    assert victims == Victims([], ["r"] if drops else [], [], waits=not drops), "r waits only when it can't drop itself"
    assert not kv._earmarks, "nothing newer frees a page"
    if not drops:
        _runner(kv).readmit("r", NODE)
        assert _queue(kv) == ["r"] and not _admitted(kv, "r"), "r didn't wait in the queue for its shortage"
    if then == "waits, then is dropped":
        _drop(kv, "r")
        returns = kv._dropped["r"].returns
        assert kv._need("r", NODE, WALK, None) == returns, "r's wait for its shortage outlived its drop"
        _remove(kv, "older")
        assert _admitted(kv, "r"), "r didn't return once its history and a page fit"
    elif then == "waits":
        _remove(kv, "older")
        assert _admitted(kv, "r") and _step(kv, "r", 1).ok, "r kept waiting after the room grew"


def test_a_request_no_move_covers_waits_in_the_queue_until_its_whole_shortage_fits():
    kv = _pool()
    _running(kv, "a", 1, arrival=0)
    _running(kv, "b", 4, arrival=1)
    _running(kv, "r", 5, arrival=2)
    short = _step(kv, "r", PAGE_SIZE + 1).reason.pages_short
    assert _preempt(kv, "r", short, droppable=lambda rid: False).waits
    _runner(kv).readmit("r", NODE)

    _remove(kv, "a")
    assert not _admitted(kv, "r"), f"r was let go once one page freed, though it was short by {short}"
    _remove(kv, "b")
    assert _admitted(kv, "r") and _step(kv, "r", PAGE_SIZE + 1).ok, "r wasn't admitted once its shortage fit"


@pytest.mark.parametrize("state", ["pending victim", "dropped", "rebuilding", "earmark's request"])
def test_removing_a_request_in_any_state_leaks_nothing(monkeypatch, state):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    kv = _pool()
    _running(kv, "r", 3, arrival=1)
    _running(kv, "v", 5, arrival=2)
    assert _step(kv, "r", 2 * PAGE_SIZE).ok
    assert _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: True).to_drop == ["v"]
    if state in ("dropped", "rebuilding"):
        _drop(kv, "v")
    if state == "rebuilding":
        _remove(kv, "r")
        assert _admitted(kv, "v")
    gone = "r" if state == "earmark's request" else "v"

    _remove(kv, gone)

    admission = _runner(kv)._admission
    left = [
        state for state in (
            kv._earmarks, kv._victims, kv._ran_short, kv._asks, kv._dropped, kv._allotments,
            admission._waiting, admission._admitted, admission._arrival, admission._numbered, admission._queues["kv"],
        ) if gone in state
    ]
    assert not left and not kv._earmarks and not kv._victims, f"{gone} left preemption state behind"
    assert kv._unused == sum(allotment.remaining for allotment in kv._allotments.values())


def test_a_request_dropped_partway_through_its_prompt_returns_with_its_whole_prompt():
    kv = _pool()
    _request(kv, "v", 5, arrival=1)
    assert _ready(kv, "v") and _step(kv, "v", 2 * PAGE_SIZE).ok

    kv.drop("v")

    assert kv._dropped["v"].returns == 5, "v would return with room for the pages it wrote and run short on the rest"


@pytest.mark.usefixtures("backend")
def test_victims_that_share_pages_are_preempted_together():
    kv = _pool()
    shared, tokens = list(range(3 * PAGE_SIZE)), {"a": range(100, 116), "b": range(200, 216)}
    _running(kv, "r", 5, arrival=1)
    _request(kv, "a", 4, arrival=2, keyed=shared + list(tokens["a"]))
    assert _ready(kv, "a") and _step(kv, "a", 4 * PAGE_SIZE).ok
    _request(kv, "b", 4, arrival=3, keyed=shared + list(tokens["b"]))
    assert _ready(kv, "b")
    inputs = ARNodeInputs(input_seq_len=4 * PAGE_SIZE)
    kv.apply_cached_prefix("b", NODE, WALK, inputs, kv.resolve_cached_prefix("b", NODE, WALK, inputs))
    assert _step(kv, "b", PAGE_SIZE).ok and kv._arena.num_free == 0, "b writes only its page past the shared three"
    short = _step(kv, "r", 2 * PAGE_SIZE).reason.pages_short

    assert _preempt(kv, "r", short, droppable=lambda rid: True).to_drop == ["b", "a"], "a victim alone frees one page"
    alone, both = kv._frees(["a"]), kv._frees(["b", "a"])
    before = kv._arena.num_free + len(kv._index.evictable())
    _drop(kv, "b")
    _drop(kv, "a")

    assert alone == 1, "the shared pages counted as freed while b still holds them"
    assert kv._arena.num_free + len(kv._index.evictable()) - before == both == 5, "the drops freed other than counted"


def test_a_page_the_failing_step_gives_back_is_no_release():
    kv = _pool()
    for rid, arrival in (("a", 0), ("r", 1), ("v", 2)):
        _running(kv, rid, 3, arrival)
    step, ctx = KVStep(segments=(Segment("a", "main", 1), Segment("r", "main", 1))), _ctx("a", "r")
    outcome = kv.admit(step, ctx)
    kv.rollback_admit(step, ctx)

    assert _preempt(kv, "r", outcome.reason.pages_short, droppable=lambda rid: True).to_drop == ["v"], (
        "the page a's row gave back counted as room, so r waits on the step that will fail again"
    )


@pytest.mark.parametrize("guaranteed, local_loops", [(False, {"decode"}), (True, {"decode"}), (True, set())])
def test_a_guaranteed_request_reserves_what_it_grows_into_on_this_worker(guaranteed, local_loops):
    kv, whole = _pool(), guaranteed and bool(local_loops)
    kv.local_loops = local_loops
    _running(kv, "x", 5, arrival=0)
    _request(kv, "r", 2, arrival=1, guaranteed=guaranteed)
    assert _ready(kv, "r") is not whole, "the request was admitted into room sized for another class"
    _remove(kv, "x")
    caps = kv._overrides["r"].loop_caps if whole else {"decode": 0}
    reserved = kv._overrides["r"].pages(None, caps, kv.config)
    assert _ready(kv, "r") and kv._allotments["r"].pages == reserved, "the reservation doesn't match the class"


def test_a_guaranteed_hit_on_pages_another_request_releases_still_grants_within_its_reservation():
    kv, tokens = _pool(), list(range(3 * PAGE_SIZE))
    _running(kv, "h", 3, arrival=0, keyed=tokens)
    _request(kv, "g", 3, arrival=1, keyed=tokens, guaranteed=True)
    assert _ready(kv, "g")
    _remove(kv, "h")
    kv._index.evict(3)  # as a third request's grant does when no page is free
    room = kv._room()
    assert _step(kv, "g", 7 * PAGE_SIZE).ok and kv._room() == room, "g's hit on h's pages left it short once h left"


def test_a_guaranteed_lease_of_pages_another_request_holds_gives_their_charge_back():
    kv, tokens = _pool(), list(range(3 * PAGE_SIZE))
    _running(kv, "h", 3, arrival=0, keyed=tokens)
    _request(kv, "g", 3, arrival=1, keyed=tokens, guaranteed=True)
    assert _ready(kv, "g")
    room, inputs = kv._room(), ARNodeInputs(input_seq_len=3 * PAGE_SIZE)

    kv.apply_cached_prefix("g", NODE, WALK, inputs, kv.resolve_cached_prefix("g", NODE, WALK, inputs))
    leased = len(kv._streams["g"]["main"].lease or ())
    assert _step(kv, "g", PAGE_SIZE).ok
    assert leased and kv._room() == room + leased, "g's charge for the pages it leases stayed out of the room"


@pytest.mark.parametrize("hit", ["held", "evictable"])
def test_a_guaranteed_request_whose_lease_is_cut_writes_the_cut_pages_within_its_reservation(hit):
    """Qwen3.5's recurrent snapshots cut a hit to what they cover, and the walk writes the cut pages itself. The
    probe charged a guaranteed request for its hits and gave that room to the pool, and a cut page another request
    still held came off the guaranteed request's allotment. That request then ran short and could neither move nor
    be dropped."""
    kv, tokens = _pool(), list(range(3 * PAGE_SIZE))
    _running(kv, "h", 3, arrival=0, keyed=tokens)
    if hit == "evictable":
        _remove(kv, "h")
    _request(kv, "g", 3, arrival=1, keyed=tokens, guaranteed=True)
    assert _ready(kv, "g")
    inputs = ARNodeInputs(input_seq_len=3 * PAGE_SIZE)
    kv.resolve_cached_prefix("g", NODE, WALK, inputs)
    if hit == "evictable":
        # "b" leases the pages "g" leased, so they stay held once "g" lets go of them
        _request(kv, "b", 3, arrival=2, keyed=tokens)
        assert _ready(kv, "b") and kv.resolve_cached_prefix("b", NODE, WALK, inputs).tokens == 2 * PAGE_SIZE
    # "x" is admitted into whatever room the probe left
    _request(kv, "x", 2, arrival=3)
    _ready(kv, "x")

    kv.apply_cached_prefix("g", NODE, WALK, inputs, CachedPrefix("main", PAGE_SIZE, PAGE_SIZE))

    assert _step(kv, "g", 2 * PAGE_SIZE).ok
    for _ in range(4):
        assert _step(kv, "g", PAGE_SIZE).ok, "g ran short of the pages its reservation held for the cut ones"


def test_a_guaranteed_request_counts_its_pages_as_reclaimable_and_is_no_eviction_candidate():
    """`reclaimable` answered 0 for a guaranteed request only on the rank that decides, so a follower's replay read
    the request's pages instead, and a teardown needed a second hook to count them."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(64)
    _running(kv, "g", 2, arrival=0, guaranteed=True)

    assert kv.reclaimable("g") == len(kv._streams["g"]["main"].pages()), "reclaimable answered by class"
    assert not kv.can_offload("g"), "the offload of the least recently active request could move g to the host"


def test_a_guaranteed_request_that_wrote_nothing_keeps_its_allotment_when_an_older_one_runs_short():
    kv = _pool()
    _running(kv, "o", 3, arrival=0)
    _request(kv, "g", 2, arrival=1, guaranteed=True)
    assert _ready(kv, "g")
    short = _step(kv, "o", 2 * PAGE_SIZE).reason.pages_short

    assert _preempt(kv, "o", short, droppable=lambda rid: False) == Victims([], [], [], waits=True), (
        "a guaranteed request gave back its allotment for an older request's shortage"
    )
    assert "g" in kv._allotments


def test_the_host_counts_as_room_only_the_cached_pages_place_can_evict():
    """A host fetch holds a run of cached pages that starts below the root, and eviction takes leaves only, so the
    pages above that run stay. Counting those pages as room would send a victim to a host that can't take
    it."""
    cpu = host._HostPool(6)
    cached, keys = cpu.arena.acquire(4), [bytes([i]) * 32 for i in range(4)]
    cpu.index.insert_run(keys, cached, None)
    cpu.arena.release(cached)
    cpu.retain_run(keys[2:])
    room = cpu.room()

    assert cpu.place("v", "main", (GroupPages(list(range(room))),), [], position=0) is not None, (
        f"the host counted {room} pages of room and couldn't place them"
    )


def test_room_the_host_set_aside_stays_when_a_host_fetch_holds_the_cached_pages_it_counted():
    """The host set a request's room aside out of cached pages, and a host fetch then held them, leaving the room
    at -3 and the request with no way off the device."""
    cpu = host._HostPool(6)
    cached, keys = cpu.arena.acquire(4), [bytes([i]) * 32 for i in range(4)]
    cpu.index.insert_run(keys, cached, None)
    cpu.arena.release(cached)
    assert cpu.reserve("r", 5)

    cpu.retain_run(keys)

    assert cpu.room() >= 0 and cpu.place("r", "main", (GroupPages(list(range(5))),), [], position=0) is not None, (
        "a host fetch held the pages set aside for r's move"
    )


def test_the_page_check_catches_a_host_page_counted_as_room_that_eviction_cant_free():
    """The host's room counts the evictable set its arena keeps current, so a page counted there that eviction
    can't free would send a victim to a host that can't take it."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(16)
    kv.assert_pages_conserved()
    kv._cpu_pool.index.evictable().add(0)

    with pytest.raises(AssertionError, match="eviction reaches 0 of the 1 pages"):
        kv.assert_pages_conserved()


def test_the_page_check_catches_a_request_on_the_host_that_kept_a_device_page():
    """A request waiting in the queue on the host holds no page on the device. A move that copied only some of its
    pages would leave the request holding a device page, which the debug page check raises on."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(64)
    _running(kv, "r", 5, arrival=1)
    _running(kv, "v", 5, arrival=2)
    short = _step(kv, "r", 1).reason.pages_short
    assert _preempt(kv, "r", short, droppable=lambda rid: False).to_host == ["v"]
    _offload(kv, "v")
    assert "v" in kv._asks and kv.is_offloaded("v")
    kv.assert_pages_conserved()
    kv._streams["v"]["main"].page_indices = kv._arena.acquire(1)

    with pytest.raises(AssertionError, match="v waits on the host, holding pages"):
        kv.assert_pages_conserved()


def test_an_older_request_on_the_host_returns_before_a_newer_one():
    kv = _pool()
    kv._cpu_pool = host._HostPool(64)
    _running(kv, "r", 4, arrival=1)
    _running(kv, "a", 3, arrival=2)
    _running(kv, "b", 3, arrival=3)
    short = _step(kv, "r", 4 * PAGE_SIZE).reason.pages_short
    assert _preempt(kv, "r", short, droppable=lambda rid: False).to_host == ["b", "a"]
    _offload(kv, "b")
    _offload(kv, "a")
    assert _queue(kv) == ["a", "b"] and not kv._allotments.keys() & {"a", "b"}, "a victim kept its allotment"

    _remove(kv, "r")

    assert _runner(kv).admit_waiting(lambda rid, node: None) == [(NODE, "a"), (NODE, "b")], "b returned ahead of a"
    assert _ready(kv, "a") and kv._allotments["a"].remaining == 1, "a didn't reserve its reload and a page"
    assert _ready(kv, "b") and not _queue(kv)


def _back_from_the_host_and_short(kv) -> int:
    """z moves to the host for x, comes back beside y once x finishes, and grows until the pool is full. Returns how
    many pages z is short."""
    kv._cpu_pool = host._HostPool(64)
    _running(kv, "x", 3, arrival=0)
    _running(kv, "y", 5, arrival=1)
    _running(kv, "z", 2, arrival=3)
    assert _preempt(kv, "x", _short(kv, "x"), droppable=lambda rid: False) == Victims(["z"], [], [])
    _offload(kv, "z")
    _remove(kv, "x")
    assert _ready(kv, "z"), "z didn't come back once its pages and a page fit"
    while (outcome := _step(kv, "z", PAGE_SIZE)).ok:
        pass
    return outcome.reason.pages_short


def test_a_request_back_from_the_host_waits_for_its_shortage_rather_than_moving_itself_again():
    """A request came back from the host once its pages and one more page fit. On its next page the pool had no
    room, so the request, the newest holding pages, moved to the host again, and again for every page after."""
    kv = _pool()
    short = _back_from_the_host_and_short(kv)

    assert _preempt(kv, "z", short, droppable=lambda rid: False) == Victims([], [], [], waits=True), (
        "z moved to the host again for its own page"
    )


def test_a_request_back_moves_itself_again_for_an_older_request_waiting_in_the_queue():
    """A request that came back waited for its shortage holding its pages, while an older request waited in the
    queue for room those pages would make. The queue admits the older request first, so neither was admitted."""
    kv = _pool()
    short = _back_from_the_host_and_short(kv)
    _request(kv, "p", 3, arrival=2)
    assert not _ready(kv, "p")

    assert _preempt(kv, "z", short, droppable=lambda rid: False) == Victims(["z"], [], []), (
        "z waited for its shortage behind p, which waits for z's pages"
    )
    _offload(kv, "z")
    assert _ready(kv, "p"), "p didn't get the room z made"


def test_a_request_back_from_a_drop_is_still_a_victim_for_an_older_request():
    kv = _pool()
    _running(kv, "o", 3, arrival=0)
    _running(kv, "v", 2, arrival=2)
    assert _step(kv, "o", 5 * PAGE_SIZE).ok
    assert _preempt(kv, "o", _short(kv, "o"), droppable=lambda rid: True).to_drop == ["v"]
    _drop(kv, "v")
    _request(kv, "a", 2, arrival=1)
    _remove(kv, "o")
    assert _ready(kv, "a") and _admitted(kv, "v")
    while (outcome := _step(kv, "a", PAGE_SIZE)).ok:
        pass

    victims = _preempt(kv, "a", outcome.reason.pages_short, droppable=lambda rid: True)

    assert "v" in (*victims.to_host, *victims.to_drop, *victims.to_ask), "v held back pages an older request needed"


def test_a_failed_reload_puts_a_request_back_from_the_host_on_the_host_again():
    """`Engine.reload_request` undoes a partial reload by offloading what it reloaded, which the KV's offload must not
    refuse for a request that came back from the host."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(64)
    _running(kv, "o", 5, arrival=0)
    _running(kv, "z", 3, arrival=1)
    assert _step(kv, "o", 2 * PAGE_SIZE).ok
    assert _preempt(kv, "o", _short(kv, "o"), droppable=lambda rid: False) == Victims(["z"], [], [])
    _offload(kv, "z")
    _remove(kv, "o")
    assert _admitted(kv, "z")
    refusing = SimpleNamespace(
        supports_eviction=True, residency=lambda rid: Residency.ON_HOST, can_reload=lambda rid: True,
        reload=lambda rid: False,
    )
    engine = Engine.__new__(Engine)
    engine._tp_follower_nodes, engine._submodules = set(), {NODE: SimpleNamespace(resources={"kv": kv, "x": refusing})}

    assert not engine.reload_request(NODE, "z")
    assert kv.residency("z") is Residency.ON_HOST, "z stayed resident in one pool and on the host in the other"


@pytest.mark.parametrize("decides", [False, True], ids=["a pool that doesn't decide", "an offload no victim chose"])
def test_an_offload_outside_preemption_reloads_as_before(decides):
    kv = _pool()
    kv._cpu_pool = host._HostPool(64)
    kv._decides = decides
    _running(kv, "r", 5, arrival=1)
    _request(kv, "w", 8, arrival=0)
    _ready(kv, "w")
    remaining = kv._allotments.get("r", manager_mod.Allotment(0)).remaining

    assert _offload(kv, "r") == 5 and "r" not in _queue(kv)

    assert kv.can_reload("r") and kv.reload("r"), "r waited in the queue behind w"
    assert kv._allotments.get("r", manager_mod.Allotment(0)).remaining == remaining, "the reload changed r's allotment"


def test_a_guaranteed_request_is_never_moved_to_the_host():
    kv = _pool()
    kv._cpu_pool = host._HostPool(64)
    _running(kv, "r", 3, arrival=0)
    _running(kv, "g", 2, arrival=1, guaranteed=True)
    short = _step(kv, "r", 2 * PAGE_SIZE).reason.pages_short

    assert _preempt(kv, "r", short, droppable=lambda rid: False) == Victims(["r"], [], []), (
        "a guaranteed request was moved to the host"
    )


@pytest.mark.parametrize("host_pages, guaranteed", [(None, True), (1, True), (64, False)], ids=[
    "no host pool", "a host pool short of its prompt", "a host pool with room",
])
def test_a_request_no_drop_can_free_reserves_its_footprint_only_when_the_host_cant_set_it_aside(
    host_pages, guaranteed,
):
    """qwen3_asr_realtime's sessions can't be rebuilt, since the model declares no rebuild policy. Reserving each
    one's whole decode cap fits 29 of its 64 sessions, though a host pool could take any of them back."""
    kv = _pool()
    kv._cpu_pool = None if host_pages is None else host._HostPool(host_pages)
    _request(kv, "r", 2, arrival=0, rebuildable=False)

    assert _ready(kv, "r")
    assert ("r" in kv._guaranteed, kv._allotments["r"].pages) == (guaranteed, 6 if guaranteed else 2), (
        "the class didn't follow the room the host can set aside"
    )
    assert host_pages is None or kv._cpu_pool.reserved == ({} if guaranteed else {"r": 2})


def test_a_victim_that_can_be_rebuilt_never_moves_into_room_set_aside_on_the_host():
    kv = _pool()
    kv._cpu_pool = host._HostPool(4)
    _running(kv, "o", 3, arrival=0)
    _running(kv, "r", 2, arrival=1, rebuildable=False)
    _running(kv, "v", 3, arrival=2)
    assert _step(kv, "o", 2 * PAGE_SIZE).ok
    short = _step(kv, "o", PAGE_SIZE).reason.pages_short

    assert _preempt(kv, "o", short, droppable=lambda rid: True) == Victims([], ["v"], []), (
        "v moved into the host room set aside for r"
    )


def test_a_request_no_drop_can_free_moves_into_its_own_room_on_a_full_host():
    kv = _pool()
    kv._cpu_pool = host._HostPool(4)
    _running(kv, "o", 3, arrival=0)
    _running(kv, "r", 2, arrival=1, rebuildable=False)
    _running(kv, "x", 2, arrival=2)
    assert _step(kv, "o", 3 * PAGE_SIZE).ok
    assert _preempt(kv, "o", _step(kv, "o", PAGE_SIZE).reason.pages_short, droppable=lambda rid: False) == (
        Victims(["x"], [], [])
    )
    assert _offload(kv, "x")
    while (outcome := _step(kv, "o", PAGE_SIZE)).ok:
        pass

    victims = _preempt(kv, "o", outcome.reason.pages_short, droppable=lambda rid: False)

    assert victims == Victims(["r"], [], []) and _offload(kv, "r"), "r had no way off the device once the host was full"


def test_the_host_sets_room_aside_when_a_request_is_admitted_and_not_while_it_waits():
    """Room set aside at a request's first ask held host room for requests still waiting in the queue."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(64)
    _running(kv, "x", 9, arrival=0)
    _request(kv, "r", 2, arrival=1, rebuildable=False)
    assert not _ready(kv, "r")
    assert kv._cpu_pool.reserved == {}, "r held room on the host while it waited in the queue"

    _remove(kv, "x")

    assert _ready(kv, "r") and kv._cpu_pool.reserved == {"r": 2}, "the host didn't set room aside for r's prompt"


def test_a_request_whose_admission_is_undone_waits_with_no_room_set_aside_on_the_host():
    """A request is admitted in every pool it asks or in none, so a pool that admitted it gives the allotment back
    when another can't."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(64)
    _request(kv, "r", 2, arrival=0, rebuildable=False)
    assert _ready(kv, "r")

    kv.unreserve("r")

    assert kv._cpu_pool.reserved == {}, "r held room on the host while it waited for another pool"


def test_the_room_the_host_sets_aside_grows_with_each_page_a_request_takes():
    """qwen3_asr_realtime's sessions each set their whole decode cap aside on the host at their first ask, so few of
    them got room, and the host's prefix tier was left little."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(64)
    _running(kv, "r", 2, arrival=0, rebuildable=False)

    assert _step(kv, "r", PAGE_SIZE).ok

    assert kv._cpu_pool.reserved == {"r": 3}, "the room set aside isn't the pages r holds"


def test_a_request_the_full_host_cant_cover_turns_guaranteed_at_that_page_and_is_counted(caplog):
    kv = _pool()
    kv._cpu_pool = host._HostPool(3)
    _running(kv, "r", 2, arrival=0, rebuildable=False)
    # s sets aside the host's last page
    _running(kv, "s", 1, arrival=1, rebuildable=False)
    assert "r" not in kv._guaranteed and kv._cpu_pool.reserved == {"r": 2, "s": 1}

    assert _step(kv, "r", PAGE_SIZE).ok

    assert "r" in kv._guaranteed and kv._cpu_pool.reserved == {"s": 1}, "r took a page its move had no room for"
    assert kv._cpu_pool.guaranteed == 1 and "no room to set aside for r's move" in caplog.text
    for _ in range(3):
        assert _step(kv, "r", PAGE_SIZE).ok, "r turned guaranteed without reserving what it grows into"


def test_a_request_neither_the_host_nor_the_device_can_cover_is_short_of_its_footprint_and_can_still_move():
    kv = _pool()
    kv._cpu_pool = host._HostPool(3)
    _running(kv, "r", 2, arrival=0, rebuildable=False)
    _running(kv, "s", 1, arrival=1, rebuildable=False)
    _running(kv, "x", 6, arrival=2)

    outcome = _step(kv, "r", PAGE_SIZE)

    # the 4 pages it grows into, less the 1 page of room
    assert not outcome.ok and outcome.reason.pages_short == 3, "r took a page its move had no room for"
    assert _preempt(kv, "r", 3, droppable=lambda rid: False) == Victims(["r"], [], []) and _offload(kv, "r"), (
        "r had no way off the device"
    )


class _StubHostRefusingB(host._HostPool):
    """Refuses stream ``b``, as a host whose room runs out partway through a move does."""

    def place(self, rid, label, *args, **kwargs):
        return None if label == "b" else super().place(rid, label, *args, **kwargs)


def test_a_victim_the_host_took_only_part_of_keeps_its_allotment():
    """A victim whose second stream the host refused was still vacated, so it asked again for its footprint on top
    of the pages that stream held, and never came back even alone."""
    kv = _pool()
    kv._cpu_pool = _StubHostRefusingB(64)
    _running(kv, "o", 3, arrival=0)
    layout = StreamLayout(spans=(PrefixSpan(3 * PAGE_SIZE, 3 * PAGE_SIZE, WALK),), growth=(("decode", 1),))
    _runner(kv).ingest_request("v", {"kv": KVReqConfig(
        kv_layout={"main": layout, "b": StreamLayout(spans=(PrefixSpan(2 * PAGE_SIZE, 2 * PAGE_SIZE, WALK),))},
        needed_labels_per_node_walk={(NODE, WALK): ["main", "b"]}, loop_caps={"decode": 4 * PAGE_SIZE},
    )}, 1)
    assert _ready(kv, "v")
    step, ctx = KVStep(segments=(Segment("v", "main", 3 * PAGE_SIZE), Segment("v", "b", 2 * PAGE_SIZE))), _ctx("v")
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)
    assert _step(kv, "o", 2 * PAGE_SIZE).ok
    assert _preempt(kv, "o", _short(kv, "o"), droppable=lambda rid: False) == Victims(["v"], [], [])
    assert _offload(kv, "v") and kv._streams["v"]["b"].pages()

    _remove(kv, "o")

    assert _ready(kv, "v"), "v, still holding b's pages, never came back even alone"
