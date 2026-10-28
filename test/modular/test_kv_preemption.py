"""A request whose grant runs short moves newer requests off the device and takes the pages they free.

A new prompt waits in the runner's queue for room, since a pool admits only its oldest waiting request.
A request that grows past what it was allotted can't wait that way: every request beside it
may be growing too, and nothing would ever finish. The newest requests go to the host when the host holds
them whole, and are otherwise dropped if they can be rebuilt from their token ids. They wait in the queue at their
arrival position. A request on the host returns once its pages and one more page fit, and a dropped request
once its whole history and one more page fit.
So the oldest request always gets its page and finishes.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import test_kv_host_index as host
from test_kv_admission import _remove, _runner
from test_kv_prefix_lease import NODE, PAGE_SIZE, WALK, _ctx, _keys, _manager, _StubTransfer

from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PrefixSpan, StreamLayout
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


def _request(kv, rid: str, pages: int, arrival: int | None, keyed: list[int] | None = None, guaranteed=False) -> None:
    """``rid``'s prompt fills ``pages`` pages, and it decodes one token per iteration."""
    tokens = pages * PAGE_SIZE
    layout = StreamLayout(
        spans=(PrefixSpan(tokens, tokens, WALK),), growth=(("decode", 1),), keyed_spans=int(bool(keyed)),
    )
    _runner(kv).ingest_request(rid, {"kv": KVReqConfig(
        prefix_keys={"main": _keys(keyed)} if keyed else None, kv_layout={"main": layout},
        loop_caps={"decode": 4 * PAGE_SIZE}, arrival=arrival, guaranteed=guaranteed,
    )})


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


def _queue(kv) -> list[str]:
    return _runner(kv)._queues["kv"]


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

    assert kv.preempt("r", _short(kv, "r"), droppable=lambda rid: True) == ["v"]
    _drop(kv, "v")

    assert kv._allotments["r"].remaining == 1, "the page v freed isn't reserved for r"
    assert _step(kv, "r", 1).ok, "the older request didn't get the page its victim freed"
    assert not _admitted(kv, "v"), "v came back into the room r needs to grow"
    _remove(kv, "r")
    # its history is the prompt's 5 pages and the sample its next step writes
    assert _admitted(kv, "v") and kv._allotments["v"].pages == 7, "v didn't reserve its history and a page"


def test_a_victim_still_to_move_is_chosen_for_no_other_request():
    kv = _pool()
    _running(kv, "r1", 3, arrival=1)
    _running(kv, "r2", 3, arrival=2)
    _running(kv, "v", 4, arrival=3)
    assert kv.preempt("r1", _short(kv, "r1"), droppable=lambda rid: True) == ["v"]

    assert kv.preempt("r2", _short(kv, "r2"), droppable=lambda rid: True) == [], "r2 took r1's victim or moved itself"
    assert kv.waits("r2"), "r2 doesn't wait for the room v's move makes"
    assert kv.preempt("r1", _short(kv, "r1"), droppable=lambda rid: True) == [], "r1 chose again before v moved"
    _drop(kv, "v")

    assert not kv.waits("r2") and not kv._earmarks and not kv._victims, "v's move didn't settle r1's earmark"
    assert _step(kv, "r1", 1).ok, "r1 didn't get the page its victim freed"
    assert _step(kv, "r2", 1).ok, "r2 didn't get a page from the room v's move made"


@pytest.mark.parametrize("then", ["drops itself", "waits", "waits, then is dropped"])
def test_the_oldest_request_drops_itself_or_waits_for_room(then):
    kv = _pool()
    _running(kv, "older", 5, arrival=0)
    _running(kv, "r", 5, arrival=1)

    victims = kv.preempt("r", _short(kv, "r"), droppable=lambda rid: then == "drops itself")

    assert victims == (["r"] if then == "drops itself" else []) and not kv._earmarks, "nothing newer frees a page"
    assert kv.waits("r") == (then != "drops itself"), "r waits only when it can't drop itself"
    if then == "waits, then is dropped":
        _drop(kv, "r")
        assert not kv.waits("r") and _queue(kv) == ["r"], "r's wait for room outlived its drop"
        _remove(kv, "older")
        assert _admitted(kv, "r"), "r didn't return once its history and a page fit"
    elif then == "waits":
        _remove(kv, "older")
        assert not kv.waits("r"), "r kept waiting after the room grew"


@pytest.mark.parametrize("state", ["pending victim", "dropped", "rebuilding", "earmark's request"])
def test_removing_a_request_in_any_state_leaks_nothing(monkeypatch, state):
    monkeypatch.setattr(manager_mod, "_DEBUG_ASSERTS", True)
    kv = _pool()
    _running(kv, "r", 3, arrival=1)
    _running(kv, "v", 5, arrival=2)
    assert _step(kv, "r", 2 * PAGE_SIZE).ok
    assert kv.preempt("r", _short(kv, "r"), droppable=lambda rid: True) == ["v"]
    if state in ("dropped", "rebuilding"):
        _drop(kv, "v")
    if state == "rebuilding":
        _remove(kv, "r")
        assert _admitted(kv, "v")
    gone = "r" if state == "earmark's request" else "v"

    _remove(kv, gone)

    runner = _runner(kv)
    left = [
        state for state in (
            kv._earmarks, kv._victims, kv._short, kv._ran_short, kv._asks, kv._dropped, kv._allotments,
            runner._waiting, runner._admitted, runner._arrival, runner._queues["kv"],
        ) if gone in state
    ]
    assert not left and not kv._earmarks and not kv._victims, f"{gone} left preemption state behind"
    assert kv._unused == sum(allotment.remaining for allotment in kv._allotments.values())


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

    assert kv.preempt("r", short, droppable=lambda rid: True) == ["b", "a"], "a victim alone frees one page"
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

    assert kv.preempt("r", outcome.reason.pages_short, droppable=lambda rid: True) == ["v"], (
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
    reserved = kv._overrides["r"].pages(None, caps, PAGE_SIZE)
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
    assert leased and kv._room() == room + leased, "g's charge for the pages it leases stayed out of the room"


@pytest.mark.parametrize("decides", [True, False])
def test_only_a_deciding_pool_keeps_a_guaranteed_request_from_the_offload(decides):
    kv = _pool()
    _running(kv, "p", 2, arrival=0)
    _running(kv, "g", 2, arrival=1, guaranteed=True)
    kv._decides = decides
    # Only a deciding pool reserves a guaranteed request's pages and drops requests to free room.
    # Any other pool frees room only through the offload, which has to be able to take any request.
    assert kv.reclaimable("p") and bool(kv.reclaimable("g")) is not decides, "the offload saw the wrong class"


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

    assert cpu.place("v", "main", list(range(room)), [], room * PAGE_SIZE, room * PAGE_SIZE) is not None, (
        f"the host counted {room} pages of room and couldn't place them"
    )


def test_the_page_check_catches_a_request_on_the_host_that_kept_a_device_page():
    """A request waiting in the queue on the host holds no page on the device. A move that copied only some of its
    pages would leave the request holding a device page, which the debug page check raises on."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(64)
    _running(kv, "r", 5, arrival=1)
    _running(kv, "v", 5, arrival=2)
    short = _step(kv, "r", 1).reason.pages_short
    assert kv.preempt("r", short, droppable=lambda rid: False) == ["v"]
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
    assert kv.preempt("r", short, droppable=lambda rid: False) == ["b", "a"]
    assert kv.moves_to_host("b") and kv.moves_to_host("a")
    _offload(kv, "b")
    _offload(kv, "a")
    assert _queue(kv) == ["a", "b"] and not kv._allotments.keys() & {"a", "b"}, "a victim kept its allotment"

    _remove(kv, "r")

    assert _runner(kv).admit_waiting(lambda rid, node: None) == [(NODE, "a"), (NODE, "b")], "b returned ahead of a"
    assert _ready(kv, "a") and kv._allotments["a"].remaining == 1, "a didn't reserve its reload and a page"
    assert _ready(kv, "b") and not _queue(kv)


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

    assert kv.preempt("r", short, droppable=lambda rid: False) == ["r"], "a guaranteed request was moved to the host"
