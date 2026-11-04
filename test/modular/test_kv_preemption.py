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
from test_kv_prefix_layout_lease import IMAGE, IMAGE_WALK, RID, TAIL, TEXT, TEXT_WALK
from test_kv_prefix_layout_lease import _Node as _LayoutNode
from test_kv_prefix_lease import NODE, PAGE_SIZE, WALK, _ctx, _keys, _manager, _StubTransfer
from test_kv_row_passes import backend  # noqa: F401, the Rust and the Python page arena

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.engine import Engine
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
    """A pool where requests can hold 10 pages, past the sink page."""
    return _manager(max_num_pages=11)


def _request(kv, rid: str, pages: int, arrival: int | None, keyed: list[int] | None = None) -> None:
    """``rid``'s prompt fills ``pages`` pages, and it decodes one token per iteration."""
    tokens = pages * PAGE_SIZE
    layout = StreamLayout(
        spans=(PrefixSpan(tokens, tokens, WALK),), growth=(("decode", 1),), keyed_spans=int(bool(keyed)),
    )
    _runner(kv).ingest_request(rid, {"kv": KVReqConfig(
        prefix_keys={"main": _keys(keyed)} if keyed else None, kv_layout={"main": layout},
        loop_caps={"decode": 4 * PAGE_SIZE},
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
    freed = kv.drop(rid, kv._laid_out_tokens(rid))
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


def _running(kv, rid: str, pages: int, arrival: int) -> None:
    """Admit ``rid`` and prefill its prompt of ``pages`` pages, ending on a full page."""
    _request(kv, rid, pages, arrival)
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
    kv.drop("b", 2 * PAGE_SIZE)
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
    assert waits == Victims([], [], [], waits=1), "r2 took r1's victim or moved itself"
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


def test_a_dropped_request_admitted_again_isnt_chosen_before_its_rebuild_lands():
    """A dropped request admitted again holds an allotment and no page until its rebuild writes. Asked to give the
    allotment back, it would leave its rebuild row running on none, and the worker would keep offering that row."""
    kv = _pool()
    _running(kv, "o", 4, arrival=0)
    _running(kv, "r", 2, arrival=1)
    _running(kv, "v", 2, arrival=2)
    assert _step(kv, "r", 2 * PAGE_SIZE).ok
    assert _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: True).to_drop == ["v"]
    _drop(kv, "v")
    assert _step(kv, "r", 1).ok
    _remove(kv, "o")
    assert _admitted(kv, "v")
    short = _step(kv, "r", 4 * PAGE_SIZE).reason.pages_short

    victims = _preempt(kv, "r", short, droppable=lambda rid: True)

    assert "v" not in victims.to_ask, "v gave back the allotment its rebuild runs on"


def test_a_request_holding_a_served_walk_in_its_lease_isnt_chosen():
    """A walk the cache serves whole never admits, so its KV lives only in the stream's lease. A newer request with
    that lease and no page, asked to give back its allotment, would release the lease, and once eviction took those
    pages its next walks would write from slot 0."""
    node = _LayoutNode()
    kv = node.kv
    _request(kv, "o", 1, arrival=0)
    assert _ready(kv, "o") and _ready(kv, RID)
    answers = [node.walk(TEXT_WALK, len(TEXT))]
    node.runner.remove_request("seed")

    kv.preempt("o", 1, [RID], droppable=lambda rid: False)
    # the older request's growth evicts the pages only the index holds
    kv._evict(len(kv._index.evictable()))
    answers += [node.walk(IMAGE_WALK, IMAGE), node.walk(TEXT_WALK, len(TAIL))]

    assert answers == [len(TEXT), IMAGE, 4 * PAGE_SIZE - len(TEXT) - IMAGE], "a move released the served walk's KV"


def test_a_request_holding_a_served_walk_in_its_lease_doesnt_move_itself():
    """With nothing newer to move, a request that ran short goes to the host itself, and the move releases its
    leases. A page on another label made it look movable while its lease held a served walk's KV."""
    node = _LayoutNode()
    kv = node.kv
    kv._cpu_pool = host._HostPool(64)
    assert _ready(kv, RID)
    node.walk(TEXT_WALK, len(TEXT))
    step, ctx = KVStep(segments=(Segment(RID, "other", PAGE_SIZE),)), _ctx(RID)
    assert kv.admit(step, ctx).ok
    kv.plan(step, ctx)
    kv.commit(step, ctx)

    victims = kv.preempt(RID, 1, [], droppable=lambda rid: False)

    assert RID not in victims.to_host, "r1 moved itself to the host, releasing the served walk's KV"


def test_a_victim_whose_move_fails_comes_off_the_earmark():
    kv = _pool()
    kv._cpu_pool = host._HostPool(4)
    _running(kv, "x", 2, arrival=0)
    _running(kv, "r", 5, arrival=1)
    _running(kv, "v", 3, arrival=2)
    assert _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: False).to_host == ["v"]
    # an offload no victim chose fills the host before v moves
    assert kv.offload("x") == 2

    assert _offload(kv, "v") == 0
    assert "r" not in kv._earmarks and not kv._victims, "v's failed move left r's earmark open, so r can't choose again"


@pytest.mark.parametrize("then", ["drops itself", "waits", "waits, then is dropped"])
def test_the_oldest_request_drops_itself_or_waits_for_room(then):
    kv = _pool()
    _running(kv, "older", 5, arrival=0)
    _running(kv, "r", 5, arrival=1)

    victims = _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: then == "drops itself")

    drops = then == "drops itself"
    assert victims == Victims([], ["r"] if drops else [], [], waits=0 if drops else 1), (
        "r waits only when it can't drop itself"
    )
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


def test_a_waiting_request_asks_for_the_room_its_grant_counted_too():
    """A grant two pages past its allotment, with one page free, is one page short. Asking for that page alone, the
    request was let go at once into the free page, and its next grant failed the same way."""
    kv = _pool()
    _running(kv, "a", 4, arrival=0)
    _running(kv, "r", 5, arrival=1)
    short = _step(kv, "r", PAGE_SIZE + 1).reason.pages_short
    assert _preempt(kv, "r", short, droppable=lambda rid: False).waits == 2, "r asked for less than its grant needs"
    _runner(kv).readmit("r", NODE)

    assert not _admitted(kv, "r"), "r was let go into the one free page, though its grant needs two"


@pytest.mark.parametrize("cover", ["a victim's move", "room freed since"])
def test_the_room_a_short_grant_counted_is_set_aside_for_it(cover):
    """A grant two pages past its allotment, with one page free, is one page short. Setting aside only the page a
    move freed, or nothing once the room had grown to cover the grant, left the free page to another request's
    growth, and the grant failed again."""
    kv = _pool()
    _running(kv, "a", 3, arrival=0)
    _running(kv, "r", 5, arrival=1)
    _running(kv, "v", 1, arrival=2)
    short = _step(kv, "r", PAGE_SIZE + 1).reason.pages_short
    if cover == "room freed since":
        _remove(kv, "v")
    victims = _preempt(kv, "r", short, droppable=lambda rid: True)
    if cover == "a victim's move":
        assert victims.to_drop == ["v"]
        _drop(kv, "v")
    _step(kv, "a", PAGE_SIZE)

    assert _step(kv, "r", PAGE_SIZE + 1).ok, "a grew into the room r's grant counted, and r's grant failed again"


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

    kv.drop("v", 2 * PAGE_SIZE)

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
