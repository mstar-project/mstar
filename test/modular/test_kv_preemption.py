"""A request whose grant runs short moves newer requests off the device and takes the pages they free.

A new prompt waits in the runner's queue for room, since a pool admits only its oldest waiting request.
A request that grows past what it was allotted can't wait that way: every request beside it
may be growing too, and nothing would ever finish. The newest requests go to the host when the host holds
them whole, and are otherwise dropped if they can be rebuilt from their token ids. A request on the host or dropped
waits outside the queue and is tried ahead of it, oldest first, and returns once its history and one more page fit.
While it is the oldest request, no newer prompt starts before it. So the oldest request always gets its page and
finishes.
"""

from __future__ import annotations

import sys
from array import array

sys.path.insert(0, ".")

import pytest
import test_kv_host_index as host
from test_kv_admission import _RUNNERS, _remove, _runner
from test_kv_prefix_layout_lease import IMAGE, IMAGE_WALK, RID, TAIL, TEXT, TEXT_WALK
from test_kv_prefix_layout_lease import _Node as _LayoutNode
from test_kv_prefix_lease import NODE, PAGE_SIZE, WALK, _ctx, _keys, _manager, _StubTransfer
from test_kv_row_passes import backend  # noqa: F401, the Rust and the Python page arena
from test_runner_admission import _StubPool

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.engine import Engine
from mstar.engine.resources.base import Rebuild
from mstar.engine.resources.kv import manager as manager_mod
from mstar.engine.resources.kv.cache import GroupPages
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PrefixSpan, StreamLayout
from mstar.engine.resources.kv.manager import Victims
from mstar.engine.resources.runner import StepRunner
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


def _returns(kv) -> list[str]:
    return list(_runner(kv)._admission._returns)


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


def test_a_dropped_request_returns_ahead_of_the_queue():
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

    assert not _ready(kv, "c") and "b" in kv._allotments, "c took the room the request dropped ahead of it returns into"


def test_a_request_on_the_host_holds_up_no_newer_prompt():
    """A request on the host returns with its history and a page, which can take long to fit. Waiting at its
    arrival, it held back every newer prompt, though they fit."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(64)
    _running(kv, "r", 5, arrival=1)
    _running(kv, "v", 5, arrival=2)
    assert _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: False).to_host == ["v"]
    _offload(kv, "v")
    _request(kv, "c", 1, arrival=3)

    assert _ready(kv, "c") and not _admitted(kv, "v"), "c waited behind v, which can't return yet"


@pytest.mark.parametrize("moved", ["dropped", "to the host"])
def test_a_request_that_moved_holds_back_newer_prompts_once_it_is_the_oldest(moved):
    """v returns with its whole history and a page. Waiting outside the queue for that, it was passed by every newer
    prompt that fit, and a steady stream of small ones kept it waiting until it timed out. Newer prompts pass it only
    while an older request runs."""
    kv = _pool()
    if moved == "to the host":
        kv._cpu_pool = host._HostPool(64)
    _running(kv, "r", 3, arrival=1)
    _running(kv, "v", 6, arrival=2)
    victims = _preempt(kv, "r", _step(kv, "r", 2 * PAGE_SIZE).reason.pages_short, droppable=lambda rid: True)
    if moved == "dropped":
        assert victims.to_drop == ["v"]
        _drop(kv, "v")
    else:
        assert victims.to_host == ["v"]
        _offload(kv, "v")
    assert _step(kv, "r", 2 * PAGE_SIZE).ok
    _running(kv, "c1", 4, arrival=3)
    _remove(kv, "r")
    _request(kv, "c2", 1, arrival=4)

    assert not _ready(kv, "c2"), "c2 passed v, the oldest request, which waits to return"
    _remove(kv, "c1")
    assert _admitted(kv, "v") and _ready(kv, "c2")


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
    assert victims.waits and not victims.to_drop, "r dropped its history for room that only newer requests take"


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


def test_a_request_whose_probe_holds_a_lease_isnt_chosen():
    """A probe moves the position counter as it takes its lease, before its walk's step admits. A newer request whose
    step was refused in between gave the lease back and came back to a smaller hit, and its walk wrote from where the
    first probe had moved the counter, which never moves back."""
    node = _LayoutNode()
    kv = node.kv
    _request(kv, "o", 1, arrival=0)
    assert _ready(kv, "o") and _ready(kv, RID)
    node.probe(TEXT_WALK, len(TEXT))
    node.runner.remove_request("seed")

    kv.preempt("o", 1, [RID], droppable=lambda rid: False)
    kv._evict(len(kv._index.evictable()))
    node.walk(TEXT_WALK, len(TEXT))

    assert node.rope.position(RID, "main") == len(TEXT), "the walk wrote past where its probe had moved the counter"


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


def test_a_request_dropped_before_its_victim_lands_returns_with_what_its_drop_recorded():
    """b runs short and moves v, then o runs short and moves b before v lands. v's move then admitted b for the
    shortage b ran short by, not the history b returns with."""
    kv = _pool()
    _running(kv, "o", 2, arrival=0)
    _running(kv, "b", 3, arrival=1)
    _running(kv, "v", 4, arrival=2)
    assert _preempt(kv, "b", _step(kv, "b", 3 * PAGE_SIZE).reason.pages_short, lambda rid: True).to_drop == ["v"]
    short = _step(kv, "o", 3 * PAGE_SIZE).reason.pages_short
    assert _preempt(kv, "o", short, droppable=lambda rid: rid == "b").to_drop == ["b"]

    _drop(kv, "b")
    _drop(kv, "v")

    assert "b" not in kv._allotments or kv._allotments["b"].pages >= kv._dropped["b"].returns, (
        "v's move admitted b with the shortage it ran short by, not what it returns with"
    )


@pytest.mark.parametrize("then", ["waits", "waits, then is dropped"])
def test_a_request_with_nothing_newer_to_move_waits_for_room(then):
    """A request that ran short with nothing newer to move dropped itself. Nothing older waited for the room, so it
    went to newer prompts, or back to the request itself, which ran short again."""
    kv = _pool()
    _running(kv, "older", 5, arrival=0)
    _running(kv, "r", 5, arrival=1)

    victims = _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: True)

    assert victims == Victims([], [], [], waits=1), "r moved itself"
    _runner(kv).readmit("r", NODE)
    assert _returns(kv) == ["r"] and not _admitted(kv, "r"), "r didn't wait for its shortage"
    if then == "waits, then is dropped":
        _drop(kv, "r")
        returns = kv._dropped["r"].returns
        assert kv._need("r", NODE, WALK, None) == returns, "r's wait for its shortage outlived its drop"
        _remove(kv, "older")
        assert _admitted(kv, "r"), "r didn't return once its history and a page fit"
    elif then == "waits":
        _remove(kv, "older")
        assert _admitted(kv, "r") and _step(kv, "r", 1).ok, "r kept waiting after the room grew"


def test_a_request_with_nothing_newer_to_move_waits_on_the_host_when_it_fits_there():
    """Waiting holding its pages, r kept them from the requests beside it. On the host it frees them, as main's
    offload-and-hold does, and it never drops itself."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(64)
    _running(kv, "older", 5, arrival=0)
    _running(kv, "r", 5, arrival=1)

    victims = _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: True)

    assert victims == Victims(["r"], [], []), "r didn't move itself to the host"
    assert _offload(kv, "r") == 5 and _returns(kv) == ["r"] and "r" not in kv._allotments
    _remove(kv, "older")
    assert _ready(kv, "r") and _step(kv, "r", 1).ok, "r didn't return once its history and a page fit"


def test_a_request_the_host_cant_take_whole_waits_holding_its_pages():
    kv = _pool()
    kv._cpu_pool = host._HostPool(4)
    _running(kv, "older", 5, arrival=0)
    _running(kv, "r", 5, arrival=1)

    victims = _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: True)

    assert victims == Victims([], [], [], waits=1), "r moved to a host that can't hold it"


def test_a_request_back_from_the_host_runs_the_grant_it_moved_for():
    """r's grant needs two pages past its allotment, and one is free. Returning with its history and one page, r went
    to the host and straight back on every pass, and its grant ran short again each time. It returns only once its
    history and both pages fit."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(64)
    _running(kv, "a", 4, arrival=0)
    _running(kv, "r", 5, arrival=1)
    short = _step(kv, "r", PAGE_SIZE + 1).reason.pages_short
    assert _preempt(kv, "r", short, droppable=lambda rid: False).to_host == ["r"]
    _offload(kv, "r")

    assert not _admitted(kv, "r"), "r came straight back into the room it freed, short of its grant"
    _remove(kv, "a")
    assert _ready(kv, "r") and _step(kv, "r", PAGE_SIZE + 1).ok, "r came back without the pages its grant needed"


@pytest.mark.usefixtures("backend")
def test_a_request_back_from_the_host_with_a_shared_prefix_runs_the_grant_it_moved_for():
    """r reads 3 of its 5 pages from a's prefix and needs two pages past its allotment, with one free. Its return
    counted the pages r holds without those 3, but its reload counts them, so r came back a page short and went to
    the host again."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(64)
    shared = list(range(3 * PAGE_SIZE))
    _running(kv, "b", 3, arrival=0)
    _request(kv, "a", 4, arrival=1, keyed=shared + list(range(100, 100 + PAGE_SIZE)))
    assert _ready(kv, "a") and _step(kv, "a", 4 * PAGE_SIZE).ok
    _request(kv, "r", 5, arrival=2, keyed=shared + list(range(200, 200 + 2 * PAGE_SIZE)))
    assert _ready(kv, "r")
    inputs = ARNodeInputs(input_seq_len=5 * PAGE_SIZE)
    kv.apply_cached_prefix("r", NODE, WALK, inputs, kv.resolve_cached_prefix("r", NODE, WALK, inputs))
    assert _step(kv, "r", 2 * PAGE_SIZE).ok and kv._arena.num_free == 1
    moves = []
    for _ in range(2):
        outcome = _step(kv, "r", PAGE_SIZE + 1)
        if outcome.ok:
            break
        assert _preempt(kv, "r", outcome.reason.pages_short, droppable=lambda rid: False).to_host == ["r"]
        moves.append(_offload(kv, "r"))
        if not _ready(kv, "r"):
            _remove(kv, "b")
            assert _ready(kv, "r")

    assert len(moves) == 1, f"r went to the host {len(moves)} times for one grant"


def _make_room(kv, droppable) -> Victims | None:
    """`KVManager.make_room` for the oldest waiting request the pool refused that moves can let in, as the worker
    calls it."""
    runner = _runner(kv)
    for refusal in runner.refusals():
        victims = kv.make_room(refusal.mark, runner.newer_than(refusal.rid), droppable)
        if victims is not None:
            return victims
    return None


def test_an_older_waiting_request_moves_a_newer_one_that_waits_holding_pages():
    """With nothing newer to move, r waits holding its pages. o, older and dropped, waits to return into pages only r
    holds, and r runs no step that could run short again. o moves r, and takes the room as r's drop lands, so r
    can't take it back first."""
    kv = _pool()
    _running(kv, "x", 2, arrival=0)
    _running(kv, "o", 3, arrival=1)
    _running(kv, "r", 5, arrival=2)
    kv._streams["r"]["main"].step_in_flight = True
    assert _preempt(kv, "x", _short(kv, "x"), droppable=lambda rid: True).to_drop == ["o"]
    kv._streams["r"]["main"].step_in_flight = False
    _drop(kv, "o")
    assert _step(kv, "r", 2 * PAGE_SIZE).ok, "r grows into what room is left"
    assert _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: True).waits, "r moved itself"
    _runner(kv).readmit("r", NODE)
    assert not _admitted(kv, "o")

    assert _make_room(kv, droppable=lambda rid: True).to_drop == ["r"]
    _drop(kv, "r")

    assert "o" in kv._allotments, "the room r's drop freed wasn't set aside for o"
    assert _admitted(kv, "o") and not _admitted(kv, "r")


def test_requests_each_short_by_several_pages_dont_wait_on_each_other_for_good():
    """Each runs short by four pages while the newer ones have a step in flight, and w1 and w2 can each free only
    three. Moving only for an older request that one move lets in, all three waited, and nothing ran again. The
    oldest moves both."""
    kv = _pool()
    rids = ["w0", "w1", "w2"]
    for arrival, (rid, pages) in enumerate(zip(rids, (4, 3, 3), strict=True)):
        _running(kv, rid, pages, arrival)
    for i, rid in enumerate(rids):
        short = _step(kv, rid, 4 * PAGE_SIZE).reason.pages_short
        for other in rids[i + 1:]:
            kv._streams[other]["main"].step_in_flight = True
        assert _preempt(kv, rid, short, droppable=lambda rid: True).waits
        for other in rids[i + 1:]:
            kv._streams[other]["main"].step_in_flight = False
        _runner(kv).readmit(rid, NODE)
    assert not any(_admitted(kv, rid) for rid in ("w0", "w1", "w2"))

    assert _make_room(kv, droppable=lambda rid: True).to_drop == ["w2", "w1"]
    _drop(kv, "w2")
    _drop(kv, "w1")

    assert _admitted(kv, "w0") and _step(kv, "w0", 4 * PAGE_SIZE).ok, "w0 didn't get the pages w1 and w2 freed"


def test_a_waiting_request_admitted_before_its_last_victim_moves_takes_no_more_room():
    """w0 waits for 4 pages and moves w2 and w1. A host move lands before a drop, so w1's 5 pages can admit w0 before
    w2 moves. w2's page then went into w0's allotment too, past what w0 asked for."""
    kv = _pool()
    rids = ["w0", "w1", "w2"]
    for arrival, (rid, pages) in enumerate(zip(rids, (4, 5, 1), strict=True)):
        _running(kv, rid, pages, arrival)
    for i, rid in enumerate(rids):
        short = _step(kv, rid, 4 * PAGE_SIZE).reason.pages_short
        for other in rids[i + 1:]:
            kv._streams[other]["main"].step_in_flight = True
        assert _preempt(kv, rid, short, droppable=lambda rid: True).waits
        for other in rids[i + 1:]:
            kv._streams[other]["main"].step_in_flight = False
        _runner(kv).readmit(rid, NODE)
    assert not any(_admitted(kv, rid) for rid in rids)
    assert _make_room(kv, droppable=lambda rid: True).to_drop == ["w2", "w1"]

    _drop(kv, "w1")
    pages = kv._allotments["w0"].pages
    _drop(kv, "w2")

    assert kv._allotments["w0"].pages == pages, "w2's move grew the allotment of a request it already admitted"
    assert _admitted(kv, "w0") and _step(kv, "w0", 4 * PAGE_SIZE).ok


def test_a_waiting_request_the_runner_admits_before_its_victims_land_takes_no_more_room():
    """w0 waits for 4 pages and moves w2 and w1. x's removal frees those 4 pages first, and the runner admits w0 into
    them. The pages w2 and w1 freed then went into w0's allotment too."""
    kv = _pool()
    rids = ["w0", "w1", "w2"]
    _running(kv, "x", 4, arrival=0)
    for arrival, rid in enumerate(rids, start=1):
        _running(kv, rid, 2, arrival)
    for i, rid in enumerate(rids):
        short = _step(kv, rid, 4 * PAGE_SIZE).reason.pages_short
        for other in rids[i + 1:]:
            kv._streams[other]["main"].step_in_flight = True
        assert _preempt(kv, rid, short, droppable=lambda rid: True).waits
        for other in rids[i + 1:]:
            kv._streams[other]["main"].step_in_flight = False
        _runner(kv).readmit(rid, NODE)
    assert not any(_admitted(kv, rid) for rid in rids)
    assert _make_room(kv, droppable=lambda rid: True).to_drop == ["w2", "w1"]
    _remove(kv, "x")
    assert _admitted(kv, "w0")
    pages = kv._allotments["w0"].pages

    _drop(kv, "w2")
    _drop(kv, "w1")

    assert kv._allotments["w0"].pages == pages, "the victims' moves grew the allotment of a request already admitted"


def test_a_request_no_move_covers_waits_until_its_whole_shortage_fits():
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


def test_a_rollback_keeps_the_allotment_of_a_request_that_waits_for_its_shortage():
    """r holds 6 of the pool's 10 pages and waits for 2 more. Once they fit, a pool on another node refused r, and the
    rollback gave back r's allotment and asked for its 6 pages and the 2 again, on top of the 6 it holds. The pool
    never had room for that."""
    kv, slots = _pool(), _StubPool(0)
    _RUNNERS[id(kv)] = (kv, StepRunner({"kv": kv, "slots": slots}, node_resources={NODE: ["kv"], "other": ["slots"]}))
    _running(kv, "o", 3, arrival=0)
    _running(kv, "r", 6, arrival=1)
    short = _step(kv, "r", PAGE_SIZE + 1).reason.pages_short
    assert _preempt(kv, "r", short, droppable=lambda rid: False).waits == 2
    _runner(kv).readmit("r", NODE)
    assert _runner(kv).admit_request("r", "other", WALK) is not None
    _remove(kv, "o")
    assert not _admitted(kv, "r"), "r started though the other node's pool refused it"

    slots.free = 1

    assert _admitted(kv, "r"), "r's pool asked for its whole history after the rollback"
    assert _step(kv, "r", PAGE_SIZE + 1).ok


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
def test_a_dropped_request_is_charged_nothing_for_hits_another_request_holds():
    """v reads 3 of its 4 pages from a's prefix. Dropped, it asked to return with its whole history and a page, though
    a holds those 3 pages, which can't be evicted, and the rebuild's probe takes them for nothing. With a shared
    prefix, that let one return into the pool at a time."""
    kv = _pool()
    shared = list(range(3 * PAGE_SIZE))
    tokens = shared + list(range(200, 200 + PAGE_SIZE))
    _request(kv, "a", 4, arrival=0, keyed=shared + list(range(100, 100 + PAGE_SIZE)))
    assert _ready(kv, "a") and _step(kv, "a", 4 * PAGE_SIZE).ok
    _running(kv, "r", 2, arrival=1)
    _running(kv, "s", 3, arrival=2)
    _request(kv, "v", 4, arrival=3, keyed=tokens)
    assert _ready(kv, "v")
    inputs = ARNodeInputs(input_seq_len=4 * PAGE_SIZE)
    kv.apply_cached_prefix("v", NODE, WALK, inputs, kv.resolve_cached_prefix("v", NODE, WALK, inputs))
    assert _step(kv, "v", PAGE_SIZE).ok and kv._arena.num_free == 0
    assert _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: True).to_drop == ["v"]
    _drop(kv, "v")
    assert _step(kv, "r", 1).ok
    _remove(kv, "s")

    assert _admitted(kv, "v"), "v waited for room for the 3 pages a holds"
    kv.prepare_rebuild("v", Rebuild(WALK, array("i", [*tokens, 0])))
    kv.apply_cached_prefix("v", NODE, WALK, inputs, kv.resolve_cached_prefix("v", NODE, WALK, inputs))
    assert _step(kv, "v", PAGE_SIZE).ok and _step(kv, "v", PAGE_SIZE + 1).ok, "v ran short of its allotment"


def _hit(kv, rid: str, pages: int) -> None:
    inputs = ARNodeInputs(input_seq_len=pages * PAGE_SIZE)
    kv.apply_cached_prefix(rid, NODE, WALK, inputs, kv.resolve_cached_prefix(rid, NODE, WALK, inputs))


@pytest.mark.usefixtures("backend")
def test_a_dropped_request_refused_before_its_rebuild_is_measured_on_the_chain_the_rebuild_lays_out():
    """v is dropped and refused, and readiness then lays out its rebuild's chain. `make_room` measured v's refusal
    against that chain before the next `reserve` rooted it, and raised ``TypeError``."""
    kv = _pool()
    shared = list(range(3 * PAGE_SIZE))
    tokens = shared + list(range(200, 200 + PAGE_SIZE))
    _request(kv, "a", 4, arrival=0, keyed=shared + list(range(100, 100 + PAGE_SIZE)))
    assert _ready(kv, "a") and _step(kv, "a", 4 * PAGE_SIZE).ok
    _running(kv, "r", 2, arrival=1)
    _running(kv, "s", 2, arrival=2)
    _request(kv, "v", 4, arrival=3, keyed=tokens)
    assert _ready(kv, "v")
    _hit(kv, "v", 4)
    assert _step(kv, "v", PAGE_SIZE).ok
    _running(kv, "w", 1, arrival=4)
    kv._streams["w"]["main"].step_in_flight = True
    assert _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: True).to_drop == ["v"]
    kv._streams["w"]["main"].step_in_flight = False
    _drop(kv, "v")
    assert _step(kv, "r", 1).ok
    assert not _admitted(kv, "v")
    outcome = _step(kv, "w", PAGE_SIZE + 1)
    assert _preempt(kv, "w", outcome.reason.pages_short, droppable=lambda rid: False).waits
    _runner(kv).readmit("w", NODE)

    kv.prepare_rebuild("v", Rebuild(WALK, array("i", [*tokens, 0])))

    assert _make_room(kv, droppable=lambda rid: True) is None, "moving w frees less than v's return needs"
    assert kv._need("v", NODE, WALK, None) == kv._dropped["v"].returns - 3, "v paid for the 3 pages a holds"


@pytest.mark.usefixtures("backend")
def test_a_waiting_request_whose_victim_held_its_hits_is_charged_for_them_once_the_victim_moves():
    """v, dropped, shares 3 prompt pages with h, which holds them, so they cost v's return nothing. v moves h, which
    waits for its shortage. h's drop leaves those pages to the index alone, where eviction can take them, so v's probe
    pays for them. v was admitted with the count from before the move, 3 pages short."""
    kv = _pool()
    shared = list(range(3 * PAGE_SIZE))
    _request(kv, "a", 4, arrival=0, keyed=shared + list(range(100, 100 + PAGE_SIZE)))
    assert _ready(kv, "a") and _step(kv, "a", 4 * PAGE_SIZE).ok
    _remove(kv, "a")
    _running(kv, "r", 2, arrival=1)
    for rid, arrival, own in (("v", 2, 200), ("h", 3, 300)):
        _request(kv, rid, 4, arrival=arrival, keyed=shared + list(range(own, own + PAGE_SIZE)))
        assert _ready(kv, rid)
        _hit(kv, rid, 4)
        assert _step(kv, rid, PAGE_SIZE).ok
    _running(kv, "w", 3, arrival=4)
    for rid in ("h", "w"):
        kv._streams[rid]["main"].step_in_flight = True
    assert _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: True).to_drop == ["v"]
    kv._streams["h"]["main"].step_in_flight = False
    _drop(kv, "v")
    assert _step(kv, "r", 1).ok
    assert _preempt(kv, "h", _step(kv, "h", PAGE_SIZE + 1).reason.pages_short, droppable=lambda rid: True).waits
    _runner(kv).readmit("h", NODE)
    assert not _admitted(kv, "v")
    assert _make_room(kv, droppable=lambda rid: True).to_drop == ["h"]

    _drop(kv, "h")

    returns = kv._dropped["v"].returns
    assert "v" not in kv._allotments or kv._allotments["v"].pages == returns, "v came back short of its hits"


def test_a_rollback_puts_back_the_allotment_of_a_request_past_it():
    """r grew a page past its allotment of 6 and then waits for 1 more. The grow counted from the 7 pages r holds, and
    a pool on another node refusing r took back only the page r asked for, so r's allotment grew to 7."""
    kv, slots = _pool(), _StubPool(0)
    _RUNNERS[id(kv)] = (kv, StepRunner({"kv": kv, "slots": slots}, node_resources={NODE: ["kv"], "other": ["slots"]}))
    _running(kv, "o", 3, arrival=0)
    _running(kv, "r", 6, arrival=1)
    assert _step(kv, "r", PAGE_SIZE).ok and kv._allotments["r"].pages == 6
    assert _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: False).waits
    _runner(kv).readmit("r", NODE)
    assert _runner(kv).admit_request("r", "other", WALK) is not None

    _remove(kv, "o")

    assert not _admitted(kv, "r")
    assert kv._allotments["r"].pages == 6, "the rollback left r an allotment of the pages it holds"


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


def test_a_page_an_earlier_row_gives_back_is_no_release():
    """Run row by row, a step admits each row as a step of its own and rolls back the rows ahead of one that fails,
    after that row has rolled back. r's grant counted the page a then gave back as room it had seen, took it, and a's
    retry ran short, which would move r for nothing."""
    kv = _pool()
    _running(kv, "a", 4, arrival=0)
    _running(kv, "r", 5, arrival=1)
    rows = {rid: (KVStep(segments=(Segment(rid, "main", 1),)), _ctx(rid)) for rid in ("a", "r")}
    assert kv.admit(*rows["a"]).ok
    outcome = kv.admit(*rows["r"])
    kv.rollback_admit(*rows["r"])
    kv.rollback_admit(*rows["a"])

    _preempt(kv, "r", outcome.reason.pages_short, droppable=lambda rid: False)

    assert _step(kv, "a", 1).ok, "r took the page a's rollback gave back, so a's retry ran short"


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
    """A request waiting on the host holds no page on the device. A move that copied only some of its
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
    assert sorted(_returns(kv)) == ["a", "b"] and not kv._allotments.keys() & {"a", "b"}, "a victim kept its allotment"

    _remove(kv, "r")

    assert _runner(kv).admit_waiting(lambda rid, node: None) == [(NODE, "a"), (NODE, "b")], "b returned ahead of a"
    assert _ready(kv, "a") and kv._allotments["a"].remaining == 1, "a didn't reserve its reload and a page"
    assert _ready(kv, "b") and not _returns(kv)


@pytest.mark.parametrize("decides", [False, True], ids=["a pool that doesn't decide", "an offload no victim chose"])
def test_an_offload_outside_preemption_reloads_as_before(decides):
    kv = _pool()
    kv._cpu_pool = host._HostPool(64)
    kv._decides = decides
    _running(kv, "r", 5, arrival=1)
    _request(kv, "w", 8, arrival=0)
    _ready(kv, "w")
    remaining = kv._allotments.get("r", manager_mod.Allotment(0)).remaining

    assert _offload(kv, "r") == 5 and "r" not in _returns(kv)

    assert kv.can_reload("r") and kv.reload("r"), "r waited to return behind w"
    assert kv._allotments.get("r", manager_mod.Allotment(0)).remaining == remaining, "the reload changed r's allotment"
