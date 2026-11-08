"""A request whose grant runs short moves newer requests off the device and takes the pages they free.

A new prompt waits in the runner's queue for room, since a pool admits only its oldest waiting request.
A request that grows past what it was allotted can't wait that way: every request beside it
may be growing too, and nothing would ever finish. The newest requests go to the host when the host holds
them whole, and are otherwise dropped if they can be rebuilt from their token ids. A request on the host waits outside
the queue and is tried ahead of it, oldest first, and returns once its pages and one more page fit. A dropped request
waits in the queue at its arrival and returns once its whole history and one more page fit.
So the oldest request always gets its page and finishes.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import test_kv_host_index as host
from test_kv_admission import _remove, _runner
from test_kv_prefix_layout_lease import IMAGE, IMAGE_WALK, RID, TAIL, TEXT, TEXT_WALK
from test_kv_prefix_layout_lease import _Node as _LayoutNode
from test_kv_prefix_lease import NODE, PAGE_SIZE, WALK, _ctx, _keys, _manager, _StubTransfer
from test_kv_release import _window_manager
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
    freed = kv.drop(rid, kv._laid_out_tokens(rid))
    _runner(kv).readmit(rid, NODE, queued=True)
    return freed


def _offload(kv, rid: str) -> int:
    freed = kv.offload(rid)
    _runner(kv).readmit(rid, NODE, queued=False)
    return freed


def _preempt(kv, rid: str, pages_short: int, droppable) -> Victims | None:
    """`KVManager.preempt` over the runner's arrival order, as the worker calls it."""
    return kv.preempt(rid, pages_short, _runner(kv).newer_than(rid), droppable)


def _queue(kv) -> list[str]:
    return _runner(kv)._admission._queues["kv"]


def _returns(kv) -> list[str]:
    return list(_runner(kv)._admission._returns)


def _step(kv, rid: str, tokens: int, label: str = "main", walk: str = WALK) -> AdmitOutcome:
    """One step of ``tokens`` for ``rid``, committed when it is admitted."""
    step, ctx = KVStep(segments=(Segment(rid, label, tokens),)), replace(_ctx(rid), graph_walk=walk)
    outcome = kv.admit(step, ctx)
    if outcome.ok:
        kv.plan(step, ctx)
        kv.commit(step, ctx)
    else:
        kv.rollback_admit(step, ctx)
    return outcome


def _running(kv, rid: str, pages: int, arrival: int, **kwargs) -> None:
    """Admit ``rid`` and prefill its prompt of ``pages`` pages, ending on a full page."""
    _request(kv, rid, pages, arrival, **kwargs)
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
    _runner(kv).readmit("b", NODE, queued=True)

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


def test_a_dropped_request_keeps_its_place_ahead_of_newer_prompts():
    """A dropped request returns with its whole history and a page. Waiting outside the queue for that, it was passed
    by every newer prompt that fit, and a steady stream of small ones kept it waiting until it timed out."""
    kv = _pool()
    _running(kv, "r", 5, arrival=1)
    _running(kv, "v", 5, arrival=2)
    assert _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: True).to_drop == ["v"]
    _drop(kv, "v")
    _request(kv, "c", 1, arrival=3)

    assert not _ready(kv, "c") and _queue(kv) == ["v", "c"], "c passed v, which waits to return"
    _remove(kv, "r")
    assert _admitted(kv, "v") and _ready(kv, "c")


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


def test_a_dropped_request_with_an_image_returns_with_pages_for_the_image_too():
    kv = _pool()
    image = PrefixSpan(3 * PAGE_SIZE, 4, WALK, kind="digest", opens_step=False)
    text = (PrefixSpan(PAGE_SIZE - 1, PAGE_SIZE - 1, WALK), PrefixSpan(1, 1, WALK, opens_step=False))
    layout, caps = StreamLayout(spans=(text[0], image, text[1]), growth=(("decode", 1),)), {"decode": 4 * PAGE_SIZE}
    _runner(kv).ingest_request("v", {"kv": KVReqConfig(kv_layout={"main": layout}, loop_caps=caps)}, arrival=1)
    assert _ready(kv, "v") and _step(kv, "v", 4 * PAGE_SIZE).ok
    _drop(kv, "v")

    # its rebuild writes the image's rows too: the 4 pages its KV held, a page for its next sample and one more
    assert _admitted(kv, "v") and kv._allotments["v"].pages == 6, "the rebuild would run out of pages in the image"


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


@pytest.mark.parametrize("then", ["waits", "waits, then is dropped"])
def test_a_request_with_nothing_newer_to_move_waits_for_room(then):
    """A request that ran short with nothing newer to move dropped itself. Nothing older waited for the room, so it
    went to newer prompts, or back to the request itself, which ran short again."""
    kv = _pool()
    _running(kv, "older", 5, arrival=0)
    _running(kv, "r", 5, arrival=1)

    victims = _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: True)

    assert victims == Victims([], [], [], waits=1), "r moved itself"
    _runner(kv).readmit("r", NODE, queued=False)
    assert _returns(kv) == ["r"] and not _admitted(kv, "r"), "r didn't wait for its shortage"
    if then == "waits, then is dropped":
        _drop(kv, "r")
        assert _queue(kv) == ["r"] and not _returns(kv), "r, dropped, kept waiting outside the queue"
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
    _runner(kv).readmit("r", NODE, queued=False)
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
        _runner(kv).readmit(rid, NODE, queued=False)
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
        _runner(kv).readmit(rid, NODE, queued=False)
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
        _runner(kv).readmit(rid, NODE, queued=False)
    assert not any(_admitted(kv, rid) for rid in rids)
    assert _make_room(kv, droppable=lambda rid: True).to_drop == ["w2", "w1"]
    _remove(kv, "x")
    assert _admitted(kv, "w0")
    pages = kv._allotments["w0"].pages

    _drop(kv, "w2")
    _drop(kv, "w1")

    assert kv._allotments["w0"].pages == pages, "the victims' moves grew the allotment of a request already admitted"


def test_a_request_no_move_covers_waits_in_the_queue_until_its_whole_shortage_fits():
    kv = _pool()
    _running(kv, "a", 1, arrival=0)
    _running(kv, "b", 4, arrival=1)
    _running(kv, "r", 5, arrival=2)
    short = _step(kv, "r", PAGE_SIZE + 1).reason.pages_short
    assert _preempt(kv, "r", short, droppable=lambda rid: False).waits
    _runner(kv).readmit("r", NODE, queued=False)

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
    _runner(kv).readmit("r", NODE, queued=False)

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


def test_a_request_dropped_on_a_pool_that_keeps_a_window_returns_with_its_window_pages_too():
    """A drop counted the first group's pages alone, where a return from the host counts every group's, so a request
    whose window held pages returned with too few and ran short again once its rebuild wrote them."""
    kv = _window_manager()
    kv.local_loops = {"decode"}
    page = kv.config.tokens_per_page
    layout = StreamLayout(spans=(PrefixSpan(2 * page, 2 * page, WALK),), growth=(("decode", 1),))
    _runner(kv).ingest_request("r", {"kv": KVReqConfig(kv_layout={"main": layout}, loop_caps={"decode": 4 * page})}, 0)
    assert _ready(kv, "r") and _step(kv, "r", 2 * page).ok
    for _ in range(2 * page):
        assert _step(kv, "r", 1).ok
    window = len(kv._streams["r"]["main"].later[0].page_indices)

    kv.drop("r", kv._laid_out_tokens("r"))

    # the first group's 5 pages of history, with the sample its next step writes, and a page more
    assert window and kv._dropped["r"].returns == window + 6, "r would return without room for its window's pages"


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

    assert _preempt(kv, "o", short, droppable=lambda rid: False) == Victims([], [], [], waits=2), (
        "a guaranteed request gave back its allotment for an older request's shortage"
    )
    assert "g" in kv._allotments


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


def test_room_the_host_set_aside_stays_when_a_host_fetch_holds_the_cached_pages_a_reload_left():
    """A reload gave its keyed host pages back to the host index, where they count as room without being free, and a
    host fetch then held them, leaving the room at -3 until the next request reserved."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(6)
    _running(kv, "r", 3, arrival=0, keyed=list(range(3 * PAGE_SIZE)), rebuildable=False)
    _running(kv, "s", 3, arrival=1, rebuildable=False)
    assert _offload(kv, "r")
    keys = kv._cpu_pool.offloaded["r"]["main"].keys
    assert kv.reload("r")

    kv._cpu_pool.retain_run(keys)

    assert kv._cpu_pool.room() >= 0, "a host fetch held the pages set aside for r's and s's moves"


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
    assert sorted(_returns(kv)) == ["a", "b"] and not kv._allotments.keys() & {"a", "b"}, "a victim kept its allotment"

    _remove(kv, "r")

    assert _runner(kv).admit_waiting(lambda rid, node: None) == [(NODE, "a"), (NODE, "b")], "b returned ahead of a"
    assert _ready(kv, "a") and kv._allotments["a"].remaining == 1, "a didn't reserve its reload and a page"
    assert _ready(kv, "b") and not _returns(kv)


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


def test_a_request_back_from_the_host_and_short_again_waits_there_for_its_shortage():
    """A request came back from the host once its pages and one more page fit, and its next page found no room. It
    moves to the host again, and comes back only once its history and that page fit, not into the room it freed."""
    kv = _pool()
    short = _back_from_the_host_and_short(kv)

    assert _preempt(kv, "z", short, droppable=lambda rid: False) == Victims(["z"], [], [])
    _offload(kv, "z")

    assert not _admitted(kv, "z"), "z came straight back into the room it freed"
    _remove(kv, "y")
    assert _ready(kv, "z") and _step(kv, "z", PAGE_SIZE).ok


def test_a_request_back_from_the_host_moves_again_for_an_older_request_waiting_in_the_queue():
    """A request that came back waited for its shortage holding its pages, while an older request waited in the
    queue for room those pages would make. The queue admits the older request first, so neither was admitted."""
    kv = _pool()
    short = _back_from_the_host_and_short(kv)
    _request(kv, "p", 3, arrival=2)
    assert not _ready(kv, "p")
    # another request's pages fill the host while z runs short
    filled = kv._cpu_pool.arena.acquire(kv._cpu_pool.room())
    assert _preempt(kv, "z", short, droppable=lambda rid: False).waits
    kv._cpu_pool.arena.release(filled)
    _runner(kv).readmit("z", NODE, queued=False)

    assert _make_room(kv, droppable=lambda rid: False) == Victims(["z"], [], []), (
        "z waited for its shortage behind p, which waits for z's pages"
    )
    _offload(kv, "z")
    assert _ready(kv, "p"), "p didn't get the room z made"


def _rebuilt(kv) -> None:
    """``v``, dropped for ``o``, is rebuilt beside ``a``, which arrived before it."""
    _running(kv, "o", 3, arrival=0)
    _running(kv, "v", 2, arrival=2)
    assert _step(kv, "o", 5 * PAGE_SIZE).ok
    assert _preempt(kv, "o", _short(kv, "o"), droppable=lambda rid: True).to_drop == ["v"]
    _drop(kv, "v")
    _request(kv, "a", 2, arrival=1)
    _remove(kv, "o")
    assert _ready(kv, "a") and _admitted(kv, "v")
    assert _step(kv, "v", 2 * PAGE_SIZE).ok
    kv.land_rebuild("v")


def test_a_request_back_from_a_drop_is_still_a_victim_for_an_older_request():
    kv = _pool()
    _rebuilt(kv)
    assert _step(kv, "v", 1).ok
    while (outcome := _step(kv, "a", PAGE_SIZE)).ok:
        pass

    victims = _preempt(kv, "a", outcome.reason.pages_short, droppable=lambda rid: True)

    assert "v" in (*victims.to_host, *victims.to_drop, *victims.to_ask), "v held back pages an older request needed"


def test_a_rebuilt_request_isnt_dropped_again_until_it_writes_a_token():
    """A rebuilt request lands as the newest request holding pages, so the next older request that ran short dropped
    it again before it kept a token, and the whole rebuild was wasted. With requests going around more than once,
    that could repeat for as long as older requests ran short."""
    kv = _pool()
    _rebuilt(kv)
    while (outcome := _step(kv, "a", PAGE_SIZE)).ok:
        pass

    victims = _preempt(kv, "a", outcome.reason.pages_short, _droppable(kv))

    assert "v" not in (*victims.to_host, *victims.to_drop, *victims.to_ask), "v's rebuild was thrown away unused"
    assert victims.waits, "a took room that v's next step needs"
    assert _step(kv, "v", 1).ok, "v's first step after its rebuild ran short"


def _droppable(kv):
    """Whether the KV allows a drop, as the worker asks through the engine."""
    return lambda rid: kv.check_drop(rid, kv._laid_out_tokens(rid)) is None


def test_an_older_request_short_of_pages_moves_a_rebuilt_one_to_the_host_rather_than_itself():
    """A rebuilt request was held back from every move until it wrote a token, though a move to the host keeps its
    rebuilt KV. An older request that ran short then copied itself to the host for the newer one."""
    kv = _pool()
    _rebuilt(kv)
    kv._cpu_pool = host._HostPool(16)
    while (outcome := _step(kv, "a", PAGE_SIZE)).ok:
        pass

    victims = _preempt(kv, "a", outcome.reason.pages_short, _droppable(kv))

    assert victims == Victims(["v"], [], []), "a moved itself to the host for a newer request"


def test_a_rebuilt_request_back_from_the_host_gives_back_its_allotment_before_it_reloads():
    """The hold read the tokens written past the rebuild, which an offload sets to 0 until the reload. A rebuilt
    request that had moved to the host and been let back in then held its allotment against every older request
    until it reloaded."""
    kv = _pool()
    _rebuilt(kv)
    kv._cpu_pool = host._HostPool(16)
    _offload(kv, "v")
    assert _admitted(kv, "v") and kv.residency("v") is Residency.ON_HOST
    while (outcome := _step(kv, "a", PAGE_SIZE)).ok:
        pass

    victims = _preempt(kv, "a", outcome.reason.pages_short, _droppable(kv))

    assert victims == Victims([], [], ["v"]), "v kept the allotment for a reload while an older request ran short"


def test_a_rebuilt_request_stays_undroppable_through_a_trip_to_the_host_until_it_writes():
    kv = _pool()
    _rebuilt(kv)
    kv._cpu_pool = host._HostPool(16)
    _offload(kv, "v")
    assert _ready(kv, "v")

    assert not _droppable(kv)("v"), "a drop would throw away the rebuild before it kept a token"
    assert _step(kv, "v", 1).ok
    assert _droppable(kv)("v"), "v stayed undroppable after writing past its rebuild"


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

    assert _offload(kv, "r") == 5 and "r" not in _returns(kv)

    assert kv.can_reload("r") and kv.reload("r"), "r waited to return behind w"
    assert kv._allotments.get("r", manager_mod.Allotment(0)).remaining == remaining, "the reload changed r's allotment"


def test_a_guaranteed_request_is_never_moved_to_the_host():
    kv = _pool()
    kv._cpu_pool = host._HostPool(64)
    _running(kv, "r", 3, arrival=0)
    _running(kv, "g", 2, arrival=1, guaranteed=True)
    short = _step(kv, "r", 2 * PAGE_SIZE).reason.pages_short

    victims = _preempt(kv, "r", short, droppable=lambda rid: False)

    assert "g" not in victims.to_host, "a guaranteed request was moved to the host"


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


def test_a_victim_that_holds_nothing_waits_with_no_room_set_aside_on_the_host():
    """A victim admitted but not yet run gave back its allotment and kept its room on the host, so r's next page found
    the host full and r was short of a guarantee it didn't need."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(4)
    _running(kv, "o", 6, arrival=0)
    _running(kv, "r", 2, arrival=1, rebuildable=False)
    _request(kv, "a", 2, arrival=2, rebuildable=False)
    assert _ready(kv, "a") and kv._cpu_pool.reserved == {"r": 2, "a": 2}

    assert _preempt(kv, "o", _short(kv, "o"), droppable=lambda rid: False).to_ask == ["a"]

    assert "a" not in kv._cpu_pool.reserved and _step(kv, "r", PAGE_SIZE).ok, (
        "a held room on the host while it waited in the queue"
    )


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


def test_a_request_the_full_host_turns_guaranteed_before_it_adopts_its_lease_grows_within_its_allotment():
    """`_cover_move` left the lease's lent pages out of the allotment it guaranteed, and `_adopt` took them off again,
    so x took room g's guarantee covered."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(4)
    tokens = list(range(3 * PAGE_SIZE))
    _running(kv, "h", 2, arrival=0, keyed=tokens[:2 * PAGE_SIZE])
    main = StreamLayout(spans=(PrefixSpan(3 * PAGE_SIZE, 3 * PAGE_SIZE, WALK),), growth=(("decode", 1),), keyed_spans=1)
    other = StreamLayout(spans=(PrefixSpan(PAGE_SIZE, PAGE_SIZE, "other"),), growth=(("decode", 1),))
    _runner(kv).ingest_request("g", {"kv": KVReqConfig(
        prefix_keys={"main": _keys(tokens)}, kv_layout={"main": main, "other": other},
        needed_labels_per_node_walk={(NODE, WALK): ["main"], (NODE, "other"): ["other"]},
        loop_caps={"decode": 2 * PAGE_SIZE}, rebuildable=False,
    )}, 1)
    assert _ready(kv, "g")
    inputs = ARNodeInputs(input_seq_len=3 * PAGE_SIZE)
    kv.apply_cached_prefix("g", NODE, WALK, inputs, kv.resolve_cached_prefix("g", NODE, WALK, inputs))
    assert kv._streams["g"]["main"].lent == set(kv._streams["g"]["main"].lease)
    # the third page is one the host can't cover
    for _ in range(3):
        assert _step(kv, "g", PAGE_SIZE, label="other", walk="other").ok
    assert "g" in kv._guaranteed
    assert _step(kv, "g", PAGE_SIZE).ok
    _running(kv, "x", kv._room(), arrival=2)

    for _ in range(2):
        assert _step(kv, "g", PAGE_SIZE).ok, "g ran short of pages its guarantee covered"


def test_a_request_neither_the_host_nor_the_device_can_cover_is_short_of_its_footprint_and_can_still_move():
    kv = _pool()
    kv._cpu_pool = host._HostPool(3)
    _running(kv, "r", 2, arrival=1, rebuildable=False)
    _running(kv, "s", 1, arrival=2, rebuildable=False)
    _running(kv, "x", 6, arrival=3)

    outcome = _step(kv, "r", PAGE_SIZE)

    # the 4 pages it grows into, less the 1 page of room
    assert not outcome.ok and outcome.reason.pages_short == 3, "r took a page its move had no room for"
    assert _preempt(kv, "r", 3, droppable=lambda rid: False) == Victims(["r"], [], []) and _offload(kv, "r"), (
        "r had no way off the device"
    )


def test_a_waiting_request_the_full_host_turns_guaranteed_moves_more_newer_requests_for_its_footprint():
    """p asks the host to set room aside for its prompt only once z2's drop has made room for that prompt. The host
    has none, so p turns guaranteed and asks for its whole footprint, which z2's pages don't cover."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(1)
    _running(kv, "o", 2, arrival=0)
    _running(kv, "z1", 4, arrival=2)
    _running(kv, "z2", 4, arrival=3)
    # z1 waits for more than z2's drop frees, so its return doesn't take that room first
    for rid, tokens in (("z2", 1), ("z1", 5 * PAGE_SIZE)):
        short = _step(kv, rid, tokens).reason.pages_short
        assert _preempt(kv, rid, short, droppable=lambda rid: False).waits
        _runner(kv).readmit(rid, NODE, queued=False)
    _request(kv, "p", 2, arrival=1, rebuildable=False)
    assert not _ready(kv, "p")
    assert _make_room(kv, droppable=lambda rid: True).to_drop == ["z2"]
    _drop(kv, "z2")
    assert "p" in kv._guaranteed and not _admitted(kv, "p")

    assert _make_room(kv, droppable=lambda rid: True).to_drop == ["z1"], "p waited for good once it turned guaranteed"
    _drop(kv, "z1")
    assert _admitted(kv, "p") and kv._allotments["p"].pages == 6


def test_a_waiting_request_that_turns_guaranteed_as_one_victim_lands_isnt_admitted_short_by_the_next():
    """p asks for its 2-page prompt and moves z2 and z1. z1's move lands first and covers the prompt, but the host has
    no room for it, so p turns guaranteed and asks for its 6-page footprint. z2's move then covered the prompt again,
    and p was admitted, guaranteed, with 2 pages."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(0)
    _running(kv, "o", 7, arrival=0)
    _running(kv, "z1", 2, arrival=2)
    _running(kv, "z2", 1, arrival=3)
    for rid in ("z2", "z1"):
        short = _step(kv, rid, 1).reason.pages_short
        assert _preempt(kv, rid, short, droppable=lambda rid: False).waits
        _runner(kv).readmit(rid, NODE, queued=False)
    _request(kv, "p", 2, arrival=1, rebuildable=False)
    assert not _ready(kv, "p")
    assert _make_room(kv, droppable=lambda rid: True).to_drop == ["z2", "z1"]

    _drop(kv, "z1")
    assert "p" in kv._guaranteed and not _admitted(kv, "p")
    _drop(kv, "z2")

    assert "p" not in kv._allotments, "p was admitted guaranteed with less than its footprint"
    for rid in ("o", "z1", "z2"):
        _remove(kv, rid)
    assert _admitted(kv, "p") and kv._allotments["p"].pages == 6


class _StubHostRefusingB(host._HostPool):
    """Refuses stream ``b``, as a host whose room runs out partway through a move does."""

    def place(self, rid, label, *args, **kwargs):
        return None if label == "b" else super().place(rid, label, *args, **kwargs)


def _moved_in_part(kv) -> None:
    """o runs short, and v, whose stream ``b`` the host refuses, moves its other stream there for o."""
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


def test_a_victim_the_host_took_only_part_of_keeps_its_allotment():
    """A victim whose second stream the host refused was still vacated, so it asked again for its footprint on top
    of the pages that stream held, and never came back even alone."""
    kv = _pool()
    _moved_in_part(kv)

    _remove(kv, "o")

    assert _ready(kv, "v"), "v, still holding b's pages, never came back even alone"


def test_a_victim_the_host_took_only_part_of_sets_what_it_freed_aside_for_the_request_it_moved_for():
    """`offload` settled such a victim with nothing freed, so the head of the queue could take the room its move
    made before the request it moved for grew into it."""
    kv = _pool()
    _moved_in_part(kv)
    _request(kv, "p", 3, arrival=2)

    _admitted(kv, "p")

    assert _step(kv, "o", 1).ok, "p took the room v's move made for o, and o ran short again"


def test_a_request_short_of_its_guarantee_asks_for_every_page_it_grows_into():
    """The host has no room to grow what it set aside for r, so r runs short of its guarantee: 4 pages past its
    allotment, of which the room covers 1. Counted as short of its grant's one page, r had s moved to the host for a
    page, and ran short of its guarantee again on its next step."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(3)
    _running(kv, "r", 2, arrival=0, rebuildable=False)
    _running(kv, "s", 1, arrival=1, rebuildable=False)
    _running(kv, "x", 6, arrival=2)
    outcome = _step(kv, "r", PAGE_SIZE)
    kv._streams["r"]["main"].step_in_flight = True

    victims = _preempt(kv, "r", outcome.reason.pages_short, droppable=lambda rid: False)

    assert victims == Victims([], [], [], waits=4), "r asked for its grant's page, not the 4 pages it grows into"
