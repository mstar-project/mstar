"""On a KV shortage in a pool that decides admission, the worker moves newer requests to the host when the host
takes them whole, drops the others if they can be rebuilt from their kept ids, and holds nothing. Everywhere else it
offloads the least recently active request and holds it for 50 ms, or, when nothing can be offloaded, holds the
request that ran short.
"""

from __future__ import annotations

import logging
import sys
from array import array
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import test_kv_host_index as host
import torch
from test_kv_admission import _remove, _runner
from test_kv_preemption import (  # noqa: F401
    PAGE_SIZE,
    _admitted,
    _ctx,
    _drop,
    _offload,
    _pool,
    _preempt,
    _queue,
    _ready,
    _request,
    _returns,
    _running,
    _short,
    _step,
    _stub_transfer,
)
from test_kv_prefix_lease import WALK, _manager

from mstar.engine.engine import Engine, PagedKVCache
from mstar.engine.resources import Residency
from mstar.engine.resources.admission import Admission
from mstar.engine.resources.kv.config import KVStep
from mstar.engine.resources.step import ASKS, Segment
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.model.rebuild import RebuildPolicy
from mstar.model.submodule_base import ARNodeInputs
from mstar.observability.recorders import EdgeTransfers
from mstar.testing_utils.worker import make_scheduler, make_worker, preempting
from mstar.worker.host_pool import HostCopy, HostSlice
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mstar.worker.rebuild import KeptIds, Prefilling, Rebuilds, policies_run_here
from mstar.worker.worker_metrics import PreemptionCounts

NODE = "LLM"
POLICY = RebuildPolicy(walk="prefill", ids_input="text_inputs", sample_output="new_token", loop="decode")


def _graphs(node_name: str = NODE) -> dict:
    """The node prefills in ``prefill`` and grows its KV in loop ``decode`` of walk ``decode``."""
    def node():
        return GraphNode(name=node_name, input_names={"text_inputs"}, outputs=[GraphEdge("out", "new_token")])
    return {"prefill": node(), "decode": Loop(section=node(), max_iters=8, outputs=[], name="decode")}


class _Engine(SimpleNamespace):
    """An engine stub the scheduler can collect in a set, as it does real engines."""
    __hash__ = object.__hash__


def _kept(tokens: int) -> KeptIds:
    """A request's prompt of ``tokens`` ids on the host, and one sample."""
    view = torch.arange(tokens, dtype=torch.long)
    return KeptIds(HostCopy(HostSlice(0, view.nbytes), view, None), array("i", [0]))


def _worker(kv):
    """The real handlers over one pool, with the engine's drop going straight to the pool and its runner."""
    def drop_request(node, rid, tokens):
        freed = kv.drop(rid, tokens)
        _runner(kv).readmit(rid, node)
        return freed

    engine = _Engine(
        drop_refusal=lambda node, rid, tokens: kv.check_drop(rid, tokens), drop_request=drop_request,
        offload_request=lambda node, rid: _offload(kv, rid),
        readmit_request=lambda node, rid: _runner(kv).readmit(rid, node),
        newer_than=lambda rid: _runner(kv).newer_than(rid),
        refusals=lambda: _runner(kv).refusals(),
        paged_kv_caches=lambda: [PagedKVCache("kv", frozenset({NODE}), kv)],
        admit_request=lambda node, rid, info: _runner(kv).admit_request(rid, node, info.graph_walk),
        admit_waiting=lambda published_of: _runner(kv).admit_waiting(published_of),
    )
    w = make_worker(
        worker_id="w",
        rebuilds=Rebuilds({NODE: POLICY}, _graphs()),
        graph_runtime=SimpleNamespace(push_back_node=lambda node, rids, wgs: None, get_rid_string=str),
        scheduler=SimpleNamespace(
            backlog={}, return_backlogged=lambda rid: None, could_refill=lambda *args: False,
            last_in_line=lambda node, walk, rids: rids[-1], prefilling=lambda node, rid, walk: None,
        ),
        request_state=SimpleNamespace(
            get_partition_for_node=lambda node: "p",
            get_fwd_info=lambda rid, p: SimpleNamespace(
                graph_walk="decode", dynamic_loop_iter_counts={"decode": 0}, resource_publish_info=None,
            ),
        ),
        engine_manager=SimpleNamespace(get_engine=lambda node: engine, engine=engine),
    )
    w._rebuilds._kept[NODE] = {rid: _kept(5 * PAGE_SIZE) for rid in "rv"}
    w.tried, w.failed = [], {}
    w.pipeline._fail_requests = w.failed.update
    w.admission._evict = lambda node, batch_ids, arrivals, last_in_line, affected_resources=None, inside_only=False: (
        w.tried.append(node)
    )
    preempting(w)
    return w


def _held(w) -> list:
    return list(w.admission.held_until)


def _refused(batch, reason, failed_resource="kv"):
    """The step of ``batch``'s rows that ``reason`` refused, with every row prepared."""
    return SimpleNamespace(
        node_name=batch.node_name, admit_error=reason, failed_resource=failed_resource,
        request_ids=list(batch.request_to_worker_graph), cached_rids=set(),
        step_context=SimpleNamespace(graph_walk=batch.graph_walk),
    )


def _preempting(case: str = "deciding kv pool", in_flight: tuple[str, ...] = ()):
    """Call `_handle_admit_failure` once r runs short beside newer v, both 5 pages on the 10-page pool."""
    kv = _pool()
    _running(kv, "r", 5, arrival=None if case == "no arrival" else 1)
    _running(kv, "v", 5, arrival=2)
    kv._decides = case != "a pool that doesn't decide"
    reason = _step(kv, "r", 1).reason
    w = _worker(kv)
    w.pipeline.in_flight_rids.update(in_flight)
    if case == "a node that keeps no ids":
        w._rebuilds = Rebuilds({}, {})
        preempting(w)
    elif case == "a node where no request keeps ids":
        w._rebuilds._kept[NODE] = {}
    elif case == "nothing to drop":
        w.engine_manager.get_engine(NODE).drop_refusal = lambda node, rid, tokens: "refused"
    rows = dict.fromkeys(("r", *in_flight), 0)
    batch = ScheduledBatch(node_name=NODE, graph_walk="decode", request_to_worker_graph=rows)
    resource = "recurrent" if case == "another resource" else "kv"
    w.pipeline._handle_admit_failure(batch, _refused(batch, reason, resource))
    return kv, w


@pytest.mark.parametrize("case", [
    "deciding kv pool", "another resource", "a pool that doesn't decide", "no arrival", "a node that keeps no ids",
    "a node where no request keeps ids",
])
def test_only_a_deciding_kv_pool_preempts(case):
    kv, w = _preempting(case)

    if case == "deciding kv pool":
        assert not w.tried, "a pool that decides chose its victim by recency"
        assert kv.residency("v") is Residency.DROPPED and not _held(w), "r's shortage didn't drop v"
        assert _step(kv, "r", 1).ok, "r didn't get the page v's drop freed"
        assert w.preemption.counts[NODE] == PreemptionCounts(victims=1, drops=1), "the drop went uncounted"
    elif case == "a node where no request keeps ids":
        # the node's policy decides how it preempts, whichever requests keep ids now
        assert not w.tried and not _held(w) and _returns(kv) == ["r"] and kv.residency("v") is Residency.RESIDENT, (
            "a node that rebuilds requests switched to the offload and the hold while none kept ids"
        )
    else:
        resident = kv.residency("v") is Residency.RESIDENT
        assert w.tried and _held(w) == ["r"] and resident, f"{case} lost the offload and the hold"


@pytest.mark.parametrize("my_walks, rebuilds", [
    pytest.param({(NODE, "prefill"), (NODE, "decode")}, True, id="a node whose walks all run here"),
    pytest.param({(NODE, "decode")}, False, id="a node whose prefill walk runs on another worker"),
])
def test_a_policy_runs_here_only_where_the_worker_runs_its_prefill_and_loop_walks(my_walks, rebuilds):
    assert bool(policies_run_here({NODE: POLICY}, _graphs(), my_walks)) == rebuilds


def test_a_victim_the_host_takes_whole_goes_there_and_its_pages_to_the_request_that_ran_short():
    kv = _pool()
    kv._cpu_pool = host._HostPool(16)
    _running(kv, "r", 5, arrival=1)
    _running(kv, "v", 5, arrival=2)
    reason = _step(kv, "r", 1).reason
    w = _worker(kv)
    batch = ScheduledBatch(node_name=NODE, graph_walk="decode", request_to_worker_graph={"r": 0})

    w.pipeline._handle_admit_failure(batch, _refused(batch, reason))

    assert kv.residency("v") is Residency.ON_HOST and not _held(w), "v wasn't moved to the host"
    assert not kv.can_reload("v"), "v could reload into the page r ran short of"
    assert _step(kv, "r", 1).ok


def test_a_request_that_can_only_wait_waits_and_is_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="mstar.worker.admission"):
        kv, w = _preempting("nothing to drop")

    assert _returns(kv) == ["r"] and "r" in kv._allotments and not _held(w)
    assert "r waits for" in caplog.text, "a request that can only wait left no line"


def test_a_victim_in_the_refused_step_drops_in_the_same_call():
    kv, w = _preempting(in_flight=("r", "v"))

    dropped = kv.residency("v") is Residency.DROPPED
    assert dropped and _step(kv, "r", 1).ok, "v waited for a step that never ran to clear"


def test_a_newer_request_that_wrote_nothing_gives_back_its_allotment_before_an_older_one_drops():
    """Without a host pool, a newer request admitted but not yet prefilled holds an allotment and no pages. It
    keeps no ids, so it can't be dropped, and the older decode dropped itself for a page the newer one could give
    back by asking again."""
    kv = _pool()
    _running(kv, "r", 5, arrival=1)
    _request(kv, "n", 5, arrival=2)
    assert _ready(kv, "n") and kv._allotments["n"].remaining == 5
    reason = _step(kv, "r", 1).reason
    w = _worker(kv)
    batch = ScheduledBatch(node_name=NODE, graph_walk="decode", request_to_worker_graph={"r": 0})

    w.pipeline._handle_admit_failure(batch, _refused(batch, reason))

    assert kv.residency("r") is Residency.RESIDENT and "n" not in kv._allotments, (
        "the older decode dropped itself while a newer request that hadn't prefilled kept its allotment")


def test_an_older_decode_short_by_a_page_drops_the_newest_request_mid_prefill_not_itself():
    """A newer request between two chunks of its prompt keeps no ids from a sample yet. It was refused as a victim,
    so the older decode dropped itself and paid a rebuild of its whole history, where the newer one reruns only
    the chunks it ran."""
    kv = _pool()
    _running(kv, "r", 5, arrival=1)
    _running(kv, "v", 5, arrival=2)
    reason = _step(kv, "r", 1).reason
    w = _worker(kv)
    del w._rebuilds._kept[NODE]["v"]
    w.scheduler.prefilling = lambda node, rid, walk: Prefilling(POLICY.walk, 5 * PAGE_SIZE, 8 * PAGE_SIZE) if (
        rid == "v") else None
    batch = ScheduledBatch(node_name=NODE, graph_walk="decode", request_to_worker_graph={"r": 0})

    w.pipeline._handle_admit_failure(batch, _refused(batch, reason))

    assert kv.residency("v") is Residency.DROPPED and kv.residency("r") is Residency.RESIDENT, (
        "the older decode dropped itself while the newest request kept its pages")
    assert _step(kv, "r", 1).ok


@pytest.mark.parametrize("why", ["kept ids", "mid-prefill"])
def test_a_victim_whose_rebuild_wouldnt_write_what_its_kv_holds_isnt_chosen(why):
    """v's KV holds 5 pages, but its kept ids, or the chunks its prompt ran, rebuild a token less. Chosen anyway,
    v either fails its drop in the move or comes back one token short."""
    kv = _pool()
    _running(kv, "r", 5, arrival=1)
    _running(kv, "v", 5, arrival=2)
    reason = _step(kv, "r", 1).reason
    w = _worker(kv)
    if why == "kept ids":
        w._rebuilds._kept[NODE]["v"] = _kept(5 * PAGE_SIZE - 1)
    else:
        del w._rebuilds._kept[NODE]["v"]
        w.scheduler.prefilling = lambda node, rid, walk: Prefilling(POLICY.walk, 5 * PAGE_SIZE - 1, 8 * PAGE_SIZE) if (
            rid == "v") else None
    batch = ScheduledBatch(node_name=NODE, graph_walk="decode", request_to_worker_graph={"r": 0})

    w.pipeline._handle_admit_failure(batch, _refused(batch, reason))

    assert kv.residency("v") is Residency.RESIDENT, "v was chosen though its rebuild doesn't write what its KV holds"
    assert kv.residency("r") is Residency.RESIDENT and _returns(kv) == ["r"], "r didn't wait for room beside v"


def test_an_older_request_that_asks_after_a_newer_one_parks_moves_it():
    """z ran short with nothing newer to move and, the host too full to take it, waits holding its pages. p, older,
    asks only afterwards, as a request held in an encoder does, and doesn't fit. z runs no step that could run short
    again, so nothing moved it and both waited for good."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(8)
    _running(kv, "x", 3, arrival=0)
    assert kv.offload("x") == 3
    _running(kv, "o", 4, arrival=1)
    _running(kv, "z", 6, arrival=3)
    assert _preempt(kv, "z", _short(kv, "z"), droppable=lambda rid: False).waits
    _runner(kv).readmit("z", NODE)
    _request(kv, "p", 5, arrival=2)
    assert not _ready(kv, "p") and not _admitted(kv, "z")
    _remove(kv, "x")
    w = _worker(kv)

    w.preemption.make_room()

    assert kv.is_offloaded("z") and _ready(kv, "p"), "z, newer and waiting, kept the pages p needs"


def test_make_room_helps_the_oldest_refused_request_moves_can_let_in():
    """o, the oldest, waits for 9 pages, which y and p, the newer requests that wait holding pages, can't free
    together. Looking only at the oldest refusal, nothing moved for anyone, though dropping p lets y in."""
    kv = _pool()
    _running(kv, "x", 4, arrival=0)
    _running(kv, "y", 3, arrival=2)
    _running(kv, "p", 3, arrival=3)
    kv._streams["p"]["main"].step_in_flight = True
    assert _preempt(kv, "y", _short(kv, "y"), droppable=lambda rid: False).waits
    kv._streams["p"]["main"].step_in_flight = False
    assert _preempt(kv, "p", _short(kv, "p"), droppable=lambda rid: False).waits
    for rid in ("y", "p"):
        _runner(kv).readmit(rid, NODE)
    _request(kv, "o", 9, arrival=1)
    assert not _ready(kv, "o") and not _admitted(kv, "y")
    w = _worker(kv)
    w._rebuilds._kept[NODE]["p"] = _kept(3 * PAGE_SIZE)

    w.preemption.make_room()

    assert kv.residency("p") is Residency.DROPPED and _admitted(kv, "y"), "y waited behind o, which no move lets in"


def test_make_room_asks_nothing_while_no_request_waits_holding_its_allotment():
    """make_room runs every pass. With no request waiting holding its allotment there is nothing it could move."""
    kv = _pool()
    _running(kv, "r", 5, arrival=1)
    _request(kv, "c", 6, arrival=2)
    assert not _ready(kv, "c")
    w = _worker(kv)
    read = []
    w.engine_manager.engine.refusals = lambda: read.append(True) or []
    # a locked section raises
    kv._lock = None

    w.preemption.make_room()

    assert not read, "make_room read the refusals with no request it could move"


def test_make_room_moves_nothing_once_the_oldest_refused_request_fits():
    """x and y wait to return from their drops, and t's removal makes room for x. The walk passed over x, which fit
    and so needed no move, and dropped p for y, which then took the room x fit in."""
    kv = _manager(max_num_pages=21)
    for arrival, (rid, pages) in enumerate((("s", 10), ("t", 3), ("x", 2), ("y", 3), ("p", 2))):
        _running(kv, rid, pages, arrival)
    kv._streams["p"]["main"].step_in_flight = True
    short = _step(kv, "s", 4 * PAGE_SIZE).reason.pages_short
    assert _preempt(kv, "s", short, droppable=lambda rid: True).to_drop == ["y", "x"]
    kv._streams["p"]["main"].step_in_flight = False
    for rid in ("y", "x"):
        _drop(kv, rid)
    assert _step(kv, "s", 4 * PAGE_SIZE).ok
    assert _preempt(kv, "p", _step(kv, "p", PAGE_SIZE + 1).reason.pages_short, droppable=lambda rid: True).waits
    _runner(kv).readmit("p", NODE)
    assert not _admitted(kv, "x") and not _admitted(kv, "y")
    _remove(kv, "t")
    w = _worker(kv)
    w._rebuilds._kept[NODE]["p"] = _kept(2 * PAGE_SIZE)

    w.preemption.make_room()

    assert kv.residency("p") is Residency.RESIDENT, "p was dropped for y, though x, older, fit"
    assert _admitted(kv, "x") and not _admitted(kv, "y"), "y took the room x fit in"


def test_make_room_walks_again_only_once_a_refusal_or_a_request_it_could_move_changes():
    """o waits to return into pages r holds, and r, waiting for its shortage, can't be dropped until its ids cover its
    KV. Each pass walked every refusal again, about 2 ms with 32 requests waiting. Walking only once a refusal's room
    grew would miss r's ids reaching the host, which grows nothing, and o would wait for good."""
    kv = _pool()
    _running(kv, "x", 2, arrival=0)
    _running(kv, "o", 3, arrival=1)
    _running(kv, "r", 5, arrival=2)
    kv._streams["r"]["main"].step_in_flight = True
    assert _preempt(kv, "x", _short(kv, "x"), droppable=lambda rid: True).to_drop == ["o"]
    kv._streams["r"]["main"].step_in_flight = False
    _drop(kv, "o")
    assert _step(kv, "r", 2 * PAGE_SIZE).ok
    assert _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: True).waits
    _runner(kv).readmit("r", NODE)
    assert not _admitted(kv, "o")
    w = _worker(kv)
    walked, make_room = [], kv.make_room
    kv.make_room = lambda *args: walked.append(args[0].rid) or make_room(*args)
    w.preemption.make_room()
    assert walked == ["o", "r"] and kv.residency("r") is Residency.RESIDENT
    walked.clear()

    w.preemption.make_room()
    assert not walked, "make_room walked the refusals again with nothing changed"
    w._rebuilds._kept[NODE]["r"] = _kept(7 * PAGE_SIZE)
    w.preemption.make_room()

    assert kv.residency("r") is Residency.DROPPED and _admitted(kv, "o"), "o waited after r could be dropped"


def test_make_room_walks_again_once_a_running_request_lets_go_of_a_page_a_waiting_one_shares():
    """o waits to return, and s, waiting for its shortage, shares its prompt's pages with h. Moving s frees too little
    while h holds them. h's removal leaves the room as it was, since s still holds every page h held, but moving s
    now frees them. The walk read only the refusals, the requests it could move and the host's room, none of which
    changed, so o waited until something else did."""
    kv = _pool()
    shared = list(range(3 * PAGE_SIZE))
    _running(kv, "x", 2, arrival=0)
    _running(kv, "o", 3, arrival=1)
    _request(kv, "h", 3, arrival=2, keyed=shared)
    assert _ready(kv, "h") and _step(kv, "h", 3 * PAGE_SIZE).ok
    _request(kv, "s", 4, arrival=3, keyed=shared + list(range(100, 100 + PAGE_SIZE)))
    assert _ready(kv, "s")
    inputs = ARNodeInputs(input_seq_len=4 * PAGE_SIZE)
    kv.apply_cached_prefix("s", NODE, WALK, inputs, kv.resolve_cached_prefix("s", NODE, WALK, inputs))
    assert _step(kv, "s", PAGE_SIZE).ok
    for rid in ("h", "s"):
        kv._streams[rid]["main"].step_in_flight = True
    short = _step(kv, "x", 2 * PAGE_SIZE).reason.pages_short
    assert _preempt(kv, "x", short, droppable=lambda rid: True).to_drop == ["o"]
    for rid in ("h", "s"):
        kv._streams[rid]["main"].step_in_flight = False
    _drop(kv, "o")
    assert _step(kv, "x", 2 * PAGE_SIZE).ok
    assert _preempt(kv, "s", _step(kv, "s", 3 * PAGE_SIZE).reason.pages_short, droppable=lambda rid: True).waits
    _runner(kv).readmit("s", NODE)
    assert not _admitted(kv, "o")
    w = _worker(kv)
    w._rebuilds._kept[NODE]["s"] = _kept(4 * PAGE_SIZE)
    w.preemption.make_room()
    assert kv.residency("s") is Residency.RESIDENT and not _admitted(kv, "o")
    room = kv._room()

    _remove(kv, "h")
    assert kv._room() == room
    w.preemption.make_room()

    assert kv.residency("s") is Residency.DROPPED, "o waited though moving s would now let it in"


class _Pool:
    """A pool that asks for every request and fits it."""

    def ask(self, rid, node_name, graph_walk):
        return ASKS

    def reserve(self, rid, node_name, graph_walk, published):
        return None


def test_a_request_admitted_on_a_node_it_never_parked_on_is_unparked():
    """Bagel's CFG nodes share one pool, so a move asks each of them again for the request, and its admission starts
    all of them. The request parked only on the node whose readiness check ran, and unparking it raised a KeyError
    on the others."""
    admission = Admission({"kv": _Pool()}, ["kv"])
    nodes = ("LLM", "LLM_cfg")
    assert all(admission.admit("q", node, "decode", None, ["kv"]) is None for node in nodes)
    for node in nodes:
        admission.readmit("q", node, ["kv"])
    engine = _Engine(admit_waiting=admission.admit_waiting)
    request_state = SimpleNamespace(
        get_partition_for_node=lambda node: "p",
        get_fwd_info=lambda rid, p: SimpleNamespace(resource_publish_info=None),
    )
    parking = make_scheduler(SimpleNamespace(get_engine=lambda node: engine), request_state=request_state).admission
    parking.parked = {"LLM": {"q"}}

    parking.let_rows_back()

    assert parking.parked == {"LLM": set()}, "q stayed parked on the node its check waited on"


def test_a_dropped_request_that_waits_for_room_is_unparked_and_rebuilt():
    """A dropped request asks again at its drop, so its next readiness check parks it in the queue. Once the pool
    admits it, the check starts its rebuild, and the rebuild's landing lets it run. A second path that readmitted
    dropped requests left the admitted request parked, and it never ran again."""
    kv, w = _preempting()
    assert kv.residency("v") is Residency.DROPPED and _step(kv, "r", 1).ok
    engine = w.engine_manager.get_engine(NODE)
    engine._runner, engine.residency = _runner(kv), lambda node, rid: kv.residency(rid)
    engine.prepare_rebuild = lambda node, rid, rebuild: None
    engine.check_ready = lambda node, rid, info, allow_reload=True: Engine.check_ready(engine, node, rid, info)
    admission = w.admission
    info = w.request_state.get_fwd_info("v", "p")

    assert not admission.check_ready(NODE, "v", info) and "v" in admission.parked[NODE], "v wasn't parked"
    _remove(kv, "r")
    admission.let_rows_back()
    assert not admission.check_ready(NODE, "v", info) and w._rebuilds.rebuilding("v", NODE), "v was never rebuilt"
    w._rebuilds.land_chunk("v", NODE, None)

    assert "v" not in admission.parked[NODE], "v's rebuild landed, but v stays parked and never runs again"


def test_a_request_on_the_host_without_a_step_of_its_own_holds_back_no_newer_request():
    """A request on the host may have no next step left on its node. It is admitted from the queue once it fits,
    so a newer request waits for that and not for the request's REMOVE."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(16)
    _running(kv, "r", 5, arrival=1)
    _running(kv, "v", 5, arrival=2)
    reason = _step(kv, "r", 1).reason
    w = _worker(kv)
    batch = ScheduledBatch(node_name=NODE, graph_walk="decode", request_to_worker_graph={"r": 0})
    w.pipeline._handle_admit_failure(batch, _refused(batch, reason))
    _remove(kv, "r")
    _request(kv, "x", 2, arrival=3)

    assert _ready(kv, "x"), "x waited behind v on the host"
    assert kv.is_offloaded("v") and "v" in kv._allotments, "v wasn't admitted ahead of x"


def test_a_request_back_from_the_host_that_gives_back_its_allotment_waits_as_a_return():
    """v, admitted to return from the host but not yet reloaded, holds an allotment and no device page, so r's
    shortage has it give the allotment back. It then waited in the queue at its arrival, holding back every newer
    prompt while it waited for its reload and a page, where every other request on the host waits outside it."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(64)
    _running(kv, "o", 4, arrival=0)
    _running(kv, "r", 3, arrival=1)
    _running(kv, "v", 3, arrival=2)
    assert _preempt(kv, "r", _short(kv, "r"), droppable=lambda rid: False).to_host == ["v"]
    _offload(kv, "v")
    _remove(kv, "o")
    assert _admitted(kv, "v") and kv.is_offloaded("v")
    reason = _step(kv, "r", 4 * PAGE_SIZE).reason
    w = _worker(kv)
    batch = ScheduledBatch(node_name=NODE, graph_walk="decode", request_to_worker_graph={"r": 0})

    w.pipeline._handle_admit_failure(batch, _refused(batch, reason))

    assert "v" not in kv._allotments, "v kept the allotment r's shortage asked it to give back"
    assert _returns(kv) == ["v"] and not _queue(kv), "v waits in the queue, though it is on the host"


def test_a_backlogged_victim_goes_back_to_the_runtime():
    runtime = SimpleNamespace(pushed=[], in_flight={})
    runtime.push_back_node = lambda node, rids, wgs: runtime.pushed.extend(rids)
    runtime.set_in_flight = lambda node, rids, wgs, flag: runtime.in_flight.update(dict.fromkeys(rids, flag))
    scheduler = MicroScheduler.__new__(MicroScheduler)
    scheduler.runtime = runtime
    scheduler.backlog = {(NODE, "decode"): ScheduledBatch(NODE, "decode", request_to_worker_graph={"v": 0, "x": 0})}

    scheduler.return_backlogged("v")

    assert runtime.pushed == ["v"] and runtime.in_flight == {"v": False}, "v's row never runs again"
    assert list(scheduler.backlog[(NODE, "decode")].request_to_worker_graph) == ["x"]


def test_preemptions_after_metrics_start_are_reported():
    """The worker hands the metrics its counts before anything is preempted."""
    w = _worker(_pool())
    w.layout, w.scheduler.pending_rows = replace(w.layout, parallel_leader_nodes=frozenset({NODE})), dict
    w.engine_manager.engine = SimpleNamespace(paged_kv_caches=list)
    w.communicator = SimpleNamespace(send=lambda peer, message: None)
    w.tensor_manager = SimpleNamespace(enable_transfer_metrics=lambda: EdgeTransfers("SHM"))
    metrics = w._start_metrics(3600.0)
    metrics.start_counting()

    w.preemption.counts[NODE].victims += 1

    (snapshot,) = [s for s in metrics.snapshot() if s.name == "mstar_kv_preemptions_total"]
    reported = dict(zip(map(tuple, snapshot.label_values), snapshot.values, strict=True))
    assert reported == {(NODE,): 1}, "the preemption went unreported"


def test_requests_short_of_pages_all_finish_and_each_goes_to_the_host_at_most_twice():
    """Six requests decode 4 pages each on a 10-page pool with a host pool, one step a pass. Moving the least
    recently active request to the host, and reloading it as soon as it fit, sent one request back and forth every
    pass, and none finished."""
    kv = _pool()
    kv._cpu_pool = host._HostPool(64)
    w = _worker(kv)
    # the real offload of the least recently active request, which the worker runs when `_preempt` returns False
    w._last_active = {}
    engine = w.engine_manager.get_engine(NODE)
    engine.evictable, engine.is_offloaded = lambda node: True, lambda node, rid: kv.is_offloaded(rid)
    engine.reclaimable = lambda node, rid, affected_resources=None: kv.reclaimable(rid)
    del w.admission._evict
    rids = [f"r{i}" for i in range(6)]
    for arrival, rid in enumerate(rids):
        _request(kv, rid, 2, arrival)
    moves, tokens, offload = Counter(), Counter(), kv.offload
    kv.offload = lambda rid: moves.update([rid]) or offload(rid)
    for i in range(300):
        ready = [rid for rid in rids if rid in kv._streams and _ready(kv, rid)]
        if not ready:
            continue
        ctx = _ctx(*ready)
        step = KVStep(tuple(Segment(rid, "main", 1 if tokens[rid] else 2 * PAGE_SIZE) for rid in ready))
        outcome = kv.admit(step, ctx)
        if not outcome.ok:
            kv.rollback_admit(step, ctx)
            batch = ScheduledBatch(node_name=NODE, graph_walk="decode", request_to_worker_graph=dict.fromkeys(ready, 0))
            w.pipeline._handle_admit_failure(batch, _refused(batch, outcome.reason))
            continue
        kv.plan(step, ctx)
        kv.commit(step, ctx)
        for rid in ready:
            w._last_active[(rid, NODE)] = float(i)
            tokens[rid] += 1
            if tokens[rid] == 3 * PAGE_SIZE:
                _remove(kv, rid)

    assert not kv._streams, "a request never finished"
    assert max(moves.values()) <= 2, f"a request went to the host more than twice: {dict(moves)}"
