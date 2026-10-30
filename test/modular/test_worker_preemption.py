"""On a KV shortage in a pool that decides admission, the worker moves newer requests to the host when the host
takes them whole, drops the others if they can be rebuilt from their kept ids, and holds nothing. Everywhere else it
offloads the least recently active request and holds it for 50 ms, or, when nothing can be offloaded, holds the
request that ran short.
"""

from __future__ import annotations

import logging
import sys
from array import array
from collections import Counter, defaultdict
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
import test_kv_host_index as host
import torch
from test_kv_admission import _remove, _runner
from test_kv_preemption import (  # noqa: F401
    PAGE_SIZE,
    _ctx,
    _drop,
    _offload,
    _pool,
    _ready,
    _request,
    _running,
    _step,
    _stub_transfer,
)

from mstar.engine.engine import PagedKVCache
from mstar.engine.resources.kv.config import KVStep
from mstar.engine.resources.step import Segment
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.model.rebuild import RebuildPolicy
from mstar.observability.recorders import EdgeTransfers
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mstar.worker.rebuild import KeptIds, Rebuilds, policies_run_here
from mstar.worker.worker import Worker
from mstar.worker.worker_metrics import PreemptionCounts

NODE = "LLM"
POLICY = RebuildPolicy(walk="prefill", ids_input="text_inputs", sample_output="new_token", loop="decode")


def _graphs(node_name: str = NODE) -> dict:
    """The node prefills in ``prefill`` and grows its KV in loop ``decode`` of walk ``decode``."""
    def node():
        return GraphNode(name=node_name, input_names={"text_inputs"}, outputs=[GraphEdge("out", "new_token")])
    return {"prefill": node(), "decode": Loop(section=node(), max_iters=8, outputs=[], name="decode")}


def _worker(kv):
    """The real handlers over one pool, with the engine's drop going straight to the pool and its runner."""
    engine = SimpleNamespace(
        drop_refusal=lambda node, rid: kv.check_drop(rid), drop_request=lambda node, rid: _drop(kv, rid),
        offload_request=lambda node, rid: _offload(kv, rid),
        admit_request=lambda node, rid, info: _runner(kv).admit_request(rid, node, info.graph_walk),
    )
    w = Worker.__new__(Worker)
    w.worker_id, w.parallel_nodes, w.parallel_leader_nodes = "w", set(), set()
    w._kv_caches, w._kv_keys = {"kv": PagedKVCache("kv", frozenset({NODE}), kv)}, {NODE: ["kv"]}
    w._rebuilds = Rebuilds({NODE: POLICY}, _graphs())
    w._rebuilds._kept[NODE] = {rid: KeptIds(torch.arange(5 * PAGE_SIZE), array("i", [0])) for rid in "rv"}
    w._in_flight_rids, w._victims, w._dropped, w._hold_logged = set(), {}, {}, {}
    w._preempted, w._preemptions, w._drop_logged, w._rid_str = {}, defaultdict(PreemptionCounts), {}, str
    w.held, w.tried = [], []
    w.scheduler = SimpleNamespace(
        backlog={}, rebuilds={}, return_backlogged=lambda rid: None, hold_requests=w.held.extend,
        could_refill=lambda *args: False, cap_batch=lambda *args: None, limit_batch=lambda *args: None,
    )
    for name in ("queue_rebuild", "take_rebuilds", "rebuild_queued"):  # the real rebuild queue
        setattr(w.scheduler, name, getattr(MicroScheduler, name).__get__(w.scheduler))
    w.request_state = SimpleNamespace(
        get_partition_for_node=lambda node: "p",
        get_fwd_info=lambda rid, p: SimpleNamespace(graph_walk="decode", dynamic_loop_iter_counts={"decode": 0}),
    )
    w.engine_manager = SimpleNamespace(get_engine=lambda node: engine)
    w._graph_runtime = SimpleNamespace(push_back_node=lambda node, rids, wgs: None)
    w._apply_removes_whose_step_landed = lambda: None
    w._apply_pending_removes_safe_to_drop = lambda rids: None
    w._try_offload_cold_request = lambda node, rids, affected_resources=None, inside_only=False: w.tried.append(node)
    return w


def _preempting(case: str = "deciding kv pool", in_flight: tuple[str, ...] = ()):
    """Call `_handle_allocation_failure` once r runs short beside newer v, both 5 pages on the 10-page pool."""
    kv = _pool()
    _running(kv, "r", 5, arrival=None if case == "no arrival" else 1)
    _running(kv, "v", 5, arrival=2)
    kv._decides = case != "a pool that doesn't decide"
    reason = _step(kv, "r", 1).reason
    w = _worker(kv)
    w._in_flight_rids = set(in_flight)
    if case == "a node that keeps no ids":
        w._rebuilds = Rebuilds({}, {})
    elif case == "a node where no request keeps ids":
        w._rebuilds._kept[NODE] = {}
    elif case == "nothing to drop":
        w.engine_manager.get_engine(NODE).drop_refusal = lambda node, rid: "refused"
    rows = dict.fromkeys(("r", *in_flight), 0)
    batch = ScheduledBatch(node_name=NODE, graph_walk="decode", request_to_worker_graph=rows)
    node_batch = SimpleNamespace(
        node_name=NODE, admit_error=reason, failed_resource="recurrent" if case == "another resource" else "kv",
        step_context=SimpleNamespace(graph_walk="decode"),
    )
    Worker._handle_allocation_failure(w, batch, node_batch)
    return kv, w


@pytest.mark.parametrize("case", [
    "deciding kv pool", "another resource", "a pool that doesn't decide", "no arrival", "a node that keeps no ids",
    "a node where no request keeps ids",
])
def test_only_a_deciding_kv_pool_preempts(case):
    kv, w = _preempting(case)

    if case == "deciding kv pool":
        assert not w.tried, "a pool that decides chose its victim by recency"
        assert w._dropped == {"v": NODE} and not w.held, "r's shortage didn't drop v"
        assert _step(kv, "r", 1).ok, "r didn't get the page v's drop freed"
        assert w._preemptions[NODE] == PreemptionCounts(victims=1, drops=1), "the drop went uncounted"
    elif case == "a node where no request keeps ids":
        # the node's policy decides how it preempts, whichever requests keep ids now
        assert not w.tried and not w.held and kv.waits("r") and kv.is_resident("v"), (
            "a node that rebuilds requests switched to the offload and the hold while none kept ids"
        )
    else:
        assert w.tried and w.held == ["r"] and kv.is_resident("v"), f"{case} lost the offload and the hold"


@pytest.mark.parametrize("my_walks, rebuilds", [
    pytest.param({(NODE, "prefill"), (NODE, "decode")}, True, id="a node whose walks all run here"),
    pytest.param({(NODE, "decode")}, False, id="a node whose prefill walk runs on another worker"),
])
def test_only_a_node_whose_rebuild_runs_here_drops_requests(my_walks, rebuilds):
    assert bool(policies_run_here({NODE: POLICY}, _graphs(), my_walks)) == rebuilds


def test_a_victim_the_host_takes_whole_goes_there_and_its_pages_to_the_request_that_ran_short():
    kv = _pool()
    kv._cpu_pool = host._HostPool(16)
    _running(kv, "r", 5, arrival=1)
    _running(kv, "v", 5, arrival=2)
    reason = _step(kv, "r", 1).reason
    w = _worker(kv)
    batch = ScheduledBatch(node_name=NODE, graph_walk="decode", request_to_worker_graph={"r": 0})
    node_batch = SimpleNamespace(node_name=NODE, admit_error=reason, failed_resource="kv")

    Worker._handle_allocation_failure(w, batch, node_batch)

    assert kv.is_offloaded("v") and not w._dropped and not w.held, "v wasn't moved to the host"
    assert not kv.can_reload("v"), "v could reload into the page r ran short of"
    assert _step(kv, "r", 1).ok


def test_a_request_that_can_only_wait_holds_nothing_and_is_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="mstar.worker.worker"):
        kv, w = _preempting("nothing to drop")

    assert kv.waits("r") and not w.held and not w._dropped
    assert "r waits for room" in caplog.text, "a request that can only wait left no line"


def test_a_victim_in_the_refused_step_drops_once_the_step_clears():
    kv, w = _preempting(in_flight=("r", "v"))
    assert not w._dropped and kv.is_resident("v"), "v dropped while its step was in flight"

    w._in_flight_rids = set()
    Worker._move_victims(w)

    assert w._dropped == {"v": NODE} and _step(kv, "r", 1).ok, "v didn't drop once its step cleared"


def test_a_dropped_request_returns_once_its_history_and_a_page_fit():
    kv, w = _preempting()
    assert _step(kv, "r", 1).ok
    w.rebuilt, w._is_tearing_down = [], lambda rid: False
    w._rebuild = lambda rid, node: w.rebuilt.append((rid, node))

    Worker._readmit_dropped(w)
    assert not w.rebuilt, "v came back into the room r needs to grow"
    _remove(kv, "r")
    Worker._readmit_dropped(w)

    assert w.rebuilt == [("v", NODE)] and kv._allotments["v"].pages == 7, "v didn't return with its history and a page"


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
    node_batch = SimpleNamespace(node_name=NODE, admit_error=reason, failed_resource="kv")
    Worker._handle_allocation_failure(w, batch, node_batch)
    _remove(kv, "r")
    _request(kv, "x", 2, arrival=3)

    assert _ready(kv, "x"), "x waited behind v on the host"
    assert kv.is_offloaded("v") and "v" in kv._allotments, "v wasn't admitted ahead of x"


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
    w.parallel_leader_nodes, w.scheduler.pending_rows = {NODE}, dict
    w.engine_manager.engine = SimpleNamespace(paged_kv_caches=list)
    w.communicator = SimpleNamespace(send=lambda peer, message: None)
    w.tensor_manager = SimpleNamespace(enable_transfer_metrics=lambda: EdgeTransfers("SHM"))
    metrics = w._start_metrics(3600.0)
    metrics.start_counting()

    w._preemptions[NODE].victims += 1

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
    w._last_active, w.eviction_policy = {}, SimpleNamespace(value="lru")
    engine = w.engine_manager.get_engine(NODE)
    engine.evictable, engine.is_offloaded = lambda node: True, lambda node, rid: kv.is_offloaded(rid)
    engine.reclaimable = lambda node, rid, affected_resources=None: kv.reclaimable(rid)
    w._try_offload_cold_request = Worker._try_offload_cold_request.__get__(w)
    w._select_eviction_victim = Worker._select_eviction_victim.__get__(w)
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
            node_batch = SimpleNamespace(
                node_name=NODE, admit_error=outcome.reason, failed_resource="kv",
                step_context=SimpleNamespace(graph_walk="decode"),
            )
            Worker._handle_allocation_failure(w, batch, node_batch)
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
