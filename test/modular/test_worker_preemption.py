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
    _queue,
    _ready,
    _request,
    _running,
    _step,
    _stub_transfer,
)

from mstar.engine.engine import Engine, PagedKVCache
from mstar.engine.resources import Residency
from mstar.engine.resources.kv.config import KVReqConfig, KVStep, PrefixSpan, StreamLayout
from mstar.engine.resources.step import Segment
from mstar.graph.base import GraphEdge, GraphNode, Loop
from mstar.model.base import KVLabel
from mstar.model.rebuild import RebuildPolicy
from mstar.observability.recorders import EdgeTransfers
from mstar.worker.host_pool import HostCopy, HostSlice
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mstar.worker.preemption import Preemption
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


class _Engine(SimpleNamespace):
    """An engine stub the scheduler can collect in a set, as it does real engines."""
    __hash__ = object.__hash__


def _kept(tokens: int) -> KeptIds:
    """A request's prompt of ``tokens`` ids on the host, and one sample."""
    view = torch.arange(tokens, dtype=torch.int32)
    return KeptIds(HostCopy(HostSlice(0, view.nbytes), view, None), torch.long, array("i", [0]))


def _worker(kv):
    """The real handlers over one pool, with the engine's drop going straight to the pool and its runner."""
    engine = _Engine(
        drop_refusal=lambda node, rid: kv.check_drop(rid), drop_request=lambda node, rid: _drop(kv, rid),
        offload_request=lambda node, rid: _offload(kv, rid),
        readmit_request=lambda node, rid: _runner(kv).readmit(rid, node),
        newer_than=lambda rid: _runner(kv).newer_than(rid),
        admit_request=lambda node, rid, info: _runner(kv).admit_request(rid, node, info.graph_walk),
        admit_waiting=lambda published_of: _runner(kv).admit_waiting(published_of),
    )
    w = Worker.__new__(Worker)
    w.worker_id, w.parallel_nodes, w.parallel_leader_nodes = "w", set(), set()
    w._rebuilds = Rebuilds({NODE: POLICY}, _graphs())
    w._rebuilds._kept[NODE] = {rid: _kept(5 * PAGE_SIZE) for rid in "rv"}
    w._in_flight_rids, w._hold_logged, w._rid_str = set(), {}, str
    w.held, w.tried, w.failed = [], [], {}
    w._is_tearing_down, w._fail_requests = lambda rid: False, w.failed.update
    w.scheduler = SimpleNamespace(
        backlog={}, return_backlogged=lambda rid: None, hold_requests=w.held.extend,
        could_refill=lambda *args: False, cap_batch=lambda *args: None, limit_batch=lambda *args: None,
    )
    w.request_state = SimpleNamespace(
        get_partition_for_node=lambda node: "p",
        get_fwd_info=lambda rid, p: SimpleNamespace(
            graph_walk="decode", dynamic_loop_iter_counts={"decode": 0}, resource_publish_info=None,
        ),
    )
    caches = SimpleNamespace(paged_kv_caches=lambda: [PagedKVCache("kv", frozenset({NODE}), kv)])
    w.engine_manager = SimpleNamespace(get_engine=lambda node: engine, engine=caches)
    _preemption(w)
    w._graph_runtime = SimpleNamespace(push_back_node=lambda node, rids, wgs: None)
    w._apply_removes_whose_step_landed = lambda: None
    w._apply_pending_removes_safe_to_drop = lambda rids: None
    w._try_offload_cold_request = lambda node, walk, rids, affected_resources=None, inside_only=False: (
        w.tried.append(node)
    )
    return w


def _preemption(w) -> None:
    """The worker's `Preemption`, over its rebuilds and its own drop."""
    w._preemption = Preemption(
        w.engine_manager, w._rebuilds, w.scheduler, w.request_state, w._log_throttled, w._rid_str,
    )


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
        _preemption(w)
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
        assert kv.residency("v") is Residency.DROPPED and not w.held, "r's shortage didn't drop v"
        assert _step(kv, "r", 1).ok, "r didn't get the page v's drop freed"
        assert w._preemption.counts[NODE] == PreemptionCounts(victims=1, drops=1), "the drop went uncounted"
    elif case == "a node where no request keeps ids":
        # the node's policy decides how it preempts, whichever requests keep ids now
        assert not w.tried and not w.held and _queue(kv) == ["r"] and kv.residency("v") is Residency.RESIDENT, (
            "a node that rebuilds requests switched to the offload and the hold while none kept ids"
        )
    else:
        resident = kv.residency("v") is Residency.RESIDENT
        assert w.tried and w.held == ["r"] and resident, f"{case} lost the offload and the hold"


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
    node_batch = SimpleNamespace(node_name=NODE, admit_error=reason, failed_resource="kv")

    Worker._handle_allocation_failure(w, batch, node_batch)

    assert kv.residency("v") is Residency.ON_HOST and not w.held, "v wasn't moved to the host"
    assert not kv.can_reload("v"), "v could reload into the page r ran short of"
    assert _step(kv, "r", 1).ok


def test_a_request_that_can_only_wait_waits_in_the_queue_and_is_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="mstar.worker.worker"):
        kv, w = _preempting("nothing to drop")

    assert _queue(kv) == ["r"] and "r" in kv._allotments and not w.held
    assert "r waits in the queue" in caplog.text, "a request that can only wait left no line"


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
    node_batch = SimpleNamespace(node_name=NODE, admit_error=reason, failed_resource="kv")

    Worker._handle_allocation_failure(w, batch, node_batch)

    assert kv.residency("r") is Residency.RESIDENT and "n" not in kv._allotments, (
        "the older decode dropped itself while a newer request that hadn't prefilled kept its allotment")


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
    scheduler = SimpleNamespace(parked={}, engine_manager=w.engine_manager, rebuilds=w._rebuilds)
    for name in ("_check_ready", "_unpark", "note_rebuilt"):
        setattr(scheduler, name, getattr(MicroScheduler, name).__get__(scheduler))
    info = w.request_state.get_fwd_info("v", "p")

    assert not scheduler._check_ready(NODE, "v", info) and "v" in scheduler.parked[NODE], "v wasn't parked"
    _remove(kv, "r")
    scheduler._unpark(w.request_state)
    assert not scheduler._check_ready(NODE, "v", info) and w._rebuilds.rebuilding("v", NODE), "v was never rebuilt"
    scheduler.note_rebuilt(NODE, "v")

    assert "v" not in scheduler.parked[NODE], "v's rebuild landed, but v stays parked and never runs again"


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
    # v comes back with its footprint, 9 of the 10 pages
    _request(kv, "x", 1, arrival=3)

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

    w._preemption.counts[NODE].victims += 1

    (snapshot,) = [s for s in metrics.snapshot() if s.name == "mstar_kv_preemptions_total"]
    reported = dict(zip(map(tuple, snapshot.label_values), snapshot.values, strict=True))
    assert reported == {(NODE,): 1}, "the preemption went unreported"


@pytest.mark.parametrize("walks, loops", [
    ({(NODE, "prefill")}, set()), ({(NODE, "prefill"), (NODE, "decode")}, {"decode"}),
], ids=["prefill worker", "worker that runs every walk"])
def test_a_pool_reserves_growth_only_for_the_loops_its_worker_runs(walks, loops):
    kv = _pool()
    model = SimpleNamespace(kv_labels=lambda: {"kv": {"main": KVLabel(growth=(("decode", "decode"),))}})
    _worker(kv)._set_kv_writes(model, walks)
    assert kv.local_loops == loops, "a guaranteed request would reserve growth its worker never runs"


def test_a_node_that_writes_none_of_a_requests_labels_leaves_it_preemptible():
    """Bagel's CFG nodes share the LLM's pool and declare no rebuild policy. A text chat writes only the LLM's
    main label, in the policy's walks, so placing the CFG nodes on the LLM's rank must not change its class."""
    kv = _pool()
    w = _worker(kv)
    w.engine_manager.engine = SimpleNamespace(
        paged_kv_caches=lambda: [PagedKVCache("kv", frozenset({NODE, "LLM_cfg_text"}), kv)],
    )
    model = SimpleNamespace(kv_labels=lambda: {"kv": {"main": KVLabel(growth=(("decode", "decode"),))}})
    w._set_kv_writes(model, {(NODE, "prefill"), (NODE, "decode"), ("LLM_cfg_text", "cfg_text")})
    layout = StreamLayout(spans=(PrefixSpan(4 * PAGE_SIZE, 4 * PAGE_SIZE, "prefill"),), growth=(("decode", 1),))
    configs = {"kv": KVReqConfig(kv_layout={"main": layout}, loop_caps={"decode": 64})}

    w._set_rebuildable(1, configs)

    assert configs["kv"].rebuildable, "a node that writes nothing for the request made it guaranteed"


@pytest.mark.parametrize("unpausable", [False, True])
def test_the_worker_keeps_ids_only_for_a_request_whose_output_can_pause(unpausable):
    kv = _pool()
    w = _worker(kv)
    model = SimpleNamespace(kv_labels=lambda: {"kv": {"main": KVLabel(growth=(("decode", "decode"),))}})
    w._set_kv_writes(model, {(NODE, "prefill"), (NODE, "decode")})
    layout = StreamLayout(spans=(PrefixSpan(4 * PAGE_SIZE, 4 * PAGE_SIZE, "prefill"),), growth=(("decode", 1),))
    configs = {"kv": KVReqConfig(kv_layout={"main": layout}, loop_caps={"decode": 64}, unpausable=unpausable)}

    w._set_rebuildable(1, configs)

    assert (1 in w._rebuilds._droppable[NODE]) is not unpausable, "the conductor's mark didn't reach the kept ids"


@pytest.mark.parametrize("layout", ["two labels", "two spans", "two loops"])
def test_a_pool_whose_layout_one_rebuild_row_cant_write_again_isnt_rebuildable(layout):
    """A drop lays the pool's one label out again as one span of the policy's walk. The second label is written
    in a walk and a loop this worker doesn't run, so only the pool's own layout refuses it."""
    kv = _pool()
    w = _worker(kv)
    model = SimpleNamespace(kv_labels=lambda: {"kv": {"main": KVLabel(growth=(("decode", "decode"),))}})
    w._set_kv_writes(model, {(NODE, "prefill"), (NODE, "decode")})
    span, growth = PrefixSpan(4 * PAGE_SIZE, 4 * PAGE_SIZE, "prefill"), (("decode", 1),)
    layouts = {"main": StreamLayout(spans=(span,), growth=growth)}
    if layout == "two labels":
        layouts["uncond"] = StreamLayout(spans=(PrefixSpan(4, 4, "cfg_text"),), growth=(("cfg_loop", 1),))
    elif layout == "two spans":  # as a prompt prefilled over two rows
        layouts["main"] = StreamLayout(spans=(span, PrefixSpan(4, 4, "prefill")), growth=growth)
    else:
        layouts["main"] = StreamLayout(spans=(span,), growth=(*growth, ("other_loop", 1)))
    configs = {"kv": KVReqConfig(kv_layout=layouts, loop_caps={"decode": 64, "cfg_loop": 4, "other_loop": 4})}

    w._set_rebuildable(1, configs)

    assert not configs["kv"].rebuildable, "a request its rebuild couldn't bring back could be dropped"


def test_an_older_request_drops_itself_rather_than_a_newer_guaranteed_one():
    kv = _pool()
    _running(kv, "r", 5, arrival=1)
    _running(kv, "v", 1, arrival=2, guaranteed=True)
    w = _worker(kv)
    batch = ScheduledBatch(node_name=NODE, graph_walk="decode", request_to_worker_graph={"r": 0})
    node_batch = SimpleNamespace(node_name=NODE, admit_error=_step(kv, "r", 1).reason, failed_resource="kv")

    Worker._handle_allocation_failure(w, batch, node_batch)

    assert (kv.residency("r"), kv.residency("v")) == (Residency.DROPPED, Residency.RESIDENT), (
        "a guaranteed request was dropped for an older page"
    )


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
