"""A KV shortage on a pool that decides admission drops newer requests and holds nothing.

On any other shortage, the worker offloads the least recently active request and holds it for 50 ms.
Without a host pool nothing can be offloaded, so it holds the request that ran short instead. Where
the worker can rebuild a request from its kept ids, it drops newer requests.
"""

from __future__ import annotations

import logging
import sys
from array import array
from collections import defaultdict
from types import SimpleNamespace

sys.path.insert(0, ".")

import pytest
from test_kv_preemption import PAGE_SIZE, _pool, _running, _step, _stub_transfer  # noqa: F401

from mstar.engine.engine import PagedKVCache
from mstar.observability.recorders import EdgeTransfers
from mstar.worker.micro_scheduler import MicroScheduler, ScheduledBatch
from mstar.worker.worker import KeptIds, Worker
from mstar.worker.worker_metrics import PreemptionCounts

NODE = "LLM"


def _worker(kv):
    """The real handlers over one pool, with the engine's drop going straight to the pool."""
    engine = SimpleNamespace(
        drop_refusal=lambda node, rid: kv.check_drop(rid), drop_request=lambda node, rid, kept: kv.drop(rid),
    )
    w = Worker.__new__(Worker)
    w.worker_id, w.parallel_nodes, w.parallel_leader_nodes = "w", set(), set()
    w._kv_caches, w._kv_keys = {"kv": PagedKVCache("kv", frozenset({NODE}), kv)}, {NODE: ["kv"]}
    w._kept_ids = {NODE: {rid: KeptIds(array("i", range(5 * PAGE_SIZE)).tobytes(), array("i", [0])) for rid in "rv"}}
    w._growth_loops = {("decode", NODE): "decode"}
    w._in_flight_rids, w._victims, w._dropped, w._rebuild_rows, w._hold_logged = set(), {}, {}, {}, {}
    w._preempted, w._preemptions, w._drop_logged, w._rid_str = {}, defaultdict(PreemptionCounts), {}, str
    w.held, w.offloaded = [], []
    w.scheduler = SimpleNamespace(backlog={}, return_backlogged=lambda rid: None, hold_requests=w.held.extend)
    w.request_state = SimpleNamespace(
        get_partition_for_node=lambda node: "p",
        get_fwd_info=lambda rid, p: SimpleNamespace(graph_walk="decode", dynamic_loop_iter_counts={"decode": 0}),
    )
    w.engine_manager = SimpleNamespace(get_engine=lambda node: engine)
    w._graph_runtime = SimpleNamespace(push_back_node=lambda node, rids, wgs: None)
    w._apply_removes_whose_step_landed = lambda: None
    w._apply_pending_removes_safe_to_drop = lambda rids: None
    w._try_offload_cold_request = lambda node, rids, affected_resources=None: w.offloaded.append(node)
    return w


def _preempting(case: str = "deciding kv pool", in_flight: tuple[str, ...] = ()):
    """r runs short beside newer v, both 5 pages on the 10-page pool, and `_handle_allocation_failure` runs."""
    kv = _pool()
    _running(kv, "r", 5, arrival=None if case == "no arrival" else 1)
    _running(kv, "v", 5, arrival=2)
    kv._decides = case != "a pool that doesn't decide"
    reason = _step(kv, "r", 1).reason
    w = _worker(kv)
    w._in_flight_rids = set(in_flight)
    if case == "a node that keeps no ids":
        w._kept_ids = {}
    elif case == "a node where no request keeps ids":
        w._kept_ids = {NODE: {}}
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
        assert w._dropped == {"v": NODE} and not w.held and not w.offloaded
        assert _step(kv, "r", 1).ok, "r didn't get the page v's drop freed"
        assert w._preemptions[NODE] == PreemptionCounts(victims=1, drops=1), "the drop went uncounted"
    else:
        assert w.offloaded and w.held == ["r"] and kv.is_resident("v"), f"{case} lost the offload and the hold"


@pytest.mark.parametrize("node, rebuilds", [
    pytest.param("llm", True, id="a node whose walks all run here"),
    pytest.param("decoder", False, id="a node with a second cache, as Whisper's decoder"),
    pytest.param("pd_decode", False, id="a node whose prefill walk runs on another worker"),
])
def test_only_a_node_a_rebuild_can_recompute_drops_requests(node, rebuilds):
    w = Worker.__new__(Worker)
    w._kv_keys = {"llm": ["kv"], "decoder": ["kv", "cross"], "pd_decode": ["kv"]}
    w._walk_inputs = {(walk, name): {"text_inputs"} for walk in ("prefill", "decode") for name in w._kv_keys}
    my_walks = {(name, walk) for walk, name in w._walk_inputs} - {("pd_decode", "prefill")}

    assert w._can_rebuild_on(node, my_walks) == rebuilds


def test_a_request_that_can_only_wait_holds_nothing_and_is_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="mstar.worker.worker"):
        kv, w = _preempting("nothing to drop")

    assert kv.waits("r") and not w.held and not w.offloaded and not w._dropped
    assert "r waits for room" in caplog.text, "a request that can only wait left no line"


def test_a_victim_in_the_refused_step_drops_once_the_step_clears():
    kv, w = _preempting(in_flight=("r", "v"))
    assert not w._dropped and kv.is_resident("v"), "v dropped while its step was in flight"

    w._in_flight_rids = set()
    Worker._drop_victims(w)

    assert w._dropped == {"v": NODE} and _step(kv, "r", 1).ok, "v didn't drop once its step cleared"


def test_a_dropped_request_returns_once_its_history_and_a_page_fit():
    kv, w = _preempting()
    assert _step(kv, "r", 1).ok
    w.rebuilt, w._is_tearing_down = [], lambda rid: False
    w._rebuild = lambda rid, node: w.rebuilt.append((rid, node))

    Worker._readmit_dropped(w)
    assert not w.rebuilt, "v came back into the room r needs to grow"
    kv.remove_request("r")
    Worker._readmit_dropped(w)

    assert w.rebuilt == [("v", NODE)] and kv._allotments["v"].pages == 7, "v didn't return with its history and a page"


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


@pytest.mark.parametrize("walks, loops", [
    ({(NODE, "prefill")}, set()), ({(NODE, "prefill"), (NODE, "decode")}, {"decode"}),
], ids=["prefill worker", "worker that runs every walk"])
def test_a_pool_reserves_growth_only_for_the_loops_its_worker_runs(walks, loops):
    kv = _pool()
    _worker(kv)._set_local_loops(walks)
    assert kv.local_loops == loops, "a guaranteed request would reserve growth its worker never runs"


def test_an_older_request_drops_itself_rather_than_a_newer_guaranteed_one():
    kv = _pool()
    _running(kv, "r", 5, arrival=1)
    _running(kv, "v", 1, arrival=2, guaranteed=True)
    w = _worker(kv)
    batch = ScheduledBatch(node_name=NODE, graph_walk="decode", request_to_worker_graph={"r": 0})
    node_batch = SimpleNamespace(node_name=NODE, admit_error=_step(kv, "r", 1).reason, failed_resource="kv")

    Worker._handle_allocation_failure(w, batch, node_batch)

    assert w._dropped == {"r": NODE} and kv.is_resident("v"), "a guaranteed request was dropped for an older page"


def test_a_tp_nodes_victim_waits_for_the_nodes_next_broadcast_which_a_pass_without_a_step_still_sends():
    """Rank 0 drops a TP node's victim when it sends the node's next step. The followers drop it at the
    same point, after every removal stamped with the step before. A pass that builds no step for the node
    still sends one with no rows, since the request the victim was chosen for waits for the drop."""
    kv = _pool()
    _running(kv, "r", 5, arrival=1)
    _running(kv, "v", 5, arrival=2)
    reason = _step(kv, "r", 1).reason
    w = _worker(kv)
    w.parallel_nodes = w.parallel_leader_nodes = {NODE}
    batch = ScheduledBatch(node_name=NODE, graph_walk="decode", request_to_worker_graph={"r": 0})

    node_batch = SimpleNamespace(node_name=NODE, admit_error=reason, failed_resource="kv")
    Worker._handle_allocation_failure(w, batch, node_batch)
    assert "v" in w._victims and kv.is_resident("v"), "rank 0 dropped ahead of the broadcast its followers drop at"

    sent = []
    w.maybe_send_zmq_to_tp_followers, w._make_executing_batch = sent.append, SimpleNamespace
    Worker._flush_tp_moves(w)
    assert [(step.node_name, step.request_ids) for step in sent] == [(NODE, [])], (
        "the drop waits for a step that never comes"
    )

    Worker._drop_victims(w, NODE)
    assert w._dropped == {"v": NODE} and not w._victims, "the broadcast didn't drop the victim it carries"
