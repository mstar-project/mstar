"""Tests for ``RequestStateManager``.

Covers:
- Inverted ``walk_node_to_worker_graph_id`` index built in __post_init__
- ``get_worker_graph_id_for_node`` uses the index (O(1) lookup, no scan)
- ``mark_node_complete`` returns the registry's ``NodeCompletionOutput``
- ``process_new_inputs`` returns leftover edges that no wg claimed
- ``stop_loops`` returns the loop-back ``set[(name, dest)]``
"""

from types import SimpleNamespace

import pytest

from mstar.communication.tensor_store import TensorStore
from mstar.communication.tensor_uuid import TensorUuidMinter
from mstar.communication.tensors import TensorCommunicationManager
from mstar.conductor.request_info import (
    CurrentForwardPassInfo,
)
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential, TensorPointerInfo
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.graph.runtime.base import (
    EdgeSpec,
    PendingLoopStop,
    RouteInput,
    SendInput,
    SpeculationPrepInput,
)
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.graph.special_destinations import EMIT_TO_CLIENT, EMPTY_DESTINATION
from mstar.model.base import WorkerGraph
from mstar.utils.containers import ParallelList
from mstar.utils.ipc_format import ConductorMessageType
from mstar.worker.node_manager_utils import (
    RequestStateManager,
)

# --- minimal stubs for tensor manager + fwd info -----------------------------

class StubTensorManager:
    """Records ref/deref calls so we can assert reference balance.

    Carries a real TensorStore: the runtime resolves uuids to descriptors
    through it when rebuilding an ingested edge.
    """

    def __init__(self):
        self.refs: dict[tuple[str, str], int] = {}
        self.tensor_store = TensorStore()

    def increment_ref(self, request_id: str, uuid: str, n: int = 1):
        key = (request_id, uuid)
        self.refs[key] = self.refs.get(key, 0) + n

    def dereference(self, request_id: str, uuid: str, n: int = 1):
        key = (request_id, uuid)
        self.refs[key] = self.refs.get(key, 0) - n

    def dereference_batch_uniform(self, uuids: list[str], n: int = 1):
        """What ``ReadySignals.clear`` and the loop caches call now: the
        whole set in one crossing, with no request id."""
        for uuid in uuids:
            key = ("", uuid)
            self.refs[key] = self.refs.get(key, 0) - n

    # Routing settles output refcounts through the manager, as production
    # does; these go straight to the real store underneath.
    set_output_ref_counts = TensorCommunicationManager.set_output_ref_counts

    # The runtime refreshes arena placement on every outgoing edge. The real
    # implementation, so the early-return this stub takes is the one under
    # test: no arena here, so nothing to copy back.
    stamps_shm_placement = False
    refresh_shm_placement = TensorCommunicationManager.refresh_shm_placement

    def set_persist(self, uuid: int, persist: bool):
        self.tensor_store.set_metadata(uuid, persist=persist)

    def dereference_batch(self, uuids: list[int], counts: list[int]):
        self.tensor_store.dereference_batch(uuids, counts)


def _fwd_info(graph_walk: str, partition: str = "default", fwd_index: int = 0):
    return CurrentForwardPassInfo(
        request_id="rid",
        graph_walk=graph_walk,
        fwd_index=fwd_index,
        random_seed=0,
        max_tokens=128,
        partition_name=partition,
    )


def _sharding_config():
    """Un-sharded base config; ``add_request`` clones it and calls ``setup``."""
    return ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={})


# --- fixtures ----------------------------------------------------------------

def _make_ar_walk_graph():
    """Single-worker, single-walk AR-shaped graph: prefill → ar_loop(ar_decode).

    The loop body's "token" output drives the loop-back AND (by name match in
    ``Loop.__post_init__``) the loop's terminal output to ``post_processor``
    on done. ``Loop.outputs`` entries whose names don't match a section-
    produced name are filtered out at construction.
    """
    return Sequential(sections=[
        GraphNode(
            name="prefill",
            input_names={"prompt"},
            outputs=[
                GraphEdge(name="token", next_node="ar_decode"),
                GraphEdge(name="kv_cache", next_node="ar_decode"),
            ],
        ),
        Loop(
            name="ar_loop",
            section=GraphNode(
                name="ar_decode",
                input_names={"token", "kv_cache"},
                outputs=[
                    GraphEdge(name="token", next_node="ar_decode"),
                    GraphEdge(name="kv_cache", next_node="ar_decode"),
                ],
            ),
            outputs=[GraphEdge(name="token", next_node="post_processor")],
            max_iters=10,
        ),
    ])


def _build(section, wg_id, walk, nodes, loops=frozenset(), worker_id="worker0"):
    """Wire a runtime + manager the way Worker does, and admit one request.

    The runtime owns the per-request queue lifecycle and the manager is handed
    the SAME queues dict, so both see one copy of the state.
    """
    worker_graph = WorkerGraph(
        section=section, graph_walks={walk}, ranks=[0], worker_graph_id=wg_id,
    )
    all_walks = {wg_id: {walk}}
    all_nodes = {wg_id: set(nodes)}
    all_loops = {wg_id: set(loops)}
    node_to_partition = dict.fromkeys(nodes, "default")

    runtime = PythonGraphRuntime(
        my_worker_id=worker_id,
        my_worker_graphs=[worker_graph],
        all_wg_ids_to_graph_walks=all_walks,
        all_wg_ids_to_dyn_loops=all_loops,
        all_wg_ids_to_nodes=all_nodes,
        node_to_partition=node_to_partition,
        sharding_config=_sharding_config(),
        tensor_manager=StubTensorManager(),
    )
    mgr = RequestStateManager(node_to_partition=node_to_partition)
    fwd_info = _fwd_info(walk)
    rid = runtime.add_request(
        request_id=fwd_info.request_id,
        partition=fwd_info.partition_name,
        graph_walk=walk,
        partition_worker_graph_ids=[wg_id],
        worker_graph_to_workers=ParallelList([wg_id], [[worker_id]]),
    )
    mgr.add_request(rid, fwd_info)
    return mgr, runtime, rid


def _make_manager(wg_id=0, graph_walk="decode", worker_id="worker0"):
    mgr, runtime, rid = _build(
        _make_ar_walk_graph(), wg_id, graph_walk,
        nodes={"prefill", "ar_decode"}, loops={"ar_loop"}, worker_id=worker_id,
    )
    return mgr, wg_id, graph_walk, runtime, rid


# --- tests -------------------------------------------------------------------

def test_inverted_index_populated_at_init():
    mgr, wg_id, walk, runtime, rid = _make_manager()
    # Both nodes should be indexed under the walk.
    assert runtime._walk_node_to_wg_id[(walk, "prefill")] == wg_id
    assert runtime._walk_node_to_wg_id[(walk, "ar_decode")] == wg_id
    # Unknown (walk, node) pairs should not be in the index.
    assert ("other_walk", "prefill") not in runtime._walk_node_to_wg_id


def test_get_worker_graph_id_uses_inverted_index():
    mgr, wg_id, walk, runtime, rid = _make_manager()
    assert runtime.get_worker_graph_id_for_node("prefill", walk) == wg_id
    assert runtime.get_worker_graph_id_for_node("ar_decode", walk) == wg_id


def test_get_worker_graph_id_raises_for_unknown_node():
    mgr, wg_id, walk, runtime, rid = _make_manager()
    # What matters is that an unknown (walk, node) raises rather than silently
    # returning some other worker graph.
    with pytest.raises(RuntimeError, match="Could not find worker graph"):
        runtime.get_worker_graph_id_for_node("mystery_node", walk)


def test_mark_node_complete_returns_node_completion_output():
    mgr, wg_id, walk, runtime, rid = _make_manager()
    # Ingest prompt → prefill, then complete prefill.
    uningested = _ingest(runtime, rid, [
        GraphEdge(name="prompt", next_node="prefill"),
    ])
    assert uningested == []  # prefill is in this wg, edge claimed

    completion = runtime._mark_node_complete(rid, wg_id, "prefill")
    # Top-level GraphNode completion returns its outputs (token + kv_cache → ar_decode)
    # with no filtered signals (prefill isn't loop-managed).
    names = sorted((e.name, e.next_node) for e in completion.output_edges)
    assert names == [("kv_cache", "ar_decode"), ("token", "ar_decode")]
    assert completion.filtered_signals == set()


def test_ingest_reports_the_index_of_an_unclaimed_signal():
    """Uningested signals come back as INDICES into the input list, which is
    how the streaming path knows which edge to hand back to its buffer."""
    mgr, wg_id, walk, runtime, rid = _make_manager()
    uningested = _ingest(runtime, rid, [
        GraphEdge(name="prompt", next_node="prefill"),
        GraphEdge(name="some_other_input", next_node="not_in_this_wg"),
    ])
    # Index 1, not index 0: no worker graph here owns "not_in_this_wg".
    assert uningested == [1]


def test_stop_loops_returns_loop_back_signal_set():
    mgr, wg_id, walk, runtime, rid = _make_manager()
    # Drive prefill → ar_decode so the loop is active.
    _ingest(runtime, rid, [GraphEdge(name="prompt", next_node="prefill")])
    runtime._mark_node_complete(rid, wg_id, "prefill")

    stopped = runtime._stop_loops_for_rid(
        rid, "default", {"ar_loop"}, last_node_run=None,
    )
    # ar_loop has two loop-back inputs: (token, ar_decode) and (kv_cache, ar_decode).
    assert stopped == {("token", "ar_decode"), ("kv_cache", "ar_decode")}
    # _finish_signal should be set on the live loop.
    wgio = runtime._queues[wg_id].per_request_queues[rid]
    assert wgio.loops["ar_loop"]._finish_signal is True


def test_stop_loops_snapshots_loop_stop_times_for_the_last_node_run():
    mgr, wg_id, walk, runtime, rid = _make_manager()
    _ingest(runtime, rid, [GraphEdge(name="prompt", next_node="prefill")])
    runtime._mark_node_complete(rid, wg_id, "prefill")
    fwd_info = mgr.get_fwd_info(rid, "default")

    runtime._stop_loops_for_rid(
        rid, "default", {"ar_loop"}, last_node_run="ar_decode",
    )
    # The snapshot lives on the runtime now, not on the wire-forwarded fwd_info.
    snapshot = runtime._loop_stop_times(rid).get("ar_loop")
    assert snapshot is not None
    assert snapshot.wg_fwd_pass_idx == fwd_info.fwd_index
    assert snapshot.loop_name_order == ["ar_loop"]


def test_loop_done_drops_loop_back_and_keeps_terminal_outputs():
    """Drive prefill and two ar_decode iterations, requesting loop termination
    before the second completes. The final completion should report loop-back
    signals in ``filtered_signals`` and return only the terminal outputs."""
    mgr, wg_id, walk, runtime, rid = _make_manager()
    _ingest(runtime, rid, [GraphEdge(name="prompt", next_node="prefill")])
    runtime._mark_node_complete(rid, wg_id, "prefill")
    # Route prefill's outputs back in.
    _ingest(runtime, rid, [
        GraphEdge(name="token", next_node="ar_decode"),
        GraphEdge(name="kv_cache", next_node="ar_decode"),
    ])
    runtime._mark_node_complete(rid, wg_id, "ar_decode")  # advance: iter 0 done

    # Now request a stop on ar_loop, then complete the next iter.
    _ingest(runtime, rid, [
        GraphEdge(name="token", next_node="ar_decode"),
        GraphEdge(name="kv_cache", next_node="ar_decode"),
    ])
    runtime._stop_loops_for_rid(rid, "default", {"ar_loop"}, None)
    completion = runtime._mark_node_complete(rid, wg_id, "ar_decode")

    assert sorted(completion.filtered_signals) == [
        ("kv_cache", "ar_decode"), ("token", "ar_decode"),
    ]
    assert [(e.name, e.next_node) for e in completion.output_edges] == [
        ("token", "post_processor"),
    ]


def test_mark_node_complete_on_empty_outputs_node_flips_is_done():
    """Regression: nodes with no declared outputs (BAGEL prefill_text,
    vae_encoder) must still drive ``is_done`` to True via mark_node_complete.

    The worker's _store_outputs_and_finish_loops used to ``continue`` over
    empty-output nodes BEFORE calling complete_loops, so the registry never
    flipped → WORKER_GRAPHS_DONE never fired → t2i hung on prefill_text.
    """
    empty_outputs_graph = GraphNode(
        name="prefill_text",
        input_names={"text_inputs"},
        outputs=[],  # no declared outputs — KV-cache-only step
    )
    wg_id = 1
    mgr, runtime, rid = _build(
        empty_outputs_graph, wg_id, "prefill_text", nodes={"prefill_text"},
    )
    _ingest(runtime, rid, [GraphEdge(name="text_inputs", next_node="prefill_text")])
    assert not runtime._queues[wg_id].is_done(rid)  # not done before complete
    runtime._mark_node_complete(rid, wg_id, "prefill_text")
    assert runtime._queues[wg_id].is_done(rid), \
        "mark_node_complete on a no-output node must flip is_done"


def test_process_node_outputs_marks_wg_done_with_all_external_outputs():
    """Regression: a wg whose just-completed node emits only special-destination
    edges (EMPTY_DESTINATION / EMIT_TO_CLIENT / streaming) must still flip to
    ``completed_worker_graph_ids`` so the worker can fire WORKER_GRAPHS_DONE.

    Previously the inverted-index routing only checked ``is_done`` on wgs
    that ingested an edge in this call, so prefill walks whose outputs all
    leave the local wg (Orpheus prefill -> EMPTY_DESTINATION + streaming to
    SNAC) never reported done and the conductor hung.
    """
    # Build a 1-node wg whose single output goes to EMPTY_DESTINATION.
    single_node_graph = GraphNode(
        name="prefill",
        input_names={"prompt"},
        outputs=[
            GraphEdge(name="new_token", next_node=EMPTY_DESTINATION,
                      conductor_new_token=True, persist=True),
        ],
    )
    wg_id = 2
    mgr, runtime, rid = _build(
        single_node_graph, wg_id, "prefill", nodes={"prefill"},
    )
    _ingest(runtime, rid, [GraphEdge(name="prompt", next_node="prefill")])
    runtime._mark_node_complete(rid, wg_id, "prefill")

    routing = runtime._process_node_outputs(
        rid,
        node_name="prefill",
        outputs=list(runtime._queues[wg_id].per_request_queues[rid].nodes["prefill"].outputs),
        graph_walk="prefill",
    )
    assert wg_id in routing.completed_worker_graph_ids, \
        "prefill wg with only EMPTY_DESTINATION outputs must still report done"


# --- loop stops on the runtime ------------------------------------------------

def test_peer_loop_stop_is_applied_only_when_newer():
    """A peer's STOP_LOOPS stops the loop only if its observation is newer than
    what this rank already has. Re-applying an older one must be a no-op, or a
    late duplicate would re-stop a loop that has since restarted."""
    _mgr, _wg_id, _walk, runtime, rid = _make_manager()
    mgr, wg_id = _mgr, _wg_id
    _ingest(runtime, rid, [GraphEdge(name="prompt", next_node="prefill")])
    runtime._mark_node_complete(rid, wg_id, "prefill")
    wgio = runtime._queues[wg_id].per_request_queues[rid]

    newer = NestedLoopIndices(
        loop_name_order=["ar_loop"], loop_indices={"ar_loop": 5},
        wg_fwd_pass_idx=1,
    )
    runtime.apply_peer_loop_stops(rid, "default", {"ar_loop": newer})
    assert wgio.loops["ar_loop"]._finish_signal is True
    assert runtime._loop_stop_times(rid)["ar_loop"] is newer

    # An older observation for the same loop is recorded but stops nothing new.
    wgio.loops["ar_loop"]._finish_signal = False
    older = NestedLoopIndices(
        loop_name_order=["ar_loop"], loop_indices={"ar_loop": 1},
        wg_fwd_pass_idx=0,
    )
    runtime.apply_peer_loop_stops(rid, "default", {"ar_loop": older})
    assert wgio.loops["ar_loop"]._finish_signal is False, \
        "an older peer stop must not re-stop the loop"


def test_peer_loop_stop_for_an_unknown_partition_is_dropped():
    _mgr, _wg_id, _walk, runtime, rid = _make_manager()
    runtime.apply_peer_loop_stops(rid, "no_such_partition", {"ar_loop": None})
    assert runtime._loop_stop_times(rid) == {}


def test_pending_loop_stops_are_recorded_and_live_one_iteration():
    _mgr, _wg_id, walk, runtime, rid = _make_manager()
    mgr, wg_id = _mgr, _wg_id
    _ingest(runtime, rid, [GraphEdge(name="prompt", next_node="prefill")])
    runtime._mark_node_complete(rid, wg_id, "prefill")

    runtime.stop_loops_batched(
        partition="default", graph_walk=walk, last_node_run="ar_decode",
        loop_names=ParallelList([rid], [["ar_loop"]]),
    )
    assert runtime.has_pending_loop_stop(rid, walk, "ar_loop")
    assert runtime.pending_loop_stop_rids(walk, "ar_loop") == {rid}
    # A different walk must not match.
    assert runtime.pending_loop_stop_rids("other_walk", "ar_loop") == set()

    runtime.clear_pending_loop_stops()
    assert not runtime.has_pending_loop_stop(rid, walk, "ar_loop")


def test_stop_for_a_loop_not_in_the_walk_is_dropped():
    """check_dyn_loop filtering: a stop naming a loop this walk does not have
    is a model bug, logged and dropped rather than raised."""
    _mgr, _wg_id, walk, runtime, rid = _make_manager()
    runtime.stop_loops_batched(
        partition="default", graph_walk=walk, last_node_run="ar_decode",
        loop_names=ParallelList([rid], [["not_a_real_loop"]]),
    )
    assert runtime.pending_loop_stop_rids(walk, "not_a_real_loop") == set()


def test_peer_loop_stop_compares_enclosing_loop_indices():
    """Same forward pass, so label_context_gt has to fall through to comparing
    the indices of the loops ENCLOSING the target. The previous assertions all
    differ in wg_fwd_pass_idx, which short-circuits before that point."""
    _mgr, _wg_id, _walk, runtime, rid = _make_manager()
    mgr, wg_id = _mgr, _wg_id
    _ingest(runtime, rid, [GraphEdge(name="prompt", next_node="prefill")])
    runtime._mark_node_complete(rid, wg_id, "prefill")
    wgio = runtime._queues[wg_id].per_request_queues[rid]

    def _at(outer_idx):
        return NestedLoopIndices(
            loop_name_order=["outer", "ar_loop"],
            loop_indices={"outer": outer_idx, "ar_loop": 0},
            wg_fwd_pass_idx=3,
        )

    runtime.apply_peer_loop_stops(rid, "default", {"ar_loop": _at(1)})
    assert wgio.loops["ar_loop"]._finish_signal is True

    wgio.loops["ar_loop"]._finish_signal = False
    runtime.apply_peer_loop_stops(rid, "default", {"ar_loop": _at(2)})
    assert wgio.loops["ar_loop"]._finish_signal is True, \
        "a later enclosing iteration is a newer stop"

    wgio.loops["ar_loop"]._finish_signal = False
    runtime.apply_peer_loop_stops(rid, "default", {"ar_loop": _at(2)})
    assert wgio.loops["ar_loop"]._finish_signal is False, \
        "the same observation twice must not re-stop"


# --- complete_and_route_batch -------------------------------------------------

def _ingest(runtime, rid, edges):
    """Feed edges through the runtime's contract entry point."""
    return runtime.ingest_inputs_batch(
        ParallelList(
            [rid] * len(edges),
            [
                EdgeSpec(
                    signal=e.name, next_node=e.next_node,
                    uuids=[i.uuid for i in e.tensor_info],
                ) for e in edges
            ],
        ),
        can_buffer=True,
    )


def _store_outputs(store, minter, rid, tensors):
    """Put tensors + descriptors in the store, as the worker's prologue does,
    and hand back the uuids. Descriptors are what let the runtime take uuids."""
    out = {}
    for name, tensor_list in tensors.items():
        uuids = []
        for tensor in tensor_list:
            uuid = minter.mint()
            store.put_tensor(rid, uuid, tensor, TensorPointerInfo(
                dims=tuple(tensor.shape), dtype=tensor.dtype,
                stride=tensor.stride(), nbytes=tensor.nbytes,
                address=tensor.data_ptr(), uuid=uuid,
                source_session_id="s", source_entity="worker_0",
            ))
            store.increment_ref(uuid, n=1)  # the safety hold
            uuids.append(uuid)
        out[name] = uuids
    return out


def test_route_batch_decodes_the_flat_rid_major_layout():
    """num_tensors is indexed [rid_i * n_signals + signal_i]. Getting that
    wrong silently attaches one rid's tensors to another's edges, so drive it
    with two rids and DIFFERENT tensor counts per signal."""
    import torch

    mgr, runtime, rid_a = _build(
        _make_ar_walk_graph(), 0, "decode",
        nodes={"prefill", "ar_decode"}, loops={"ar_loop"},
    )
    rid_b = runtime.add_request(
        request_id="rid_b", partition="default", graph_walk="decode",
        partition_worker_graph_ids=[0],
        worker_graph_to_workers=ParallelList([0], [["worker0"]]),
    )
    mgr.add_request(rid_b, _fwd_info("decode"))
    for rid in (rid_a, rid_b):
        _ingest(runtime, rid, [GraphEdge(name="prompt", next_node="prefill")]
        )

    store, minter = TensorStore(), TensorUuidMinter("worker_0")
    # Two signals with asymmetric counts: with one signal, or with matching
    # counts, a rid-major and a signal-major read give the SAME answer and the
    # test proves nothing.
    a = _store_outputs(store, minter, rid_a, {
        "token": [torch.ones(2), torch.ones(3)], "kv_cache": [torch.ones(4)],
    })
    b = _store_outputs(store, minter, rid_b, {
        "token": [torch.ones(5)],
        "kv_cache": [torch.ones(6), torch.ones(7), torch.ones(8)],
    })
    signals = ["kv_cache", "token"]  # sorted, as the worker builds it
    flat = (
        a["kv_cache"] + a["token"] + b["kv_cache"] + b["token"]
    )
    num_tensors = [
        len(a["kv_cache"]), len(a["token"]),
        len(b["kv_cache"]), len(b["token"]),
    ]

    out = runtime.complete_and_route_batch(
        RouteInput(
            partition="default", graph_walk="decode", node_name="prefill",
            output_signals=signals,
            wg_ids=ParallelList([rid_a, rid_b], [0, 0]),
            tensors=flat,
            num_tensors=num_tensors,
        ),
        store,
    )
    completion = runtime._peek_completion(out.completion_id)

    # Each rid's edges carry exactly its own uuids, per signal.
    for rid, own in ((rid_a, a), (rid_b, b)):
        by_signal: dict[str, list[int]] = {}
        for edge in completion.routing[rid].routed_to_this_worker_graph:
            by_signal.setdefault(edge.name, []).extend(
                i.uuid for i in edge.tensor_info
            )
        assert by_signal == own, f"{rid} got the wrong slice of the flat list"


def test_route_batch_parks_routing_under_a_fresh_completion_id():
    import torch

    mgr, runtime, rid = _build(
        _make_ar_walk_graph(), 0, "decode",
        nodes={"prefill", "ar_decode"}, loops={"ar_loop"},
    )
    _ingest(runtime, rid, [GraphEdge(name="prompt", next_node="prefill")])
    store, minter = TensorStore(), TensorUuidMinter("worker_0")
    uuids = _store_outputs(store, minter, rid, {"token": [torch.ones(2)]})["token"]

    out = runtime.complete_and_route_batch(
        RouteInput(
            partition="default", graph_walk="decode", node_name="prefill",
            output_signals=["token"],
            wg_ids=ParallelList([rid], [0]),
            tensors=uuids, num_tensors=[1],
        ),
        store,
    )
    assert out.completion_id not in (0,), "ids start above the unset sentinel"
    state = runtime._peek_completion(out.completion_id)
    assert state.node_name == "prefill" and state.graph_walk == "decode"
    # peek does not consume; send_outputs is what pops the entry, so peeking
    # twice has to keep working.
    assert runtime._peek_completion(out.completion_id) is state


# --- send_outputs -------------------------------------------------------------

class _RecordingCommunicator:
    def __init__(self):
        self.sent: list[tuple[str, object]] = []

    def send(self, entity_id, msg=None, **kwargs):
        self.sent.append((entity_id, msg if msg is not None else kwargs))


def _route_one(runtime, mgr, rid, store, minter, signal="token"):
    import torch

    uuids = _store_outputs(store, minter, rid, {signal: [torch.ones(2)]})[signal]
    out = runtime.complete_and_route_batch(
        RouteInput(
            partition="default", graph_walk="decode", node_name="prefill",
            output_signals=[signal],
            wg_ids=ParallelList([rid], [0]),
            tensors=uuids, num_tensors=[1],
        ),
        store,
    )
    return out, uuids


def _send_input(runtime, rid, completion_id, fwd_info):
    return SendInput(
        completion_id=completion_id,
        per_request_info=ParallelList([rid], [fwd_info]),
        new_token_counts=ParallelList([rid], [{}]),
    )


def test_send_outputs_consumes_the_completion():
    """complete_and_route_batch parks the routing and send_outputs is what
    frees it. The worker peeks in between, so a double-pop here would only
    show up on the live path."""
    mgr, runtime, rid = _build(
        _make_ar_walk_graph(), 0, "decode",
        nodes={"prefill", "ar_decode"}, loops={"ar_loop"},
    )
    _ingest(runtime, rid, [GraphEdge(name="prompt", next_node="prefill")])
    runtime._communicator = _RecordingCommunicator()
    store, minter = TensorStore(), TensorUuidMinter("worker_0")

    out, _ = _route_one(runtime, mgr, rid, store, minter)
    # Peeking must not consume.
    runtime._peek_completion(out.completion_id)
    runtime._peek_completion(out.completion_id)

    runtime.send_outputs(_send_input(runtime, rid, out.completion_id, _fwd_info("decode")))
    with pytest.raises(KeyError):
        runtime._peek_completion(out.completion_id)


def test_persist_signals_are_buffered_until_a_worker_graph_finishes():
    """They ride WORKER_GRAPHS_DONE rather than going out on their own, so a
    persist cannot race the message announcing it."""
    single = GraphNode(
        name="prefill", input_names={"prompt"},
        outputs=[GraphEdge(name="token", next_node=EMPTY_DESTINATION,
                           persist=True)],
    )
    mgr, runtime, rid = _build(single, 0, "decode", nodes={"prefill"})
    _ingest(runtime, rid, [GraphEdge(name="prompt", next_node="prefill")])
    comm = _RecordingCommunicator()
    runtime._communicator = comm
    store, minter = TensorStore(), TensorUuidMinter("worker_0")

    out, uuids = _route_one(runtime, mgr, rid, store, minter)
    runtime.send_outputs(_send_input(runtime, rid, out.completion_id, _fwd_info("decode")))

    wgd = [
        m for e, m in comm.sent
        if e == "conductor"
        and m.message_type == ConductorMessageType.WORKER_GRAPHS_DONE
    ]
    assert len(wgd) == 1, "the finished worker graph must report exactly once"
    assert wgd[0].body.persist_signals["token"][0].uuid == uuids[0]
    # Flushed, so a second pass does not resend them.
    assert runtime._request_info[rid].pending_persist_signals == []


# --- speculate_node -----------------------------------------------------------

def test_speculate_node_finds_the_downstream_target():
    """prefill's outputs feed ar_decode, so speculating from prefill offers
    ar_decode as the next node."""
    mgr, runtime, rid = _build(
        _make_ar_walk_graph(), 0, "decode",
        nodes={"prefill", "ar_decode"}, loops={"ar_loop"},
    )
    _ingest(runtime, rid, [GraphEdge(name="prompt", next_node="prefill")])

    out = runtime.speculate_node("prefill", "decode", rid)
    assert [o.node_name for o in out] == ["ar_decode"]
    assert out[0].graph_walk == "decode"


def test_speculate_node_skips_a_target_that_opted_out_of_async():
    mgr, runtime, rid = _build(
        _make_ar_walk_graph(), 0, "decode",
        nodes={"prefill", "ar_decode"}, loops={"ar_loop"},
    )
    _ingest(runtime, rid, [GraphEdge(name="prompt", next_node="prefill")])
    wgio = runtime._queues[0].per_request_queues[rid]
    wgio.nodes["ar_decode"].enable_async_scheduling = False

    assert runtime.speculate_node("prefill", "decode", rid) == []


def test_speculate_node_refuses_a_parallel_target_without_tp_async():
    """A parallel node is only a valid target as a leader-side same-node
    loop-back under TP async; a transition INTO one gives a follower no
    in-flight batch to rebuild a head from."""
    mgr, runtime, rid = _build(
        _make_ar_walk_graph(), 0, "decode",
        nodes={"prefill", "ar_decode"}, loops={"ar_loop"},
    )
    _ingest(runtime, rid, [GraphEdge(name="prompt", next_node="prefill")])

    runtime.set_node_metadata(
        parallel_nodes={"ar_decode"},
        parallel_leader_nodes={"ar_decode"},
        tp_async_nodes=set(),  # feature off
    )
    assert runtime.speculate_node("prefill", "decode", rid) == []

    # Even with TP async on, a TRANSITION into the parallel node is refused:
    # only a same-node loop-back qualifies.
    runtime.set_node_metadata(
        parallel_nodes={"ar_decode"},
        parallel_leader_nodes={"ar_decode"},
        tp_async_nodes={"ar_decode"},
    )
    assert runtime.speculate_node("prefill", "decode", rid) == []


def test_speculate_node_returns_nothing_for_an_unknown_rid():
    mgr, runtime, rid = _build(
        _make_ar_walk_graph(), 0, "decode",
        nodes={"prefill", "ar_decode"}, loops={"ar_loop"},
    )
    assert runtime.speculate_node("prefill", "decode", rid + 999) == []


# --- prep_spec_rids -----------------------------------------------------------

def _prep(runtime, rids, streaming=(), room=None,
          curr="prefill", spec="ar_decode"):
    per_rid: dict[int, list] = {r: [] for r in rids}
    for rid, spec_edge in streaming:
        per_rid[rid].append(spec_edge)
    flat = [e for r in rids for e in per_rid[r]]
    return runtime.prep_spec_rids(SpeculationPrepInput(
        spec_node_name=spec,
        curr_node_name=curr,
        graph_walk="decode",
        rids=rids,
        room_for_continuing=room,
        streaming_edges=flat,
        streaming_edges_per_rid=[len(per_rid[r]) for r in rids],
    ))


def _spec_ready_runtime():
    """A request sitting at prefill, whose outputs make ar_decode speculatable."""
    mgr, runtime, rid = _build(
        _make_ar_walk_graph(), 0, "decode",
        nodes={"prefill", "ar_decode"}, loops={"ar_loop"},
    )
    _ingest(runtime, rid, [GraphEdge(name="prompt", next_node="prefill")])
    return mgr, runtime, rid


def test_prep_returns_the_worker_graph_per_ready_rid():
    _mgr, runtime, rid = _spec_ready_runtime()
    out = _prep(runtime, [rid])
    assert out.ready_rids == [rid]
    assert out.wg_ids == [0], "wg_ids is parallel to ready_rids"
    assert len(out.input_edges_per_rid) == len(out.ready_rids)
    assert sum(out.input_edges_per_rid) == len(out.input_edges)


def test_prep_respects_room_for_continuing():
    """The backlog has first claim; rids past the cap are skipped BEFORE any
    streaming ingest, so there is nothing to roll back for them."""
    _mgr, runtime, rid = _spec_ready_runtime()
    assert _prep(runtime, [rid], room=0).ready_rids == []
    assert _prep(runtime, [rid], room=1).ready_rids == [rid]


def _in_loop_runtime():
    """Drive the request INTO ar_decode, so speculating ar_decode -> ar_decode
    is a loop-BACK. The loop filters only apply to a new loop iteration, which
    a transition into the loop (prefill -> ar_decode) is not."""
    mgr, runtime, rid = _spec_ready_runtime()
    runtime._mark_node_complete(rid, 0, "prefill")
    _ingest(runtime, rid, [
        GraphEdge(name="token", next_node="ar_decode"),
        GraphEdge(name="kv_cache", next_node="ar_decode"),
    ])
    return mgr, runtime, rid


def test_prep_skips_a_rid_whose_loop_already_has_a_pending_stop():
    _mgr, runtime, rid = _in_loop_runtime()
    # Sanity: the loop-back IS speculatable before the stop, or the assertion
    # below would pass for the wrong reason.
    assert _prep(
        runtime, [rid], curr="ar_decode", spec="ar_decode"
    ).ready_rids == [rid]

    runtime._pending_loop_stops.add(PendingLoopStop(rid, "decode", "ar_loop"))
    assert _prep(
        runtime, [rid], curr="ar_decode", spec="ar_decode"
    ).ready_rids == [], \
        "a loop with a stop pending has no further iterations to speculate"


def test_prep_skips_a_rid_on_its_loops_final_iteration():
    _mgr, runtime, rid = _in_loop_runtime()
    assert _prep(
        runtime, [rid], curr="ar_decode", spec="ar_decode"
    ).ready_rids == [rid]

    runtime._queues[0].per_request_queues[rid].loops[
        "ar_loop"
    ]._finish_signal = True
    assert _prep(
        runtime, [rid], curr="ar_decode", spec="ar_decode"
    ).ready_rids == []


def test_prep_does_not_apply_the_loop_filter_to_a_transition_into_the_loop():
    """prefill -> ar_decode enters the loop, so it is not a new iteration and
    a pending stop must not suppress it."""
    _mgr, runtime, rid = _spec_ready_runtime()
    runtime._pending_loop_stops.add(PendingLoopStop(rid, "decode", "ar_loop"))
    assert _prep(runtime, [rid]).ready_rids == [rid]


def test_prep_rolls_back_the_streaming_ingest_when_the_node_is_not_ready():
    """The rollback is what lets a later normal scheduling consume the chunk.
    If the ingest were left in place, the chunk would be stranded in a slot
    of a node that never ran."""
    _mgr, runtime, rid = _spec_ready_runtime()
    wgio = runtime._queues[0].per_request_queues[rid]
    node = wgio.nodes["ar_decode"]

    # Force not-ready: ar_decode needs token AND kv_cache, so offer only a
    # signal it does not take, which cannot complete it.
    before_ready = set(node.ready_signals.ready_names)
    out = _prep(runtime, [rid], streaming=[(
        rid, EdgeSpec(signal="token", next_node="ar_decode", uuids=[]),
    )])

    if not out.ready_rids:
        assert set(node.ready_signals.ready_names) == before_ready, \
            "a failed prep must leave no streaming chunk behind"
        assert out.consumed_streaming_edge_idxs == [], \
            "nothing was consumed, so the caller returns every chunk"


def test_prep_reports_consumed_streaming_edges_by_index():
    """The caller hands back whatever is NOT consumed, so the indices have to
    line up with the flat input list."""
    _mgr, runtime, rid = _spec_ready_runtime()
    out = _prep(runtime, [rid], streaming=[(
        rid, EdgeSpec(signal="token", next_node="ar_decode", uuids=[]),
    )])
    assert all(
        0 <= i < 1 for i in out.consumed_streaming_edge_idxs
    ), "indices must be into the flat streaming_edges list"


# --- routing settles refcounts through the tensor manager -------------------

def _build_on_real_manager(section, nodes, tmp_path, loops=frozenset()):
    """A runtime over a real (file-SHM, CPU) tensor manager, so refcount
    changes run the manager's teardown rather than a stub's bookkeeping."""
    from mstar.communication.tensors import SharedMemoryCommunicationManager

    tm = SharedMemoryCommunicationManager(
        my_entity_id="worker_0", hostname="localhost", device="cpu",
        communicator=SimpleNamespace(send=lambda *a, **k: None),
        shm_dir=str(tmp_path),
    )
    worker_graph = WorkerGraph(
        section=section, graph_walks={"w"}, ranks=[0], worker_graph_id=0,
    )
    runtime = PythonGraphRuntime(
        my_worker_id="worker_0",
        my_worker_graphs=[worker_graph],
        all_wg_ids_to_graph_walks={0: {"w"}},
        all_wg_ids_to_dyn_loops={0: set(loops)},
        all_wg_ids_to_nodes={0: set(nodes)},
        node_to_partition=dict.fromkeys(nodes, "default"),
        sharding_config=_sharding_config(),
        tensor_manager=tm,
    )
    rid = runtime.add_request(
        request_id="r", partition="default", graph_walk="w",
        partition_worker_graph_ids=[0],
        worker_graph_to_workers=ParallelList([0], [["worker_0"]]),
    )
    tm.register_request(rid, runtime.get_sharding_config(rid))
    return tm, runtime, rid


def _complete(tm, runtime, rid, node, outputs, signals):
    import torch  # noqa: F401  (outputs are tensors)

    stored = tm.store_and_return_tensor_info_batch(
        [rid], {rid: outputs}, signals,
    )
    tm.increment_ref_batch_uniform(stored.flat_uuids, 1)
    out = runtime.complete_and_route_batch(
        RouteInput(
            partition="default", graph_walk="w", node_name=node,
            output_signals=signals, wg_ids=ParallelList([rid], [0]),
            tensors=stored.flat_uuids, num_tensors=stored.num_tensors,
        ),
        tm.tensor_store,
    )
    return stored, out


def test_an_output_routed_to_no_consumer_is_freed_at_routing(tmp_path):
    """The safety hold is settled to the real fanout through the manager, so
    an output nothing consumes is torn down now -- not when the request ends."""
    import torch

    section = GraphNode(
        name="n", input_names={"x"},
        outputs=[GraphEdge(name="junk", next_node=EMPTY_DESTINATION)],
    )
    tm, runtime, rid = _build_on_real_manager(section, {"n"}, tmp_path)
    _ingest(runtime, rid, [GraphEdge(name="x", next_node="n")])

    stored, _ = _complete(tm, runtime, rid, "n", {"junk": [torch.ones(3)]}, ["junk"])
    [uuid] = stored.flat_uuids
    assert not tm.tensor_store.check_uuid_presence(uuid)


def test_a_new_token_signal_on_two_edges_is_counted_once(tmp_path):
    """One output routed to two destinations is two edges carrying the SAME
    tensors; reporting both would double every token count."""
    import torch

    section = GraphNode(
        name="n", input_names={"x"},
        outputs=[
            GraphEdge(name="tok", next_node=EMPTY_DESTINATION,
                      conductor_new_token=True),
            GraphEdge(name="tok", next_node=EMIT_TO_CLIENT,
                      conductor_new_token=True),
        ],
    )
    tm, runtime, rid = _build_on_real_manager(section, {"n"}, tmp_path)
    _ingest(runtime, rid, [GraphEdge(name="x", next_node="n")])

    stored, out = _complete(
        tm, runtime, rid, "n", {"tok": [torch.ones(4)]}, ["tok"],
    )
    assert out.new_token_output_idxs == [0]


# --- a loop edge carries tensors earlier batches produced -------------------
#
# The batch that finishes a loop is not the batch that produced everything the
# loop emits: ``accumulated_outputs`` gathers one tensor per iteration, and a
# ``Loop.outputs`` edge fed by a non-terminal body node was filled before the
# terminal node ran. RouteOutput used to report these as indices into the
# completing batch's tensors, which silently dropped exactly those -- nothing
# staged them, and the reader (api server, or a peer worker) got a missing
# tensor for a request that then hung.
#
# new_token_output_idxs stays index-based on purpose; the test below pins
# that down rather than leaving it looking like the same oversight.

def _rollout_runtime(tmp_path, accumulated_edge, max_iters=2, extra_nodes=()):
    """encode -> loop(step) with ``pred`` accumulated across iterations.

    ``extra_nodes`` are on this worker but outside the section, which is how a
    streaming consumer gets a sharding group to fan out to.

    Returns the per-iteration uuids (one per ``step`` completion, oldest
    first) and the RouteOutput of the completion that finished the loop.
    """
    import torch

    section = Sequential(sections=[
        GraphNode(
            name="encode", input_names={"video"},
            outputs=[GraphEdge(name="pred", next_node="step")],
        ),
        Loop(
            name="roll",
            section=GraphNode(
                name="step", input_names={"pred"},
                outputs=[GraphEdge(name="pred", next_node="step")],
            ),
            outputs=[],
            accumulated_outputs=[accumulated_edge],
            max_iters=max_iters,
        ),
    ])
    tm, runtime, rid = _build_on_real_manager(
        section, {"encode", "step", *extra_nodes}, tmp_path, loops={"roll"},
    )
    _ingest(runtime, rid, [GraphEdge(name="video", next_node="encode")])
    _complete(tm, runtime, rid, "encode", {"pred": [torch.ones(2)]}, ["pred"])

    per_iter: list[int] = []
    out = None
    for _ in range(max_iters):
        stored, out = _complete(
            tm, runtime, rid, "step", {"pred": [torch.ones(2)]}, ["pred"],
        )
        per_iter.append(stored.flat_uuids[0])
    return tm, runtime, rid, per_iter, out


def test_accumulated_loop_outputs_are_staged_for_every_iteration(tmp_path):
    """Regression: only the last iteration's tensor used to be staged, so the
    api server read a tensor the producer had never written (vjepa2
    prefill_video_rollout with rollout_horizon > 1)."""
    tm, _runtime, rid, per_iter, out = _rollout_runtime(
        tmp_path,
        GraphEdge(name="pred", next_node=EMIT_TO_CLIENT, persist=True),
    )
    first, last = per_iter
    assert first != last

    assert sorted(out.register_uuids) == sorted(per_iter), (
        "every tensor on an outgoing edge has to be staged, not just the "
        "ones this batch produced"
    )
    assert out.register_rids == [rid] * len(per_iter)
    # And really readable afterwards: staging is what writes the SHM file
    # the reader opens, which is where the FileNotFoundError came from.
    tm.register_for_send_uuids(ParallelList([rid], [list(out.register_uuids)]))
    for uuid in (first, last):
        assert tm.tensor_store.is_registered(uuid)


def test_new_token_outputs_stay_within_the_producing_batch(tmp_path):
    """Not the same fix: a new-token edge reports the batch that MINTED the
    tokens, so a loop re-emitting cached tensors at completion must not count
    them again. Indices into this batch's tensors say exactly that, and cost
    no strings at the boundary."""
    _tm, _runtime, _rid, per_iter, out = _rollout_runtime(
        tmp_path,
        GraphEdge(name="pred", next_node=EMIT_TO_CLIENT,
                  conductor_new_token=True),
    )
    # The completing batch minted one tensor; the earlier iteration's is on
    # the same edge and is deliberately not named.
    assert out.new_token_output_idxs == [0], (
        f"only this batch's tensor counts, not all of {per_iter}"
    )


def test_accumulated_local_streaming_carries_every_iteration(tmp_path):
    """The stream buffer has to receive every iteration's chunk, in order --
    an accumulated edge flushes them all at once and only the last was minted
    by the completing batch."""
    _tm, _runtime, rid, per_iter, out = _rollout_runtime(
        tmp_path,
        GraphEdge(name="pred", next_node="vocoder", is_streaming=True),
        extra_nodes=("vocoder",),
    )
    assert list(out.local_streaming_by_signal) == ["pred"], (
        "one key per stream is what keeps the edge name off the per-chunk path"
    )
    per_signal = out.local_streaming_by_signal["pred"]
    assert per_signal.values == per_iter, "oldest chunk first"
    assert per_signal.keys == [rid] * len(per_iter)


def test_a_tensor_on_two_outgoing_edges_is_staged_once(tmp_path):
    """Dedupe is by uuid across the whole batch: a tensor that both persists
    and emits must not be staged (and D2H-copied) twice."""
    import torch

    section = GraphNode(
        name="n", input_names={"x"},
        outputs=[
            GraphEdge(name="y", next_node=EMPTY_DESTINATION, persist=True),
            GraphEdge(name="y", next_node=EMIT_TO_CLIENT),
        ],
    )
    tm, runtime, rid = _build_on_real_manager(section, {"n"}, tmp_path)
    _ingest(runtime, rid, [GraphEdge(name="x", next_node="n")])

    stored, out = _complete(tm, runtime, rid, "n", {"y": [torch.ones(3)]}, ["y"])
    assert out.register_uuids == stored.flat_uuids
    assert out.register_rids == [rid]
def test_results_report_the_loop_context_from_before_the_completion(tmp_path):
    """Marking the node complete advances its loop, so the runtime has to
    snapshot the loop context first: a result emitted on iteration k is
    labelled k, not k + 1."""
    import torch

    section = Loop(
        name="gen_loop",
        section=GraphNode(
            name="gen", input_names={"tok"},
            outputs=[
                GraphEdge(name="tok", next_node="gen"),
                GraphEdge(name="out", next_node=EMIT_TO_CLIENT),
            ],
        ),
        outputs=[],
        max_iters=10,
    )
    tm, runtime, rid = _build_on_real_manager(section, {"gen"}, tmp_path)
    comm = _RecordingCommunicator()
    runtime._communicator = comm
    _ingest(runtime, rid, [GraphEdge(name="tok", next_node="gen")])
    wgio = runtime._queues[0].per_request_queues[rid]
    before = wgio.get_nested_loop_idxs_for_node("gen")

    _, out = _complete(
        tm, runtime, rid, "gen",
        {"tok": [torch.ones(1)], "out": [torch.ones(1)]}, ["out", "tok"],
    )
    assert wgio.get_nested_loop_idxs_for_node("gen") != before, (
        "the completion should have advanced the loop, or this proves nothing"
    )
    runtime.send_outputs(SendInput(
        completion_id=out.completion_id,
        per_request_info=ParallelList([rid], [_fwd_info("w")]),
        new_token_counts=ParallelList([rid], [{}]),
    ))
    [result] = [
        m for _entity, m in comm.sent if m.message_type == "result_tensors"
    ]
    assert result.body.loop_indices == before


def test_cleanup_takes_the_consumed_node_out_of_the_ready_set():
    """Its inputs are gone once consumed, so leaving the name ready would let
    the scheduler pop it again with nothing to run on."""
    mgr, runtime, rid = _build(
        _make_ar_walk_graph(), 0, "decode",
        nodes={"prefill", "ar_decode"}, loops={"ar_loop"},
    )
    _ingest(runtime, rid, [GraphEdge(name="prompt", next_node="prefill")])
    wgio = runtime._queues[0].per_request_queues[rid]
    assert "prefill" in wgio.ready_node_names

    runtime.cleanup_consumed_inputs("prefill", [rid], [0])
    assert "prefill" not in wgio.ready_node_names
