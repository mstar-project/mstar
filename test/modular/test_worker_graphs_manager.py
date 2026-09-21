"""Tests for ``WorkerGraphsManager``.

Covers:
- Inverted ``walk_node_to_worker_graph_id`` index built in __post_init__
- ``get_worker_graph_id_for_node`` uses the index (O(1) lookup, no scan)
- ``mark_node_complete`` returns the registry's ``NodeCompletionOutput``
- ``process_new_inputs`` returns leftover edges that no wg claimed
- ``stop_loops`` returns the loop-back ``set[(name, dest)]``
"""

import pytest

from mstar.communication.tensor_store import TensorStore
from mstar.communication.tensor_uuid import TensorUuidMinter
from mstar.conductor.request_info import (
    CurrentForwardPassInfo,
)
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential, TensorPointerInfo
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.graph.runtime.base import RouteInput, SendInput
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.graph.special_destinations import EMPTY_DESTINATION
from mstar.model.base import WorkerGraph
from mstar.utils.containers import ParallelList
from mstar.utils.ipc_format import ConductorMessageType
from mstar.worker.node_manager_utils import (
    WorkerGraphsManager,
)

# --- minimal stubs for tensor manager + fwd info -----------------------------

class StubTensorManager:
    """Records ref/deref calls so we can assert reference balance."""

    def __init__(self):
        self.refs: dict[tuple[str, str], int] = {}

    def increment_ref(self, request_id: str, uuid: str, n: int = 1):
        key = (request_id, uuid)
        self.refs[key] = self.refs.get(key, 0) + n

    def dereference(self, request_id: str, uuid: str, n: int = 1):
        key = (request_id, uuid)
        self.refs[key] = self.refs.get(key, 0) - n


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
    mgr = WorkerGraphsManager(
        queues=runtime.queues,
        per_request_info={},
        base_sharding_config=_sharding_config(),
        worker_id=worker_id,
        all_worker_graph_ids_to_graph_walks=all_walks,
        all_worker_graph_ids_to_nodes=all_nodes,
        all_worker_graph_ids_to_dyn_loops=all_loops,
        node_to_partition=node_to_partition,
    )
    fwd_info = _fwd_info(walk)
    rid = runtime.add_request(
        request_id=fwd_info.request_id,
        partition=fwd_info.partition_name,
        graph_walk=walk,
        partition_worker_graph_ids=[wg_id],
        worker_graph_to_workers=ParallelList([wg_id], [[worker_id]]),
    )
    mgr.add_request(
        rid=rid,
        partition_worker_graph_ids=[wg_id],
        worker_graph_to_workers={wg_id: [worker_id]},
        current_fwd_info=fwd_info,
    )
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
    leftovers = mgr.process_new_inputs(rid, [
        GraphEdge(name="prompt", next_node="prefill"),
    ])
    assert leftovers == []  # prefill is in this wg, edge claimed

    completion = runtime._mark_node_complete(rid, wg_id, "prefill")
    # Top-level GraphNode completion returns its outputs (token + kv_cache → ar_decode)
    # with no filtered signals (prefill isn't loop-managed).
    names = sorted((e.name, e.next_node) for e in completion.output_edges)
    assert names == [("kv_cache", "ar_decode"), ("token", "ar_decode")]
    assert completion.filtered_signals == set()


def test_process_new_inputs_leftovers_when_destination_unknown():
    mgr, wg_id, walk, runtime, rid = _make_manager()
    leftovers = mgr.process_new_inputs(rid, [
        GraphEdge(name="prompt", next_node="prefill"),
        GraphEdge(name="some_other_input", next_node="not_in_this_wg"),
    ])
    # The unknown-destination edge isn't claimed by any wg on this manager.
    assert len(leftovers) == 1
    assert leftovers[0].next_node == "not_in_this_wg"


def test_stop_loops_returns_loop_back_signal_set():
    mgr, wg_id, walk, runtime, rid = _make_manager()
    # Drive prefill → ar_decode so the loop is active.
    mgr.process_new_inputs(rid, [GraphEdge(name="prompt", next_node="prefill")])
    runtime._mark_node_complete(rid, wg_id, "prefill")

    stopped = runtime._stop_loops_for_rid(
        rid, "default", {"ar_loop"}, last_node_run=None,
    )
    # ar_loop has two loop-back inputs: (token, ar_decode) and (kv_cache, ar_decode).
    assert stopped == {("token", "ar_decode"), ("kv_cache", "ar_decode")}
    # _finish_signal should be set on the live loop.
    wgio = mgr.queues[wg_id].per_request_queues[rid]
    assert wgio.loops["ar_loop"]._finish_signal is True


def test_stop_loops_snapshots_loop_stop_times_for_the_last_node_run():
    mgr, wg_id, walk, runtime, rid = _make_manager()
    mgr.process_new_inputs(rid, [GraphEdge(name="prompt", next_node="prefill")])
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
    mgr.process_new_inputs(rid, [GraphEdge(name="prompt", next_node="prefill")])
    runtime._mark_node_complete(rid, wg_id, "prefill")
    # Route prefill's outputs back in.
    mgr.process_new_inputs(rid, [
        GraphEdge(name="token", next_node="ar_decode"),
        GraphEdge(name="kv_cache", next_node="ar_decode"),
    ])
    runtime._mark_node_complete(rid, wg_id, "ar_decode")  # advance: iter 0 done

    # Now request a stop on ar_loop, then complete the next iter.
    mgr.process_new_inputs(rid, [
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
    mgr.process_new_inputs(rid, [GraphEdge(name="text_inputs", next_node="prefill_text")])
    assert not mgr.queues[wg_id].is_done(rid)  # not done before complete
    runtime._mark_node_complete(rid, wg_id, "prefill_text")
    assert mgr.queues[wg_id].is_done(rid), \
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
    mgr.process_new_inputs(rid, [GraphEdge(name="prompt", next_node="prefill")])
    runtime._mark_node_complete(rid, wg_id, "prefill")

    routing = runtime._process_node_outputs(
        rid,
        node_name="prefill",
        outputs=list(mgr.queues[wg_id].per_request_queues[rid].nodes["prefill"].outputs),
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
    mgr.process_new_inputs(rid, [GraphEdge(name="prompt", next_node="prefill")])
    runtime._mark_node_complete(rid, wg_id, "prefill")
    wgio = runtime.queues[wg_id].per_request_queues[rid]

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
    mgr.process_new_inputs(rid, [GraphEdge(name="prompt", next_node="prefill")])
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
    mgr.process_new_inputs(rid, [GraphEdge(name="prompt", next_node="prefill")])
    runtime._mark_node_complete(rid, wg_id, "prefill")
    wgio = runtime.queues[wg_id].per_request_queues[rid]

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
    mgr.add_request(
        rid=rid_b, partition_worker_graph_ids=[0],
        worker_graph_to_workers={0: ["worker0"]},
        current_fwd_info=_fwd_info("decode"),
    )
    for rid in (rid_a, rid_b):
        mgr.process_new_inputs(
            rid, [GraphEdge(name="prompt", next_node="prefill")]
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
    completion = runtime.peek_completion(out.completion_id)

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
    mgr.process_new_inputs(rid, [GraphEdge(name="prompt", next_node="prefill")])
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
    state = runtime.peek_completion(out.completion_id)
    assert state.node_name == "prefill" and state.graph_walk == "decode"
    # peek does not consume; send_outputs is what pops the entry, so peeking
    # twice has to keep working.
    assert runtime.peek_completion(out.completion_id) is state


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
        nested_loop_indices=ParallelList([rid], [None]),
    )


def test_send_outputs_consumes_the_completion():
    """complete_and_route_batch parks the routing and send_outputs is what
    frees it. The worker peeks in between, so a double-pop here would only
    show up on the live path."""
    mgr, runtime, rid = _build(
        _make_ar_walk_graph(), 0, "decode",
        nodes={"prefill", "ar_decode"}, loops={"ar_loop"},
    )
    mgr.process_new_inputs(rid, [GraphEdge(name="prompt", next_node="prefill")])
    runtime._communicator = _RecordingCommunicator()
    store, minter = TensorStore(), TensorUuidMinter("worker_0")

    out, _ = _route_one(runtime, mgr, rid, store, minter)
    # Peeking must not consume.
    runtime.peek_completion(out.completion_id)
    runtime.peek_completion(out.completion_id)

    runtime.send_outputs(_send_input(runtime, rid, out.completion_id, _fwd_info("decode")))
    with pytest.raises(KeyError):
        runtime.peek_completion(out.completion_id)


def test_persist_signals_are_buffered_until_a_worker_graph_finishes():
    """They ride WORKER_GRAPHS_DONE rather than going out on their own, so a
    persist cannot race the message announcing it."""
    single = GraphNode(
        name="prefill", input_names={"prompt"},
        outputs=[GraphEdge(name="token", next_node=EMPTY_DESTINATION,
                           persist=True)],
    )
    mgr, runtime, rid = _build(single, 0, "decode", nodes={"prefill"})
    mgr.process_new_inputs(rid, [GraphEdge(name="prompt", next_node="prefill")])
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
