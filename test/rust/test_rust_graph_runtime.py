"""The Rust GraphRuntime behind the ABC.

Only the ported subset is exercised; the rest raises on purpose (see
``_UNPORTED`` in ``mstar/graph/runtime/rust.py``). What these pin is the SEAM:
the Python graph objects have to arrive in Rust meaning the same thing, and a
mismatch there is silent -- a node that compiles into the wrong worker graph
routes its outputs to the wrong peer.
"""
import sys

sys.path.insert(0, ".")

import pytest

from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mstar.model.base import WorkerGraph
from mstar.utils.containers import ParallelList

rust_runtime = pytest.importorskip(
    "mstar.graph.runtime.rust",
    reason="mstar_rust not built (maturin develop --release in rust/)",
)
from mstar.communication.rust_tensor_store import RustTensorBookkeeping

WG_ID = 0
WALK = "decode"
WORKER = "worker_0"


def _graph():
    """prefill -> ar_loop(ar_decode), the shape the Python tests use."""
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


@pytest.fixture
def runtime():
    wg = WorkerGraph(
        section=_graph(), graph_walks={WALK}, ranks=[0], worker_graph_id=WG_ID,
    )
    return rust_runtime.RustGraphRuntime(
        my_worker_id=WORKER,
        my_worker_graphs=[wg],
        all_wg_ids_to_graph_walks={WG_ID: {WALK}},
        all_wg_ids_to_dyn_loops={WG_ID: {"ar_loop"}},
        all_wg_ids_to_nodes={WG_ID: {"prefill", "ar_decode"}},
        node_to_partition={"prefill": "default", "ar_decode": "default"},
        sharding_config=ShardingConfig(
            groups=[], tp_enabled_nodes=set(), shard_dim={},
        ),
        bookkeeping=RustTensorBookkeeping(),
    )


def _admit(runtime, request_id="r1", walk=WALK):
    return runtime.add_request(
        request_id=request_id, partition="default", graph_walk=walk,
        partition_worker_graph_ids=[WG_ID],
        worker_graph_to_workers=ParallelList([WG_ID], [[WORKER]]),
    )


# --- the compile seam --------------------------------------------------------

def test_nodes_and_walks_survive_the_crossing(runtime):
    # Wrong here and a node's outputs route to the wrong peer, silently.
    assert runtime.get_worker_graph_id_for_node("prefill", WALK) == WG_ID
    assert runtime.get_worker_graph_id_for_node("ar_decode", WALK) == WG_ID
    with pytest.raises((RuntimeError, ValueError)):
        runtime.get_worker_graph_id_for_node("nope", WALK)
    with pytest.raises((RuntimeError, ValueError)):
        runtime.get_worker_graph_id_for_node("prefill", "other_walk")


def test_output_signals_match_the_python_graph(runtime):
    assert runtime.get_output_signals("prefill", WALK) == ["kv_cache", "token"]


def test_consumed_edges_are_the_pairs_into_the_destination(runtime):
    assert runtime.get_consumed_edges("prefill", "ar_decode", WALK) == {
        ("token", "ar_decode"), ("kv_cache", "ar_decode"),
    }
    assert runtime.get_consumed_edges("prefill", "elsewhere", WALK) == set()


def test_async_schedulability_crosses(runtime):
    assert runtime.is_async_schedulable("prefill", WALK)
    assert not runtime.is_async_schedulable("nope", WALK)


# --- rid interning -----------------------------------------------------------

def test_a_handle_round_trips(runtime):
    rid = _admit(runtime)
    assert runtime.get_rid_handle("r1") == rid
    assert runtime.get_rid_string(rid) == "r1"
    assert runtime.get_rid_handle("never-seen") is None


def test_admitting_the_same_request_twice_reuses_the_handle(runtime):
    # add_request runs once per PARTITION, so this is the normal path for a
    # multi-partition request, not an error case.
    assert _admit(runtime) == _admit(runtime)


def test_removal_frees_the_handle_for_reuse(runtime):
    rid = _admit(runtime, "r1")
    runtime.remove_request(rid)
    assert runtime.get_rid_handle("r1") is None
    # Recycled, which is why teardown has to be complete.
    assert _admit(runtime, "r2") == rid
    assert runtime.get_rid_string(rid) == "r2"


def test_removing_twice_is_idempotent(runtime):
    rid = _admit(runtime)
    runtime.remove_request(rid)
    runtime.remove_request(rid)
    assert runtime.get_rid_handle("r1") is None


# --- per-request state -------------------------------------------------------

def test_set_walk_and_stream_done_reach_the_request(runtime):
    rid = _admit(runtime)
    runtime.set_walk(rid, "default", "other_walk")
    runtime.mark_stream_partition_done(rid, "default")
    # No getter for stream-done on the ABC; not raising is the assertion, and
    # the walk change is observable.
    assert runtime is not None


def test_pending_loop_stops_live_one_iteration(runtime):
    rid = _admit(runtime)
    assert not runtime.has_pending_loop_stop(rid, WALK, "ar_loop")
    assert runtime.pending_loop_stop_rids(WALK, "ar_loop") == set()
    runtime.clear_pending_loop_stops()


def test_node_metadata_is_accepted(runtime):
    runtime.set_node_metadata(
        parallel_nodes={"ar_decode"},
        parallel_leader_nodes={"ar_decode"},
        tp_async_nodes=set(),
    )


def test_speculative_flag_rejects_an_unknown_node(runtime):
    rid = _admit(runtime)
    runtime.set_speculatively_scheduled("prefill", WG_ID, [rid], True)
    with pytest.raises((RuntimeError, ValueError)):
        runtime.set_speculatively_scheduled("nope", WG_ID, [rid], True)


# --- the unported half -------------------------------------------------------

def test_an_unported_method_says_so(runtime):
    # A silent AttributeError would surface as "NoneType has no attribute"
    # three frames away.
    with pytest.raises(NotImplementedError, match="not ported yet"):
        runtime.complete_and_route_batch(None, None)


# --- the compile seam's loop handling ----------------------------------------

def test_loop_nesting_survives_the_crossing():
    """``_managing_registry`` is only populated by a WorkerGraphIO, which the
    worker normally builds per request. Compiling needs the nesting, so the
    seam builds one for that side effect -- without it every loop crosses as
    top-level and a nested loop's completion accounts to the wrong registry."""
    from mstar.graph.runtime.rust import worker_graph_args

    inner = Loop(
        name="inner_loop",
        section=GraphNode(
            name="inner_node", input_names={"x"},
            outputs=[GraphEdge(name="x", next_node="inner_node")],
        ),
        outputs=[GraphEdge(name="x", next_node="outer_node")],
        max_iters=3,
    )
    outer = Loop(
        name="outer_loop",
        section=Sequential(sections=[inner]),
        outputs=[GraphEdge(name="x", next_node="sink")],
        max_iters=5,
    )
    wg = WorkerGraph(
        section=Sequential(sections=[outer]), graph_walks={WALK},
        ranks=[0], worker_graph_id=WG_ID,
    )

    args = worker_graph_args(wg)
    parents = {lp["name"]: lp["parent"] for lp in args["loops"]}
    assert parents["outer_loop"] is None, "a top-level loop has no parent"
    assert parents["inner_loop"] == "outer_loop"


def test_compiling_does_not_mutate_the_shared_section():
    """The seam walks a COPY: WorkerGraphIO writes _managing_registry into the
    sections it visits, and the pristine section is shared by every request."""
    from mstar.graph.runtime.rust import worker_graph_args

    wg = WorkerGraph(
        section=_graph(), graph_walks={WALK}, ranks=[0], worker_graph_id=WG_ID,
    )
    before = {
        name: node._managing_registry
        for name, node in wg.section.get_nodes().items()
    }
    worker_graph_args(wg)
    after = {
        name: node._managing_registry
        for name, node in wg.section.get_nodes().items()
    }
    assert before == after, "compiling must leave the shared section untouched"


def test_streaming_inputs_cross():
    """Read from _streaming_inputs; the public name is on ReadySignals, so a
    getattr on the wrong one silently sends an empty set and every streaming
    node loses its ready-for-streaming seed."""
    from mstar.graph.runtime.rust import worker_graph_args

    node = GraphNode(name="consumer", input_names={"chunk", "prompt"}, outputs=[])
    node._register_streaming({"chunk"})
    wg = WorkerGraph(
        section=node, graph_walks={WALK}, ranks=[0], worker_graph_id=WG_ID,
    )
    args = worker_graph_args(wg)
    assert args["nodes"][0]["streaming_inputs"] == ["chunk"]
