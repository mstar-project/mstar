"""The Rust and Python runtimes must answer identically.

The strongest check available: drive both through the same sequence and
compare every answer. A divergence here is silent in production -- a request
scheduled on one and not the other, or a refcount that differs by one -- so
the value is in running the SAME script against both rather than asserting
hand-written expectations twice.
"""
import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.communication.tensor_store import PythonTensorBookkeeping
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential, TensorPointerInfo
from mstar.graph.runtime.base import EdgeSpec, RouteInput, SpeculationPrepInput
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.model.base import WorkerGraph
from mstar.utils.containers import ParallelList

pytest.importorskip(
    "mstar_rust",
    reason="mstar_rust not built (maturin develop --release in rust/)",
)
from mstar.communication.tensor_store import RustTensorBookkeeping
from mstar.graph.runtime import rust as rust_runtime

WG_ID = 0
WALK = "decode"
WORKER = "worker_0"
NODES = {"prefill", "ar_decode"}


def _graph():
    return Sequential(sections=[
        GraphNode(
            name="prefill", input_names={"prompt"},
            outputs=[
                GraphEdge(name="token", next_node="ar_decode"),
                GraphEdge(name="kv_cache", next_node="ar_decode"),
            ],
        ),
        Loop(
            name="ar_loop",
            section=GraphNode(
                name="ar_decode", input_names={"token", "kv_cache"},
                outputs=[
                    GraphEdge(name="token", next_node="ar_decode"),
                    GraphEdge(name="kv_cache", next_node="ar_decode"),
                ],
            ),
            outputs=[GraphEdge(name="token", next_node="post_processor")],
            max_iters=4,
        ),
    ])


class _StubTensorManager:
    """Just enough of TensorCommunicationManager: the Python runtime reaches
    through it to dereference, where Rust goes straight to the bookkeeper."""

    def __init__(self, bookkeeping):
        from mstar.communication.tensor_store import TensorStore
        self.tensor_store = TensorStore(bookkeeping=bookkeeping)

    def dereference(self, uuid, n=1):
        self.tensor_store.dereference(uuid, n=n)

    def increment_ref(self, uuid, n=1):
        self.tensor_store.increment_ref(uuid, n=n)

    def get_tensor(self, uuid):
        return self.tensor_store.get_tensor(uuid)


def _sharding():
    return ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={})


def _python():
    book = PythonTensorBookkeeping()
    tm = _StubTensorManager(book)
    wg = WorkerGraph(
        section=_graph(), graph_walks={WALK}, ranks=[0], worker_graph_id=WG_ID,
    )
    rt = PythonGraphRuntime(
        my_worker_id=WORKER, my_worker_graphs=[wg],
        all_wg_ids_to_graph_walks={WG_ID: {WALK}},
        all_wg_ids_to_dyn_loops={WG_ID: {"ar_loop"}},
        all_wg_ids_to_nodes={WG_ID: NODES},
        node_to_partition=dict.fromkeys(NODES, "default"),
        sharding_config=_sharding(),
        tensor_manager=tm,
        communicator=None,
    )
    return rt, book, tm.tensor_store


def _rust():
    book = RustTensorBookkeeping()
    wg = WorkerGraph(
        section=_graph(), graph_walks={WALK}, ranks=[0], worker_graph_id=WG_ID,
    )
    rt = rust_runtime.RustGraphRuntime(
        my_worker_id=WORKER, my_worker_graphs=[wg],
        all_wg_ids_to_graph_walks={WG_ID: {WALK}},
        all_wg_ids_to_dyn_loops={WG_ID: {"ar_loop"}},
        all_wg_ids_to_nodes={WG_ID: NODES},
        node_to_partition=dict.fromkeys(NODES, "default"),
        sharding_config=_sharding(),
        bookkeeping=book,
    )
    return rt, book, None


@pytest.fixture(params=["python", "rust"])
def pair(request):
    """(runtime, bookkeeping, store). The store is only read by the Python
    runtime -- Rust holds a share of the bookkeeper and ignores the
    argument -- but both are handed it so the sequences stay identical."""
    return _python() if request.param == "python" else _rust()


def _info(uuid):
    return TensorPointerInfo(
        dims=[4], dtype=torch.float16, nbytes=8, address=0, stride=(1,),
        uuid=uuid, source_session_id="h:1", source_entity=WORKER,
    )


def _admit(rt, rid="r1"):
    return rt.add_request(
        request_id=rid, partition="default", graph_walk=WALK,
        partition_worker_graph_ids=[WG_ID],
        worker_graph_to_workers=ParallelList([WG_ID], [[WORKER]]),
    )


def _spec(signal, node, uuids=()):
    return EdgeSpec(
        signal=signal, next_node=node, uuids=list(uuids),
        is_final_streaming_chunk=False,
    )


def _ready(rt):
    """Normalised so the two orderings compare."""
    return sorted(
        (r.node_name, r.graph_walk, sorted(r.rids))
        for r in rt.get_ready_nodes(set())
    )


# --- the sequences -----------------------------------------------------------

def test_admit_ingest_schedule(pair):
    rt, _book, _store = pair
    rid = _admit(rt)
    assert rt.get_rid_handle("r1") == rid
    assert rt.get_rid_string(rid) == "r1"
    assert _ready(rt) == []

    assert rt.ingest_inputs_batch(
        ParallelList([rid], [_spec("prompt", "prefill")])
    ) == []
    assert _ready(rt) == [("prefill", WALK, [rid])]
    assert rt.has_ready_excluding(set())
    assert not rt.has_ready_excluding({rid})

    popped = rt.pop_rids("prefill", WALK, [rid])
    assert popped.wg_ids.keys == [rid]
    assert popped.wg_ids.values == [WG_ID]
    assert [e.signal for e in popped.input_edges] == ["prompt"]
    assert _ready(rt) == []


def test_structural_lookups_agree(pair):
    rt, _book, _store = pair
    assert rt.get_worker_graph_id_for_node("prefill", WALK) == WG_ID
    assert rt.get_output_signals("prefill", WALK) == ["kv_cache", "token"]
    assert rt.get_consumed_edges("prefill", "ar_decode", WALK) == {
        ("token", "ar_decode"), ("kv_cache", "ar_decode"),
    }
    assert rt.is_async_schedulable("prefill", WALK)


def test_refusals_agree(pair):
    rt, _book, _store = pair
    rid = _admit(rt)
    # unknown node, an input the node does not take, unknown rid
    assert rt.ingest_inputs_batch(
        ParallelList([rid], [_spec("x", "no_such_node")])
    ) == [0]
    assert rt.ingest_inputs_batch(
        ParallelList([rid], [_spec("not_an_input", "prefill")])
    ) == [0]
    assert rt.ingest_inputs_batch(
        ParallelList([424242], [_spec("prompt", "prefill")])
    ) == [0]


def test_one_handle_per_request_across_partitions(pair):
    """The conductor sends one NewRequest PER PARTITION, so add_request runs
    several times for one request and must return the same handle.

    Minting a fresh one each time orphans the previous handle's queues -- a
    deepcopy of the whole graph section per extra partition, never freed,
    because remove_request only ever sees the last handle. On a three-
    partition model that is two leaked graphs per request, forever.
    """
    rt, _book, _store = pair
    handles = [
        rt.add_request(
            request_id="r", partition=p, graph_walk=WALK,
            partition_worker_graph_ids=[WG_ID],
            worker_graph_to_workers=ParallelList([WG_ID], [[WORKER]]),
        )
        for p in ("Thinker", "Talker", "Code2Wav")
    ]
    assert len(set(handles)) == 1, f"one request, {len(set(handles))} handles"
    assert rt.get_rid_handle("r") == handles[0]

    # And teardown is complete: the handle is reusable, so nothing may be left
    # keyed by it.
    rt.remove_request(handles[0])
    assert rt.get_rid_handle("r") is None
    assert rt.add_request(
        request_id="r2", partition="Thinker", graph_walk=WALK,
        partition_worker_graph_ids=[WG_ID],
        worker_graph_to_workers=ParallelList([WG_ID], [[WORKER]]),
    ) == handles[0]


def test_handle_recycling_agrees(pair):
    rt, _book, _store = pair
    rid = _admit(rt, "r1")
    rt.remove_request(rid)
    assert rt.get_rid_handle("r1") is None
    rt.remove_request(rid)  # idempotent
    assert _admit(rt, "r2") == rid
    assert rt.get_rid_string(rid) == "r2"


def test_speculation_agrees(pair):
    rt, _book, _store = pair
    rid = _admit(rt)
    out = rt.speculate_node("prefill", WALK, rid)
    assert [(o.node_name, o.is_new_loop_iter, o.loop_name) for o in out] == [
        ("ar_decode", False, "ar_loop")
    ]
    back = rt.speculate_node("ar_decode", WALK, rid)
    assert [(o.node_name, o.is_new_loop_iter) for o in back] == [
        ("ar_decode", True)
    ]
    # Speculating must leave nothing behind.
    assert _ready(rt) == []


def test_loop_iters_agree(pair):
    rt, _book, _store = pair
    rid = _admit(rt)
    got = rt.get_dynamic_loop_iters([rid], "default")
    assert got.keys == [rid]
    assert got.values == [{"ar_loop": 0}]


def test_routing_agrees(pair):
    rt, book, store = pair
    rid = _admit(rt)
    rt.ingest_inputs_batch(ParallelList([rid], [_spec("prompt", "prefill")]))
    rt.pop_rids("prefill", WALK, [rid])

    for u in (1, 2):
        book.put_tensor(u, _info(u))
        book.increment_ref(u, 1)  # the safety hold

    out = rt.complete_and_route_batch(
        RouteInput(
            partition="default", graph_walk=WALK, node_name="prefill",
            output_signals=["kv_cache", "token"],
            wg_ids=ParallelList([rid], [WG_ID]),
            tensors=[1, 2], num_tensors=[1, 1],
        ),
        store,
    )
    assert out.completion_id > 0
    # Both route prefill's outputs into ar_decode locally.
    assert _ready(rt) == [("ar_decode", WALK, [rid])]
    # One local consumer each: the hold became one real reference.
    assert not book.can_gc(1)
    assert not book.can_gc(2)


def test_prep_agrees(pair):
    rt, _book, _store = pair
    rid = _admit(rt)
    rt.ingest_inputs_batch(ParallelList([rid], [_spec("prompt", "prefill")]))
    rt.pop_rids("prefill", WALK, [rid])
    rt.ingest_inputs_batch(ParallelList(
        [rid, rid],
        [_spec("token", "ar_decode"), _spec("kv_cache", "ar_decode")],
    ))

    out = rt.prep_spec_rids(SpeculationPrepInput(
        spec_node_name="ar_decode", curr_node_name="ar_decode",
        graph_walk=WALK, rids=[rid], room_for_continuing=None,
        streaming_edges=[], streaming_edges_per_rid=[0],
    ))
    assert out.ready_rids == [rid]
    assert out.wg_ids == [WG_ID]
    assert sum(out.input_edges_per_rid) == len(out.input_edges)

    capped = rt.prep_spec_rids(SpeculationPrepInput(
        spec_node_name="ar_decode", curr_node_name="ar_decode",
        graph_walk=WALK, rids=[rid], room_for_continuing=0,
        streaming_edges=[], streaming_edges_per_rid=[0],
    ))
    assert capped.ready_rids == []


def test_cleanup_agrees(pair):
    rt, book, store = pair
    rid = _admit(rt)
    book.put_tensor(5, _info(5))
    book.increment_ref(5, 1)
    rt.ingest_inputs_batch(
        ParallelList([rid], [_spec("prompt", "prefill", uuids=[5])])
    )
    rt.cleanup_consumed_inputs("prefill", [rid], [WG_ID])
    assert book.can_gc(5), "the consumed input was dereferenced"
    # And the slot is free again.
    assert rt.ingest_inputs_batch(
        ParallelList([rid], [_spec("prompt", "prefill")]), can_buffer=False
    ) == []


# --- stale handles -----------------------------------------------------------

@pytest.mark.parametrize("call", [
    pytest.param(lambda rt, r: rt.remove_request(r), id="remove_request"),
    pytest.param(lambda rt, r: rt.get_rid_string(r), id="get_rid_string"),
    pytest.param(lambda rt, r: rt.set_walk(r, "default", "w"), id="set_walk"),
    pytest.param(
        lambda rt, r: rt.mark_stream_partition_done(r, "default"),
        id="mark_stream_partition_done"),
    pytest.param(
        lambda rt, r: rt.get_dynamic_loop_iters([r], "default"),
        id="get_dynamic_loop_iters"),
    pytest.param(
        lambda rt, r: rt.cleanup_consumed_inputs("prefill", [r], [WG_ID]),
        id="cleanup_consumed_inputs"),
    pytest.param(
        lambda rt, r: rt.reset_outputs("prefill", [r], [WG_ID]),
        id="reset_outputs"),
    pytest.param(
        lambda rt, r: rt.push_back_node("prefill", [r], [WG_ID]),
        id="push_back_node"),
    pytest.param(
        lambda rt, r: rt.set_speculatively_scheduled(
            "prefill", WG_ID, [r], True),
        id="set_speculatively_scheduled"),
    pytest.param(
        lambda rt, r: rt.pop_rids("prefill", WALK, [r]), id="pop_rids"),
    pytest.param(
        lambda rt, r: rt.pop_rids("prefill", WALK, [r], check_ready=True),
        id="pop_rids_checked"),
    pytest.param(
        lambda rt, r: rt.speculate_node("prefill", WALK, r),
        id="speculate_node"),
    pytest.param(
        lambda rt, r: rt.get_spec_target("prefill", "ar_decode", WALK, r),
        id="get_spec_target"),
    pytest.param(
        lambda rt, r: rt.has_pending_loop_stop(r, WALK, "ar_loop"),
        id="has_pending_loop_stop"),
    pytest.param(
        lambda rt, r: rt.apply_peer_loop_stops(r, "default", {}),
        id="apply_peer_loop_stops"),
    pytest.param(
        lambda rt, r: rt.stop_loops_batched(
            partition="default", graph_walk=WALK, last_node_run="ar_decode",
            loop_names=ParallelList([r], [["ar_loop"]])),
        id="stop_loops_batched"),
    pytest.param(
        lambda rt, r: rt.prep_spec_rids(SpeculationPrepInput(
            spec_node_name="ar_decode", curr_node_name="prefill",
            graph_walk=WALK, rids=[r], room_for_continuing=None,
            streaming_edges=[], streaming_edges_per_rid=[0])),
        id="prep_spec_rids"),
    pytest.param(
        lambda rt, r: rt.prep_follow_spec_rids(SpeculationPrepInput(
            spec_node_name="ar_decode", curr_node_name="prefill",
            graph_walk=WALK, rids=[r], room_for_continuing=None,
            streaming_edges=[], streaming_edges_per_rid=[0])),
        id="prep_follow_spec_rids"),
    pytest.param(
        lambda rt, r: rt.ingest_inputs_batch(
            ParallelList([r], [_spec("prompt", "prefill")])),
        id="ingest_inputs_batch"),
])
def test_a_stale_handle_never_panics(pair, call):
    """A handle can outlive its request: a message for a rid this rank already
    removed is a benign race the Python side has always tolerated.

    On the Rust side an unchecked index panics ACROSS the FFI boundary, which
    is far worse than an exception -- so every entry point is checked.
    """
    rt, _book, _store = pair
    _admit(rt)  # so the tables are non-empty and a bad index is really out of range
    try:
        call(rt, 9999)
    except Exception as e:  # noqa: BLE001 - the point is what it must NOT be
        assert "Panic" not in type(e).__name__, (
            f"panicked across the FFI boundary: {e}"
        )


@pytest.mark.parametrize("cleanup_first", [True, False])
def test_an_input_is_released_whichever_order_runs(pair, cleanup_first):
    """The worker cleans up before routing, but the release must not DEPEND on
    that order: completion also clears a top-level node's inputs, and clearing
    without dereferencing leaks the reference silently."""
    rt, book, store = pair
    rid = _admit(rt)
    book.put_tensor(100, _info(100))
    book.increment_ref(100, 1)
    rt.ingest_inputs_batch(
        ParallelList([rid], [_spec("prompt", "prefill", uuids=[100])])
    )
    rt.pop_rids("prefill", WALK, [rid])

    def cleanup():
        rt.cleanup_consumed_inputs("prefill", [rid], [WG_ID])

    def route():
        for u in (200, 201):
            book.put_tensor(u, _info(u))
            book.increment_ref(u, 1)
        rt.complete_and_route_batch(
            RouteInput(
                partition="default", graph_walk=WALK, node_name="prefill",
                output_signals=["kv_cache", "token"],
                wg_ids=ParallelList([rid], [WG_ID]),
                tensors=[200, 201], num_tensors=[1, 1],
            ),
            store,
        )

    if cleanup_first:
        cleanup()
        route()
    else:
        route()
        cleanup()

    assert book.can_gc(100), "the consumed input leaked a reference"


def test_removal_purges_routing_parked_for_a_send_that_never_ran(pair):
    """An exception between complete_and_route_batch and send_outputs
    abandons the parked routing.

    Handles are RECYCLED, so a stale entry is not merely a leak: the next
    request to get that integer would have another request's outputs sent
    under its name.
    """
    rt, _book, store = pair
    for i in range(20):
        rid = _admit(rt, f"r{i}")
        rt.ingest_inputs_batch(
            ParallelList([rid], [_spec("prompt", "prefill")])
        )
        rt.pop_rids("prefill", WALK, [rid])
        rt.complete_and_route_batch(
            RouteInput(
                partition="default", graph_walk=WALK, node_name="prefill",
                output_signals=[], wg_ids=ParallelList([rid], [WG_ID]),
                tensors=[], num_tensors=[],
            ),
            store,
        )
        rt.remove_request(rid)  # aborted before the send

    parked = (
        len(rt._completions) if hasattr(rt, "_completions")
        else rt._rust.num_parked_completions()
    )
    assert parked == 0, f"{parked} completions left parked"
