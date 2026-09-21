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
from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential, TensorPointerInfo
from mstar.graph.runtime.base import EdgeSpec, RouteInput, SpeculationPrepInput
from mstar.graph.runtime.python import PythonGraphRuntime
from mstar.graph.special_destinations import EMIT_TO_CLIENT
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


def _graph_target_opts_out():
    """Same shape, but the speculation TARGET refuses async scheduling.

    The main fixture has every node async-enabled, so it cannot tell the
    source-side check from the destination-side one -- they agree on every
    input it produces.
    """
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
                enable_async_scheduling=False,
                outputs=[
                    GraphEdge(name="token", next_node="ar_decode"),
                    GraphEdge(name="kv_cache", next_node="ar_decode"),
                ],
            ),
            outputs=[GraphEdge(name="token", next_node="post_processor")],
            max_iters=4,
        ),
    ])


@pytest.fixture(params=["python", "rust"])
def opted_out(request):
    wg = WorkerGraph(
        section=_graph_target_opts_out(), graph_walks={WALK}, ranks=[0],
        worker_graph_id=WG_ID,
    )
    common = dict(
        my_worker_id=WORKER, my_worker_graphs=[wg],
        all_wg_ids_to_graph_walks={WG_ID: {WALK}},
        all_wg_ids_to_dyn_loops={WG_ID: {"ar_loop"}},
        all_wg_ids_to_nodes={WG_ID: NODES},
        node_to_partition=dict.fromkeys(NODES, "default"),
        sharding_config=_sharding(),
    )
    if request.param == "python":
        book = PythonTensorBookkeeping()
        return PythonGraphRuntime(
            tensor_manager=_StubTensorManager(book), communicator=None,
            **common,
        )
    return rust_runtime.RustGraphRuntime(
        bookkeeping=RustTensorBookkeeping(), **common,
    )


def test_a_target_that_refuses_async_is_never_speculated(opted_out):
    """``enable_async_scheduling=False`` on the DESTINATION.

    Rust only consulted the source's flag, and the source is already filtered
    by the caller -- so the real check was missing and a node that opted out
    got speculated into, to be dropped again per rid.
    """
    rid = _admit(opted_out)
    assert opted_out.speculate_node("prefill", WALK, rid) == []


def test_the_per_request_sharding_config_agrees(pair):
    """``register_request`` and the TP fan-out paths need the Python object.

    Rust derives its own copy for routing, but that one cannot come back out,
    so the shim derives a second from the same input by the same rule. This
    pins them together -- and that the config is dropped on removal, since
    handles are recycled.
    """
    rt, _book, _store = pair
    rid = _admit(rt)
    cfg = rt.get_sharding_config(rid)
    assert cfg is not None
    assert cfg.groups == []            # the fixture is un-sharded
    assert cfg is not _sharding(), "must not hand back the shared base"

    rt.remove_request(rid)
    assert rt.get_sharding_config(rid) is None


def test_an_unknown_rid_has_no_sharding_config(pair):
    # Teardown and TP fan-out can race a removal; both runtimes answer None
    # rather than raising.
    rt, _book, _store = pair
    assert rt.get_sharding_config(9999) is None


# --- a worker graph completes many times over one request --------------------

@pytest.fixture(params=["python", "rust"])
def one_node(request):
    """A worker graph whose single node finishes it in one completion.

    The main fixture runs each request through exactly one pass, so it cannot
    see whether a finished worker graph is RESET afterwards.
    """
    section = GraphNode(
        name="only", input_names={"prompt"},
        outputs=[GraphEdge(name="out", next_node=EMIT_TO_CLIENT)],
    )
    wg = WorkerGraph(
        section=section, graph_walks={WALK}, ranks=[0], worker_graph_id=WG_ID,
    )
    common = dict(
        my_worker_id=WORKER, my_worker_graphs=[wg],
        all_wg_ids_to_graph_walks={WG_ID: {WALK}},
        all_wg_ids_to_dyn_loops={WG_ID: set()},
        all_wg_ids_to_nodes={WG_ID: {"only"}},
        node_to_partition={"only": "default"},
        sharding_config=_sharding(),
    )
    if request.param == "python":
        book = PythonTensorBookkeeping()
        tm = _StubTensorManager(book)
        rt = PythonGraphRuntime(
            tensor_manager=tm, communicator=None, **common,
        )
        return rt, book, tm.tensor_store
    book = RustTensorBookkeeping()
    return rust_runtime.RustGraphRuntime(bookkeeping=book, **common), book, None


def _one_pass(rt, book, store, rid, uuid):
    """Drive the single node through a full pass; return the completed wgs."""
    rt.ingest_inputs_batch(ParallelList([rid], [_spec("prompt", "only")]))
    rt.pop_rids("only", WALK, [rid])
    book.put_tensor(uuid, _info(uuid))
    book.increment_ref(uuid, 1)  # the safety hold
    out = rt.complete_and_route_batch(
        RouteInput(
            partition="default", graph_walk=WALK, node_name="only",
            output_signals=["out"],
            wg_ids=ParallelList([rid], [WG_ID]),
            tensors=[uuid], num_tensors=[1],
        ),
        store,
    )
    return out


def test_a_worker_graph_completes_on_every_pass_not_just_the_first(one_node):
    """``is_done`` latches unless the finished graph is reset.

    A worker graph runs many times over a request. Rust never called
    RequestState::reset, so after the first completion root_entity_done
    early-returned on is_done, node flags and loop counters never cleared, and
    the request reported done once and could never become ready again.
    """
    rt, book, store = one_node
    rid = _admit(rt)

    first = _one_pass(rt, book, store, rid, uuid=1)
    assert first.completion_id > 0
    # The pass consumed it: nothing is ready until new input arrives.
    assert _ready(rt) == []

    # Second pass. Without the reset the node's completed flag and the
    # worker graph's is_done both latch, so this ingest goes nowhere.
    rt.ingest_inputs_batch(ParallelList([rid], [_spec("prompt", "only")]))
    assert _ready(rt) == [("only", WALK, [rid])], "the graph must run again"

    second = _one_pass(rt, book, store, rid, uuid=2)
    assert second.completion_id != first.completion_id


# --- a tensor read by N workers needs N references ---------------------------

@pytest.fixture(params=["python", "rust"])
def two_consumers(request):
    """One local node whose output goes to a node owned by TWO workers.

    Every other fixture puts a single worker on each node, so an edge and a
    destination are indistinguishable there -- which is exactly the confusion
    the Rust refcount had.
    """
    section = GraphNode(
        name="only", input_names={"prompt"},
        outputs=[GraphEdge(name="out", next_node="remote")],
    )
    wg = WorkerGraph(
        section=section, graph_walks={WALK}, ranks=[0], worker_graph_id=WG_ID,
    )
    common = dict(
        my_worker_id=WORKER, my_worker_graphs=[wg],
        all_wg_ids_to_graph_walks={WG_ID: {WALK}, 1: {WALK}},
        all_wg_ids_to_dyn_loops={WG_ID: set(), 1: set()},
        all_wg_ids_to_nodes={WG_ID: {"only"}, 1: {"remote"}},
        node_to_partition={"only": "default", "remote": "default"},
        # "remote" is a real TP group of two; shard_dim is empty so the
        # tensor is replicated to both rather than split.
        sharding_config=ShardingConfig(
            groups=[ShardingGroup(nodes={"remote"}, tp_size=2)],
            tp_enabled_nodes=set(), shard_dim={},
        ),
    )
    if request.param == "python":
        book = PythonTensorBookkeeping()
        tm = _StubTensorManager(book)
        return (
            PythonGraphRuntime(tensor_manager=tm, communicator=None, **common),
            book, tm.tensor_store,
        )
    book = RustTensorBookkeeping()
    return rust_runtime.RustGraphRuntime(bookkeeping=book, **common), book, None


def test_a_tensor_read_by_two_workers_holds_two_references(two_consumers):
    """Rust counted one reference per EDGE, Python one per destination WORKER.

    The per-worker expansion happens later, in take_send_plan, so the count
    was settled before the fanout existed. With two readers the hold drops to
    1 and the first release frees a tensor the other is still reading.
    """
    rt, book, store = two_consumers
    # "remote" runs on two workers for this request.
    rid = rt.add_request(
        request_id="r1", partition="default", graph_walk=WALK,
        partition_worker_graph_ids=[WG_ID, 1],
        worker_graph_to_workers=ParallelList(
            [WG_ID, 1], [[WORKER], ["worker1", "worker2"]],
        ),
    )
    rt.ingest_inputs_batch(ParallelList([rid], [_spec("prompt", "only")]))
    rt.pop_rids("only", WALK, [rid])

    book.put_tensor(1, _info(1))
    book.increment_ref(1, 1)  # the safety hold
    rt.complete_and_route_batch(
        RouteInput(
            partition="default", graph_walk=WALK, node_name="only",
            output_signals=["out"],
            wg_ids=ParallelList([rid], [WG_ID]),
            tensors=[1], num_tensors=[1],
        ),
        store,
    )

    # Two outstanding reads: one release must not free it.
    book.dereference(1, 1)
    assert not book.can_gc(1), "still being read by the second worker"
    book.dereference(1, 1)
    assert book.can_gc(1)


# --- several requests in one batch --------------------------------------------

def test_a_batch_of_three_routes_and_stays_schedulable(pair):
    """Every other case admits one rid, so nothing exercises a real batch.

    Drives three requests through prefill together, then checks both runtimes
    agree on what became ready and that each request can run again.
    """
    rt, book, store = pair
    rids = [_admit(rt, f"r{i}") for i in range(3)]
    assert len(set(rids)) == 3, "handles must be distinct"

    rt.ingest_inputs_batch(ParallelList(
        rids, [_spec("prompt", "prefill") for _ in rids],
    ))
    assert _ready(rt) == [("prefill", WALK, sorted(rids))]
    rt.pop_rids("prefill", WALK, rids)
    assert _ready(rt) == [], "popped nodes must leave the ready set"

    # One tensor per rid per signal, flat and rid-major.
    signals = ["kv_cache", "token"]
    uuids, num = [], []
    for i, _rid in enumerate(rids):
        for s_i, _sig in enumerate(signals):
            u = 1 + i * len(signals) + s_i
            book.put_tensor(u, _info(u))
            book.increment_ref(u, 1)  # the safety hold
            uuids.append(u)
            num.append(1)

    out = rt.complete_and_route_batch(
        RouteInput(
            partition="default", graph_walk=WALK, node_name="prefill",
            output_signals=signals,
            wg_ids=ParallelList(rids, [WG_ID] * len(rids)),
            tensors=uuids, num_tensors=num,
        ),
        store,
    )
    assert out.completion_id > 0
    # Each rid's outputs went to its OWN ar_decode, so all three are ready.
    assert _ready(rt) == [("ar_decode", WALK, sorted(rids))]
    # Every tensor has exactly its one local consumer -- a rid-major decoding
    # slip would pile references on one rid and free another's early.
    for u in uuids:
        assert not book.can_gc(u), f"uuid {u} lost its reference"


def test_a_batch_completes_every_request_not_just_the_first(pair):
    """The completion has to name all three requests.

    A batch that reports only the first rid leaves the others' worker graphs
    un-reset, so they never become ready again -- the request simply stops.
    """
    rt, book, store = pair
    rids = [_admit(rt, f"r{i}") for i in range(3)]
    rt.ingest_inputs_batch(ParallelList(
        rids, [_spec("prompt", "prefill") for _ in rids],
    ))
    rt.pop_rids("prefill", WALK, rids)
    signals = ["kv_cache", "token"]
    uuids, num = [], []
    for i in range(len(rids)):
        for s_i in range(len(signals)):
            u = 100 + i * len(signals) + s_i
            book.put_tensor(u, _info(u))
            book.increment_ref(u, 1)
            uuids.append(u)
            num.append(1)
    rt.complete_and_route_batch(
        RouteInput(
            partition="default", graph_walk=WALK, node_name="prefill",
            output_signals=signals,
            wg_ids=ParallelList(rids, [WG_ID] * len(rids)),
            tensors=uuids, num_tensors=num,
        ),
        store,
    )
    # Now run the loop body for all three at once and do it again.
    rt.pop_rids("ar_decode", WALK, rids)
    assert _ready(rt) == []


# --- a walk transition, which is the whole of a text-to-text request ----------

@pytest.fixture(params=["python", "rust"])
def two_walks(request):
    """Two worker graphs, one per walk, both holding the same node.

    This is the t2t shape: prefill runs, the conductor advances the partition
    to decode, and the SAME node has to become ready again in the other walk.
    Every other fixture has a single walk, so nothing exercises the handover.
    """
    def wg(wg_id, walk):
        return WorkerGraph(
            section=GraphNode(
                name="LLM", input_names={"text_inputs"},
                outputs=[GraphEdge(name="new_token", next_node=EMIT_TO_CLIENT,
                                   persist=True)],
            ),
            graph_walks={walk}, ranks=[0], worker_graph_id=wg_id,
        )
    mine = [wg(0, "prefill"), wg(1, "decode")]
    common = dict(
        my_worker_id=WORKER, my_worker_graphs=mine,
        all_wg_ids_to_graph_walks={0: {"prefill"}, 1: {"decode"}},
        all_wg_ids_to_dyn_loops={0: set(), 1: set()},
        all_wg_ids_to_nodes={0: {"LLM"}, 1: {"LLM"}},
        node_to_partition={"LLM": "default"},
        sharding_config=_sharding(),
    )
    if request.param == "python":
        book = PythonTensorBookkeeping()
        tm = _StubTensorManager(book)
        return (PythonGraphRuntime(tensor_manager=tm, communicator=None,
                                   **common), book, tm.tensor_store)
    book = RustTensorBookkeeping()
    return rust_runtime.RustGraphRuntime(bookkeeping=book, **common), book, None


def _run_walk(rt, book, store, rid, walk, wg_id, uuid):
    rt.ingest_inputs_batch(ParallelList([rid], [_spec("text_inputs", "LLM")]))
    ready = _ready(rt)
    rt.pop_rids("LLM", walk, [rid])
    book.put_tensor(uuid, _info(uuid))
    book.increment_ref(uuid, 1)
    rt.complete_and_route_batch(
        RouteInput(
            partition="default", graph_walk=walk, node_name="LLM",
            output_signals=["new_token"],
            wg_ids=ParallelList([rid], [wg_id]),
            tensors=[uuid], num_tensors=[1],
        ),
        store,
    )
    return ready


def test_the_node_becomes_ready_again_in_the_next_walk(two_walks):
    """The handover a t2t request depends on.

    Stuck here the request simply stops: prefill runs once, the walk advances,
    and the node never becomes ready in the new walk -- no error anywhere.
    """
    rt, book, store = two_walks
    rid = rt.add_request(
        request_id="r1", partition="default", graph_walk="prefill",
        partition_worker_graph_ids=[0, 1],
        worker_graph_to_workers=ParallelList([0, 1], [[WORKER], [WORKER]]),
    )
    assert _run_walk(rt, book, store, rid, "prefill", 0, uuid=1) == [
        ("LLM", "prefill", [rid])
    ]

    # The conductor advances the partition.
    rt.set_walk(rid, "default", "decode")
    assert _run_walk(rt, book, store, rid, "decode", 1, uuid=2) == [
        ("LLM", "decode", [rid])
    ], "the node never became ready in the new walk"


def test_a_batch_transitions_together(two_walks):
    """Same handover with three requests, which is how it actually runs."""
    rt, book, store = two_walks
    rids = [
        rt.add_request(
            request_id=f"r{i}", partition="default", graph_walk="prefill",
            partition_worker_graph_ids=[0, 1],
            worker_graph_to_workers=ParallelList([0, 1], [[WORKER], [WORKER]]),
        )
        for i in range(3)
    ]
    rt.ingest_inputs_batch(ParallelList(
        rids, [_spec("text_inputs", "LLM") for _ in rids],
    ))
    assert _ready(rt) == [("LLM", "prefill", sorted(rids))]
    rt.pop_rids("LLM", "prefill", rids)
    uuids = list(range(10, 10 + len(rids)))
    for u in uuids:
        book.put_tensor(u, _info(u))
        book.increment_ref(u, 1)
    rt.complete_and_route_batch(
        RouteInput(
            partition="default", graph_walk="prefill", node_name="LLM",
            output_signals=["new_token"],
            wg_ids=ParallelList(rids, [0] * len(rids)),
            tensors=uuids, num_tensors=[1] * len(rids),
        ),
        store,
    )
    for rid in rids:
        rt.set_walk(rid, "default", "decode")
    rt.ingest_inputs_batch(ParallelList(
        rids, [_spec("text_inputs", "LLM") for _ in rids],
    ))
    assert _ready(rt) == [("LLM", "decode", sorted(rids))]


# --- one walk spanning two LOCAL worker graphs (the i2t shape) ----------------

@pytest.fixture(params=["python", "rust"])
def two_local_graphs(request):
    """``encoder`` and ``LLM`` in SEPARATE worker graphs on the SAME worker.

    BAGEL's prefill_vit is exactly this: vit_encoder is one graph, LLM is
    another, both on rank 0, both in one walk. Rust compiles ``Dest::Local``
    per worker graph, so the cross-graph edge reads as External -- it has to be
    resolved against the local worker-graph index or the LLM node never gets
    its input.
    """
    enc = WorkerGraph(
        section=GraphNode(
            name="encoder", input_names={"image_inputs"},
            outputs=[GraphEdge(name="img_emb", next_node="LLM")],
        ),
        graph_walks={WALK}, ranks=[0], worker_graph_id=0,
    )
    llm = WorkerGraph(
        section=GraphNode(
            name="LLM", input_names={"img_emb"},
            outputs=[GraphEdge(name="new_token", next_node=EMIT_TO_CLIENT,
                               persist=True)],
        ),
        graph_walks={WALK}, ranks=[0], worker_graph_id=1,
    )
    common = dict(
        my_worker_id=WORKER, my_worker_graphs=[enc, llm],
        all_wg_ids_to_graph_walks={0: {WALK}, 1: {WALK}},
        all_wg_ids_to_dyn_loops={0: set(), 1: set()},
        all_wg_ids_to_nodes={0: {"encoder"}, 1: {"LLM"}},
        node_to_partition={"encoder": "default", "LLM": "default"},
        sharding_config=_sharding(),
    )
    if request.param == "python":
        book = PythonTensorBookkeeping()
        tm = _StubTensorManager(book)
        return (PythonGraphRuntime(tensor_manager=tm, communicator=None,
                                   **common), book, tm.tensor_store)
    book = RustTensorBookkeeping()
    return rust_runtime.RustGraphRuntime(bookkeeping=book, **common), book, None


def test_an_edge_into_a_sibling_local_graph_is_ingested_not_only_sent(
    two_local_graphs,
):
    """The i2t stall: the encoder's output has to reach LLM on this worker.

    Routed only onto the wire, LLM never becomes ready -- and the tensor's
    reference accounting comes out wrong too, which is how the same uuid ends
    up freed while a ready slot still names it.
    """
    rt, book, store = two_local_graphs
    rid = rt.add_request(
        request_id="r1", partition="default", graph_walk=WALK,
        partition_worker_graph_ids=[0, 1],
        worker_graph_to_workers=ParallelList([0, 1], [[WORKER], [WORKER]]),
    )
    rt.ingest_inputs_batch(ParallelList([rid], [_spec("image_inputs", "encoder")]))
    assert _ready(rt) == [("encoder", WALK, [rid])]
    rt.pop_rids("encoder", WALK, [rid])

    book.put_tensor(1, _info(1))
    book.increment_ref(1, 1)  # the safety hold
    rt.complete_and_route_batch(
        RouteInput(
            partition="default", graph_walk=WALK, node_name="encoder",
            output_signals=["img_emb"],
            wg_ids=ParallelList([rid], [0]),
            tensors=[1], num_tensors=[1],
        ),
        store,
    )

    # The sibling graph's node got the input.
    assert _ready(rt) == [("LLM", WALK, [rid])], "LLM never received img_emb"
    # And it is still held: one local consumer, not zero.
    assert not book.can_gc(1), "freed while LLM's ready slot still names it"
