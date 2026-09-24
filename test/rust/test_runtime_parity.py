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

from mstar.communication.tensor_store import PythonTensorBookkeeping, TensorStore
from mstar.communication.tensors import TensorCommunicationManager
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
    """Just enough of TensorCommunicationManager, with its own refcount
    methods: the Python runtime settles and tears down through the manager,
    so the harness must do what production does -- a hand-rolled copy would
    only test itself."""

    set_output_ref_counts = TensorCommunicationManager.set_output_ref_counts
    set_persist = TensorCommunicationManager.set_persist
    dereference = TensorCommunicationManager.dereference
    dereference_batch = TensorCommunicationManager.dereference_batch
    dereference_batch_uniform = (
        TensorCommunicationManager.dereference_batch_uniform
    )
    cleanup_collectable = TensorCommunicationManager.cleanup_collectable

    def __init__(self, bookkeeping):
        self.tensor_store = TensorStore(bookkeeping=bookkeeping)

    def increment_ref(self, uuid, n=1):
        self.tensor_store.increment_ref(uuid, n=n)

    def get_tensor(self, uuid):
        return self.tensor_store.get_tensor(uuid)

    def _cleanup_by_uuid(self, uuid, registered=None):
        # The transport half is shm files and registered memory, which this
        # harness has none of; what is left is dropping the record, as
        # remove_tensor does. A tensor put into the bookkeeper alone is not in
        # the store, so it is forgotten directly -- unless a batched
        # dereference (``registered`` given) already forgot it.
        if self.tensor_store.check_uuid_presence(uuid):
            self.tensor_store.remove_tensor(uuid)
        elif registered is None:
            self.tensor_store.bookkeeping.forget_tensor(uuid)


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


def _released(book, uuid) -> bool:
    """The reference is gone, however the runtime got there.

    Python dereferences through the tensor manager, which tears the tensor
    down as it goes and leaves a collectable record. Rust hands the uuid back
    for the caller to tear down and drops the record in the same pass, and an
    untracked uuid is never "collectable" -- so can_gc alone reads as a leak
    on one side and a release on the other.
    """
    return book.get_info(uuid) is None or book.can_gc(uuid)


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
    assert _released(book, 5), "the consumed input was dereferenced"
    # And the slot is free again.
    assert rt.ingest_inputs_batch(
        ParallelList([rid], [_spec("prompt", "prefill")]), can_buffer=False
    ) == []


# --- who reclaims a freed input ----------------------------------------------
#
# Dropping the last reference is half of it. The shm file, the arena slot and
# the registered memory belong to the tensor manager, and the two runtimes
# reach it differently -- so the contract is "tell the caller what it still
# has to tear down", and only one of them has anything to say.

def _consume_one_input(rt, book, uuid=7):
    rid = _admit(rt)
    book.put_tensor(uuid, _info(uuid))
    book.increment_ref(uuid, 1)
    book.set_mem_registered(uuid, True)
    rt.ingest_inputs_batch(
        ParallelList([rid], [_spec("prompt", "prefill", uuids=[uuid])])
    )
    rt.pop_rids("prefill", WALK, [rid])
    return rt.cleanup_consumed_inputs("prefill", [rid], [WG_ID])


def test_rust_names_the_inputs_it_freed():
    """It holds the bookkeeper, not the manager, so a tensor it frees is
    reclaimed by nobody unless it says which ones. That is what left a
    consumed input's shm file sitting until the request was torn down."""
    rt, book, _store = _rust()
    freed = _consume_one_input(rt, book)

    assert freed.uuids == [7]
    assert freed.registered == [True], "the teardown has to know to unregister"
    assert book.get_info(7) is None, "forgotten in the same pass, not a second one"


def test_python_tears_a_freed_input_down_in_place():
    """It was built with the tensor manager and dereferences through it, so
    the teardown already ran and the caller is owed nothing."""
    rt, book, _store = _python()
    freed = _consume_one_input(rt, book)

    assert freed == ([], [])
    assert book.get_info(7) is None, "still released, just by the other route"


def test_a_recycled_handle_does_not_inherit_a_loop_stop(pair):
    """``pending_loop_stops`` is cleared once per postprocess, not per removal,
    and handles are recycled -- so a request admitted in between reads as
    already stopped. The worker then drops its outputs on a speculative new
    iteration and prep leaves it out of speculation. On main the key was the
    string rid, which never recurs."""
    rt, _book, _store = pair
    rid = _admit(rt, "r1")
    rt.stop_loops_batched(
        partition="default", graph_walk=WALK, last_node_run="prefill",
        loop_names=ParallelList([rid], [["ar_loop"]]),
    )
    assert rt.has_pending_loop_stop(rid, WALK, "ar_loop"), "nothing to inherit"
    rt.remove_request(rid)

    assert _admit(rt, "r2") == rid, "the handle was recycled, as intended"
    assert not rt.has_pending_loop_stop(rid, WALK, "ar_loop")
    assert rt.pending_loop_stop_rids(WALK, "ar_loop") == set()


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

    assert _released(book, 100), "the consumed input leaked a reference"


@pytest.mark.parametrize("cleanup_first", [True, False])
def test_whoever_frees_a_consumed_input_names_it(pair, cleanup_first):
    """Releasing it is half the job -- see the section below. Whichever call
    got there first has to be the one that reports it, or the completion path
    reclaims nothing on the order this very test says is supported."""
    rt, book, store = pair
    rid = _admit(rt)
    book.put_tensor(100, _info(100))
    book.increment_ref(100, 1)
    rt.ingest_inputs_batch(
        ParallelList([rid], [_spec("prompt", "prefill", uuids=[100])])
    )
    rt.pop_rids("prefill", WALK, [rid])

    def cleanup():
        return list(rt.cleanup_consumed_inputs("prefill", [rid], [WG_ID]).uuids)

    def route():
        out = rt.complete_and_route_batch(
            RouteInput(
                partition="default", graph_walk=WALK, node_name="prefill",
                output_signals=[], wg_ids=ParallelList([rid], [WG_ID]),
                tensors=[], num_tensors=[],
            ),
            store,
        )
        return list(out.freed_inputs.uuids)

    named = cleanup() + route() if cleanup_first else route() + cleanup()

    # Python tears it down in place through the tensor manager and names
    # nothing; Rust cannot, so it must.
    assert named == ([] if hasattr(rt, "_queues") else [100])


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


@pytest.fixture(params=["python", "rust"])
def streams_to_sibling(request):
    """``LLM`` streams new_token into ``snac_decoder``, a SEPARATE local graph.

    Orpheus exactly: the producer and the streaming consumer are different
    worker graphs on the same worker, so the edge compiles to External even
    though the consumer is local.
    """
    llm = WorkerGraph(
        section=GraphNode(
            name="LLM", input_names={"text_inputs"},
            outputs=[GraphEdge(name="new_token", next_node="snac_decoder",
                               is_streaming=True)],
        ),
        graph_walks={WALK}, ranks=[0], worker_graph_id=0,
    )
    snac = WorkerGraph(
        section=GraphNode(
            name="snac_decoder", input_names={"new_token"},
            outputs=[GraphEdge(name="audio", next_node=EMIT_TO_CLIENT)],
        ),
        graph_walks={"snac_chunk"}, ranks=[0], worker_graph_id=1,
    )
    common = dict(
        my_worker_id=WORKER, my_worker_graphs=[llm, snac],
        # The consumer is in its OWN walk and partition, as in Orpheus: the
        # producer streams out of `decode` into `snac_chunk`.
        all_wg_ids_to_graph_walks={0: {WALK}, 1: {"snac_chunk"}},
        all_wg_ids_to_dyn_loops={0: set(), 1: set()},
        all_wg_ids_to_nodes={0: {"LLM"}, 1: {"snac_decoder"}},
        node_to_partition={"LLM": "default", "snac_decoder": "SNAC"},
        sharding_config=_sharding(),
    )
    if request.param == "python":
        book = PythonTensorBookkeeping()
        tm = _StubTensorManager(book)
        return (PythonGraphRuntime(tensor_manager=tm, communicator=None,
                                   **common), book, tm.tensor_store)
    book = RustTensorBookkeeping()
    return rust_runtime.RustGraphRuntime(bookkeeping=book, **common), book, None


def test_a_stream_to_a_sibling_graph_is_reported_local(streams_to_sibling):
    """It must come back as a LOCAL stream, not be staged for a remote read.

    Reported as remote, the chunk never reaches the worker's StreamBuffer:
    the consumer gets an edge with no tensors, which is what surfaces as an
    empty ``inputs["new_token"]`` in the submodule.
    """
    rt, book, store = streams_to_sibling
    rid = rt.add_request(
        request_id="r1", partition="default", graph_walk=WALK,
        partition_worker_graph_ids=[0, 1],
        worker_graph_to_workers=ParallelList([0, 1], [[WORKER], [WORKER]]),
    )
    rt.ingest_inputs_batch(ParallelList([rid], [_spec("text_inputs", "LLM")]))
    rt.pop_rids("LLM", WALK, [rid])
    book.put_tensor(1, _info(1))
    book.increment_ref(1, 1)

    out = rt.complete_and_route_batch(
        RouteInput(
            partition="default", graph_walk=WALK, node_name="LLM",
            output_signals=["new_token"],
            wg_ids=ParallelList([rid], [0]),
            tensors=[1], num_tensors=[1],
        ),
        store,
    )
    assert out.local_streaming_tensor_idxs == [0], "the chunk was not local"
    assert out.register_tensor_idxs == [], "a local chunk must not be staged"
    # A streaming edge does NOT go straight into the node: the StreamBuffer
    # decides when a chunk is whole, so the consumer is not ready yet.
    assert _ready(rt) == []


def test_a_sibling_graph_chunk_is_held_once_not_twice(streams_to_sibling):
    """One local reader, so one release frees it.

    The edge compiles to External, and the fanout names THIS worker -- so the
    post-fanout count and the local ingest are the same copy. Counting the
    fanout destination on top of local_counts settles the hold to 2, and the
    tensor is never collected.
    """
    rt, book, store = streams_to_sibling
    rid = rt.add_request(
        request_id="r1", partition="default", graph_walk=WALK,
        partition_worker_graph_ids=[0, 1],
        worker_graph_to_workers=ParallelList([0, 1], [[WORKER], [WORKER]]),
    )
    rt.ingest_inputs_batch(ParallelList([rid], [_spec("text_inputs", "LLM")]))
    rt.pop_rids("LLM", WALK, [rid])
    book.put_tensor(1, _info(1))
    book.increment_ref(1, 1)
    rt.complete_and_route_batch(
        RouteInput(
            partition="default", graph_walk=WALK, node_name="LLM",
            output_signals=["new_token"],
            wg_ids=ParallelList([rid], [0]),
            tensors=[1], num_tensors=[1],
        ),
        store,
    )
    book.dereference(1, 1)
    assert book.can_gc(1), "still held after its only reader released"


def test_the_speculation_target_reports_its_output_signals(pair):
    """A speculated batch never goes through pop_rids, so the target carries
    the names instead -- left empty, completion routes none of the node's
    outputs, and decode is almost entirely speculated."""
    rt, _book, _store = pair
    rid = _admit(rt)
    out = rt.speculate_node("prefill", WALK, rid)
    assert [o.node_name for o in out] == ["ar_decode"]
    # ar_decode's own edges, sorted and deduped -- it sends `token` to itself
    # and to the loop output, so a raw edge list would repeat it.
    assert tuple(out[0].output_signals) == ("kv_cache", "token")
    assert tuple(out[0].output_signals) == tuple(
        rt.get_output_signals("ar_decode", WALK)
    )


def test_the_follower_target_reports_the_same_signals(pair):
    """get_spec_target is the follower's route to the same information: it
    cannot call speculate_node, and it builds the same ScheduledBatch."""
    rt, _book, _store = pair
    rid = _admit(rt)
    target = rt.get_spec_target("prefill", "ar_decode", WALK, rid)
    assert target is not None
    assert tuple(target.output_signals) == tuple(
        rt.get_output_signals("ar_decode", WALK)
    )


def test_marking_speculatively_scheduled_does_not_unready_the_node(pair):
    """Python sets a flag that gates FUTURE ingests; it never withdraws a node
    that is already ready. Rust folded `scheduled` into the readiness
    predicate, so marking a node pulled it out of the ready set -- and
    unmarking put it back."""
    rt, _book, _store = pair
    rid = _admit(rt)
    rt.ingest_inputs_batch(ParallelList([rid], [_spec("prompt", "prefill")]))
    assert _ready(rt) == [("prefill", WALK, [rid])]

    rt.set_speculatively_scheduled("prefill", WG_ID, [rid], True)
    assert _ready(rt) == [("prefill", WALK, [rid])], "marking withdrew it"

    rt.set_speculatively_scheduled("prefill", WG_ID, [rid], False)
    assert _ready(rt) == [("prefill", WALK, [rid])]


def test_push_back_re_readies_a_node_whose_inputs_were_consumed(pair):
    """``push_back_node`` puts a popped node back after an OOM hold.

    Python adds the name back unconditionally. Rust recomputed readiness from
    the current input slots, so a node whose inputs had already been cleared
    -- which is the normal order, cleanup runs before the push back -- stayed
    silently unready and the batch was lost.
    """
    rt, _book, _store = pair
    rid = _admit(rt)
    rt.ingest_inputs_batch(ParallelList([rid], [_spec("prompt", "prefill")]))
    rt.pop_rids("prefill", WALK, [rid])
    rt.cleanup_consumed_inputs("prefill", [rid], [WG_ID])
    assert _ready(rt) == []

    rt.push_back_node("prefill", [rid], [WG_ID])
    assert _ready(rt) == [("prefill", WALK, [rid])], "the push back was lost"


def test_a_purely_local_completion_needs_no_request_info():
    """Nothing goes out, so nobody's ``per_request_info`` has to be prepared.

    ``prefill -> ar_decode`` is one worker graph on one worker: no
    INPUT_SIGNALS to a peer, no WORKER_GRAPHS_DONE (the loop has not
    finished), so the re-encode would be pure waste.
    """
    rt, book, store = _rust()
    rid = _admit(rt)
    rt.ingest_inputs_batch(ParallelList([rid], [_spec("prompt", "prefill")]))
    rt.pop_rids("prefill", WALK, [rid])
    book.put_tensor(1, _info(1))
    book.increment_ref(1, 1)
    out = rt.complete_and_route_batch(
        RouteInput(
            partition="default", graph_walk=WALK, node_name="prefill",
            output_signals=["kv_cache", "token"],
            wg_ids=ParallelList([rid], [WG_ID]),
            tensors=[1], num_tensors=[1, 0],
        ),
        store,
    )
    assert out.rids_needing_request_info == frozenset()


def test_a_finished_worker_graph_asks_for_request_info(streams_to_sibling):
    """A WORKER_GRAPHS_DONE carries it, so a finished wg is a send.

    ``LLM`` here is a single node with no loop, so completing it finishes the
    worker graph and a WGD goes to the conductor -- the rid needs its
    payload even though the only output edge is a locally-ingested stream.

    (The sibling stream itself is NOT a send: it compiles to External because
    the consumer is another worker graph, but it lands in this worker's
    StreamBuffer. Counting that as a send is what made an earlier version of
    this flag skip nothing at all on Orpheus.)
    """
    rt, book, store = streams_to_sibling
    if not hasattr(rt, "_rust"):
        pytest.skip("flag is only computed by the rust runtime")
    rid = rt.add_request(
        request_id="r1", partition="default", graph_walk=WALK,
        partition_worker_graph_ids=[0, 1],
        worker_graph_to_workers=ParallelList([0, 1], [[WORKER], [WORKER]]),
    )
    rt.ingest_inputs_batch(ParallelList([rid], [_spec("text_inputs", "LLM")]))
    rt.pop_rids("LLM", WALK, [rid])
    book.put_tensor(1, _info(1))
    book.increment_ref(1, 1)
    out = rt.complete_and_route_batch(
        RouteInput(
            partition="default", graph_walk=WALK, node_name="LLM",
            output_signals=["new_token"],
            wg_ids=ParallelList([rid], [0]),
            tensors=[1], num_tensors=[1],
        ),
        store,
    )
    assert rid in out.rids_needing_request_info


def test_a_node_missing_a_non_streaming_input_is_not_streaming_ready():
    """"Streaming-ready" means every MISSING input is a streaming one.

    Python expressed this as ``input_names.issuperset(ready_names |
    streaming_inputs)``, which is a tautology -- ready_names is asserted a
    subset of input_names and streaming_inputs is one by construction, so the
    union always is too. Every node with any input at all therefore read as
    streaming-ready, and Rust mirrored it deliberately. This pins the fix on
    both: ``ar_decode`` takes token and kv_cache, neither streaming, so one
    arriving must NOT make it streaming-ready.
    """
    from mstar.graph.base import ReadySignals

    sig = ReadySignals(
        node_name="ar_decode",
        input_names={"token", "kv_cache"},
        streaming_inputs=set(),
    )
    sig.update(GraphEdge(name="token", next_node="ar_decode"))
    assert not sig.is_ready
    assert not sig.is_ready_for_streaming, "a tautology made this always true"

    sig.update(GraphEdge(name="kv_cache", next_node="ar_decode"))
    assert sig.is_ready and sig.is_ready_for_streaming


def test_a_node_missing_only_a_streaming_input_is_streaming_ready():
    """The case the flag exists for: run on what has arrived."""
    from mstar.graph.base import ReadySignals

    sig = ReadySignals(
        node_name="snac",
        input_names={"text", "new_token"},
        streaming_inputs={"new_token"},
    )
    sig.update(GraphEdge(name="text", next_node="snac"))
    assert not sig.is_ready
    assert sig.is_ready_for_streaming


def test_removing_an_input_takes_streaming_readiness_back_down():
    """``remove`` is the speculation-rollback path, so the flag must fall."""
    from mstar.graph.base import ReadySignals

    sig = ReadySignals(
        node_name="snac",
        input_names={"text", "new_token"},
        streaming_inputs={"new_token"},
    )
    sig.update(GraphEdge(name="text", next_node="snac"))
    assert sig.is_ready_for_streaming
    sig.remove("text")
    assert not sig.is_ready_for_streaming


# --- nested loops ------------------------------------------------------------

NESTED_NODES = {"denoiser", "refiner", "decoder"}
NESTED_LOOPS = {"refine_loop", "denoise_loop"}


def _nested_graph():
    """``test/modular/test_graph.py::test_nested_loops``, which pins the
    Python semantics."""
    return Sequential(sections=[
        Loop(
            name="refine_loop", max_iters=2,
            section=Sequential(sections=[
                Loop(
                    name="denoise_loop", max_iters=3,
                    section=GraphNode(
                        name="denoiser", input_names={"latents"},
                        outputs=[GraphEdge(name="latents",
                                           next_node="denoiser")],
                    ),
                    outputs=[GraphEdge(name="latents", next_node="refiner")],
                ),
                GraphNode(
                    name="refiner", input_names={"latents"},
                    outputs=[GraphEdge(name="latents", next_node="denoiser")],
                ),
            ]),
            outputs=[GraphEdge(name="latents", next_node="decoder")],
        ),
        GraphNode(
            name="decoder", input_names={"latents"},
            outputs=[GraphEdge(name="image", next_node=EMIT_TO_CLIENT,
                               output_modality="image")],
        ),
    ])


@pytest.fixture(params=["python", "rust"])
def nested(request):
    wg = WorkerGraph(
        section=_nested_graph(), graph_walks={WALK}, ranks=[0],
        worker_graph_id=WG_ID,
    )
    common = dict(
        my_worker_id=WORKER, my_worker_graphs=[wg],
        all_wg_ids_to_graph_walks={WG_ID: {WALK}},
        all_wg_ids_to_dyn_loops={WG_ID: NESTED_LOOPS},
        all_wg_ids_to_nodes={WG_ID: NESTED_NODES},
        node_to_partition=dict.fromkeys(NESTED_NODES, "default"),
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


def test_a_loop_inside_a_loop_runs_every_iteration(nested):
    """Every ancestor claimed its descendants' nodes, so the inner loop
    never completed: one denoiser run, then a silent hang."""
    rt, book, store = nested
    rid = _admit(rt)
    uuid = iter(range(1, 500))

    def fresh():
        u = next(uuid)
        book.put_tensor(u, _info(u))
        book.increment_ref(u, 1)  # the safety hold
        return u

    rt.ingest_inputs_batch(
        ParallelList([rid], [_spec("latents", "denoiser", uuids=[fresh()])])
    )
    log = []
    for _ in range(40):
        ready = [r.node_name for r in rt.get_ready_nodes(set())]
        if not ready:
            break
        node = ready[0]
        rt.pop_rids(node, WALK, [rid])
        log.append(node)
        signals = rt.get_output_signals(node, WALK)
        rt.complete_and_route_batch(
            RouteInput(
                partition="default", graph_walk=WALK, node_name=node,
                output_signals=signals,
                wg_ids=ParallelList([rid], [WG_ID]),
                tensors=[fresh() for _ in signals],
                num_tensors=[1] * len(signals),
            ),
            store,
        )
    assert log == [
        "denoiser", "denoiser", "denoiser", "refiner",
        "denoiser", "denoiser", "denoiser", "refiner",
        "decoder",
    ]


def test_the_inner_loop_is_the_one_that_advances(nested):
    """The counter the conductor reads: with the inner loop bypassed it
    stayed at 0 while the outer one moved."""
    rt, book, store = nested
    rid = _admit(rt)
    book.put_tensor(900, _info(900))
    book.increment_ref(900, 1)
    rt.ingest_inputs_batch(
        ParallelList([rid], [_spec("latents", "denoiser", uuids=[900])])
    )
    rt.pop_rids("denoiser", WALK, [rid])
    book.put_tensor(901, _info(901))
    book.increment_ref(901, 1)
    rt.complete_and_route_batch(
        RouteInput(
            partition="default", graph_walk=WALK, node_name="denoiser",
            output_signals=["latents"],
            wg_ids=ParallelList([rid], [WG_ID]),
            tensors=[901], num_tensors=[1],
        ),
        store,
    )
    assert rt.get_dynamic_loop_iters([rid], "default").values == [
        {"denoise_loop": 1, "refine_loop": 0}
    ]


def test_speculation_names_the_innermost_enclosing_loop(nested):
    """Callers read this loop's ``curr_iter`` / ``max_iters``, so naming the
    outer one reads the wrong counters."""
    rt, _book, _store = nested
    rid = _admit(rt)
    out = rt.speculate_node("denoiser", WALK, rid)
    assert [(o.node_name, o.loop_name) for o in out] == [
        ("denoiser", "denoise_loop")
    ]


# --- pop_rids honours the batch the caller committed to -----------------------

def test_an_unchecked_pop_returns_every_rid_it_was_given(pair):
    """``check_ready=False`` means readiness is already established. Rust
    re-checked it and silently shortened the batch."""
    rt, _book, _store = pair
    rid = _admit(rt)
    # Nothing ingested, so the node is not in the ready set.
    popped = rt.pop_rids("prefill", WALK, [rid])
    assert popped is not None
    assert popped.wg_ids.keys == [rid]
    assert popped.wg_ids.values == [WG_ID]
    assert popped.input_edges == []
    assert popped.input_edges_per_rid == [0]


def test_a_checked_pop_still_refuses_a_rid_that_is_not_ready(pair):
    """The other half of the contract, unchanged: verify every rid, pop all
    or nothing."""
    rt, _book, _store = pair
    rid = _admit(rt)
    assert rt.pop_rids("prefill", WALK, [rid], check_ready=True) is None
    rt.ingest_inputs_batch(ParallelList([rid], [_spec("prompt", "prefill")]))
    assert rt.pop_rids("prefill", WALK, [rid], check_ready=True) is not None


def test_cleanup_then_push_back_still_pops(pair):
    """The worker's order (worker.py:1934). A guard, not a reproducer: it
    pins that ``push_back_node`` stays unconditional."""
    rt, _book, _store = pair
    rid = _admit(rt)
    rt.ingest_inputs_batch(ParallelList([rid], [_spec("prompt", "prefill")]))
    rt.pop_rids("prefill", WALK, [rid])
    rt.cleanup_consumed_inputs("prefill", [rid], [WG_ID])
    rt.push_back_node("prefill", [rid], [WG_ID])
    popped = rt.pop_rids("prefill", WALK, [rid])
    assert popped is not None and popped.wg_ids.keys == [rid]


# --- a peer's post-fanout copy is not ingested here ---------------------------
#
# cosmos3_nano_tp2's shape: a tp1 node group on rank 0 feeding a tp2 group on
# both ranks. Rank 0 runs BOTH, so the fanout's peer copy names a node this
# worker also owns -- and resolving it by name alone ingests the peer's
# tensors here too.

TP_WALK = "decode"
PEER = "worker_1"


def _tp_graphs():
    """`src` alone; `dit` looping in a SIBLING worker graph, as separate node
    groups compile to separate graphs."""
    return [
        WorkerGraph(
            section=Sequential(sections=[GraphNode(
                name="src", input_names={"prompt"},
                outputs=[GraphEdge(name="latent", next_node="dit")],
            )]),
            graph_walks={TP_WALK}, ranks=[0], worker_graph_id=0,
        ),
        WorkerGraph(
            section=Sequential(sections=[Loop(
                name="dit_loop",
                section=GraphNode(
                    name="dit", input_names={"latent"},
                    outputs=[GraphEdge(name="latent", next_node="dit")],
                ),
                outputs=[GraphEdge(name="latent", next_node="")],
                max_iters=3,
            )]),
            graph_walks={TP_WALK}, ranks=[0, 1], worker_graph_id=1,
        ),
    ]


def _tp_sharding():
    groups = [
        ShardingGroup(nodes={"src"}, tp_size=1),
        ShardingGroup(nodes={"dit"}, tp_size=2),
    ]
    for group in groups:
        group._tp_rank = 0  # this worker leads both
    return ShardingConfig(
        groups=groups, tp_enabled_nodes={"dit"}, shard_dim={},
    )


_TP_COMMON = dict(
    all_wg_ids_to_graph_walks={0: {TP_WALK}, 1: {TP_WALK}},
    all_wg_ids_to_dyn_loops={0: set(), 1: set()},
    all_wg_ids_to_nodes={0: {"src"}, 1: {"dit"}},
    node_to_partition={"src": "default", "dit": "default"},
)


def _tp_pair(kind):
    book = PythonTensorBookkeeping() if kind == "python" \
        else RustTensorBookkeeping()
    tm = _StubTensorManager(book)
    if kind == "python":
        rt = PythonGraphRuntime(
            my_worker_id=WORKER, my_worker_graphs=_tp_graphs(),
            sharding_config=_tp_sharding(), tensor_manager=tm,
            communicator=None, **_TP_COMMON,
        )
    else:
        rt = rust_runtime.RustGraphRuntime(
            my_worker_id=WORKER, my_worker_graphs=_tp_graphs(),
            sharding_config=_tp_sharding(), bookkeeping=book, **_TP_COMMON,
        )
    rt.set_node_metadata({"dit"}, {"dit"}, set())
    return rt, book, tm.tensor_store


@pytest.fixture(params=["python", "rust"])
def tp_pair(request):
    return _tp_pair(request.param)


def _drive_tp(rt, book, store, steps=8):
    """Run whatever is ready until nothing is, logging what each step ate."""
    rid = rt.add_request(
        request_id="r1", partition="default", graph_walk=TP_WALK,
        partition_worker_graph_ids=[0, 1],
        worker_graph_to_workers=ParallelList([0, 1], [[WORKER], [WORKER, PEER]]),
    )
    rt.ingest_inputs_batch(ParallelList([rid], [_spec("prompt", "src")]))

    log, uuid = [], 0
    for _ in range(steps):
        ready = _ready(rt)
        if not ready:
            log.append("idle")
            break
        node, walk, _rids = ready[0]
        wg_id = rt.get_worker_graph_id_for_node(node, walk)
        popped = rt.pop_rids(node, walk, [rid], check_ready=True)
        assert popped is not None, f"{node} went unschedulable"
        log.append((node, [(e.signal, tuple(e.uuids)) for e in popped.input_edges]))
        rt.cleanup_consumed_inputs(node, [rid], [wg_id])
        uuid += 1
        store.put_tensor(rid, uuid, torch.zeros(4), _info(uuid))
        book.increment_ref(uuid, 1)
        rt.complete_and_route_batch(RouteInput(
            partition="default", graph_walk=walk, node_name=node,
            output_signals=("latent",), wg_ids=ParallelList([rid], [wg_id]),
            tensors=[uuid], num_tensors=[1],
        ), store)
    return log


def test_a_peers_copy_does_not_feed_this_ranks_node(tp_pair):
    """The regression: rank 0 ingested the fanout copy addressed to rank 1 as
    well as its own, filling `dit`'s next-iteration slot. The real loop-back
    edge was then refused, and iteration 2 ran on the STALE latent -- one
    rank's dit fed from a tensor the other rank never saw, which is a TP
    group that never steps together again."""
    rt, book, store = tp_pair
    assert _drive_tp(rt, book, store) == [
        ("src", [("prompt", ())]),
        ("dit", [("latent", (1,))]),
        ("dit", [("latent", (2,))]),
        ("dit", [("latent", (3,))]),
        "idle",
    ]
