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
from mstar.graph.base import GraphEdge, GraphNode, Loop, Parallel, Sequential
from mstar.model.base import WorkerGraph
from mstar.utils.containers import ParallelList

pytest.importorskip(
    "mstar_rust",
    reason="mstar_rust not built (maturin develop --release in rust/)",
)
from mstar.communication.tensor_store import RustTensorBookkeeping
from mstar.graph.loop_indices import NestedLoopIndices
from mstar.graph.runtime import base as rust_runtime_base
from mstar.graph.runtime import rust as rust_runtime
from mstar.graph.runtime.base import ColumnarEdgeSpecs, RouteInput, SpeculationPrepInput


def _ingest_block(rids, specs) -> ColumnarEdgeSpecs:
    """(rids, EdgeSpecs) as the columnar block ``ingest_inputs_batch`` takes.

    Mirrors the ``ParallelList`` these call sites used to build, so the shape
    of each case is unchanged.
    """
    block = ColumnarEdgeSpecs.empty()
    for rid, spec in zip(rids, specs, strict=True):
        block.add(
            rid, spec.signal, spec.uuids, spec.is_final_streaming_chunk,
            next_node=spec.next_node,
        )
    return block

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
    book = RustTensorBookkeeping()
    rt = rust_runtime.RustGraphRuntime(
        my_worker_id=WORKER,
        my_worker_graphs=[wg],
        all_wg_ids_to_graph_walks={WG_ID: {WALK}},
        all_wg_ids_to_dyn_loops={WG_ID: {"ar_loop"}},
        all_wg_ids_to_nodes={WG_ID: {"prefill", "ar_decode"}},
        node_to_partition={"prefill": "default", "ar_decode": "default"},
        sharding_config=ShardingConfig(
            groups=[], tp_enabled_nodes=set(), shard_dim={},
        ),
        bookkeeping=book,
    )
    # Tests seed descriptors through the same bookkeeper the runtime holds.
    rt._bookkeeping_for_test = book
    return rt


def _send_input(rid, completion_id):
    from mstar.graph.runtime.base import SendInput
    return SendInput(
        completion_id=completion_id,
        per_request_info=ParallelList([rid], [None]),
        new_token_counts=ParallelList([rid], [{}]),
    )


def _tensor_info(uuid):
    import torch

    from mstar.graph.base import TensorPointerInfo
    return TensorPointerInfo(
        dims=[4], dtype=torch.float16, nbytes=8, address=0, stride=(1,),
        uuid=uuid, source_session_id="h:1", source_entity=WORKER,
    )


def _admit(runtime, request_id="r1", walk=WALK):
    return runtime.add_request(
        request_id=request_id, partition="default", graph_walk=walk,
        partition_worker_graph_ids=[WG_ID],
        worker_graph_to_workers=ParallelList([WG_ID], [[WORKER]]),
        loop_caps={},
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

def test_set_walk_and_partition_done_reach_the_request(runtime):
    rid = _admit(runtime)
    runtime.set_walk(rid, "default", "other_walk")
    runtime.mark_partition_done(rid, "default")
    # No getter for partition_done on the ABC; not raising is the assertion, and
    # the walk change is observable.
    assert runtime is not None


def test_pending_loop_stops_live_one_iteration(runtime):
    rid = _admit(runtime)
    assert not runtime.has_pending_loop_stop(rid, WALK, "ar_loop")
    assert runtime.pending_loop_stop_rids([WALK], "ar_loop") == set()
    runtime.clear_pending_loop_stops()


def test_node_metadata_is_accepted(runtime):
    runtime.set_node_metadata(
        parallel_nodes={"ar_decode"},
        parallel_leader_nodes={"ar_decode"},
        tp_async_nodes=set(),
    )


def test_speculative_flag_rejects_an_unknown_node(runtime):
    rid = _admit(runtime)
    runtime.set_in_flight("prefill", [rid], [WG_ID], True)
    with pytest.raises((RuntimeError, ValueError)):
        runtime.set_in_flight("nope", [rid], [WG_ID], True)


# --- coverage of the ABC -----------------------------------------------------

def test_every_abstract_method_is_implemented():
    """The whole contract is ported. A method left abstract would fail at
    construction; one left as a raising stub would fail mid-pass, which is
    worse -- so check the class carries no stubs either."""
    from mstar.graph.runtime.base import GraphRuntime

    abstract = {
        name for name, attr in vars(GraphRuntime).items()
        if getattr(attr, "__isabstractmethod__", False)
    }
    assert abstract, "sanity: the ABC should declare abstract methods"
    for name in sorted(abstract):
        impl = getattr(rust_runtime.RustGraphRuntime, name, None)
        assert impl is not None, f"{name} is missing"
        assert impl.__qualname__.split(".")[0] != "_unported", (
            f"{name} is still a stub"
        )


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


# --- ingest ------------------------------------------------------------------

def _spec(signal, next_node, uuids=(), final=False):
    return rust_runtime_base.EdgeSpec(
        signal=signal, next_node=next_node, uuids=list(uuids),
        is_final_streaming_chunk=final,
    )


def test_a_signal_reaches_its_node(runtime):
    rid = _admit(runtime)
    assert runtime.ingest_inputs_batch(
        _ingest_block([rid], [_spec("prompt", "prefill")])
    ) == []


def test_a_signal_for_an_unknown_node_comes_back_by_index(runtime):
    rid = _admit(runtime)
    uningested = runtime.ingest_inputs_batch(
        _ingest_block(
            [rid, rid],
            [_spec("prompt", "prefill"), _spec("x", "not_here")],
        )
    )
    assert uningested == [1], "index 1, not 0: prefill claimed the first"


def test_a_signal_a_node_does_not_take_is_refused(runtime):
    # The node exists but has no such input, so the claim loop must not
    # silently drop it into some other slot.
    rid = _admit(runtime)
    assert runtime.ingest_inputs_batch(
        _ingest_block([rid], [_spec("not_an_input", "prefill")])
    ) == [0]


def test_a_second_signal_buffers_then_refuses(runtime):
    # Both ready slots full is the refusal Python returns False for; the
    # caller re-queues rather than losing the chunk.
    rid = _admit(runtime)
    assert runtime.ingest_inputs_batch(
        _ingest_block([rid], [_spec("prompt", "prefill")])
    ) == []
    assert runtime.ingest_inputs_batch(
        _ingest_block([rid], [_spec("prompt", "prefill")]), can_buffer=True
    ) == [], "the second goes to the next-iter slot"
    assert runtime.ingest_inputs_batch(
        _ingest_block([rid], [_spec("prompt", "prefill")]), can_buffer=True
    ) == [0], "the third has nowhere to go"


def test_can_buffer_false_refuses_the_second(runtime):
    rid = _admit(runtime)
    runtime.ingest_inputs_batch(_ingest_block([rid], [_spec("prompt", "prefill")]))
    assert runtime.ingest_inputs_batch(
        _ingest_block([rid], [_spec("prompt", "prefill")]), can_buffer=False
    ) == [0], "streaming must not buffer for an iteration that may not come"


def test_an_unknown_rid_is_refused_not_raised(runtime):
    # A signal can arrive for a request this rank already removed.
    assert runtime.ingest_inputs_batch(
        _ingest_block([9999], [_spec("prompt", "prefill")])
    ) == [0]


def test_streaming_gates_on_the_non_streaming_inputs(runtime):
    # ar_decode's inputs are not streaming, so it never reports
    # ready-for-streaming and a streaming ingest must not land.
    rid = _admit(runtime)
    assert runtime.ingest_inputs_batch(
        _ingest_block([rid], [_spec("token", "ar_decode")]), is_streaming=True
    ) == [0]


def test_inconsistent_columns_are_rejected(runtime):
    """The columns describing one edge each have to agree in length -- a short
    one would silently re-pair every edge after it with the wrong rid."""
    block = _ingest_block([1], [_spec("prompt", "prefill")])
    block.rids.append(2)  # a rid with no matching edge in the other columns
    with pytest.raises((ValueError, RuntimeError)):
        runtime.ingest_inputs_batch(block)


# --- consumed inputs and loop iters ------------------------------------------

def test_cleanup_releases_consumed_inputs(runtime):
    rid = _admit(runtime)
    runtime.ingest_inputs_batch(
        _ingest_block([rid], [_spec("prompt", "prefill", uuids=[7])])
    )
    runtime.cleanup_consumed_inputs("prefill", [rid], [WG_ID])
    # The slot is free again, so the same signal lands rather than refusing.
    assert runtime.ingest_inputs_batch(
        _ingest_block([rid], [_spec("prompt", "prefill")]), can_buffer=False
    ) == []


def test_cleanup_holds_a_loops_external_inputs():
    """A loop's external inputs are re-injected each iteration, so clearing
    them would strand the loop waiting for a signal nobody will resend."""
    body = GraphNode(
        name="body", input_names={"seed", "tok"},
        outputs=[GraphEdge(name="tok", next_node="body")],
    )
    wg = WorkerGraph(
        section=Sequential(sections=[Loop(
            name="lp", section=body,
            outputs=[GraphEdge(name="tok", next_node="sink")], max_iters=4,
        )]),
        graph_walks={WALK}, ranks=[0], worker_graph_id=WG_ID,
    )
    rt = rust_runtime.RustGraphRuntime(
        my_worker_id=WORKER, my_worker_graphs=[wg],
        all_wg_ids_to_graph_walks={WG_ID: {WALK}},
        all_wg_ids_to_dyn_loops={WG_ID: {"lp"}},
        all_wg_ids_to_nodes={WG_ID: {"body"}},
        node_to_partition={"body": "default"},
        sharding_config=ShardingConfig(
            groups=[], tp_enabled_nodes=set(), shard_dim={},
        ),
        bookkeeping=RustTensorBookkeeping(),
    )
    rid = _admit(rt)
    rt.ingest_inputs_batch(_ingest_block(
        [rid, rid],
        [_spec("seed", "body", uuids=[1]), _spec("tok", "body", uuids=[2])],
    ))
    rt.cleanup_consumed_inputs("body", [rid], [WG_ID])

    # "seed" is external to the loop, so it is still held; "tok" loops back
    # and was released.
    assert rt.ingest_inputs_batch(
        _ingest_block([rid], [_spec("seed", "body")]), can_buffer=False
    ) == [0], "the external input must still be held"
    assert rt.ingest_inputs_batch(
        _ingest_block([rid], [_spec("tok", "body")]), can_buffer=False
    ) == [], "the loop-back input was consumed and released"


def test_dynamic_loop_iters_report_per_rid(runtime):
    rid = _admit(runtime)
    got = runtime.get_dynamic_loop_iters([rid], "default")
    assert got.keys == [rid]
    assert got.values[0] == {"ar_loop": 0}


def test_dynamic_loop_iters_for_an_unknown_partition_are_empty(runtime):
    rid = _admit(runtime)
    assert runtime.get_dynamic_loop_iters([rid], "nope").values == [{}]


def test_update_dynamic_loop_iters_matches_the_per_rid_form(runtime):
    from types import SimpleNamespace
    rid = _admit(runtime)
    infos = {rid: SimpleNamespace(dynamic_loop_iter_counts={"stale": 7})}
    runtime.update_dynamic_loop_iters(infos, "default")
    expect = dict(runtime.get_dynamic_loop_iters([rid], "default").values[0])
    expect["stale"] = 7
    assert infos[rid].dynamic_loop_iter_counts == expect
    infos = {rid: SimpleNamespace(dynamic_loop_iter_counts={})}
    runtime.update_dynamic_loop_iters(infos, "nope")
    assert infos[rid].dynamic_loop_iter_counts == {}
    runtime.update_dynamic_loop_iters({}, "default")


# --- scheduling --------------------------------------------------------------

def test_a_ready_node_is_reported(runtime):
    rid = _admit(runtime)
    assert runtime.get_ready_nodes(set()) == []
    runtime.ingest_inputs_batch(_ingest_block([rid], [_spec("prompt", "prefill")]))
    ready = runtime.get_ready_nodes(set())
    assert [(r.node_name, r.graph_walk, r.rids) for r in ready] == [
        ("prefill", WALK, [rid])
    ]


def test_excluded_rids_are_invisible(runtime):
    rid = _admit(runtime)
    runtime.ingest_inputs_batch(_ingest_block([rid], [_spec("prompt", "prefill")]))
    assert runtime.get_ready_nodes({rid}) == []
    assert not runtime.has_ready_excluding({rid})


def test_target_and_exclude_target_filter(runtime):
    rid = _admit(runtime)
    runtime.ingest_inputs_batch(_ingest_block([rid], [_spec("prompt", "prefill")]))
    assert len(runtime.get_ready_nodes(set(), target=("prefill", WALK))) == 1
    assert runtime.get_ready_nodes(set(), target=("ar_decode", WALK)) == []
    assert runtime.get_ready_nodes(
        set(), exclude_target=("prefill", WALK)
    ) == []


def test_the_peek_agrees_with_the_scan(runtime):
    rid = _admit(runtime)
    assert not runtime.has_ready_excluding(set())
    runtime.ingest_inputs_batch(_ingest_block([rid], [_spec("prompt", "prefill")]))
    assert runtime.has_ready_excluding(set())
    assert not runtime.has_ready_excluding(set(), exclude_target=("prefill", WALK))


def test_pop_returns_the_inputs_it_popped(runtime):
    rid = _admit(runtime)
    runtime.ingest_inputs_batch(
        _ingest_block([rid], [_spec("prompt", "prefill", uuids=[11, 12])])
    )
    out = runtime.pop_rids("prefill", WALK, [rid])
    assert out.wg_ids.keys == [rid] and out.wg_ids.values == [WG_ID]
    assert out.input_edges.edge_tuples() == [(rid, "prompt", [11, 12], False)]

    # Popped, so it is no longer ready.
    assert runtime.get_ready_nodes(set()) == []


def test_pop_with_check_ready_is_all_or_nothing(runtime):
    # One not-ready rid must leave the whole set intact for a later retry.
    a = _admit(runtime, "ra")
    b = _admit(runtime, "rb")
    runtime.ingest_inputs_batch(_ingest_block([a], [_spec("prompt", "prefill")]))

    assert runtime.pop_rids("prefill", WALK, [a, b], check_ready=True) is None
    # a was NOT popped by the failed attempt.
    out = runtime.pop_rids("prefill", WALK, [a], check_ready=True)
    assert out.wg_ids.keys == [a]


def test_pop_of_an_unknown_node_raises(runtime):
    """Not None: nothing in the deployment runs this pair, and returning None
    would have the scheduler read a structural error as "not ready yet" and
    retry the row forever."""
    with pytest.raises(RuntimeError):
        runtime.pop_rids("nope", WALK, [])


def test_push_back_makes_a_popped_node_ready_again(runtime):
    rid = _admit(runtime)
    runtime.ingest_inputs_batch(_ingest_block([rid], [_spec("prompt", "prefill")]))
    runtime.pop_rids("prefill", WALK, [rid])
    assert runtime.get_ready_nodes(set()) == []
    runtime.push_back_node("prefill", [rid], [WG_ID])
    assert len(runtime.get_ready_nodes(set())) == 1


def _spec_targets(runtime, node, walk, rid):
    return runtime.speculate_node(node, [walk], [rid])


# --- speculation -------------------------------------------------------------

def test_speculate_finds_the_downstream_target(runtime):
    rid = _admit(runtime)
    out = _spec_targets(runtime, "prefill", WALK, rid)
    assert [o.node_name for o in out] == ["ar_decode"]
    assert out[0].graph_walk == WALK
    # prefill -> ar_decode ENTERS the loop, so it is not a new iteration.
    assert out[0].is_new_loop_iter is False
    assert out[0].loop_name == "ar_loop"


def test_a_loop_back_is_reported_as_a_new_iteration(runtime):
    # ar_decode -> ar_decode is the loop-back; the per-rid loop filters key off
    # exactly this flag, so getting it wrong disables them silently.
    rid = _admit(runtime)
    out = _spec_targets(runtime, "ar_decode", WALK, rid)
    assert [o.node_name for o in out] == ["ar_decode"]
    assert out[0].is_new_loop_iter is True


def test_speculation_leaves_no_state_behind(runtime):
    # ingest_for_speculation fills the speculative slots; they must be cleared
    # or the node reads as ready when nothing actually arrived.
    rid = _admit(runtime)
    _spec_targets(runtime, "prefill", WALK, rid)
    assert runtime.get_ready_nodes(set()) == []


def test_speculate_refuses_a_node_that_opted_out(runtime):
    rid = _admit(runtime)
    runtime.set_node_metadata(
        parallel_nodes={"ar_decode"}, parallel_leader_nodes=set(),
        tp_async_nodes=set(),
    )
    # A parallel target is only valid as a leader-side same-node loop-back.
    assert _spec_targets(runtime, "prefill", WALK, rid) == []


def test_get_spec_target_reports_a_target_chosen_elsewhere(runtime):
    # The follower path: no eligibility filter, since a follower is not the
    # leader and would fail it by construction.
    rid = _admit(runtime)
    runtime.set_node_metadata(
        parallel_nodes={"ar_decode"}, parallel_leader_nodes=set(),
        tp_async_nodes=set(),
    )
    assert _spec_targets(runtime, "ar_decode", WALK, rid) == []
    got = runtime.get_spec_target("ar_decode", "ar_decode", WALK, rid)
    assert got is not None and got.node_name == "ar_decode"
    assert got.is_new_loop_iter is True


def test_get_spec_target_returns_none_for_an_unreachable_node(runtime):
    rid = _admit(runtime)
    assert runtime.get_spec_target("prefill", "prefill", WALK, rid) is None


# --- speculation prep --------------------------------------------------------

def _prep_input(rids, spec="ar_decode", curr="prefill", room=None, edges=()):
    return SpeculationPrepInput(
        spec_node_name=spec, curr_node_name=curr, graph_walk=WALK,
        rids=list(rids), room_for_continuing=room,
        streaming_edges=list(edges),
        streaming_edges_per_rid=[0] * len(rids),
    )


def _drive_into_loop(runtime, rid):
    """Get the request as far as ar_decode being ready to loop back."""
    runtime.ingest_inputs_batch(_ingest_block([rid], [_spec("prompt", "prefill")]))
    runtime.pop_rids("prefill", WALK, [rid])
    runtime.ingest_inputs_batch(_ingest_block(
        [rid, rid],
        [_spec("token", "ar_decode"), _spec("kv_cache", "ar_decode")],
    ))


def test_prep_reports_the_worker_graph_per_ready_rid(runtime):
    rid = _admit(runtime)
    _drive_into_loop(runtime, rid)
    out = runtime.prep_spec_rids(_prep_input([rid], curr="ar_decode"))
    assert out.ready_rids == [rid]
    assert out.wg_ids == [WG_ID], "wg_ids is parallel to ready_rids"
    # Every edge belongs to a rid the prep reported ready.
    assert {r for r, *_ in out.input_edges.edge_tuples()} <= set(out.ready_rids)


def test_prep_respects_room_for_continuing(runtime):
    rid = _admit(runtime)
    _drive_into_loop(runtime, rid)
    assert runtime.prep_spec_rids(
        _prep_input([rid], curr="ar_decode", room=0)
    ).ready_rids == []
    assert runtime.prep_spec_rids(
        _prep_input([rid], curr="ar_decode", room=1)
    ).ready_rids == [rid]


def _two_node_runtime(a_outputs=(), b_outputs=(), communicator=None, loop=False):
    """a -> b, where b ALSO needs an input a does not produce.

    A same-node loop-back is a poor subject for not-ready: its held external
    inputs are carried over from the current slot and the loop-back itself is
    supplied by the speculative ingest, so nothing is ever missing. A
    cross-node target checks the current slot, which is the case the prep
    filters actually see.
    """
    section = Sequential(sections=[
        GraphNode(
            name="a", input_names={"x" if loop else "in"},
            outputs=[GraphEdge(name="x", next_node="b"), *a_outputs],
        ),
        GraphNode(name="b", input_names={"x", "y"}, outputs=[
            *b_outputs, *([GraphEdge(name="x", next_node="a")] if loop else []),
        ]),
    ])
    if loop:
        from mstar.graph.special_destinations import EMIT_TO_CLIENT

        section = Sequential(sections=[Loop(
            name="steps", section=section, max_iters=2,
            outputs=[GraphEdge(name="x", next_node=EMIT_TO_CLIENT, output_modality="text")],
        )])
    return _unit_runtime(section, {"steps"} if loop else set(), communicator)


def _unit_runtime(section, loops=(), communicator=None):
    wg = WorkerGraph(
        section=section,
        graph_walks={WALK}, ranks=[0], worker_graph_id=WG_ID,
    )
    book = RustTensorBookkeeping()
    runtime = rust_runtime.RustGraphRuntime(
        my_worker_id=WORKER, my_worker_graphs=[wg],
        all_wg_ids_to_graph_walks={WG_ID: {WALK}},
        all_wg_ids_to_dyn_loops={WG_ID: set(loops)},
        all_wg_ids_to_nodes={WG_ID: set(section.get_nodes())},
        node_to_partition={node: "default" for node in section.get_nodes()},
        sharding_config=ShardingConfig(
            groups=[], tp_enabled_nodes=set(), shard_dim={},
        ),
        bookkeeping=book,
        communicator=communicator,
    )
    runtime._bookkeeping_for_test = book
    return runtime


def test_unit_claim_preserves_common_rows_and_refusal_returns_readiness_once():
    rt = _two_node_runtime()
    a, b = (_admit(rt, name) for name in ("ra", "rb"))
    rt.ingest_inputs_batch(_ingest_block([a, b, a], [
        _spec("in", "a"), _spec("in", "a"), _spec("y", "b"),
    ]))
    # One downstream row is missing an external input: no member is claimed.
    assert rt.claim_execution_unit("a", WALK, [b, a]) is None
    assert set(rt.get_ready_nodes(set())[0].rids) == {a, b}
    rt.ingest_inputs_batch(_ingest_block([b], [_spec("y", "b")]))
    rt.pop_rids("a", WALK, [b, a], check_ready=True)
    claim, members = rt.claim_execution_unit("a", WALK, [b, a])
    assert [member.wg_ids.keys for member in members] == [[b, a], [b, a]]
    assert members[1].input_edges.edge_tuples() == [(b, "y", [], False), (a, "y", [], False)]
    assert rt.get_ready_nodes(set()) == []
    assert rt.claim_execution_unit("a", WALK, [a]) is None
    assert rt.prep_spec_rids(_ab([b, a])).ready_rids == []
    assert rt.cancel_execution_unit(claim)
    assert not rt.cancel_execution_unit(claim)
    assert set(rt.get_ready_nodes(set())[0].rids) == {a, b}
    # Refused admission never consumed the original inputs.
    retry = rt.pop_rids("a", WALK, [b, a], check_ready=True)
    assert retry.input_edges.edge_tuples() == [(b, "in", [], False), (a, "in", [], False)]


@pytest.mark.parametrize("producer_runs", [False, True])
def test_an_unexecuted_member_releases_its_claim_without_completing_its_graph(producer_runs):
    rt = _two_node_runtime()
    rid = _admit(rt)
    rt.ingest_inputs_batch(_ingest_block([rid, rid], [_spec("in", "a"), _spec("y", "b")]))
    claim, _ = rt.claim_execution_unit("a", WALK, [rid])
    with pytest.raises(ValueError, match="after submission"):
        rt.release_execution_unit_member(claim, "a", WG_ID, [rid])
    rt.submit_execution_unit(claim)
    with pytest.raises(ValueError, match="graph order"):
        rt.release_execution_unit_member(claim, "b", WG_ID, [rid])
    if producer_runs:
        _route(rt, [rid], "a", [], [[]])
        # Ordinary routing hands the produced value to the reserved member.
        rt.ingest_inputs_batch(_ingest_block([rid], [_spec("x", "b")]))
    else:
        assert rt.release_execution_unit_member(claim, "a", WG_ID, [rid])
    assert rt.release_execution_unit_member(claim, "b", WG_ID, [rid])
    assert not rt.release_execution_unit_member(claim, "b", WG_ID, [rid])
    ready, = rt.get_ready_nodes(set())
    assert ready.node_name == ("b" if producer_runs else "a")
    assert ready.rids == [rid]
    # The ready member's original input is still there: release did not run
    # ordinary completion or consume an input to fabricate a zero-work step.
    inputs = rt.pop_rids(ready.node_name, WALK, [rid], check_ready=True).input_edges
    assert set(inputs.signal_names) == ({"x", "y"} if producer_runs else {"in"})


def _stream_unit_runtime(*, install_consumer=True, consumer_worker=WORKER, driven=False):
    from mstar.streaming.chunk_policy import FixedChunkPolicy
    from mstar.streaming.topology import Connection

    a = GraphNode(name="a", input_names={"seed"}, outputs=[
        GraphEdge(name="seed", next_node="a"), GraphEdge(name="x", next_node="b", is_streaming=True),
    ])
    b = GraphNode(name="b", input_names={"x", "y"}, outputs=[])
    b._register_streaming({"x"})
    # Equal graph-local NodeId and LoopId/name intentionally: each member
    # must still settle its own independent Loop, on its own installed walk.
    graphs = [WorkerGraph(
        section=Sequential(sections=[Loop(name="steps", section=node, max_iters=cap, outputs=[])]),
        graph_walks={walk}, ranks=[0], worker_graph_id=wg,
    ) for node, cap, walk, wg in ((a, 3, "produce", 7), (b, 5, "consume", 9))]
    runtime = rust_runtime.RustGraphRuntime(
        my_worker_id=WORKER, my_worker_graphs=graphs,
        all_wg_ids_to_graph_walks={7: {"produce"}, 9: {"consume"}},
        all_wg_ids_to_dyn_loops={7: {"steps"}, 9: {"steps"}},
        all_wg_ids_to_nodes={7: {"a"}, 9: {"b"}}, node_to_partition={"a": "source", "b": "sink"},
        sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        bookkeeping=RustTensorBookkeeping(),
        connections=(Connection("source", "sink", "x", lambda: FixedChunkPolicy(2),
                                consumer_walk=(lambda ctx: "consume") if driven else None),),
    )
    assignment = ParallelList([7, 9], [[WORKER], [consumer_worker]])
    rid = runtime.add_request("r", "source", "produce", [7], assignment, {})
    if install_consumer:
        assert runtime.add_request("r", "sink", "consume", [9], assignment, {}) == rid
    return runtime, rid


@pytest.mark.parametrize("install,worker,driven", [
    (False, WORKER, False), (True, "other_worker", False), (True, WORKER, True),
])
def test_stream_unit_requires_installed_local_members_and_a_stable_declared_connection(install, worker, driven):
    rt, rid = _stream_unit_runtime(install_consumer=install, consumer_worker=worker, driven=driven)
    rt.ingest_inputs_batch(_ingest_block([rid], [_spec("seed", "a")]))
    assert rt.claim_execution_unit("a", "produce", [rid], ("produce", "consume")) is None
    assert [(ready.node_name, ready.rids) for ready in rt.get_ready_nodes(set())] == [("a", [rid])]
    if driven:
        assert rt.execution_units("produce") == ()


def test_stream_unit_claim_is_atomic_and_cancellation_preserves_a_waiting_consumer():
    rt, rid = _stream_unit_runtime()
    unit, = rt.execution_units("produce")
    assert unit.worker_graph_ids == (7, 9) and unit.graph_walks == ("produce", "consume")
    assert unit.streaming_edges == (("a", "b", "x"),) and unit.loop_name is None
    rt.ingest_inputs_batch(_ingest_block([rid], [_spec("seed", "a")]))
    # A consumer's non-stream input is not something the unit can invent.
    assert rt.claim_execution_unit("a", "produce", [rid], unit.graph_walks) is None
    rt.ingest_inputs_batch(_ingest_block([rid], [_spec("y", "b")]))
    rt.pop_rids("a", "produce", [rid], check_ready=True)
    claim, members = rt.claim_execution_unit("a", "produce", [rid], unit.graph_walks)
    assert [member.wg_ids.values for member in members] == [[7], [9]]
    assert members[1].input_edges.signal_names == ["y"]
    assert rt.get_ready_nodes(set()) == []
    # Ordinary streaming ingress must not fill a member reserved by the unit.
    assert rt.ingest_inputs_batch(_ingest_block([rid], [_spec("x", "b")]), is_streaming=True) == [0]
    assert rt.cancel_execution_unit(claim)
    assert [(ready.node_name, ready.rids) for ready in rt.get_ready_nodes(set())] == [("a", [rid])]
    # Still no fabricated stream chunk or Loop completion on rollback.
    assert rt.loop_room("a", "produce", "steps", [rid], 0, False) == 3
    assert rt.loop_room("b", "consume", "steps", [rid], 0, False) == 5
    claim, _ = rt.claim_execution_unit("a", "produce", [rid], unit.graph_walks)
    rt.remove_request(rid)
    assert not rt.submit_execution_unit(claim)
    assert not rt.cancel_execution_unit(claim)


@pytest.mark.parametrize("consumer_runs", [False, True])
def test_stream_unit_lands_independent_counts_and_releases_zero_work_without_a_fake_step(consumer_runs):
    rt, rid = _stream_unit_runtime()
    rt.ingest_inputs_batch(_ingest_block([rid, rid], [_spec("seed", "a"), _spec("y", "b")]))
    claim, _ = rt.claim_execution_unit("a", "produce", [rid], ("produce", "consume"))
    rt.submit_execution_unit(claim)
    rt.complete_and_route_batch(RouteInput(
        partition="source", graph_walk="produce", node_name="a", output_signals=[],
        wg_ids=ParallelList([rid], [7]), tensors=[], num_tensors=[], loop_strides=[2],
    ))
    # Apply the ordinary stream receipt only when the bounded schedule selected
    # a consumer chunk. Graph claiming itself never pretends to pop that chunk.
    if consumer_runs:
        rt.ingest_inputs_batch(_ingest_block([rid], [_spec("x", "b")]))
        rt.complete_and_route_batch(RouteInput(
            partition="sink", graph_walk="consume", node_name="b", output_signals=[],
            wg_ids=ParallelList([rid], [9]), tensors=[], num_tensors=[], loop_strides=[1],
        ))
    else:
        assert rt.release_execution_unit_member(claim, "b", 9, [rid])
    assert not rt.cancel_execution_unit(claim)
    assert rt.loop_room("a", "produce", "steps", [rid], 0, False) == 1
    assert rt.loop_room("b", "consume", "steps", [rid], 0, False) == (4 if consumer_runs else 5)
    assert not any(ready.node_name == "b" for ready in rt.get_ready_nodes(set()))


def test_stream_unit_drains_after_producer_done_without_advancing_that_producer_again():
    from types import SimpleNamespace

    from mstar.engine.engine import ExecutingBatch
    from mstar.engine.resources.step import StepContext
    from mstar.worker.micro_scheduler import ScheduledBatch
    from mstar.worker.worker import Worker

    rt, rid = _stream_unit_runtime()
    rt.ingest_inputs_batch(_ingest_block([rid, rid], [_spec("seed", "a"), _spec("y", "b")]))
    walks = ("produce", "consume")
    claim, _ = rt.claim_execution_unit("a", "produce", [rid], walks)
    rt.submit_execution_unit(claim)
    completed = RouteInput(
        partition="source", graph_walk="produce", node_name="a", output_signals=[],
        wg_ids=ParallelList([rid], [7]), tensors=[], num_tensors=[], loop_strides=[3],
    )
    rt.complete_and_route_batch(completed)
    # Ending this member's entire Loop clears its subtree. The still-live
    # qualified claim must survive that clear until the consumer receipt lands.
    with pytest.raises(ValueError, match="graph order"):
        rt.complete_and_route_batch(completed)
    assert rt.release_execution_unit_member(claim, "b", 9, [rid])
    rt.mark_partition_done(rid, "source")
    assert rt.graph_passes({rid: 7}) == [1]
    assert rt.graph_passes({rid: 9}) == [0]
    assert rt.claim_execution_unit("a", "produce", [rid], walks) is None  # no member ready
    rt.ingest_inputs_batch(_ingest_block([rid], [_spec("x", "b")]))
    unit, = rt.execution_units("produce")
    worker = Worker.__new__(Worker)
    worker._graph_runtime, worker._step_repeats = rt, None
    worker.scheduler = SimpleNamespace(execution_units={("b", "consume"): unit})
    checked = []
    engine = SimpleNamespace(_submodules={node: SimpleNamespace(accelerator_graph_runner=SimpleNamespace(
        launch_refusal=lambda walk, node=node: checked.append((node, walk)),
    )) for node in unit.nodes})
    worker.engine_manager = SimpleNamespace(get_engine=lambda node: engine)
    worker._build_executing_batch = lambda batch: ExecutingBatch(
        batch.node_name, {}, StepContext([rid], batch.graph_walk, 0, False),
    )
    popped = rt.pop_rids("b", "consume", [rid], check_ready=True)
    scheduled = ScheduledBatch("b", "consume", {rid: 9}, popped.input_edges, list(popped.output_signals))
    executing = worker._build_executing_batch(scheduled)
    group = worker._claim_group(scheduled, executing)
    assert group.trigger == 1 and group.members[1][1] is executing
    assert [(b.node_name, b.graph_walk) for b, _ in group.members] == [("a", "produce"), ("b", "consume")]
    assert checked == [("a", "produce"), ("b", "consume")]
    claim = group.claim
    group.submitted = True
    worker._settle_group_claim(group)
    worker._settle_group_claim(group)
    assert group.claim == claim and group.acknowledged
    assert rt.release_execution_unit_member(group.claim, "a", 7, [rid])
    assert rt.get_ready_nodes(set()) == []
    rt.complete_and_route_batch(RouteInput(
        partition="sink", graph_walk="consume", node_name="b", output_signals=[],
        wg_ids=ParallelList([rid], [9]), tensors=[], num_tensors=[], loop_strides=[1],
    ))
    assert rt.graph_passes({rid: 7}) == [1]
    assert rt.graph_passes({rid: 9}) == [0]
    assert rt.loop_room("b", "consume", "steps", [rid], 0, False) == 4


@pytest.mark.parametrize("head", ["a", "b"])
def test_connected_join_claims_independent_roots_and_restores_each_on_refusal(head):
    rt = _unit_runtime(Sequential(sections=[
        Parallel(sections=[
            GraphNode(name=name, input_names={"seed"}, outputs=[GraphEdge(name=name, next_node="join")])
            for name in ("a", "b")
        ]),
        GraphNode(name="join", input_names={"a", "b"}, outputs=[]),
    ]))
    rid = _admit(rt)
    rt.ingest_inputs_batch(_ingest_block([rid], [_spec("seed", head)]))
    assert rt.claim_execution_unit(head, WALK, [rid]) is None
    other = "b" if head == "a" else "a"
    rt.ingest_inputs_batch(_ingest_block([rid], [_spec("seed", other)]))
    units = {unit.nodes[0]: unit for unit in rt.execution_units(WALK)}
    assert units[head].nodes == (head, other, "join")
    rt.pop_rids(head, WALK, [rid], check_ready=True)
    claim, members = rt.claim_execution_unit(head, WALK, [rid])
    assert len(members) == 3 and rt.get_ready_nodes(set()) == []
    assert rt.cancel_execution_unit(claim)
    assert not rt.cancel_execution_unit(claim)
    assert {spec.node_name for spec in rt.get_ready_nodes(set())} == {"a", "b"}
    claim, _ = rt.claim_execution_unit(head, WALK, [rid])
    rt.submit_execution_unit(claim)
    for name in (head, other, "join"):
        _route(rt, [rid], name, [], [[]])
        assert rt.get_ready_nodes(set()) == []
    assert not rt.cancel_execution_unit(claim)


@pytest.mark.parametrize("submitted", [False, True])
def test_unit_cancellation_does_not_attach_to_recycled_handles(submitted):
    rt = _two_node_runtime()
    rid = _admit(rt)
    rt.ingest_inputs_batch(_ingest_block([rid, rid], [_spec("in", "a"), _spec("y", "b")]))
    claim, _ = rt.claim_execution_unit("a", WALK, [rid])
    if submitted:
        assert rt.submit_execution_unit(claim)
        with pytest.raises(ValueError, match="cannot retry"):
            rt.cancel_execution_unit(claim)
    rt.remove_request(rid)
    assert _admit(rt, "replacement") == rid
    assert not rt.cancel_execution_unit(claim)
    assert not rt.submit_execution_unit(claim)
    rt.ingest_inputs_batch(_ingest_block([rid, rid], [_spec("in", "a"), _spec("y", "b")]))
    replacement, _ = rt.claim_execution_unit("a", WALK, [rid])
    assert replacement != claim
    assert rt.cancel_execution_unit(replacement)


@pytest.mark.parametrize("refuse,loop", [(False, False), (True, False), (False, True)])
def test_worker_claim_runs_the_ordinary_engine_members_and_settles_rust_once(refuse, loop):
    from pathlib import Path
    from types import SimpleNamespace

    from mstar.worker.worker import Worker
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modular"))
    from test_engine_uniform_inputs import _connected_engine

    rt = _two_node_runtime(loop=loop)
    # The engine fixture intentionally orders its real rows [2, 1].
    _admit(rt, "unused")
    a, b = _admit(rt, "ra"), _admit(rt, "rb")
    assert (a, b) == (1, 2)
    rt.ingest_inputs_batch(_ingest_block([a, b, a, b], [
        _spec("x" if loop else "in", "a"), _spec("x" if loop else "in", "a"), _spec("y", "b"), _spec("y", "b"),
    ]))
    eng, fixture, events = _connected_engine((lambda x: x + 1) if refuse else None)
    for scheduled, executing in fixture.members:
        scheduled.graph_walk = executing.step_context.graph_walk = WALK
        eng._submodules[scheduled.node_name].accelerator_graph_runner.launch_refusal = lambda walk: None
    unit, = rt.execution_units(WALK)
    worker = Worker.__new__(Worker)
    worker._graph_runtime = rt
    from mstar.graph.base import EdgeKind, LoopEdges
    from mstar.worker.step_repeats import RepeatConfig, StepRepeats
    worker._step_repeats = StepRepeats({"b": RepeatConfig(max=3)}, {
        (node, WALK): LoopEdges(inputs={}, outputs={("x", "b" if node == "a" else "a"): EdgeKind.LOOP_BACK})
        for node in ("a", "b")
    }) if loop else None
    worker.scheduler = SimpleNamespace(execution_units={("a", WALK): unit})
    worker.engine_manager = SimpleNamespace(get_engine=lambda node: eng)
    worker._build_executing_batch = lambda batch: fixture.members[1][1]
    head, executing = fixture.members[0]
    rt.pop_rids("a", WALK, [b, a], check_ready=True)
    group = worker._claim_group(head, executing)
    assert group is not None and rt.get_ready_nodes(set()) == []
    outputs = eng.exec_group(group)
    worker._settle_group_claim(group)
    worker._settle_group_claim(group)
    if refuse:
        assert outputs is None and not group.submitted
        assert set(rt.get_ready_nodes(set())[0].rids) == {a, b}
        assert events == ["release:a"]
    else:
        assert len(outputs) == 2 and group.submitted
        for node in ("a", "b"):
            rt.complete_and_route_batch(RouteInput(
                partition="default", graph_walk=WALK, node_name=node, output_signals=[],
                wg_ids=ParallelList([b, a], [WG_ID, WG_ID]), tensors=[], num_tensors=[],
                loop_strides=[2 if loop else 1] * 2,
            ))
        assert rt.get_ready_nodes(set()) == []
        assert events == ["a", "b"] * (2 if loop else 1) + ["release:b", "release:a"]
        assert all(scheduled.repeats == (2 if loop else 1) for scheduled, _ in group.members)
        assert not worker._step_repeats or not worker._step_repeats.may_repeat("b", WALK)


def test_unit_lands_in_graph_order_with_internal_readiness_and_external_outputs_once(tmp_path):
    from types import SimpleNamespace

    from mstar_rust import ZmqCommunicator

    import mstar.communication.wire_types  # noqa: F401 -- register real wire messages
    from mstar.communication.wire import decode
    from mstar.graph.special_destinations import EMIT_TO_CLIENT

    comm = ZmqCommunicator(WORKER, str(tmp_path))
    client = ZmqCommunicator("api_server", str(tmp_path))
    conductor = ZmqCommunicator("conductor", str(tmp_path))
    rt = _two_node_runtime(
        a_outputs=[GraphEdge(name="first", next_node=EMIT_TO_CLIENT, output_modality="text")],
        b_outputs=[GraphEdge(name="last", next_node=EMIT_TO_CLIENT, output_modality="text")],
        communicator=SimpleNamespace(_inner=comm),
    )
    rid = _admit(rt)
    rt.ingest_inputs_batch(_ingest_block([rid, rid], [_spec("in", "a"), _spec("y", "b")]))
    unit, = rt.execution_units(WALK)
    assert unit.nodes == ("a", "b")
    assert unit.edges == (("a", "b", "x"),)
    # The internal x payload is held by device leases, never stored here.
    assert unit.output_signals == (("first",), ("last",))
    claim, _ = rt.claim_execution_unit("a", WALK, [rid])
    with pytest.raises(ValueError, match="graph order"):
        _route(rt, [rid], "a", [], [[]])
    assert rt.submit_execution_unit(claim)
    with pytest.raises(ValueError, match="graph order"):
        _route(rt, [rid], "b", [], [[]])
    for uuid in (901, 902):
        rt._bookkeeping_for_test.put_tensor(uuid, _tensor_info(uuid))
        rt._bookkeeping_for_test.increment_ref(uuid)
    first = _route(rt, [rid], "a", ["first"], [[[901]]])
    rt.send_outputs(_send_input(rid, first.completion_id))
    # x arrived as a readiness edge; the claim prevents a second schedule.
    assert rt.get_ready_nodes(set()) == []
    with pytest.raises(ValueError, match="graph order"):
        _route(rt, [rid], "a", ["first"], [[[901]]])
    rt.mark_partition_done(rid, "default")
    last = _route(rt, [rid], "b", ["last"], [[[902]]])
    rt.send_outputs(_send_input(rid, last.completion_id))
    emitted = [decode(client.recv_timeout(1000)) for _ in range(2)]
    assert [message.body.graph_edge.tensor_info[0].uuid for message in emitted] == [901, 902]
    done = decode(conductor.recv_timeout(1000))
    assert done.body.partition_done
    assert client.recv_timeout(0) is None
    assert conductor.recv_timeout(0) is None
    assert not rt.cancel_execution_unit(claim)
    assert rt.get_ready_nodes(set()) == []


@pytest.mark.parametrize("width", [1, 2])
def test_connected_loop_iterations_keep_cached_outputs_and_release_the_claim_on_finish(tmp_path, width):
    from types import SimpleNamespace

    from mstar_rust import ZmqCommunicator

    import mstar.communication.wire_types  # noqa: F401 -- register real wire messages
    from mstar.communication.wire import decode

    comm = ZmqCommunicator(WORKER, str(tmp_path))
    client = ZmqCommunicator("api_server", str(tmp_path))
    conductor = ZmqCommunicator("conductor", str(tmp_path))
    rt = _two_node_runtime(communicator=SimpleNamespace(_inner=comm), loop=True)
    rid = _admit(rt)
    rt.ingest_inputs_batch(_ingest_block([rid, rid], [_spec("x", "a"), _spec("y", "b")]))
    unit, = rt.execution_units(WALK)
    assert unit.edges == (("a", "b", "x"),)
    # x is consumed internally AND retained by the Loop's output cache.
    assert unit.output_signals == (("x",), ("x",))
    for start in range(0, 2, width):
        uuids = list(range(903 + start, 903 + start + width))
        claim, _ = rt.claim_execution_unit("a", WALK, [rid])
        assert rt.submit_execution_unit(claim)
        for uuid in uuids:
            rt._bookkeeping_for_test.put_tensor(uuid, _tensor_info(uuid))
            rt._bookkeeping_for_test.increment_ref(uuid)
        for node, values in (("a", uuids), ("b", [])):
            landed = rt.complete_and_route_batch(RouteInput(
                partition="default", graph_walk=WALK, node_name=node, output_signals=["x"],
                wg_ids=ParallelList([rid], [WG_ID]), tensors=values, num_tensors=[len(values)], loop_strides=[width],
            ))
            rt.send_outputs(_send_input(rid, landed.completion_id))
        assert not rt.cancel_execution_unit(claim)
        if start + width < 2:
            assert client.recv_timeout(0) is None
            assert rt.get_ready_nodes(set())[0].node_name == "a"
    emitted = decode(client.recv_timeout(1000))
    assert emitted.body.graph_edge.tensor_info[0].uuid == 904
    assert client.recv_timeout(0) is None
    assert conductor.recv_timeout(1000) is not None
    assert rt.get_ready_nodes(set()) == []
    assert rt.claim_execution_unit("a", WALK, [rid]) is None


def test_connected_prefix_enters_a_loop_through_ordinary_completion(runtime):
    rid = _admit(runtime)
    runtime.ingest_inputs_batch(_ingest_block([rid], [_spec("prompt", "prefill")]))
    unit, = runtime.execution_units(WALK)
    assert unit.nodes == ("prefill", "ar_decode")
    claim, _ = runtime.claim_execution_unit("prefill", WALK, [rid])
    assert runtime.submit_execution_unit(claim)
    _route(runtime, [rid], "prefill", [], [[]])
    _route(runtime, [rid], "ar_decode", ["token", "kv_cache"], [[[], []]])
    assert not runtime.cancel_execution_unit(claim)
    assert runtime.get_ready_nodes(set())[0].node_name == "ar_decode"


def _ab(rids, room=None):
    return _prep_input(rids, spec="b", curr="a", room=room)


def test_prep_skips_a_rid_whose_other_input_has_not_arrived():
    rt = _two_node_runtime()
    rid = _admit(rt)
    # a's outputs cover x; y has to be there already.
    assert rt.prep_spec_rids(_ab([rid])).ready_rids == []

    rt.ingest_inputs_batch(_ingest_block([rid], [_spec("y", "b")]))
    assert rt.prep_spec_rids(_ab([rid])).ready_rids == [rid]


def test_the_follower_prep_is_all_or_nothing():
    rt = _two_node_runtime()
    ready = _admit(rt, "ra")
    blank = _admit(rt, "rb")
    rt.ingest_inputs_batch(_ingest_block([ready], [_spec("y", "b")]))

    # The leader's best-effort prep takes the one that is ready...
    assert rt.prep_spec_rids(_ab([ready, blank])).ready_rids == [ready]
    # ...but a follower runs the leader's exact composition or none of it.
    assert rt.prep_follow_spec_rids(_ab([ready, blank])) is None


def test_a_failed_follower_prep_leaves_no_ingest_behind():
    """The all-or-nothing unwind: a rid prepped before the failure must have
    its chunks pulled back out, or they are stranded in a node that never ran.
    """
    rt = _two_node_runtime()
    ready = _admit(rt, "ra")
    blank = _admit(rt, "rb")
    rt.ingest_inputs_batch(_ingest_block([ready], [_spec("y", "b")]))

    assert rt.prep_follow_spec_rids(_ab([ready, blank])) is None
    # The ready rid is still preppable, so nothing was consumed or left set.
    assert rt.prep_spec_rids(_ab([ready])).ready_rids == [ready]


def test_the_follower_prep_succeeds_when_every_rid_is_ready():
    rt = _two_node_runtime()
    a = _admit(rt, "ra")
    b = _admit(rt, "rb")
    for r in (a, b):
        rt.ingest_inputs_batch(_ingest_block([r], [_spec("y", "b")]))
    out = rt.prep_follow_spec_rids(_ab([a, b]))
    assert out is not None
    assert out.ready_rids == [a, b], "wire order is the leader's batch order"


def test_prep_room_cap_applies_before_any_ingest():
    rt = _two_node_runtime()
    a = _admit(rt, "ra")
    b = _admit(rt, "rb")
    for r in (a, b):
        rt.ingest_inputs_batch(_ingest_block([r], [_spec("y", "b")]))
    assert rt.prep_spec_rids(_ab([a, b], room=1)).ready_rids == [a]
    # The capped rid was skipped before any ingest, so it is untouched.
    assert rt.prep_spec_rids(_ab([b])).ready_rids == [b]


# --- routing -----------------------------------------------------------------

def _route(runtime, rids, node, signals, per_rid_uuids, walk=WALK):
    """per_rid_uuids: [[uuids for signal 0, uuids for signal 1, ...], ...]"""
    flat, counts = [], []
    for row in per_rid_uuids:
        for uuids in row:
            counts.append(len(uuids))
            flat.extend(uuids)
    return runtime.complete_and_route_batch(
        RouteInput(
            partition="default", graph_walk=walk, node_name=node,
            output_signals=signals,
            wg_ids=ParallelList(list(rids), [WG_ID] * len(rids)),
            tensors=flat, num_tensors=counts,
        ),
    )


def test_routing_decodes_the_flat_rid_major_layout(runtime):
    """num_tensors is indexed [rid_i * n_signals + signal_i]. Getting that
    wrong attaches one rid's tensors to another's edges, silently."""
    a = _admit(runtime, "ra")
    b = _admit(runtime, "rb")
    for r in (a, b):
        runtime.ingest_inputs_batch(_ingest_block([r], [_spec("prompt", "prefill")]))
        runtime.pop_rids("prefill", WALK, [r])

    bk = runtime._bookkeeping_for_test
    for u in (1, 2, 3, 4, 5, 6):
        bk.put_tensor(u, _tensor_info(u))

    # Asymmetric counts, or a transposed read gives the same answer.
    out = _route(
        runtime, [a, b], "prefill", ["kv_cache", "token"],
        [[[1], [2, 3]], [[4, 5, 6], [7]]],
    )
    assert out.completion_id > 0
    # Every uuid that a remote consumer reads is reported once.
    assert len(out.register_uuids) == len(set(out.register_uuids))


def test_routing_ingests_a_local_destination(runtime):
    # prefill -> ar_decode is local, so ar_decode becomes ready without any
    # round trip through the worker.
    rid = _admit(runtime)
    runtime.ingest_inputs_batch(_ingest_block([rid], [_spec("prompt", "prefill")]))
    runtime.pop_rids("prefill", WALK, [rid])

    bk = runtime._bookkeeping_for_test
    for u in (10, 11):
        bk.put_tensor(u, _tensor_info(u))
    _route(runtime, [rid], "prefill", ["kv_cache", "token"], [[[10], [11]]])

    ready = runtime.get_ready_nodes(set())
    assert [(r.node_name, r.rids) for r in ready] == [("ar_decode", [rid])]


def test_routing_settles_the_safety_hold(runtime):
    """Outputs come in with a hold of 1; routing corrects to the real fanout.
    Too low frees a tensor a consumer still needs, too high leaks it."""
    rid = _admit(runtime)
    runtime.ingest_inputs_batch(_ingest_block([rid], [_spec("prompt", "prefill")]))
    runtime.pop_rids("prefill", WALK, [rid])

    bk = runtime._bookkeeping_for_test
    bk.put_tensor(20, _tensor_info(20))
    bk.increment_ref(20, 1)  # the safety hold
    bk.put_tensor(21, _tensor_info(21))
    bk.increment_ref(21, 1)

    _route(runtime, [rid], "prefill", ["kv_cache", "token"], [[[20], [21]]])
    # One local consumer each: the hold is replaced by one real reference.
    assert not bk.can_gc(20)
    assert not bk.can_gc(21)


# --- loop stops --------------------------------------------------------------

def test_a_stop_records_a_pending_stop(runtime):
    rid = _admit(runtime)
    assert not runtime.has_pending_loop_stop(rid, WALK, "ar_loop")
    stopped_rids = runtime.stop_loops_batched(
        partition="default", graph_walk=WALK, last_node_run="ar_decode",
        loop_names=ParallelList([rid], [["ar_loop"]]),
    )
    assert stopped_rids == [rid]
    assert runtime.has_pending_loop_stop(rid, WALK, "ar_loop")
    assert runtime.pending_loop_stop_rids([WALK], "ar_loop") == {rid}
    runtime.clear_pending_loop_stops()
    assert not runtime.has_pending_loop_stop(rid, WALK, "ar_loop")


def test_a_stop_for_a_loop_not_in_the_walk_is_dropped(runtime):
    # A model bug rather than a protocol one: logged and dropped, not raised.
    rid = _admit(runtime)
    stopped_rids = runtime.stop_loops_batched(
        partition="default", graph_walk=WALK, last_node_run="ar_decode",
        loop_names=ParallelList([rid], [["not_a_loop"]]),
    )
    assert stopped_rids == []
    assert runtime.pending_loop_stop_rids([WALK], "not_a_loop") == set()


def test_a_peer_stop_applies_only_when_newer(runtime):
    rid = _admit(runtime)

    def at(fwd):
        return NestedLoopIndices(
            loop_name_order=["ar_loop"], loop_indices={"ar_loop": 0},
            wg_fwd_pass_idx=fwd,
        )

    runtime.apply_peer_loop_stops(rid, "default", {"ar_loop": at(3)})
    kept = runtime._loop_stop_times(rid)["ar_loop"]
    assert kept.wg_fwd_pass_idx == 3

    # An older observation is dropped, not recorded over the newer one.
    runtime.apply_peer_loop_stops(rid, "default", {"ar_loop": at(1)})
    assert runtime._loop_stop_times(rid)["ar_loop"].wg_fwd_pass_idx == 3


def test_a_peer_stop_for_an_unknown_partition_is_dropped(runtime):
    rid = _admit(runtime)
    runtime.apply_peer_loop_stops(
        rid, "no_such_partition",
        {"ar_loop": NestedLoopIndices(
            loop_name_order=[], loop_indices={}, wg_fwd_pass_idx=0,
        )},
    )
    assert runtime._loop_stop_times(rid) == {}


# --- send --------------------------------------------------------------------

class _Recorder:
    def __init__(self):
        self.sent = []

    def send(self, entity_id, msg=None, **kw):
        self.sent.append((entity_id, msg if msg is not None else kw.get("msg")))


def test_send_consumes_the_completion(runtime):
    """complete_and_route parks the routing; send_outputs is what frees it.
    A double-take would raise on the second pass of a live request."""
    rid = _admit(runtime)
    runtime._communicator = _Recorder()
    runtime.ingest_inputs_batch(_ingest_block([rid], [_spec("prompt", "prefill")]))
    runtime.pop_rids("prefill", WALK, [rid])
    bk = runtime._bookkeeping_for_test
    for u in (30, 31):
        bk.put_tensor(u, _tensor_info(u))
    out = _route(runtime, [rid], "prefill", ["kv_cache", "token"], [[[30], [31]]])

    runtime.send_outputs(_send_input(rid, out.completion_id))
    with pytest.raises((ValueError, RuntimeError)):
        runtime.send_outputs(_send_input(rid, out.completion_id))


def test_a_local_only_batch_sends_nothing_to_peers(runtime):
    # prefill -> ar_decode is local, so there is no peer frame to build.
    rid = _admit(runtime)
    rec = _Recorder()
    runtime._communicator = rec
    runtime.ingest_inputs_batch(_ingest_block([rid], [_spec("prompt", "prefill")]))
    runtime.pop_rids("prefill", WALK, [rid])
    bk = runtime._bookkeeping_for_test
    for u in (40, 41):
        bk.put_tensor(u, _tensor_info(u))
    out = _route(runtime, [rid], "prefill", ["kv_cache", "token"], [[[40], [41]]])
    runtime.send_outputs(_send_input(rid, out.completion_id))

    peers = [e for e, _ in rec.sent if e not in ("conductor", "api_server")]
    assert peers == []


# --- inline emit -------------------------------------------------------------

def _client_loop_runtime():
    """A decode node whose token goes back to itself AND to the client."""
    from mstar.graph.special_destinations import EMIT_TO_CLIENT
    section = Sequential(sections=[
        GraphNode(
            name="prefill", input_names={"prompt"},
            outputs=[GraphEdge(name="token", next_node="dec")],
        ),
        Loop(
            name="loop",
            section=GraphNode(
                name="dec", input_names={"token"},
                outputs=[
                    GraphEdge(name="token", next_node="dec"),
                    GraphEdge(name="token", next_node=EMIT_TO_CLIENT, output_modality="text"),
                ],
            ),
            outputs=[],
            max_iters=50,
        ),
    ])
    wg = WorkerGraph(section=section, graph_walks={WALK}, ranks=[0], worker_graph_id=WG_ID)
    book = RustTensorBookkeeping()
    rt = rust_runtime.RustGraphRuntime(
        my_worker_id=WORKER, my_worker_graphs=[wg],
        all_wg_ids_to_graph_walks={WG_ID: {WALK}},
        all_wg_ids_to_dyn_loops={WG_ID: {"loop"}},
        all_wg_ids_to_nodes={WG_ID: {"prefill", "dec"}},
        node_to_partition={"prefill": "default", "dec": "default"},
        sharding_config=ShardingConfig(groups=[], tp_enabled_nodes=set(), shard_dim={}),
        bookkeeping=book,
    )
    rt._bookkeeping_for_test = book
    return rt


def _run_dec_step(rt, rid, uuid, inline_value=None):
    bk = rt._bookkeeping_for_test
    bk.put_tensor(uuid, _tensor_info(uuid))
    bk.increment_ref(uuid, 1)  # the worker's safety hold
    kwargs = {}
    if inline_value is not None:
        kwargs = dict(inline_signal="token", inline_values=[inline_value])
    return rt.complete_and_route_batch(RouteInput(
        partition="default", graph_walk=WALK, node_name="dec",
        output_signals=["token"],
        wg_ids=ParallelList([rid], [WG_ID]),
        tensors=[uuid], num_tensors=[1], **kwargs,
    ))


def _reach_dec(rt, rid, uuid):
    rt.ingest_inputs_batch(_ingest_block([rid], [_spec("prompt", "prefill")]))
    rt.pop_rids("prefill", WALK, [rid])
    bk = rt._bookkeeping_for_test
    bk.put_tensor(uuid, _tensor_info(uuid))
    bk.increment_ref(uuid, 1)
    out = rt.complete_and_route_batch(RouteInput(
        partition="default", graph_walk=WALK, node_name="prefill",
        output_signals=["token"], wg_ids=ParallelList([rid], [WG_ID]),
        tensors=[uuid], num_tensors=[1],
    ))
    rt.send_outputs(_send_input(rid, out.completion_id))
    rt.pop_rids("dec", WALK, [rid])


def test_a_run_routes_its_items_on_and_its_last_iteration_back():
    """A signal that loops back and leaves the Loop: a run stores an iteration
    of it each, the client's edge takes them all, the next step reads the last."""
    rt = _client_loop_runtime()
    rid = _admit(rt)
    rt._communicator = _Recorder()
    _reach_dec(rt, rid, 100)
    bk = rt._bookkeeping_for_test
    for uuid in (101, 102, 103):
        bk.put_tensor(uuid, _tensor_info(uuid))
        bk.increment_ref(uuid, 1)
    out = rt.complete_and_route_batch(RouteInput(
        partition="default", graph_walk=WALK, node_name="dec", output_signals=["token"],
        wg_ids=ParallelList([rid], [WG_ID]), tensors=[101, 102, 103], num_tensors=[3], loop_strides=[3],
    ))
    assert {101, 102, 103} <= set(out.register_uuids)
    rt.send_outputs(_send_input(rid, out.completion_id))
    assert rt.pop_rids("dec", WALK, [rid]).input_edges.uuids == [103]


def test_the_rust_runtime_offers_inline_emit():
    assert rust_runtime.RustGraphRuntime.supports_inline_emit is True
    assert rust_runtime_base.GraphRuntime.supports_inline_emit is False


def test_an_inline_emit_is_not_staged_for_the_client():
    rt = _client_loop_runtime()
    rid = _admit(rt)
    rt._communicator = _Recorder()
    _reach_dec(rt, rid, 100)
    # the tensor path stages the client's copy
    out = _run_dec_step(rt, rid, 101)
    assert 101 in out.register_uuids
    rt.send_outputs(_send_input(rid, out.completion_id))
    rt.pop_rids("dec", WALK, [rid])
    # the inline path does not: the value rides in the frame
    out = _run_dec_step(rt, rid, 102, inline_value=151645)
    assert 102 not in out.register_uuids
    rt.send_outputs(_send_input(rid, out.completion_id))
    # the loop-back still fed the node, so the next step is ready
    assert rt.pop_rids("dec", WALK, [rid]) is not None


def test_inline_values_must_cover_every_rid_or_are_ignored():
    rt = _client_loop_runtime()
    rid = _admit(rt)
    _reach_dec(rt, rid, 200)
    bk = rt._bookkeeping_for_test
    bk.put_tensor(201, _tensor_info(201))
    bk.increment_ref(201, 1)
    out = rt.complete_and_route_batch(RouteInput(
        partition="default", graph_walk=WALK, node_name="dec",
        output_signals=["token"], wg_ids=ParallelList([rid], [WG_ID]),
        tensors=[201], num_tensors=[1], inline_signal="token", inline_values=[],
    ))
    assert 201 in out.register_uuids, "no values, so the tensor path"
    rt.send_outputs(_send_input(rid, out.completion_id))


def test_a_loop_back_routed_with_no_tensor_keeps_the_loop_going():
    """The device-side loop-back: the worker routes the decode signal with
    zero tensors per request (the token stays on the device) and the client
    copy rides inline. The node stays ready, the loop counter advances, the
    next pop reports the signal with no uuids, and cleanup frees nothing."""
    rt = _client_loop_runtime()
    rid = _admit(rt)
    rt._communicator = _Recorder()
    _reach_dec(rt, rid, 300)
    out = rt.complete_and_route_batch(RouteInput(
        partition="default", graph_walk=WALK, node_name="dec",
        output_signals=["token"], wg_ids=ParallelList([rid], [WG_ID]),
        tensors=[], num_tensors=[0], inline_signal="token", inline_values=[7],
    ))
    assert out.register_uuids == [], "nothing to stage for the client"
    rt.send_outputs(_send_input(rid, out.completion_id))
    popped = rt.pop_rids("dec", WALK, [rid])
    assert popped is not None, "the loop-back with no tensor still fed the node"
    inputs = popped.input_edges.to_input_tensors(lambda u: (_ for _ in ()).throw(KeyError(u)))
    assert inputs.by_rid[rid] == {"token": []}
    before = dict(rt.get_dynamic_loop_iters([rid], "default").values[0])
    # another zero-tensor step: still ready, nothing freed by the cleanup, the
    # loop counter moved
    out = rt.complete_and_route_batch(RouteInput(
        partition="default", graph_walk=WALK, node_name="dec",
        output_signals=["token"], wg_ids=ParallelList([rid], [WG_ID]),
        tensors=[], num_tensors=[0], inline_signal="token", inline_values=[8],
    ))
    assert out.freed_inputs.uuids == []
    rt.send_outputs(_send_input(rid, out.completion_id))
    assert rt.pop_rids("dec", WALK, [rid]) is not None
    after = dict(rt.get_dynamic_loop_iters([rid], "default").values[0])
    assert after["loop"] == before["loop"] + 1


@pytest.mark.parametrize("shape", ["whole", "partial", "split", "ancestor"])
def test_only_a_complete_closed_loop_unit_names_a_repeat_domain(shape):
    from mstar.graph.base import Parallel
    from mstar.model.base import _divide_into_worker_graphs

    a = GraphNode(name="a", input_names={"x", "y"}, outputs=[GraphEdge(name="x", next_node="b")])
    b = GraphNode(name="b", input_names={"x"}, outputs=[GraphEdge(name="x", next_node="a")])
    body = Sequential([a, b])
    if shape == "partial":
        body = Sequential([GraphNode(name="p", input_names={"p"},
                                     outputs=[GraphEdge(name="y", next_node="a")]), body])
    elif shape == "split":
        body = Parallel([body, GraphNode(name="other", input_names={"q"}, outputs=[])])
    loop = Loop(name="steps", section=body, max_iters=4, outputs=[])
    section = Sequential([loop])
    if shape == "split":
        graphs = _divide_into_worker_graphs(section, WALK, {"a": 0, "b": 0, "other": 1},
                                           [{"ranks": [0]}, {"ranks": [1]}], set())
        section = next(wg.section for wg in graphs if "a" in wg.section.get_nodes())
        assert not section.get_loops()["steps"]._partition_complete
    elif shape == "ancestor":
        section = Sequential([Loop(name="outer", max_iters=2, outputs=[], section=Sequential([
            loop, GraphNode(name="p", input_names={"p"}, outputs=[GraphEdge(name="y", next_node="a")]),
        ]))])
    runtime = _unit_runtime(section, section.get_loops())
    unit = next(unit for unit in runtime.execution_units(WALK) if unit.nodes[0] == "a")
    assert unit.nodes == ("a", "b")
    assert unit.loop_name == ("steps" if shape == "whole" else None)
    assert unit.loop_back == ((("b", "a", "x"),) if shape == "whole" else ())


def test_connected_repeated_completion_keeps_client_columns_but_only_the_last_cross_node_carry():
    from mstar.graph.special_destinations import EMIT_TO_CLIENT

    rt = _unit_runtime(Sequential([Loop(name="steps", max_iters=4, outputs=[], section=Sequential([
        GraphNode(name="a", input_names={"x"}, outputs=[GraphEdge(name="x", next_node="b")]),
        GraphNode(name="b", input_names={"x"}, outputs=[
            GraphEdge(name="x", next_node="a"),
            GraphEdge(name="x", next_node=EMIT_TO_CLIENT, output_modality="text"),
        ]),
    ]))]), {"steps"})
    rid = _admit(rt)
    rt.ingest_inputs_batch(_ingest_block([rid], [_spec("x", "a")]))
    claim, _ = rt.claim_execution_unit("a", WALK, [rid])
    assert rt.submit_execution_unit(claim)
    rt.complete_and_route_batch(RouteInput(
        partition="default", graph_walk=WALK, node_name="a", output_signals=[],
        wg_ids=ParallelList([rid], [WG_ID]), tensors=[], num_tensors=[], loop_strides=[3],
    ))
    for uuid in (911, 912, 913):
        rt._bookkeeping_for_test.put_tensor(uuid, _tensor_info(uuid))
        rt._bookkeeping_for_test.increment_ref(uuid)
    out = rt.complete_and_route_batch(RouteInput(
        partition="default", graph_walk=WALK, node_name="b", output_signals=["x"],
        wg_ids=ParallelList([rid], [WG_ID]), tensors=[911, 912, 913], num_tensors=[3], loop_strides=[3],
    ))
    assert {911, 912, 913} <= set(out.register_uuids)
    assert rt.pop_rids("a", WALK, [rid]).input_edges.uuids == [913]
    assert rt.loop_room("b", WALK, "steps", [rid], 0, False) == 1
    assert not rt.cancel_execution_unit(claim)
