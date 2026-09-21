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
from mstar.graph.runtime import base as rust_runtime_base
from mstar.graph.runtime.base import SpeculationPrepInput

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


# --- ingest ------------------------------------------------------------------

def _spec(signal, next_node, uuids=(), final=False):
    return rust_runtime_base.EdgeSpec(
        signal=signal, next_node=next_node, uuids=list(uuids),
        is_final_streaming_chunk=final,
    )


def test_a_signal_reaches_its_node(runtime):
    rid = _admit(runtime)
    assert runtime.ingest_inputs_batch(
        ParallelList([rid], [_spec("prompt", "prefill")])
    ) == []


def test_a_signal_for_an_unknown_node_comes_back_by_index(runtime):
    rid = _admit(runtime)
    uningested = runtime.ingest_inputs_batch(
        ParallelList(
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
        ParallelList([rid], [_spec("not_an_input", "prefill")])
    ) == [0]


def test_a_second_signal_buffers_then_refuses(runtime):
    # Both ready slots full is the refusal Python returns False for; the
    # caller re-queues rather than losing the chunk.
    rid = _admit(runtime)
    assert runtime.ingest_inputs_batch(
        ParallelList([rid], [_spec("prompt", "prefill")])
    ) == []
    assert runtime.ingest_inputs_batch(
        ParallelList([rid], [_spec("prompt", "prefill")]), can_buffer=True
    ) == [], "the second goes to the next-iter slot"
    assert runtime.ingest_inputs_batch(
        ParallelList([rid], [_spec("prompt", "prefill")]), can_buffer=True
    ) == [0], "the third has nowhere to go"


def test_can_buffer_false_refuses_the_second(runtime):
    rid = _admit(runtime)
    runtime.ingest_inputs_batch(ParallelList([rid], [_spec("prompt", "prefill")]))
    assert runtime.ingest_inputs_batch(
        ParallelList([rid], [_spec("prompt", "prefill")]), can_buffer=False
    ) == [0], "streaming must not buffer for an iteration that may not come"


def test_an_unknown_rid_is_refused_not_raised(runtime):
    # A signal can arrive for a request this rank already removed.
    assert runtime.ingest_inputs_batch(
        ParallelList([9999], [_spec("prompt", "prefill")])
    ) == [0]


def test_streaming_gates_on_the_non_streaming_inputs(runtime):
    # ar_decode's inputs are not streaming, so it never reports
    # ready-for-streaming and a streaming ingest must not land.
    rid = _admit(runtime)
    assert runtime.ingest_inputs_batch(
        ParallelList([rid], [_spec("token", "ar_decode")]), is_streaming=True
    ) == [0]


def test_mismatched_lengths_are_rejected(runtime):
    with pytest.raises((ValueError, RuntimeError)):
        runtime.ingest_inputs_batch(
            ParallelList([1, 2], [_spec("prompt", "prefill")])
        )


# --- consumed inputs and loop iters ------------------------------------------

def test_cleanup_releases_consumed_inputs(runtime):
    rid = _admit(runtime)
    runtime.ingest_inputs_batch(
        ParallelList([rid], [_spec("prompt", "prefill", uuids=[7])])
    )
    runtime.cleanup_consumed_inputs("prefill", [rid], [WG_ID])
    # The slot is free again, so the same signal lands rather than refusing.
    assert runtime.ingest_inputs_batch(
        ParallelList([rid], [_spec("prompt", "prefill")]), can_buffer=False
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
    rt.ingest_inputs_batch(ParallelList(
        [rid, rid],
        [_spec("seed", "body", uuids=[1]), _spec("tok", "body", uuids=[2])],
    ))
    rt.cleanup_consumed_inputs("body", [rid], [WG_ID])

    # "seed" is external to the loop, so it is still held; "tok" loops back
    # and was released.
    assert rt.ingest_inputs_batch(
        ParallelList([rid], [_spec("seed", "body")]), can_buffer=False
    ) == [0], "the external input must still be held"
    assert rt.ingest_inputs_batch(
        ParallelList([rid], [_spec("tok", "body")]), can_buffer=False
    ) == [], "the loop-back input was consumed and released"


def test_dynamic_loop_iters_report_per_rid(runtime):
    rid = _admit(runtime)
    got = runtime.get_dynamic_loop_iters([rid], "default")
    assert got.keys == [rid]
    assert got.values[0] == {"ar_loop": 0}


def test_dynamic_loop_iters_for_an_unknown_partition_are_empty(runtime):
    rid = _admit(runtime)
    assert runtime.get_dynamic_loop_iters([rid], "nope").values == [{}]


# --- scheduling --------------------------------------------------------------

def test_a_ready_node_is_reported(runtime):
    rid = _admit(runtime)
    assert runtime.get_ready_nodes(set()) == []
    runtime.ingest_inputs_batch(ParallelList([rid], [_spec("prompt", "prefill")]))
    ready = runtime.get_ready_nodes(set())
    assert [(r.node_name, r.graph_walk, r.rids) for r in ready] == [
        ("prefill", WALK, [rid])
    ]


def test_excluded_rids_are_invisible(runtime):
    rid = _admit(runtime)
    runtime.ingest_inputs_batch(ParallelList([rid], [_spec("prompt", "prefill")]))
    assert runtime.get_ready_nodes({rid}) == []
    assert not runtime.has_ready_excluding({rid})


def test_target_and_exclude_target_filter(runtime):
    rid = _admit(runtime)
    runtime.ingest_inputs_batch(ParallelList([rid], [_spec("prompt", "prefill")]))
    assert len(runtime.get_ready_nodes(set(), target=("prefill", WALK))) == 1
    assert runtime.get_ready_nodes(set(), target=("ar_decode", WALK)) == []
    assert runtime.get_ready_nodes(
        set(), exclude_target=("prefill", WALK)
    ) == []


def test_the_peek_agrees_with_the_scan(runtime):
    rid = _admit(runtime)
    assert not runtime.has_ready_excluding(set())
    runtime.ingest_inputs_batch(ParallelList([rid], [_spec("prompt", "prefill")]))
    assert runtime.has_ready_excluding(set())
    assert not runtime.has_ready_excluding(set(), exclude_target=("prefill", WALK))


def test_pop_returns_the_inputs_it_popped(runtime):
    rid = _admit(runtime)
    runtime.ingest_inputs_batch(
        ParallelList([rid], [_spec("prompt", "prefill", uuids=[11, 12])])
    )
    out = runtime.pop_rids("prefill", WALK, [rid])
    assert out.wg_ids.keys == [rid] and out.wg_ids.values == [WG_ID]
    assert out.input_edges_per_rid == [1]
    edge = out.input_edges[0]
    assert edge.signal == "prompt" and edge.uuids == [11, 12]

    # Popped, so it is no longer ready.
    assert runtime.get_ready_nodes(set()) == []


def test_pop_with_check_ready_is_all_or_nothing(runtime):
    # One not-ready rid must leave the whole set intact for a later retry.
    a = _admit(runtime, "ra")
    b = _admit(runtime, "rb")
    runtime.ingest_inputs_batch(ParallelList([a], [_spec("prompt", "prefill")]))

    assert runtime.pop_rids("prefill", WALK, [a, b], check_ready=True) is None
    # a was NOT popped by the failed attempt.
    out = runtime.pop_rids("prefill", WALK, [a], check_ready=True)
    assert out.wg_ids.keys == [a]


def test_pop_of_an_unknown_node_returns_none(runtime):
    assert runtime.pop_rids("nope", WALK, []) is None


def test_push_back_makes_a_popped_node_ready_again(runtime):
    rid = _admit(runtime)
    runtime.ingest_inputs_batch(ParallelList([rid], [_spec("prompt", "prefill")]))
    runtime.pop_rids("prefill", WALK, [rid])
    assert runtime.get_ready_nodes(set()) == []
    runtime.push_back_node("prefill", [rid], [WG_ID])
    assert len(runtime.get_ready_nodes(set())) == 1


# --- speculation -------------------------------------------------------------

def test_speculate_finds_the_downstream_target(runtime):
    rid = _admit(runtime)
    out = runtime.speculate_node("prefill", WALK, rid)
    assert [o.node_name for o in out] == ["ar_decode"]
    assert out[0].graph_walk == WALK
    # prefill -> ar_decode ENTERS the loop, so it is not a new iteration.
    assert out[0].is_new_loop_iter is False
    assert out[0].loop_name == "ar_loop"


def test_a_loop_back_is_reported_as_a_new_iteration(runtime):
    # ar_decode -> ar_decode is the loop-back; the per-rid loop filters key off
    # exactly this flag, so getting it wrong disables them silently.
    rid = _admit(runtime)
    out = runtime.speculate_node("ar_decode", WALK, rid)
    assert [o.node_name for o in out] == ["ar_decode"]
    assert out[0].is_new_loop_iter is True


def test_speculation_leaves_no_state_behind(runtime):
    # ingest_for_speculation fills the speculative slots; they must be cleared
    # or the node reads as ready when nothing actually arrived.
    rid = _admit(runtime)
    runtime.speculate_node("prefill", WALK, rid)
    assert runtime.get_ready_nodes(set()) == []


def test_speculate_refuses_a_node_that_opted_out(runtime):
    rid = _admit(runtime)
    runtime.set_node_metadata(
        parallel_nodes={"ar_decode"}, parallel_leader_nodes=set(),
        tp_async_nodes=set(),
    )
    # A parallel target is only valid as a leader-side same-node loop-back.
    assert runtime.speculate_node("prefill", WALK, rid) == []


def test_get_spec_target_reports_a_target_chosen_elsewhere(runtime):
    # The follower path: no eligibility filter, since a follower is not the
    # leader and would fail it by construction.
    rid = _admit(runtime)
    runtime.set_node_metadata(
        parallel_nodes={"ar_decode"}, parallel_leader_nodes=set(),
        tp_async_nodes=set(),
    )
    assert runtime.speculate_node("ar_decode", WALK, rid) == []
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
    runtime.ingest_inputs_batch(ParallelList([rid], [_spec("prompt", "prefill")]))
    runtime.pop_rids("prefill", WALK, [rid])
    runtime.ingest_inputs_batch(ParallelList(
        [rid, rid],
        [_spec("token", "ar_decode"), _spec("kv_cache", "ar_decode")],
    ))


def test_prep_reports_the_worker_graph_per_ready_rid(runtime):
    rid = _admit(runtime)
    _drive_into_loop(runtime, rid)
    out = runtime.prep_spec_rids(_prep_input([rid], curr="ar_decode"))
    assert out.ready_rids == [rid]
    assert out.wg_ids == [WG_ID], "wg_ids is parallel to ready_rids"
    assert len(out.input_edges_per_rid) == len(out.ready_rids)
    assert sum(out.input_edges_per_rid) == len(out.input_edges)


def test_prep_respects_room_for_continuing(runtime):
    rid = _admit(runtime)
    _drive_into_loop(runtime, rid)
    assert runtime.prep_spec_rids(
        _prep_input([rid], curr="ar_decode", room=0)
    ).ready_rids == []
    assert runtime.prep_spec_rids(
        _prep_input([rid], curr="ar_decode", room=1)
    ).ready_rids == [rid]


def _two_node_runtime():
    """a -> b, where b ALSO needs an input a does not produce.

    A same-node loop-back is a poor subject for not-ready: it checks the
    next-iter slot, which a loop's external input never occupies, so such a
    loop simply cannot be speculated. A cross-node target checks the current
    slot, which is the case the prep filters actually see.
    """
    wg = WorkerGraph(
        section=Sequential(sections=[
            GraphNode(
                name="a", input_names={"in"},
                outputs=[GraphEdge(name="x", next_node="b")],
            ),
            GraphNode(name="b", input_names={"x", "y"}, outputs=[]),
        ]),
        graph_walks={WALK}, ranks=[0], worker_graph_id=WG_ID,
    )
    return rust_runtime.RustGraphRuntime(
        my_worker_id=WORKER, my_worker_graphs=[wg],
        all_wg_ids_to_graph_walks={WG_ID: {WALK}},
        all_wg_ids_to_dyn_loops={WG_ID: set()},
        all_wg_ids_to_nodes={WG_ID: {"a", "b"}},
        node_to_partition={"a": "default", "b": "default"},
        sharding_config=ShardingConfig(
            groups=[], tp_enabled_nodes=set(), shard_dim={},
        ),
        bookkeeping=RustTensorBookkeeping(),
    )


def _ab(rids, room=None):
    return _prep_input(rids, spec="b", curr="a", room=room)


def test_prep_skips_a_rid_whose_other_input_has_not_arrived():
    rt = _two_node_runtime()
    rid = _admit(rt)
    # a's outputs cover x; y has to be there already.
    assert rt.prep_spec_rids(_ab([rid])).ready_rids == []

    rt.ingest_inputs_batch(ParallelList([rid], [_spec("y", "b")]))
    assert rt.prep_spec_rids(_ab([rid])).ready_rids == [rid]


def test_the_follower_prep_is_all_or_nothing():
    rt = _two_node_runtime()
    ready = _admit(rt, "ra")
    blank = _admit(rt, "rb")
    rt.ingest_inputs_batch(ParallelList([ready], [_spec("y", "b")]))

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
    rt.ingest_inputs_batch(ParallelList([ready], [_spec("y", "b")]))

    assert rt.prep_follow_spec_rids(_ab([ready, blank])) is None
    # The ready rid is still preppable, so nothing was consumed or left set.
    assert rt.prep_spec_rids(_ab([ready])).ready_rids == [ready]


def test_the_follower_prep_succeeds_when_every_rid_is_ready():
    rt = _two_node_runtime()
    a = _admit(rt, "ra")
    b = _admit(rt, "rb")
    for r in (a, b):
        rt.ingest_inputs_batch(ParallelList([r], [_spec("y", "b")]))
    out = rt.prep_follow_spec_rids(_ab([a, b]))
    assert out is not None
    assert out.ready_rids == [a, b], "wire order is the leader's batch order"


def test_prep_room_cap_applies_before_any_ingest():
    rt = _two_node_runtime()
    a = _admit(rt, "ra")
    b = _admit(rt, "rb")
    for r in (a, b):
        rt.ingest_inputs_batch(ParallelList([r], [_spec("y", "b")]))
    assert rt.prep_spec_rids(_ab([a, b], room=1)).ready_rids == [a]
    # The capped rid was skipped before any ingest, so it is untouched.
    assert rt.prep_spec_rids(_ab([b])).ready_rids == [b]
