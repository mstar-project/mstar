"""Step-by-step scripts of the graph runtime.

``test_runtime_parity.py`` asserts sequences of calls end to end. These stop
after each operation and assert what a worker acts on at that point: ready
nodes, what pop_rids hands back, the uuids routing asks to register, and which
uuids the bookkeeper still holds. A wrong reference shows up at the step that
caused it.
"""
import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.communication.tensor_store import TensorStore
from mstar.distributed.base import ShardingConfig, ShardingGroup
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential, TensorPointerInfo
from mstar.graph.runtime.base import (
    ColumnarEdgeSpecs,
    EdgeSpec,
    RouteInput,
    SpeculationPrepInput,
)
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import WorkerGraph
from mstar.utils.containers import ParallelList

pytest.importorskip(
    "mstar_rust",
    reason="mstar_rust not built (maturin develop --release in rust/)",
)
from mstar.communication.tensor_store import RustTensorBookkeeping
from mstar.graph.runtime import rust as rust_runtime

WG_ID, WALK, WORKER, PEER = 0, "decode", "worker_0", "worker_1"


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

def _loop_graph():
    """An AR loop whose token is BOTH emitted to the client and fed back as
    the next iteration's input -- the higgs/orpheus shape, and the one where a
    tensor is persisted for delivery while the loop still needs it."""
    return Sequential(sections=[
        GraphNode(
            name="prefill", input_names={"prompt"},
            outputs=[GraphEdge(name="token", next_node="ar_decode")],
        ),
        Loop(
            name="ar_loop",
            section=GraphNode(
                name="ar_decode", input_names={"token"},
                outputs=[GraphEdge(name="token", next_node="ar_decode")],
            ),
            outputs=[GraphEdge(name="token", next_node="post_processor")],
            max_iters=4,
        ),
    ])


def _nodes(graph):
    return set(graph.get_nodes())


def _build(graph, tp_rank=None):
    """``tp_rank`` puts this worker at that rank of a two-rank TP group over
    every node; None leaves it alone, which reads as rank 0."""
    groups = []
    if tp_rank is not None:
        groups = [ShardingGroup(nodes=_nodes(graph), tp_size=2)]
        groups[0]._tp_rank = tp_rank
    wg = WorkerGraph(
        section=graph, graph_walks={WALK}, ranks=[0] if tp_rank is None else [0, 1],
        worker_graph_id=WG_ID)
    book = RustTensorBookkeeping()
    store = TensorStore(bookkeeping=book)
    runtime = rust_runtime.RustGraphRuntime(
        my_worker_id=WORKER, my_worker_graphs=[wg],
        all_wg_ids_to_graph_walks={WG_ID: {WALK}},
        all_wg_ids_to_dyn_loops={WG_ID: set(graph.get_loops())},
        all_wg_ids_to_nodes={WG_ID: _nodes(graph)},
        node_to_partition=dict.fromkeys(_nodes(graph), "default"),
        sharding_config=ShardingConfig(
            groups=groups, tp_enabled_nodes=set(), shard_dim={}),
        bookkeeping=book,
    )
    return runtime, book, store


def _info(uuid):
    return TensorPointerInfo(
        dims=[4], dtype=torch.float16, nbytes=8, address=0, stride=(1,),
        uuid=uuid, source_session_id="h:1", source_entity=WORKER)



def _finish_teardown(store, freed):
    """What the worker does with a runtime's FreedTensors: hand them to
    TensorCommunicationManager.cleanup_collectable, whose bookkeeper half is
    mark_forgotten (the rest is shm files and memory unregistration, which
    this harness has none of)."""
    if freed.uuids:
        store.mark_forgotten(list(freed.uuids))


class Script:
    """One runtime, driven the way the worker drives it."""

    def __init__(self, graph, tp_rank=None):
        self.rt, self.book, self.store = _build(graph, tp_rank)
        self.workers = {None: [WORKER], 0: [WORKER, PEER], 1: [PEER, WORKER]}[tp_rank]

    # -- operations ---------------------------------------------------------
    def admit(self, rid_str="r1"):
        return self.rt.add_request(
            request_id=rid_str, partition="default", graph_walk=WALK,
            partition_worker_graph_ids=[WG_ID],
            worker_graph_to_workers=ParallelList([WG_ID], [self.workers]), loop_caps={})

    def put(self, uuid):
        """Store a tensor so routing has something to refer to."""
        self.store.put_tensor(request_id=1, uuid=uuid, tensor=torch.zeros(4),
                              info=_info(uuid))
        # The worker's safety hold: _postprocess_batch stores the outputs and
        # immediately does increment_ref_batch_uniform(uuids, 1), so a tensor
        # is never collectable between being stored and being routed. Without
        # it every uuid sits at refcount 0 from birth and the free paths fire
        # at points production never hits.
        self.store.increment_ref(uuid, n=1)

    def ingest(self, rid, signal, node, uuids=()):
        spec = EdgeSpec(signal=signal, next_node=node, uuids=list(uuids),
                        is_final_streaming_chunk=False)
        return self.rt.ingest_inputs_batch(_ingest_block([rid], [spec]))

    def stream(self, rid, signal, node, uuids=()):
        """A chunk from a StreamBuffer, ingested as the worker does: for this
        iteration only, gated on streaming readiness. Returns what was left
        uningested."""
        spec = EdgeSpec(signal=signal, next_node=node, uuids=list(uuids),
                        is_final_streaming_chunk=False)
        return self.rt.ingest_inputs_batch(
            _ingest_block([rid], [spec]), can_buffer=False, is_streaming=True)

    def ready(self):
        return sorted(
            (x.node_name, x.graph_walk, sorted(x.rids))
            for x in self.rt.get_ready_nodes(set()))

    def pop(self, node, rid):
        p = self.rt.pop_rids(node, WALK, [rid])
        if p is None:
            return None
        # next_node is not in the block: a pop reports edges for the node the
        # caller named, so it was the same string every time.
        return (p.wg_ids.keys, p.wg_ids.values,
                sorted((sig, node, list(uuids))
                       for _r, sig, uuids, _f
                       in p.input_edges.edge_tuples()))

    def route(self, node, rid, signals, uuids):
        out = self.rt.complete_and_route_batch(
            RouteInput(
                partition="default", graph_walk=WALK, node_name=node,
                output_signals=list(signals),
                wg_ids=ParallelList([rid], [WG_ID]),
                tensors=list(uuids),
                num_tensors=[len(uuids)] * len(signals),
            ))
        _finish_teardown(self.store, out.freed_inputs)
        return (sorted(out.register_uuids), sorted(out.register_rids),
                sorted(out.new_token_output_idxs),
                {k: (v.keys, v.values)
                 for k, v in out.local_streaming_by_signal.items()})

    def cleanup(self, node, rid):
        _finish_teardown(self.store, self.rt.cleanup_consumed_inputs(node, [rid], [WG_ID]))

    def held(self, uuids):
        """The uuids the bookkeeper still tracks and would not collect."""
        return [u for u in uuids if self.book.get_info(u) is not None and not self.book.can_gc(u)]


@pytest.fixture
def lock():
    return Script(_loop_graph())


def test_a_loop_carried_tensor_is_held_until_its_iteration_consumes_the_next(lock):
    """The token is fed back. A reference dropped or leaked here means a worker
    resolving a freed uuid, or a tensor that is never collected."""
    rid = lock.admit()
    lock.put(101)
    lock.ingest(rid, "prompt", "prefill", [101])
    lock.pop("prefill", rid)
    for i, uuid in enumerate((201, 202, 203)):
        lock.put(uuid)
        lock.route("prefill" if i == 0 else "ar_decode", rid, ["token"], [uuid])
        assert lock.ready() == [("ar_decode", WALK, [rid])]
        lock.pop("ar_decode", rid)
        assert lock.held([101, 201, 202, 203]) == [uuid]


def _emit_loop_graph():
    """Same AR loop, but the token is ALSO persisted and emitted to the client.

    This is the higgs/orpheus shape: one tensor is simultaneously (a) held at
    the conductor for delivery and (b) the next iteration's input. Persist
    edges are excluded from the routed refcount because the conductor computes
    that reference when it unpersists, so a miscount frees the tensor while the
    loop still needs it.
    """
    return Sequential(sections=[
        GraphNode(
            name="prefill", input_names={"prompt"},
            outputs=[GraphEdge(name="token", next_node="ar_decode")],
        ),
        Loop(
            name="ar_loop",
            section=GraphNode(
                name="ar_decode", input_names={"token"},
                outputs=[
                    GraphEdge(name="token", next_node="ar_decode"),
                    GraphEdge(name="token", next_node=EMIT_TO_CLIENT,
                              persist=True, conductor_new_token=True,
                              output_modality="text"),
                ],
            ),
            outputs=[GraphEdge(name="token", next_node="post_processor")],
            max_iters=4,
        ),
    ])


@pytest.fixture
def emit_lock():
    return Script(_emit_loop_graph())


def test_a_persisted_and_loop_carried_token_stays_held(emit_lock):
    """The higgs shape. A token routed to BOTH the client and the next
    iteration stays held through the iteration that consumes it -- freeing it
    on unpersist while the loop's input slot points at it is the KeyError seen
    in production."""
    lk = emit_lock
    rid = lk.admit()
    lk.put(101)
    lk.ingest(rid, "prompt", "prefill", [101])
    lk.pop("prefill", rid)
    lk.put(201)
    lk.route("prefill", rid, ["token"], [201])
    assert lk.ready() == [("ar_decode", WALK, [rid])]
    lk.pop("ar_decode", rid)
    # ar_decode emits a token that is persisted AND fed back
    lk.put(202)
    lk.route("ar_decode", rid, ["token"], [202])
    assert lk.held([101, 201, 202]) == [202]
    assert lk.ready() == [("ar_decode", WALK, [rid])]
    lk.pop("ar_decode", rid)
    assert lk.held([101, 201, 202]) == [202]


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_only_the_tp_leader_counts_new_tokens(tp_rank):
    """A follower doesn't send the emitted token to the client, so it frees its
    copy; counting it as a new token anyway made the worker look up a freed
    tensor (a KeyError on every Command A+ TP8 request)."""
    lk = Script(_emit_loop_graph(), tp_rank=tp_rank)
    rid = lk.admit()
    lk.put(101)
    lk.ingest(rid, "prompt", "prefill", [101])
    lk.pop("prefill", rid)
    lk.put(201)
    lk.route("prefill", rid, ["token"], [201])
    lk.pop("ar_decode", rid)
    lk.put(202)
    _, _, new_tokens, _ = lk.route("ar_decode", rid, ["token"], [202])
    assert bool(new_tokens) == (tp_rank == 0)


def test_repeated_emit_iterations_hold_one_reference_each(emit_lock):
    """Several loop iterations with an emitted token each time: a reference
    that leaks or is dropped once shows up as drift after a few passes. Every
    token stays persisted for delivery, and each new one joins them."""
    lk = emit_lock
    rid = lk.admit()
    lk.put(101)
    lk.ingest(rid, "prompt", "prefill", [101])
    lk.pop("prefill", rid)
    lk.put(201)
    lk.route("prefill", rid, ["token"], [201])
    lk.pop("ar_decode", rid)
    held = []
    for uuid in (202, 203, 204):
        lk.put(uuid)
        lk.route("ar_decode", rid, ["token"], [uuid])
        assert lk.ready() == [("ar_decode", WALK, [rid])]
        lk.pop("ar_decode", rid)
        held.append(uuid)
        assert lk.held([201, 202, 203, 204]) == held


def test_cleanup_then_route_after_an_emit(emit_lock):
    """cleanup_consumed_inputs on a node whose input was also emitted, in the
    order the worker actually uses: cleanup, then route."""
    lk = emit_lock
    rid = lk.admit()
    lk.put(101)
    lk.ingest(rid, "prompt", "prefill", [101])
    lk.pop("prefill", rid)
    lk.put(201)
    lk.route("prefill", rid, ["token"], [201])
    lk.pop("ar_decode", rid)
    lk.put(202)
    # Production order: _postprocess_batch calls _cleanup_consumed_inputs
    # (worker.py) BEFORE complete_and_route_batch, not after.
    lk.cleanup("ar_decode", rid)
    lk.route("ar_decode", rid, ["token"], [202])
    assert lk.ready() == [("ar_decode", WALK, [rid])]
    assert lk.held([101, 201, 202]) == [202]


def test_route_then_cleanup_after_an_emit(emit_lock):
    """The SAME steps with route before cleanup -- an order the worker does not
    produce (_postprocess_batch cleans up first). Clearing a node's inputs has
    to take it out of the ready set whichever order runs."""
    lk = emit_lock
    rid = lk.admit()
    lk.put(101)
    lk.ingest(rid, "prompt", "prefill", [101])
    lk.pop("prefill", rid)
    lk.put(201)
    lk.route("prefill", rid, ["token"], [201])
    lk.pop("ar_decode", rid)
    lk.put(202)
    lk.route("ar_decode", rid, ["token"], [202])
    assert lk.ready() == [("ar_decode", WALK, [rid])]
    lk.cleanup("ar_decode", rid)
    assert lk.ready() == []
    assert lk.held([101, 201, 202]) == [202]


def test_cleanup_without_pop_clears_the_readiness(lock):
    """Cleanup on a node sitting in the ready set, never popped. The narrowest
    form of the same rule: clearing the inputs must clear the readiness."""
    rid = lock.admit()
    lock.put(101)
    lock.ingest(rid, "prompt", "prefill", [101])
    assert lock.ready() == [("prefill", WALK, [rid])]
    lock.cleanup("prefill", rid)
    assert lock.ready() == []
    assert lock.held([101]) == []


def _spec(rt, node, rid):
    return sorted(
        (s.node_name, s.graph_walk, s.is_new_loop_iter, s.loop_name,
         tuple(s.output_signals))
        for s in rt.speculate_node(node, [WALK], [rid])
    )


def test_speculation_targets_across_a_loop(lock):
    """The target offered at each point of a request: into the loop from
    prefill, then the same node's next iteration, before and after each pop and
    route. A step with no target skips speculation entirely, so the worker pays
    for a whole fresh batch instead."""
    into_the_loop = [("ar_decode", WALK, False, "ar_loop", ("token",))]
    next_iteration = [("ar_decode", WALK, True, "ar_loop", ("token",))]
    rid = lock.admit()
    lock.put(101)
    lock.ingest(rid, "prompt", "prefill", [101])
    assert _spec(lock.rt, "prefill", rid) == into_the_loop
    lock.pop("prefill", rid)
    assert _spec(lock.rt, "prefill", rid) == into_the_loop
    lock.put(201)
    lock.route("prefill", rid, ["token"], [201])
    assert _spec(lock.rt, "ar_decode", rid) == next_iteration
    for uuid in (202, 203):
        lock.pop("ar_decode", rid)
        assert _spec(lock.rt, "ar_decode", rid) == next_iteration
        lock.put(uuid)
        lock.route("ar_decode", rid, ["token"], [uuid])
        assert _spec(lock.rt, "ar_decode", rid) == next_iteration


def _external_input_loop_graph():
    """The wan22 / waypoint shape: the loop node has an external input ``ctx``
    the loop re-injects every iteration, next to its own loop-back ``token``."""
    return Sequential(sections=[
        GraphNode(
            name="prefill", input_names={"prompt"},
            outputs=[
                GraphEdge(name="token", next_node="ar_decode"),
                GraphEdge(name="ctx", next_node="ar_decode"),
            ],
        ),
        Loop(
            name="ar_loop",
            section=GraphNode(
                name="ar_decode", input_names={"token", "ctx"},
                outputs=[GraphEdge(name="token", next_node="ar_decode")],
            ),
            outputs=[GraphEdge(name="token", next_node="post_processor")],
            max_iters=4,
        ),
    ])


def _prep_same_node(rt, node, rid):
    out = rt.prep_spec_rids(SpeculationPrepInput(
        spec_node_name=node, curr_node_name=node, graph_walk=WALK, rids=[rid],
        room_for_continuing=None, streaming_edges=[], streaming_edges_per_rid=[0],
    ))
    return (
        list(out.ready_rids),
        sorted((sig, list(uuids)) for _r, sig, uuids, _f
               in out.input_edges.edge_tuples()),
    )


def test_same_node_speculation_with_a_held_external_input():
    """A loop's external input lives in the current slot and is re-injected
    every iteration; it never reaches the next-iter slot. The same-node next
    iteration is still a speculation target, and the held input goes over with
    the prepped edges (Rust: held_mask)."""
    lock = Script(_external_input_loop_graph())
    rid = lock.admit()
    lock.put(101)
    lock.ingest(rid, "prompt", "prefill", [101])
    lock.pop("prefill", rid)
    lock.put(201)
    lock.put(301)

    out = lock.rt.complete_and_route_batch(RouteInput(
        partition="default", graph_walk=WALK, node_name="prefill",
        output_signals=["token", "ctx"], wg_ids=ParallelList([rid], [WG_ID]),
        tensors=[201, 301], num_tensors=[1, 1],
    ))
    _finish_teardown(lock.store, out.freed_inputs)
    assert lock.ready() == [("ar_decode", WALK, [rid])]
    lock.pop("ar_decode", rid)
    targets = _spec(lock.rt, "ar_decode", rid)
    assert [(t[0], t[2]) for t in targets] == [("ar_decode", True)]
    assert _prep_same_node(lock.rt, "ar_decode", rid) == ([rid], [("ctx", [301])])
    assert _prep_rows_same_node(lock.rt, "ar_decode", rid) == ([rid], WG_ID, 0, [("ctx", [301])])
    assert lock.held([101, 201, 301]) == [201, 301]


def _prep_rows_same_node(rt, node, rid, room=None):
    out = rt.prep_spec_rows(node, node, WALK, [rid], room)
    return (
        list(out.ready_rids), out.wg_id, out.spec_id,
        None if out.input_edges is None else sorted(
            (sig, list(uuids)) for _r, sig, uuids, _f in out.input_edges.edge_tuples()
        ),
    )


def test_speculative_flag_survives_completion(lock):
    """The flag marking a node speculatively scheduled for N+1 must SURVIVE
    the completion of step N -- its rids are still in flight.

    ``State::complete`` used to set ``st.scheduled = false``. Because
    ``refresh_ready`` gates the ready-set ADD on that flag, clearing it let the
    runtime report a node ready whose rids were mid-flight, and
    ``get_ready_nodes`` handed them back to the scheduler as fresh work. Every
    speculation then ran a full get_ready_nodes -> _assemble_batch -> pop_rids
    round trip that ended in ``push_back_node`` -- the branch worker.py
    documents as "Shouldn't happen". Measured on bagel bs32 before the fix:
    30.89 of 31.01 "fresh" rids were the in-flight set.

    Asserted on the flag itself rather than on ``get_ready_nodes``: reaching
    the observable ready state needs the loop to advance, which this harness
    does not drive, so a readiness assertion here passes vacuously whether or
    not the bug is present.
    """
    rid = lock.admit()
    lock.put(101)
    lock.ingest(rid, "prompt", "prefill", [101])
    lock.pop("prefill", rid)
    lock.put(201)
    lock.route("prefill", rid, ["token"], [201])
    lock.pop("ar_decode", rid)

    def flag():
        return lock.rt.is_in_flight("ar_decode", WG_ID, rid)
    assert flag() is False

    lock.rt.set_in_flight("ar_decode", [rid], [WG_ID], True)
    assert flag() is True

    # Step N completes. The flag must not be cleared by it.
    lock.put(202)
    lock.route("ar_decode", rid, ["token"], [202])
    assert flag() is True, (
        "completion cleared the speculative-scheduling flag; the node's rids "
        "are still in flight and it will be reported ready again"
    )

    # Only the worker clearing it explicitly ends the speculation.
    lock.rt.set_in_flight("ar_decode", [rid], [WG_ID], False)
    assert flag() is False


def _streaming_loop_graph():
    """A loop whose only member takes nothing but a stream -- a vocoder
    decoding chunk by chunk inside the loop.

    ``hidden`` is also declared one of the loop's external inputs, which a
    hand-built graph can do (``_divide_into_worker_graphs`` strips streams
    from them): the runtime itself must not re-inject a streamed chunk.
    """
    talker = GraphNode(
        name="talker", input_names={"hidden"},
        outputs=[GraphEdge(name="audio", next_node=EMIT_TO_CLIENT)],
    )
    talker._register_streaming({"hidden"})
    return Sequential(sections=[
        GraphNode(
            name="prefill", input_names={"prompt"},
            outputs=[GraphEdge(name="hidden", next_node="talker",
                               is_streaming=True)],
        ),
        Loop(name="talk_loop", section=talker, outputs=[], max_iters=3),
    ])


def test_an_all_streaming_loop_member_takes_a_new_chunk_every_iteration():
    """Each iteration consumes the NEXT chunk, and nothing else.

    Re-injecting the first chunk as a loop external input on every advance
    made the node re-run on it and refuse the real next chunk (its slot was
    full).
    """
    lock = Script(_streaming_loop_graph())
    rid = lock.admit()
    for i in range(3):
        chunk, out = 100 + i, 200 + i
        lock.put(chunk)
        assert lock.stream(rid, "hidden", "talker", [chunk]) == [], (
            f"iteration {i}: the chunk was refused"
        )
        assert lock.ready() == [("talker", WALK, [rid])]
        popped = lock.pop("talker", rid)
        assert popped[2] == [("hidden", "talker", [chunk])], (
            f"iteration {i}: ran on a stale chunk"
        )
        lock.cleanup("talker", rid)
        lock.put(out)
        lock.route("talker", rid, ["audio"], [out])
        # Not ready again until the next chunk arrives.
        assert lock.ready() == []
        assert lock.held([chunk]) == []
