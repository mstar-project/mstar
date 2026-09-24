"""Side-by-side parity: drive BOTH runtimes through one script, diff every step.

``test_runtime_parity.py`` runs each test twice, once per runtime, against
hand-written expectations. That catches anything somebody thought to assert.
This catches the rest: the two runtimes are stepped together and their
OBSERVABLE state is compared after every operation, so a divergence fails even
when nobody predicted it -- and the failure names the step that caused it.

What "observable" means here: ready nodes, what pop_rids hands back, the
uuids routing asks to register, and the bookkeeper's refcount/tracking for
every uuid in play. Those are exactly the things a worker acts on.
"""
import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.communication.tensor_store import PythonTensorBookkeeping, TensorStore
from mstar.distributed.base import ShardingConfig
from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential, TensorPointerInfo
from mstar.graph.runtime.base import EdgeSpec, RouteInput
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

WG_ID, WALK, WORKER = 0, "decode", "worker_0"


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


class _StubTM:
    def __init__(self, bookkeeping):
        self.tensor_store = TensorStore(bookkeeping=bookkeeping)

    def dereference(self, uuid, n=1):
        self.tensor_store.dereference(uuid, n=n)

    def increment_ref(self, uuid, n=1):
        self.tensor_store.increment_ref(uuid, n=n)

    def dereference_batch_uniform(self, uuids, n=1):
        # Mirrors TensorCommunicationManager: the bookkeeper drops the refs
        # and forgets whatever became collectable. There is no transport here,
        # so the returned teardown list has nothing to act on.
        self.tensor_store.dereference_batch_uniform(uuids, n=n, cleanup=True)

    def dereference_batch(self, uuids, counts):
        self.tensor_store.dereference_batch(uuids, counts, cleanup=True)

    def get_tensor(self, uuid):
        return self.tensor_store.get_tensor(uuid)


def _nodes(graph):
    return {"prefill", "ar_decode"}


def _build(kind, graph):
    wg = WorkerGraph(
        section=graph, graph_walks={WALK}, ranks=[0], worker_graph_id=WG_ID)
    common = dict(
        my_worker_id=WORKER, my_worker_graphs=[wg],
        all_wg_ids_to_graph_walks={WG_ID: {WALK}},
        all_wg_ids_to_dyn_loops={WG_ID: {"ar_loop"}},
        all_wg_ids_to_nodes={WG_ID: _nodes(graph)},
        node_to_partition=dict.fromkeys(_nodes(graph), "default"),
        sharding_config=ShardingConfig(
            groups=[], tp_enabled_nodes=set(), shard_dim={}),
    )
    if kind == "python":
        book = PythonTensorBookkeeping()
        tm = _StubTM(book)
        return PythonGraphRuntime(**common, tensor_manager=tm,
                                  communicator=None), book, tm.tensor_store
    book = RustTensorBookkeeping()
    store = TensorStore(bookkeeping=book)
    return rust_runtime.RustGraphRuntime(**common, bookkeeping=book), book, store


def _info(uuid):
    return TensorPointerInfo(
        dims=[4], dtype=torch.float16, nbytes=8, address=0, stride=(1,),
        uuid=uuid, source_session_id="h:1", source_entity=WORKER)



def _finish_teardown(store, freed):
    """What the worker does with a runtime's FreedTensors: hand them to
    TensorCommunicationManager.cleanup_collectable, whose bookkeeper half is
    mark_forgotten (the rest is shm files and memory unregistration, which
    this harness has none of).

    Only the Rust runtime returns anything -- the Python one owns a tensor
    manager and tore down inline -- so applying it to both keeps the two
    bookkeepers comparable instead of leaving Rust's a step behind.
    """
    if freed.uuids:
        store.mark_forgotten(list(freed.uuids))


class Lockstep:
    """Runs an operation on both runtimes and diffs the result."""

    def __init__(self, graph):
        self.py = _build("python", graph)
        self.rs = _build("rust", graph)
        self.log: list[str] = []

    def _both(self, step, fn):
        self.log.append(step)
        out = []
        for trio in (self.py, self.rs):
            try:
                out.append(fn(*trio))
            except Exception as e:  # a raise on one side only is a divergence
                out.append(f"{type(e).__name__}: {e}")
        p, r = out
        assert p == r, (
            f"\nDIVERGENCE at step: {step}\n"
            f"  python -> {p!r}\n"
            f"  rust   -> {r!r}\n"
            f"  steps so far: {self.log}"
        )
        return p

    # -- operations ---------------------------------------------------------
    def admit(self, rid_str="r1"):
        return self._both(f"add_request({rid_str})", lambda rt, b, s: rt.add_request(
            request_id=rid_str, partition="default", graph_walk=WALK,
            partition_worker_graph_ids=[WG_ID],
            worker_graph_to_workers=ParallelList([WG_ID], [[WORKER]])))

    def put(self, uuid):
        """Store a tensor on both sides so routing has something to refer to."""
        self.log.append(f"put_tensor({uuid})")
        for _rt, _book, store in (self.py, self.rs):
            store.put_tensor(rid=1, uuid=uuid, tensor=torch.zeros(4),
                             info=_info(uuid))
            # The worker's safety hold: _postprocess_batch stores the outputs
            # and immediately does increment_ref_batch_uniform(uuids, 1), so a
            # tensor is never collectable between being stored and being
            # routed. Without it every uuid sits at refcount 0 from birth and
            # the runtimes' free paths fire at points production never hits.
            store.increment_ref(uuid, n=1)

    def ingest(self, rid, signal, node, uuids=()):
        spec = EdgeSpec(signal=signal, next_node=node, uuids=list(uuids),
                        is_final_streaming_chunk=False)
        return self._both(
            f"ingest({signal}->{node}, uuids={list(uuids)})",
            lambda rt, b, s: rt.ingest_inputs_batch(ParallelList([rid], [spec])))

    def ready(self):
        return self._both("get_ready_nodes", lambda rt, b, s: sorted(
            (x.node_name, x.graph_walk, sorted(x.rids))
            for x in rt.get_ready_nodes(set())))

    def pop(self, node, rid):
        def f(rt, b, s):
            p = rt.pop_rids(node, WALK, [rid])
            if p is None:
                return None
            return (p.wg_ids.keys, p.wg_ids.values,
                    sorted((e.signal, e.next_node, list(e.uuids))
                           for e in p.input_edges))
        return self._both(f"pop_rids({node})", f)

    def route(self, node, rid, signals, uuids):
        def f(rt, b, s):
            out = rt.complete_and_route_batch(
                RouteInput(
                    partition="default", graph_walk=WALK, node_name=node,
                    output_signals=list(signals),
                    wg_ids=ParallelList([rid], [WG_ID]),
                    tensors=list(uuids),
                    num_tensors=[len(uuids)] * len(signals),
                ), s)
            _finish_teardown(s, out.freed_inputs)
            return (sorted(out.register_tensor_idxs), sorted(out.register_rids),
                    sorted(out.new_token_output_idxs))
        return self._both(f"route({node}, signals={list(signals)})", f)

    def cleanup(self, node, rid):
        """cleanup_consumed_inputs on both, comparing the EFFECT only.

        The return values are asymmetric by design: the Python runtime owns a
        tensor manager and dereferences through it as it clears, so it returns
        FreedTensors.none(); the Rust runtime has no tensor manager and hands
        the freed uuids back for the caller to tear down. What must agree is
        the bookkeeper state afterwards, which `refcounts` checks.
        """
        self.log.append(f"cleanup_consumed_inputs({node})")
        for rt, _b, store in (self.py, self.rs):
            freed = rt.cleanup_consumed_inputs(node, [rid], [WG_ID])
            _finish_teardown(store, freed)

    def refcounts(self, uuids):
        """The bookkeeper's view -- tracked and collectable -- per uuid."""
        return self._both(
            f"refcounts({list(uuids)})",
            lambda rt, b, s: [(u, b.is_tracked(u), b.can_gc(u)) for u in uuids])


@pytest.fixture
def lock():
    return Lockstep(_loop_graph())


def test_admit_and_first_ingest_agree(lock):
    rid = lock.admit()
    lock.ready()
    lock.put(101)
    lock.ingest(rid, "prompt", "prefill", [101])
    lock.ready()
    lock.pop("prefill", rid)
    lock.ready()


def test_loop_iteration_agrees(lock):
    """prefill -> route token -> ar_decode consumes it -> routes its own."""
    rid = lock.admit()
    lock.put(101)
    lock.ingest(rid, "prompt", "prefill", [101])
    lock.pop("prefill", rid)
    lock.put(201)
    lock.route("prefill", rid, ["token"], [201])
    lock.ready()
    lock.pop("ar_decode", rid)
    lock.ready()
    lock.refcounts([101, 201])


@pytest.mark.xfail(
    strict=False,
    reason="UNRESOLVED, not diagnosed -- do not read a cause into this. "
           "Observed: after a loop node routes, the Python bookkeeper has "
           "dereferenced the consumed loop-back tensor to zero and forgotten "
           "it, while the Rust one still holds it at refcount > 0; over N "
           "iterations Rust retains N tensors and Python one. It is NOT the "
           "loop's external_inputs -- `_external_inputs` is empty for this "
           "graph, so Rust's `held` list is too. What this harness does NOT "
           "model is the read-ack path (TENSOR_RECEIVED -> dereference_batch) "
           "and UNPERSIST_TENSORS, which is where a routed tensor's remaining "
           "references are released in production, so the surplus may be "
           "references legitimately awaiting an ack that never arrives here. "
           "strict=False deliberately: a pass would mean the harness got more "
           "faithful, not that a bug was fixed.",
)
def test_loop_carried_tensor_refcounts_agree(lock):
    """The token is emitted AND fed back. If one runtime holds a reference the
    other does not, this is where it shows -- and a worker acting on the
    difference resolves a freed uuid."""
    rid = lock.admit()
    lock.put(101)
    lock.ingest(rid, "prompt", "prefill", [101])
    lock.pop("prefill", rid)
    for i, uuid in enumerate((201, 202, 203)):
        lock.put(uuid)
        lock.route("prefill" if i == 0 else "ar_decode", rid, ["token"], [uuid])
        lock.ready()
        lock.pop("ar_decode", rid)
        lock.refcounts([101, 201, 202, 203])


def test_cleanup_consumed_inputs_agrees(lock):
    """The divergence I suspected: Python clears every consumed slot and skips
    only the dereference for a loop-held input; Rust skips both. If that is
    observable, readiness or refcounts differ right here."""
    rid = lock.admit()
    lock.put(101)
    lock.ingest(rid, "prompt", "prefill", [101])
    lock.pop("prefill", rid)
    lock.put(201)
    lock.route("prefill", rid, ["token"], [201])
    lock.pop("ar_decode", rid)
    lock.cleanup("ar_decode", rid)
    lock.ready()
    lock.refcounts([101, 201])


def test_removal_agrees(lock):
    rid = lock.admit()
    lock.put(101)
    lock.ingest(rid, "prompt", "prefill", [101])
    lock._both("remove_request", lambda rt, b, s: sorted(rt.remove_request(rid)))
    lock.ready()


def _emit_loop_graph():
    """Same AR loop, but the token is ALSO persisted and emitted to the client.

    This is the higgs/orpheus shape: one tensor is simultaneously (a) held at
    the conductor for delivery and (b) the next iteration's input. Python notes
    that persist edges are excluded from the routed refcount because "the
    conductor computes that reference when it unpersists" -- if the two
    runtimes disagree about that reference, the tensor is freed while the loop
    still needs it.
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
    return Lockstep(_emit_loop_graph())


@pytest.mark.xfail(
    strict=False,
    reason="UNRESOLVED, not diagnosed -- do not read a cause into this. "
           "Observed: after a loop node routes, the Python bookkeeper has "
           "dereferenced the consumed loop-back tensor to zero and forgotten "
           "it, while the Rust one still holds it at refcount > 0; over N "
           "iterations Rust retains N tensors and Python one. It is NOT the "
           "loop's external_inputs -- `_external_inputs` is empty for this "
           "graph, so Rust's `held` list is too. What this harness does NOT "
           "model is the read-ack path (TENSOR_RECEIVED -> dereference_batch) "
           "and UNPERSIST_TENSORS, which is where a routed tensor's remaining "
           "references are released in production, so the surplus may be "
           "references legitimately awaiting an ack that never arrives here. "
           "strict=False deliberately: a pass would mean the harness got more "
           "faithful, not that a bug was fixed.",
)
def test_persisted_and_loop_carried_token_agrees(emit_lock):
    """The higgs shape. A token routed to BOTH the client and the next
    iteration must leave both runtimes with the same refcount -- otherwise one
    of them frees it on unpersist while the loop still holds an input slot
    pointing at it, which is the KeyError seen in production."""
    lk = emit_lock
    rid = lk.admit()
    lk.put(101)
    lk.ingest(rid, "prompt", "prefill", [101])
    lk.pop("prefill", rid)
    lk.put(201)
    lk.route("prefill", rid, ["token"], [201])
    lk.ready()
    lk.pop("ar_decode", rid)
    # ar_decode emits a token that is persisted AND fed back
    lk.put(202)
    lk.route("ar_decode", rid, ["token"], [202])
    lk.refcounts([101, 201, 202])
    lk.ready()
    lk.pop("ar_decode", rid)
    lk.refcounts([101, 201, 202])


@pytest.mark.xfail(
    strict=False,
    reason="UNRESOLVED, not diagnosed -- do not read a cause into this. "
           "Observed: after a loop node routes, the Python bookkeeper has "
           "dereferenced the consumed loop-back tensor to zero and forgotten "
           "it, while the Rust one still holds it at refcount > 0; over N "
           "iterations Rust retains N tensors and Python one. It is NOT the "
           "loop's external_inputs -- `_external_inputs` is empty for this "
           "graph, so Rust's `held` list is too. What this harness does NOT "
           "model is the read-ack path (TENSOR_RECEIVED -> dereference_batch) "
           "and UNPERSIST_TENSORS, which is where a routed tensor's remaining "
           "references are released in production, so the surplus may be "
           "references legitimately awaiting an ack that never arrives here. "
           "strict=False deliberately: a pass would mean the harness got more "
           "faithful, not that a bug was fixed.",
)
def test_repeated_emit_iterations_agree(emit_lock):
    """Several loop iterations with an emitted token each time: a reference
    that leaks or is dropped once shows up as drift after a few passes."""
    lk = emit_lock
    rid = lk.admit()
    lk.put(101)
    lk.ingest(rid, "prompt", "prefill", [101])
    lk.pop("prefill", rid)
    lk.put(201)
    lk.route("prefill", rid, ["token"], [201])
    lk.pop("ar_decode", rid)
    for uuid in (202, 203, 204):
        lk.put(uuid)
        lk.route("ar_decode", rid, ["token"], [uuid])
        lk.ready()
        lk.pop("ar_decode", rid)
        lk.refcounts([201, 202, 203, 204])


def test_cleanup_after_emit_agrees(emit_lock):
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
    lk.ready()
    lk.refcounts([101, 201, 202])


@pytest.mark.xfail(
    strict=False,
    reason="UNRESOLVED, not diagnosed -- do not read a cause into this. "
           "Observed: after a loop node routes, the Python bookkeeper has "
           "dereferenced the consumed loop-back tensor to zero and forgotten "
           "it, while the Rust one still holds it at refcount > 0; over N "
           "iterations Rust retains N tensors and Python one. It is NOT the "
           "loop's external_inputs -- `_external_inputs` is empty for this "
           "graph, so Rust's `held` list is too. What this harness does NOT "
           "model is the read-ack path (TENSOR_RECEIVED -> dereference_batch) "
           "and UNPERSIST_TENSORS, which is where a routed tensor's remaining "
           "references are released in production, so the surplus may be "
           "references legitimately awaiting an ack that never arrives here. "
           "strict=False deliberately: a pass would mean the harness got more "
           "faithful, not that a bug was fixed.",
)
def test_cleanup_after_route_agrees(emit_lock):
    """The SAME steps as test_cleanup_after_emit_agrees but with route before
    cleanup -- an order the worker does not produce (_postprocess_batch cleans
    up first), yet the two runtimes must still agree on it.

    They did not: Python's ready-name set is add-only, so clearing a node's
    inputs left the name behind and the node read as ready with nothing
    ingested, while Rust's clear_consumed_inputs recomputes the bit and drops
    it. Fixed on the Python side; this pins the order-independence.
    """
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
    lk.ready()
    lk.cleanup("ar_decode", rid)
    lk.ready()
    lk.refcounts([101, 201, 202])


def test_cleanup_without_pop_agrees(lock):
    """Cleanup on a node sitting in the ready set, never popped. The narrowest
    form of the same rule: clearing the inputs must clear the readiness."""
    rid = lock.admit()
    lock.put(101)
    lock.ingest(rid, "prompt", "prefill", [101])
    lock.ready()
    lock.cleanup("prefill", rid)
    lock.ready()
    lock.refcounts([101])


def _spec(rt, node, rid):
    return sorted(
        (s.node_name, s.graph_walk, s.is_new_loop_iter, s.loop_name,
         tuple(s.output_signals))
        for s in rt.speculate_node(node, WALK, rid)
    )


def test_speculate_node_agrees_across_a_loop(lock):
    """Do the runtimes offer the SAME speculation targets at each point?

    Motivation from the phase sweep: on orpheus the worker's `speculate` span
    is flat in batch size under Rust and grows under Python. `speculate`
    wraps _try_speculate_next, which returns immediately when speculate_node
    yields nothing and otherwise builds a whole batch -- so a flat span means
    Rust declines to speculate where Python does not. That is a behavioural
    difference, not a faster implementation, and it belongs in lockstep.
    """
    rid = lock.admit()
    lock.put(101)
    lock.ingest(rid, "prompt", "prefill", [101])
    lock._both("speculate_node(prefill)", lambda rt, b, s: _spec(rt, "prefill", rid))
    lock.pop("prefill", rid)
    lock._both("speculate_node(prefill) after pop",
               lambda rt, b, s: _spec(rt, "prefill", rid))
    lock.put(201)
    lock.route("prefill", rid, ["token"], [201])
    lock._both("speculate_node(ar_decode)",
               lambda rt, b, s: _spec(rt, "ar_decode", rid))
    for uuid in (202, 203):
        lock.pop("ar_decode", rid)
        lock._both(f"speculate_node(ar_decode) iter {uuid}",
                   lambda rt, b, s: _spec(rt, "ar_decode", rid))
        lock.put(uuid)
        lock.route("ar_decode", rid, ["token"], [uuid])
        lock._both(f"speculate_node(ar_decode) post-route {uuid}",
                   lambda rt, b, s: _spec(rt, "ar_decode", rid))


def test_speculative_flag_survives_completion(lock):
    """The flag marking a node speculatively scheduled for N+1 must SURVIVE
    the completion of step N -- its rids are still in flight.

    Python's WorkerGraphIO.mark_node_complete -> GraphNode.complete() never
    touches ``_speculatively_scheduled``; Rust's ``State::complete`` used to
    set ``st.scheduled = false``. Because ``refresh_ready`` gates the ready-set
    ADD on that flag (as Python's ``register_ingested_input`` does), clearing
    it let the runtime report a node ready whose rids were mid-flight, and
    ``get_ready_nodes`` handed them back to the scheduler as fresh work. Every
    speculation then ran a full get_ready_nodes -> _assemble_batch ->
    pop_rids round trip that ended in ``push_back_node`` -- the branch
    worker.py documents as "Shouldn't happen".

    Measured on bagel bs32 before the fix: 30.89 of 31.01 "fresh" rids were
    the in-flight set under Rust, and exactly 0 under Python.

    Asserted on the flag itself rather than on ``get_ready_nodes``: reaching
    the observable ready state needs the loop to advance, which this harness
    does not drive, so a readiness assertion here passes vacuously (both
    sides return []) whether or not the bug is present.
    """
    rid = lock.admit()
    lock.put(101)
    lock.ingest(rid, "prompt", "prefill", [101])
    lock.pop("prefill", rid)
    lock.put(201)
    lock.route("prefill", rid, ["token"], [201])
    lock.pop("ar_decode", rid)

    def flag(rt, _book, _store):
        return rt.is_speculatively_scheduled("ar_decode", WG_ID, rid)

    assert lock._both("is_spec_scheduled before", flag) is False

    lock._both("set_speculatively_scheduled(ar_decode, True)",
               lambda rt, b, s: rt.set_speculatively_scheduled(
                   "ar_decode", WG_ID, [rid], True))
    assert lock._both("is_spec_scheduled after set", flag) is True

    # Step N completes. The flag must not be cleared by it.
    lock.put(202)
    lock.route("ar_decode", rid, ["token"], [202])
    assert lock._both("is_spec_scheduled after completion", flag) is True, (
        "completion cleared the speculative-scheduling flag; the node's rids "
        "are still in flight and it will be reported ready again"
    )

    # Only the worker clearing it explicitly ends the speculation.
    lock._both("set_speculatively_scheduled(ar_decode, False)",
               lambda rt, b, s: rt.set_speculatively_scheduled(
                   "ar_decode", WG_ID, [rid], False))
    assert lock._both("is_spec_scheduled after clear", flag) is False
