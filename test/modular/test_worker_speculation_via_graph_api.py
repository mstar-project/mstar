"""Pin down the graph-layer surface that the worker's speculation path uses.

These tests do not spin up a worker — instead they exercise the
``ingest_for_speculation`` / ``ready_for_speculation`` API in the exact shape
that ``Worker._try_speculate_next`` relies on:

  - Per-rid wgios (deep-copied from the same graph) maintain independent
    ``speculative_signals`` state, so ingesting on one rid does not leak into
    another.
  - Ingesting every loop-back output of a node populates the strict gate to
    fire with ``is_new_loop_iter=True``, which is what the worker checks.
  - ``clear_speculative_inputs`` resets the slot so the next outer iter's
    ingestion isn't a no-op (the ``edge.name in node.speculative_signals
    .ready_names`` short-circuit in ``WorkerGraphIO.ingest_for_speculation``).
  - On the gate-failure rollback path: partial ingestion does NOT advance the
    strict gate even when ``ready_signals`` happens to be full from the
    in-flight iter (the canonical readiness invariant, re-validated for the
    worker's call pattern).
"""
from copy import deepcopy

import pytest

from mminf.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mminf.graph.graph_io import WorkerGraphIO


def _ar_loop_graph(max_iters: int = 5):
    """Two-input AR decode loop (token + kv_cache), all loop-back, no streaming."""
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
            outputs=[],
            max_iters=max_iters,
        ),
    ])


def _per_rid_wgios(num_rids: int) -> list[WorkerGraphIO]:
    """Mirror ``WorkerGraphQueues.add_request`` — deepcopy the section per rid
    so each gets independent state."""
    section_template = _ar_loop_graph()
    return [WorkerGraphIO(deepcopy(section_template)) for _ in range(num_rids)]


def _ingest_all_loop_back(wgio: WorkerGraphIO, node_name: str) -> None:
    """Mirror the per-rid loop the worker uses: ingest every loop-back output
    of ``node_name`` into ``wgio``'s ``speculative_signals``."""
    rid_node = wgio.nodes[node_name]
    for edge in rid_node.outputs:
        if edge.next_node == rid_node.name:
            wgio.ingest_for_speculation(edge)


def test_per_rid_speculative_state_is_isolated():
    """The worker assumes per-rid wgios are independent: deep-copying the
    graph section yields wgios whose ``speculative_signals`` slots do not
    share state.

    Without this, ingesting anticipated outputs for one rid would leak into
    another rid's gate decision — a bug that would make the worker's gate
    check on the first rid an unreliable proxy for the rest of the batch."""
    wgios = _per_rid_wgios(num_rids=3)
    _ingest_all_loop_back(wgios[0], "ar_decode")

    # First rid's gate fires.
    spec0 = wgios[0].ready_for_speculation
    assert len(spec0) == 1
    assert spec0[0].node_name == "ar_decode"
    assert spec0[0].is_new_loop_iter is True

    # Other rids: untouched.
    for wgio in wgios[1:]:
        assert wgio.ready_for_speculation == []
        assert wgio.nodes["ar_decode"].speculative_signals.ready_names == set()


def test_first_rid_gate_match_shape():
    """The worker inspects ``first_wgio.ready_for_speculation`` for the
    specific ``SpeculativeNodeInfo`` shape: node_name matching the in-flight
    batch's node_name AND is_new_loop_iter=True. Pin both fields down."""
    wgio = _per_rid_wgios(1)[0]
    _ingest_all_loop_back(wgio, "ar_decode")

    matches = [
        info for info in wgio.ready_for_speculation
        if info.node_name == "ar_decode" and info.is_new_loop_iter
    ]
    assert len(matches) == 1


def test_clear_speculative_inputs_resets_for_next_outer_iter():
    """The worker's end-of-function clear (and the abort-path clear) must
    reset the slot, otherwise the next outer iter's ingest_for_speculation
    hits the ``edge.name in node.speculative_signals.ready_names`` short-
    circuit and becomes a no-op — gate would silently never fire after the
    first call."""
    wgio = _per_rid_wgios(1)[0]
    _ingest_all_loop_back(wgio, "ar_decode")
    assert wgio.ready_for_speculation  # fires once

    wgio.clear_speculative_inputs()
    assert wgio.ready_for_speculation == []
    assert wgio.nodes["ar_decode"].speculative_signals.ready_names == set()

    # Next outer iter: ingestion must work again.
    _ingest_all_loop_back(wgio, "ar_decode")
    assert len(wgio.ready_for_speculation) == 1


def test_gate_fails_when_only_partial_outputs_ingested():
    """If the spec node has multiple loop-back outputs and the worker only
    ingests SOME of them, the strict gate must refuse — the worker's
    rollback path depends on this so we don't speculate on an incompletely
    anticipated node."""
    wgio = _per_rid_wgios(1)[0]
    # Pre-load ready_signals to simulate the in-flight iter holding its
    # inputs; the strict gate is on speculative_signals ALONE, not the union.
    ar_decode = wgio.nodes["ar_decode"]
    ar_decode.ready_signals.update(GraphEdge(name="token", next_node="ar_decode"))
    ar_decode.ready_signals.update(GraphEdge(name="kv_cache", next_node="ar_decode"))
    assert ar_decode.ready_signals.is_ready

    # Worker ingests only ONE of the two anticipated outputs.
    wgio.ingest_for_speculation(GraphEdge(name="token", next_node="ar_decode"))
    assert wgio.ready_for_speculation == []


def test_gate_fires_on_full_ingest_after_partial_was_attempted():
    """Sequence: partial ingest → no gate, full ingest → gate fires. Mirrors
    the worker's per-rid loop iterating all of ``rid_node.outputs`` even when
    the first few don't yet cover ``input_names``."""
    wgio = _per_rid_wgios(1)[0]
    wgio.ingest_for_speculation(GraphEdge(name="token", next_node="ar_decode"))
    assert wgio.ready_for_speculation == []

    wgio.ingest_for_speculation(GraphEdge(name="kv_cache", next_node="ar_decode"))
    ready = wgio.ready_for_speculation
    assert len(ready) == 1
    assert ready[0].is_new_loop_iter is True


def test_speculative_signals_sees_tensor_info_via_shared_reference():
    """The promotion path depends on a strict invariant:
    ``ingest_for_speculation`` stores the producer's edge BY REFERENCE in
    ``speculative_signals.ready_inputs[name]``. When
    ``_store_outputs_and_finish_loops`` later mutates
    ``producer.outputs[i].tensor_info`` in place, the spec slot's tensor_info
    reflects that mutation — this is what enables
    ``_gather_spec_inputs_from_speculative_signals`` to read N's outputs
    without a separate thread step.

    Bug shape this guards against: if any layer ever started DEEPCOPYING the
    edge during ingest_for_speculation (or update), the spec gather would
    read a stale empty tensor_info and silently drop continuing rids.
    """
    from mminf.graph.base import TensorPointerInfo

    wgio = _per_rid_wgios(1)[0]
    rid_node = wgio.nodes["ar_decode"]
    # The producer edge for the loop-back ``token`` output.
    producer_token_edge = next(
        e for e in rid_node.outputs if e.name == "token" and e.next_node == "ar_decode"
    )
    assert producer_token_edge.tensor_info == []

    # Worker calls ingest_for_speculation on the producer's edge.
    wgio.ingest_for_speculation(producer_token_edge)
    spec_slot_edge = rid_node.speculative_signals.ready_inputs["token"]
    # Must be the SAME object (by reference, not a copy).
    assert spec_slot_edge is producer_token_edge

    # Simulate _store_outputs_and_finish_loops appending tensor_info on the
    # producer side (mutation in place, NOT replacement of the list).
    fake_info = TensorPointerInfo(
        dims=[1], dtype="float32", nbytes=4, address=0,
        stride=[1], uuid="uuid-deadbeef", source_session_id="test",
        source_entity="test",
    )
    producer_token_edge.tensor_info.append(fake_info)

    # Gather sees the new tensor_info via the spec slot.
    gathered = wgio.nodes["ar_decode"].speculative_signals.ready_inputs["token"]
    assert gathered.tensor_info == [fake_info]
    assert gathered.tensor_info is producer_token_edge.tensor_info  # same list


def test_speculative_node_info_carries_loop_name():
    """``SpeculativeNodeInfo.loop_name`` must be populated for nodes that
    live inside a Loop (so the worker can look up ``wgio.loops[loop_name]``
    to check ``curr_iter``/``max_iters``/``_finish_signal`` before deciding
    whether to speculate), and ``None`` for top-level nodes.
    """
    # In-loop node: ar_decode inside ar_loop.
    wgio = _per_rid_wgios(1)[0]
    _ingest_all_loop_back(wgio, "ar_decode")
    ready = wgio.ready_for_speculation
    assert len(ready) == 1
    assert ready[0].node_name == "ar_decode"
    assert ready[0].loop_name == "ar_loop"


def test_speculative_node_info_loop_name_none_for_top_level():
    """A top-level node (no enclosing Loop) reports ``loop_name=None``."""
    # Graph: a single top-level node with a loop-back to itself (no Loop
    # wrapper). It's structurally degenerate but exercises the None path.
    graph = Sequential(sections=[
        GraphNode(
            name="solo",
            input_names={"x"},
            outputs=[GraphEdge(name="x", next_node="solo")],
        ),
    ])
    wgio = WorkerGraphIO(graph)
    wgio.ingest_for_speculation(GraphEdge(name="x", next_node="solo"))
    ready = wgio.ready_for_speculation
    # Top-level node has no LoopStateRegistry → loop_name None.
    # (is_new_loop_iter is also False here because the edge isn't in any
    # Loop's _loop_back_inputs — there's no Loop.)
    assert len(ready) == 1
    assert ready[0].node_name == "solo"
    assert ready[0].loop_name is None
    assert ready[0].is_new_loop_iter is False


def test_loop_state_signals_last_iter_via_curr_iter_plus_one():
    """Sanity check that the worker's drop condition
    (``curr_iter + 1 >= max_iters``) correctly identifies "iter N is the
    last iter" for a Loop. Pin this so the worker-side filter doesn't drift
    out of sync with how Loop.complete_iter decides done."""
    graph = _ar_loop_graph(max_iters=3)
    wgio = WorkerGraphIO(graph)

    loop = wgio.loops["ar_loop"]
    # iter 0 about to run: curr_iter=0, 0+1<3 → not last.
    assert loop.curr_iter == 0
    assert not (loop.curr_iter + 1 >= loop.max_iters)

    # Simulate two iter advances (what complete_iter does for non-terminal iters).
    loop.curr_iter = 1
    assert not (loop.curr_iter + 1 >= loop.max_iters)
    loop.curr_iter = 2
    # iter 2 about to run: 2+1==3==max_iters → last iter, spec for iter 3 wasted.
    assert loop.curr_iter + 1 >= loop.max_iters


def test_loop_finish_signal_also_signals_last_iter():
    """``_finish_signal=True`` means the loop terminates at the next
    ``complete_iter`` regardless of ``curr_iter``. The worker drops spec in
    that case too."""
    graph = _ar_loop_graph(max_iters=10)
    wgio = WorkerGraphIO(graph)
    loop = wgio.loops["ar_loop"]
    # Plenty of iters left by count, but external stop fired.
    assert loop.curr_iter + 1 < loop.max_iters
    wgio.register_loop_finish_signal("ar_loop")
    assert loop._finish_signal is True


def test_prefill_to_decode_transition_gate_shape():
    """A prefill → decode forward transition is the canonical non-same-node
    speculation case. The worker ingests prefill's outputs (which target
    ar_decode); ar_decode becomes speculatively ready.

    ``is_new_loop_iter`` is True because ``("token", "ar_decode")`` matches
    ``ar_loop._loop_back_inputs`` (the loop's input edge set; populated by
    ``GraphNode.get_inputs_outputs`` whenever an input name also appears in
    the node's outputs). The graph layer doesn't distinguish "prefill's
    output to ar_decode" from "ar_decode's loop-back to itself" — both fire
    a new loop iter. The worker distinguishes the two cases at a different
    level: ``gate_match.node_name == batch_N.node_name`` means same-node
    loop-body; ``!=`` means forward transition.
    """
    wgio = _per_rid_wgios(1)[0]
    prefill_node = wgio.nodes["prefill"]
    # Ingest every prefill output (the worker's per-rid loop does this for
    # every edge in ``sample_node.outputs``, not just loop-back).
    for edge in prefill_node.outputs:
        wgio.ingest_for_speculation(edge)

    ready = wgio.ready_for_speculation
    assert len(ready) == 1
    assert ready[0].node_name == "ar_decode"
    assert ready[0].is_new_loop_iter is True  # enters ar_loop's iter 0
    assert ready[0].loop_name == "ar_loop"
    # The key difference from same-node-loop-body: the gate target's name
    # differs from prefill's. The worker uses this to decide whether to
    # treat the spec as a loop continuation or a forward transition.
    assert ready[0].node_name != prefill_node.name


def test_consumed_edges_pair_shape_for_forward_transition():
    """The worker computes ``consumed_edges`` for the spec batch as
    ``{(edge.name, edge.next_node) for edge in sample_node.outputs if
    edge.next_node == spec_target_node_name}``. Verify the shape: for a
    prefill→decode transition, the consumed pairs target ar_decode."""
    wgio = _per_rid_wgios(1)[0]
    prefill_node = wgio.nodes["prefill"]
    spec_target_name = "ar_decode"

    consumed = {
        (edge.name, edge.next_node)
        for edge in prefill_node.outputs
        if edge.next_node == spec_target_name
    }
    assert consumed == {("token", "ar_decode"), ("kv_cache", "ar_decode")}
    # All next_node fields equal the spec target — the worker-side filter in
    # ``_fast_postprocess_route`` uses these pairs to skip routing prefill's
    # outputs to ar_decode's ``ready_signals`` (the spec already consumed
    # them via ``speculative_signals``).
    assert all(dest == spec_target_name for _, dest in consumed)


def test_non_loop_back_node_does_not_appear_in_ready_for_speculation():
    """The worker only ingests outputs whose ``next_node == node.name`` (the
    loop-back filter in its per-rid loop). A non-loop-back downstream
    destination would only see one of its inputs and never satisfy the strict
    gate. This is the worker-side invariant that lets us avoid spurious spec
    fires on, e.g., the prefill → ar_decode hop."""
    # Build a graph where prefill outputs land on ar_decode (non-loop-back
    # destination from prefill's perspective).
    section = _ar_loop_graph()
    wgio = WorkerGraphIO(section)

    # Ingest prefill's outputs (mirror what the worker would do IF it were
    # speculating on prefill — which it would NOT, since prefill has no
    # loop-back outputs; but we want to confirm the safety net.)
    prefill_node = wgio.nodes["prefill"]
    for edge in prefill_node.outputs:
        if edge.next_node == prefill_node.name:
            wgio.ingest_for_speculation(edge)

    # ar_decode received nothing speculative; no gate.
    assert wgio.ready_for_speculation == []
