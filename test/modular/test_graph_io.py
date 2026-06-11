"""Regression tests for bug fixes in graph/base.py + graph_io.py.

Each test pins down a specific behavior that the prior implementation got
wrong; if any of these regress, the underlying bugs have re-surfaced.
"""

from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mstar.graph.graph_io import WorkerGraphIO


def _ar_loop_graph(max_iters: int = 5):
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


def _drive(io: WorkerGraphIO, initial: list[GraphEdge], on_step=None,
           max_steps: int = 50):
    pending = list(initial)
    step = 0
    while not io.wg_state_registry.is_done and step < max_steps:
        remaining = []
        for edge in pending:
            if not io.ingest_input(edge):
                remaining.append(edge)
        pending = remaining

        ready = list(io.ready_node_names)
        if not ready:
            return step
        name = ready[0]
        io.ready_node_names.discard(name)
        step += 1
        if on_step:
            on_step(name, io)
        completion = io.mark_node_complete(name)
        for edge in completion.output_edges:
            if (edge.name, edge.next_node) in completion.filtered_signals:
                continue
            if edge.next_node in io.nodes:
                pending.append(edge)
    return step


def test_clear_resets_finish_signal():
    """After register_loop_finish_signal + clear(), the loop must run its
    full max_iters on the next forward pass — the finish signal does not
    persist."""
    io = WorkerGraphIO(_ar_loop_graph(max_iters=10))
    decode_count = [0]

    def on_step(name, io_mgr):
        if name != "ar_decode":
            return
        decode_count[0] += 1
        if decode_count[0] == 3:
            io_mgr.register_loop_finish_signal("ar_loop")

    _drive(io, [GraphEdge(name="prompt", next_node="prefill")], on_step=on_step)
    assert decode_count[0] == 3
    assert io.loops["ar_loop"]._finish_signal is True

    io.clear()
    assert io.loops["ar_loop"]._finish_signal is False
    assert io.loops["ar_loop"].is_done is False
    assert io.loops["ar_loop"].curr_iter == 0

    def count_only(name, _io):
        if name == "ar_decode":
            decode_count[0] += 1

    decode_count[0] = 0
    _drive(io, [GraphEdge(name="prompt", next_node="prefill")], on_step=count_only)
    # No finish signal this run; should run the full max_iters=10.
    assert decode_count[0] == 10


def test_top_level_node_ready_signals_clear_on_complete():
    """A top-level (non-loop) GraphNode's ready_signals must be cleared
    after mark_node_complete, so a future ingest doesn't fall through to
    ready_next_iter and leak across forward passes."""
    graph = Sequential(sections=[
        GraphNode(
            name="encode",
            input_names={"prompt"},
            outputs=[GraphEdge(name="hidden", next_node="decode")],
        ),
        GraphNode(
            name="decode",
            input_names={"hidden"},
            outputs=[GraphEdge(name="output", next_node="EMIT_TO_CLIENT")],
        ),
    ])
    io = WorkerGraphIO(graph)
    io.ingest_input(GraphEdge(name="prompt", next_node="encode"))
    assert io.nodes["encode"].ready_signals.ready_names == {"prompt"}

    io.ready_node_names.discard("encode")
    io.mark_node_complete("encode")

    # After completion, ready_signals must be empty so the same name can be
    # ingested for the next forward pass without falling through to
    # ready_next_iter (which has no semantic meaning for a top-level node).
    assert io.nodes["encode"].ready_signals.ready_names == set()
    assert io.nodes["encode"].ready_signals.is_ready is False


def test_speculation_strict_mode_partial_spec_not_ready():
    """A speculative ingest that covers only some inputs must NOT mark the
    node as ready, even when ready_signals already has the rest. The gate
    is on speculative_signals alone."""
    io = WorkerGraphIO(_ar_loop_graph())

    # Simulate the running iter having ingested both inputs (via prefill →
    # ar_decode, then we manually mark the slot full).
    ar_decode = io.nodes["ar_decode"]
    ar_decode.ready_signals.update(
        GraphEdge(name="token", next_node="ar_decode")
    )
    ar_decode.ready_signals.update(
        GraphEdge(name="kv_cache", next_node="ar_decode")
    )
    assert ar_decode.ready_signals.is_ready

    # Now speculatively ingest ONLY one of the two anticipated outputs.
    io.ingest_for_speculation(GraphEdge(name="token", next_node="ar_decode"))
    # Strict gate: speculative_signals alone doesn't cover input_names, so
    # the node must not appear in ready_for_speculation despite the union
    # with ready_signals being a full set.
    assert io.ready_for_speculation == []

    io.ingest_for_speculation(GraphEdge(name="kv_cache", next_node="ar_decode"))
    ready = io.ready_for_speculation
    assert len(ready) == 1
    assert ready[0].node_name == "ar_decode"
    assert ready[0].is_new_loop_iter is True


def test_triple_deliver_returns_false():
    """ingest_input called three times for the same edge name returns False
    on the third call — only ready_signals + ready_next_iter slots exist.
    Soft rejection (rather than raising) lets streaming back-pressure work:
    the worker's StreamBuffer re-queues the rejected edge until the consumer
    catches up. Same return semantics is used for the cross-walk persist case
    (edge name not in destination node's input_names)."""
    graph = Sequential(sections=[
        GraphNode(
            name="x",
            input_names={"a"},
            outputs=[],
        ),
    ])
    io = WorkerGraphIO(graph)
    edge = GraphEdge(name="a", next_node="x")
    assert io.ingest_input(edge) is True
    assert io.ingest_input(edge) is True
    assert io.ingest_input(edge) is False


def test_ingest_input_rejects_unknown_input_name():
    """An edge whose name is not in the destination node's input_names
    returns False (does NOT raise). Mirrors the Q3-Omni ``talker_input_embeds
    → Talker`` cross-walk persist edge that lands on a Talker node in the
    current walk that doesn't take that input."""
    graph = Sequential(sections=[
        GraphNode(
            name="x",
            input_names={"a"},
            outputs=[],
        ),
    ])
    io = WorkerGraphIO(graph)
    assert io.ingest_input(GraphEdge(name="not_an_input", next_node="x")) is False
    # The node's ready state must be untouched.
    assert io.nodes["x"].ready_signals.ready_names == set()


def test_streaming_inputs_propagate_to_ready_signals():
    """When _register_streaming is called on a GraphNode, the existing
    ReadySignals instances must see the new streaming names. Regression
    test for the rebound-set bug."""
    node = GraphNode(
        name="x",
        input_names={"a", "b"},
        outputs=[],
    )
    # Initially no streaming inputs known.
    assert node.ready_signals.streaming_inputs == set()

    # Simulate what _divide_into_worker_graphs does.
    node._register_streaming({"a"})

    # The ReadySignals objects captured the set BY REFERENCE in __post_init__,
    # so this should now see "a" without any further plumbing.
    assert node.ready_signals.streaming_inputs == {"a"}
    assert node.ready_next_iter.streaming_inputs == {"a"}
    assert node.speculative_signals.streaming_inputs == {"a"}
    assert node.consumes_stream is True
