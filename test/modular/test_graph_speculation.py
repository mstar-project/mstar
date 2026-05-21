"""Smoke tests for WorkerGraphIO speculation API.

Originally hand-authored by the refactor lead at
~/Downloads/disaggregation_research/multimodal_inference/smoke_test_graph_speculation.py.
"""
from mminf.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mminf.graph.graph_io import WorkerGraphIO


def _make_ar_graph():
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
            outputs=[GraphEdge(name="tokens", next_node="EMIT_TO_CLIENT")],
            max_iters=1000,
        ),
    ])


def test_loop_back_speculation():
    io = WorkerGraphIO(_make_ar_graph())
    io.ingest_for_speculation(GraphEdge(name="token", next_node="ar_decode"))
    io.ingest_for_speculation(GraphEdge(name="kv_cache", next_node="ar_decode"))

    ready = io.ready_for_speculation
    assert len(ready) == 1
    assert ready[0].node_name == "ar_decode"
    assert ready[0].is_new_loop_iter is True


def test_non_loop_back_speculation():
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
    io.ingest_for_speculation(GraphEdge(name="hidden", next_node="decode"))

    ready = io.ready_for_speculation
    assert len(ready) == 1
    assert ready[0].node_name == "decode"
    assert ready[0].is_new_loop_iter is False


def test_partial_speculation_not_ready():
    io = WorkerGraphIO(_make_ar_graph())
    io.ingest_for_speculation(GraphEdge(name="token", next_node="ar_decode"))
    assert io.ready_for_speculation == []


def test_clear_speculative_inputs_wipes_buffers():
    io = WorkerGraphIO(_make_ar_graph())
    io.ingest_for_speculation(GraphEdge(name="token", next_node="ar_decode"))
    io.ingest_for_speculation(GraphEdge(name="kv_cache", next_node="ar_decode"))
    assert len(io.ready_for_speculation) == 1

    io.clear_speculative_inputs()
    assert io.ready_for_speculation == []
    assert not io.nodes["ar_decode"].speculative_signals.ready_names
    assert not io._nodes_with_speculative_inputs
    assert not io._speculative_ready


def test_duplicate_speculative_ingestion_is_noop():
    io = WorkerGraphIO(_make_ar_graph())
    e = GraphEdge(name="token", next_node="ar_decode")
    io.ingest_for_speculation(e)
    io.ingest_for_speculation(e)

    assert io.nodes["ar_decode"].speculative_signals.ready_names == {"token"}
    assert len(io.ready_for_speculation) == 0


def test_speculation_outside_graph_ignored():
    io = WorkerGraphIO(_make_ar_graph())
    io.ingest_for_speculation(GraphEdge(name="tokens", next_node="EMIT_TO_CLIENT"))
    assert io.ready_for_speculation == []
    assert not io._nodes_with_speculative_inputs


def test_graph_clear_wipes_node_speculative_buffer():
    io = WorkerGraphIO(_make_ar_graph())
    io.ingest_for_speculation(GraphEdge(name="token", next_node="ar_decode"))
    io.ingest_for_speculation(GraphEdge(name="kv_cache", next_node="ar_decode"))
    assert len(io.ready_for_speculation) == 1

    io.clear()
    assert not io.nodes["ar_decode"].speculative_signals.ready_names
    # WG-level tracking is NOT cleared by wg_state_registry.clear() — the caller
    # must use clear_speculative_inputs() when discarding a live spec schedule.
