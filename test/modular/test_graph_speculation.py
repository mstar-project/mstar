"""Smoke tests for WorkerGraphIO speculation API.

Originally hand-authored by the refactor lead at
~/Downloads/disaggregation_research/multimodal_inference/smoke_test_graph_speculation.py.
"""
import pytest

from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mstar.graph.graph_io import WorkerGraphIO


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
    ready = io.ingest_for_speculation([
        GraphEdge(name="token", next_node="ar_decode"),
        GraphEdge(name="kv_cache", next_node="ar_decode"),
    ], "ar_decode")

    assert len(ready) == 1
    assert ready[0].node_name == "ar_decode"
    assert ready[0].is_new_loop_iter is True
    assert ready[0].loop_name == "ar_loop"


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
    ready = io.ingest_for_speculation(
        [GraphEdge(name="hidden", next_node="decode")], "encode"
    )

    assert len(ready) == 1
    assert ready[0].node_name == "decode"
    assert ready[0].is_new_loop_iter is False
    assert ready[0].loop_name is None


def test_partial_speculation_not_ready():
    io = WorkerGraphIO(_make_ar_graph())
    ready = io.ingest_for_speculation(
        [GraphEdge(name="token", next_node="ar_decode")], "ar_decode"
    )
    assert ready == []


def test_clear_speculative_inputs_wipes_buffers():
    io = WorkerGraphIO(_make_ar_graph())
    ready = io.ingest_for_speculation([
        GraphEdge(name="token", next_node="ar_decode"),
        GraphEdge(name="kv_cache", next_node="ar_decode"),
    ], "ar_decode")
    assert len(ready) == 1

    io.clear_speculative_inputs()
    assert not io.nodes["ar_decode"].speculative_signals.ready_names
    assert not io._nodes_with_speculative_inputs

    # The buffer is really gone: one edge alone no longer completes the node.
    ready = io.ingest_for_speculation(
        [GraphEdge(name="token", next_node="ar_decode")], "ar_decode"
    )
    assert ready == []


def test_duplicate_speculative_ingestion_raises():
    """Re-ingesting an edge without an intervening clear is a caller bug.

    Both worker call sites clear the speculative buffers immediately after
    ingesting, so a second ingest of the same name means the schedule leaked.
    """
    io = WorkerGraphIO(_make_ar_graph())
    e = GraphEdge(name="token", next_node="ar_decode")
    io.ingest_for_speculation([e], "ar_decode")
    assert io.nodes["ar_decode"].speculative_signals.ready_names == {"token"}

    with pytest.raises(AssertionError, match="already has ready input named token"):
        io.ingest_for_speculation([e], "ar_decode")

    # Clearing first makes the same ingest legal again.
    io.clear_speculative_inputs()
    assert io.ingest_for_speculation([e], "ar_decode") == []
    assert io.nodes["ar_decode"].speculative_signals.ready_names == {"token"}


def test_speculation_outside_graph_ignored():
    io = WorkerGraphIO(_make_ar_graph())
    ready = io.ingest_for_speculation(
        [GraphEdge(name="tokens", next_node="EMIT_TO_CLIENT")], "ar_decode"
    )
    assert ready == []
    assert not io._nodes_with_speculative_inputs


def test_graph_clear_wipes_node_speculative_buffer():
    io = WorkerGraphIO(_make_ar_graph())
    ready = io.ingest_for_speculation([
        GraphEdge(name="token", next_node="ar_decode"),
        GraphEdge(name="kv_cache", next_node="ar_decode"),
    ], "ar_decode")
    assert len(ready) == 1

    io.clear()
    assert not io.nodes["ar_decode"].speculative_signals.ready_names
    # WG-level tracking is NOT cleared by wg_state_registry.clear() — the caller
    # must use clear_speculative_inputs() when discarding a live spec schedule.


def _make_dit_graph():
    """A denoise loop whose step needs a conditioning input from outside the loop
    (re-injected every iteration) next to the loop-carried latents."""
    return Sequential(sections=[
        GraphNode(
            name="encode",
            input_names={"prompt"},
            outputs=[GraphEdge(name="text_embeds", next_node="dit")],
        ),
        Loop(
            name="denoise",
            section=GraphNode(
                name="dit",
                input_names={"text_embeds", "latents"},
                outputs=[GraphEdge(name="latents", next_node="dit")],
            ),
            outputs=[GraphEdge(name="latents", next_node="vae")],
            max_iters=4,
        ),
        GraphNode(
            name="vae",
            input_names={"latents"},
            outputs=[GraphEdge(name="image", next_node="EMIT_TO_CLIENT")],
        ),
    ])


def test_loop_back_speculation_counts_the_loops_persisted_inputs():
    io = WorkerGraphIO(_make_dit_graph())
    dit = io.get_node("dit")
    # nothing has arrived: the loop-back alone cannot make the next step ready
    assert io.ingest_for_speculation([GraphEdge(name="latents", next_node="dit")], "dit") == []
    io.clear_speculative_inputs()
    # the conditioning arrives (the loop marks it as re-injected every iteration)
    assert io.ingest_input(GraphEdge(name="text_embeds", next_node="dit"))
    assert dit.persisted_input_names() == {"text_embeds"}
    ready = io.ingest_for_speculation([GraphEdge(name="latents", next_node="dit")], "dit")
    assert [(r.node_name, r.is_new_loop_iter, r.loop_name) for r in ready] == [("dit", True, "denoise")]
    assert dit.is_ready_for_speculation(check_next_iter=True, allow_streaming=False)
    # the loop-back is what the speculation supplies; the persisted conditioning
    # alone does not make the next step ready
    io.clear_speculative_inputs()
    assert not dit.is_ready_for_speculation(check_next_iter=True, allow_streaming=False)
    # the next-iteration slot is untouched: the loop re-injects the input itself
    assert dit.ready_next_iter.ready_names == set()
