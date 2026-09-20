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


# ── speculative loop indices: what a speculated step should read ──────────────

def _nested_graph():
    # refine_loop ⊃ [denoise_loop(denoiser), refiner]. refiner -> denoiser is
    # refine_loop's loop-back, so speculating from refiner into denoiser
    # advances refine_loop and restarts denoise_loop at 0.
    return Sequential(sections=[
        Loop(
            name="refine_loop",
            section=Sequential(sections=[
                Loop(
                    name="denoise_loop",
                    section=GraphNode(
                        name="denoiser",
                        input_names={"latents"},
                        outputs=[GraphEdge(name="latents", next_node="denoiser")],
                    ),
                    outputs=[GraphEdge(name="latents", next_node="refiner")],
                    max_iters=3,
                ),
                GraphNode(
                    name="refiner",
                    input_names={"latents"},
                    outputs=[GraphEdge(name="latents", next_node="denoiser")],
                ),
            ]),
            outputs=[GraphEdge(name="latents", next_node="decoder")],
            max_iters=2,
        ),
        GraphNode(
            name="decoder",
            input_names={"latents"},
            outputs=[GraphEdge(name="image", next_node="EMIT_TO_CLIENT")],
        ),
    ])


def test_speculative_loop_indices_flat_loop_back_advances_the_loop():
    io = WorkerGraphIO(_make_ar_graph())
    io.loops["ar_loop"].curr_iter = 5
    ready = io.ingest_for_speculation([
        GraphEdge(name="token", next_node="ar_decode"),
        GraphEdge(name="kv_cache", next_node="ar_decode"),
    ], "ar_decode")
    assert ready[0].advancing_loop_name == "ar_loop"
    assert io.speculative_loop_indices(ready[0]) == {"ar_loop": 6}
    # a prediction only: the real routing still advances the io
    assert io.get_loop_indices() == {"ar_loop": 5}


def test_speculative_loop_indices_forward_transition_is_unchanged():
    io = WorkerGraphIO(_make_ar_graph())
    ready = io.ingest_for_speculation([
        GraphEdge(name="token", next_node="ar_decode"),
        GraphEdge(name="kv_cache", next_node="ar_decode"),
    ], "prefill")
    assert ready[0].is_new_loop_iter is False
    assert ready[0].advancing_loop_name is None
    assert io.speculative_loop_indices(ready[0]) == {"ar_loop": 0}


def test_speculative_loop_indices_nested_outer_advance_restarts_inner():
    io = WorkerGraphIO(_nested_graph())
    # denoise_loop just ran its last iteration (0..2) inside refine iter 0
    io.loops["denoise_loop"].curr_iter = 2
    io.loops["refine_loop"].curr_iter = 0
    ready = io.ingest_for_speculation(
        [GraphEdge(name="latents", next_node="denoiser")], "refiner"
    )
    assert len(ready) == 1
    info = ready[0]
    assert info.node_name == "denoiser"
    assert info.is_new_loop_iter is True
    assert info.loop_name == "denoise_loop"           # the destination's loop
    assert info.advancing_loop_name == "refine_loop"  # the one that moves
    assert io.speculative_loop_indices(info) == {
        "refine_loop": 1, "denoise_loop": 0,
    }


def test_speculative_loop_indices_nested_inner_loop_back():
    io = WorkerGraphIO(_nested_graph())
    io.loops["denoise_loop"].curr_iter = 1
    io.loops["refine_loop"].curr_iter = 1
    ready = io.ingest_for_speculation(
        [GraphEdge(name="latents", next_node="denoiser")], "denoiser"
    )
    info = ready[0]
    assert info.advancing_loop_name == "denoise_loop"
    assert io.speculative_loop_indices(info) == {
        "refine_loop": 1, "denoise_loop": 2,
    }
