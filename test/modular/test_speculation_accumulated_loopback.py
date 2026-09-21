"""A loop that gathers its per-iteration output for the client (``Loop.accumulated_outputs``)
routes that output back to its own node without listing it as an input. The regular
ingest declines such an edge, and the speculative ingest has to skip it too instead of
asserting, or async scheduling breaks for every model that accumulates tokens."""
from copy import deepcopy

from mstar.graph.base import GraphEdge, GraphNode, Loop, Sequential
from mstar.graph.graph_io import WorkerGraphIO
from mstar.graph.special_destinations import EMIT_TO_CLIENT


def _accumulating_decode_graph(max_iters: int = 4) -> Sequential:
    return Sequential(sections=[
        GraphNode(
            name="prefill",
            input_names={"prompt"},
            outputs=[GraphEdge(name="text_inputs", next_node="decode")],
        ),
        Loop(
            name="decode_loop",
            section=GraphNode(
                name="decode",
                input_names={"text_inputs"},
                outputs=[
                    GraphEdge(name="new_token", next_node="decode"),
                    GraphEdge(name="text_inputs", next_node="decode"),
                ],
            ),
            outputs=[],
            accumulated_outputs=[
                GraphEdge(name="new_token", next_node=EMIT_TO_CLIENT, output_modality="text"),
            ],
            max_iters=max_iters,
        ),
    ])


def test_speculation_skips_the_accumulated_loop_back():
    wgio = WorkerGraphIO(deepcopy(_accumulating_decode_graph()))
    node = wgio.nodes["decode"]

    ready = wgio.ingest_for_speculation(node.outputs, node.name)

    assert [info.node_name for info in ready] == ["decode"]
    assert ready[0].is_new_loop_iter is True
    assert set(node.speculative_signals.ready_names) == {"text_inputs"}


def test_regular_ingest_declines_the_accumulated_loop_back():
    wgio = WorkerGraphIO(deepcopy(_accumulating_decode_graph()))
    edge = GraphEdge(name="new_token", next_node="decode")

    assert wgio.ingest_input(edge) is False
    assert "new_token" not in wgio.nodes["decode"].ready_signals.ready_names
