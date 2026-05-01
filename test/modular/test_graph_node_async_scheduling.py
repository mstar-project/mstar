from mminf.graph.base import GraphEdge, GraphNode


def test_clone_for_next_iter_preserves_async_scheduling_flag():
    node = GraphNode(
        name="LLM",
        input_ids={"text_inputs"},
        outputs=[GraphEdge(name="text_inputs", next_node="LLM")],
        enable_async_scheduling=False,
    )

    clone = node.clone_for_next_iter()

    assert clone.enable_async_scheduling is False
