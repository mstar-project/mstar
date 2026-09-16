"""Regression test for the same-node speculation gate with loop-external inputs.

Before the fix, ``GraphNode.is_ready_for_speculation(check_next_iter=True)``
required every input name to be in ``ready_next_iter`` or
``speculative_signals``. Loop-external inputs (e.g. wan22's ``text_embeds_*``,
waypoint's controller streams) never land in ``ready_next_iter`` — they're
re-injected unchanged into ``ready_signals`` every iteration
(``Loop.ingest_external_input`` / ``complete_iter``) — so a same-node loop
node with any external input could never be proposed as a same-node
speculation target. See ``docs`` for the plan (F3) this fixes.
"""

from mstar.graph.base import GraphEdge, GraphNode, Loop, SpeculativeNodeInfo
from mstar.graph.graph_io import WorkerGraphIO


def _wan22_shaped_loop():
    """A wan22-shaped rollout loop: one node, 2 external (persisted) inputs
    plus 2 loop-back inputs to itself."""
    dit = GraphNode(
        name="dit",
        input_names={"text", "lat", "t"},
        outputs=[
            GraphEdge(next_node="dit", name="lat"),
            GraphEdge(next_node="dit", name="t"),
        ],
    )
    return dit, Loop(name="L", section=dit, max_iters=10, outputs=[])


def test_loop_external_persisted_input_satisfies_next_iter_speculation():
    dit, loop = _wan22_shaped_loop()
    io = WorkerGraphIO(loop)
    for name in ("text", "lat", "t"):
        io.ingest_input(GraphEdge(next_node="dit", name=name, persist=True))

    ready = io.ingest_for_speculation(dit.outputs, "dit")

    assert ready == [
        SpeculativeNodeInfo(node_name="dit", is_new_loop_iter=True, loop_name="L")
    ]


def test_top_level_external_input_without_persist_for_loop_is_not_speculation_ready():
    # A node with an external input outside of any Loop never gets
    # `_persist_for_loop` set (that flag is only set by
    # `Loop.ingest_external_input`), so it must still fail the next-iter gate.
    top = GraphNode(
        name="top",
        input_names={"a", "b"},
        outputs=[GraphEdge(next_node="top", name="b")],
    )
    io = WorkerGraphIO(top)
    io.ingest_input(GraphEdge(next_node="top", name="a", persist=True))

    assert not top.is_ready_for_speculation(check_next_iter=True)
