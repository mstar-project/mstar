"""Unit tests for ``WorkerGraphQueues.stop_loops`` per-loop loop-back attribution.

Issue #3 from PR #78: the previous implementation collected every self-loop
edge on the running node, regardless of which loop was being stopped. A node
that participates in two distinct dynamic loops with disjoint loop-back edges
would lose the surviving loop's loop-back tensor when the other loop stopped.

The fix: ``stop_loops`` now returns the ``(edge.name, edge.next_node)`` pairs
of the loop-back signals belonging to the loops that actually matched the
``loop_names`` filter, so the caller can scope the routing-drop to those
edges only.

These tests don't touch the worker — they drive ``WorkerGraphQueues.stop_loops``
directly with a hand-built section to exercise the walker logic in isolation.
"""

from __future__ import annotations

import pytest

pytest.skip(
    "Depends on deleted mstar.graph.request_queues and on DynamicLoop "
    "(merged into Loop). The same per-loop loop-back attribution behavior "
    "is now covered by WorkerGraphIO + register_loop_finish_signal tests.",
    allow_module_level=True,
)

import sys  # noqa: E402

sys.path.insert(0, ".")

from mstar.graph.request_queues import PerRequestNodeQueues  # noqa: E402

from mstar.graph.base import (  # noqa: E402
    DynamicLoop,
    GraphEdge,
    GraphNode,
    Parallel,
)
from mstar.worker.node_manager_utils import WorkerGraphQueues  # noqa: E402


def _make_queues(section, rid: str = "rid") -> WorkerGraphQueues:
    """Build a minimal ``WorkerGraphQueues`` whose only request points at
    ``section``. ``worker_graph`` and ``tensor_manager`` are stubbed to None
    because none of the code paths exercised here touch them.
    """
    return WorkerGraphQueues(
        worker_graph_id="wg",
        graph_walks={"decode"},
        worker_graph=None,
        per_request_queues={
            rid: PerRequestNodeQueues(
                waiting=section,
                full_section=section,
                worker_graph_id="wg",
            ),
        },
        tensor_manager=None,
    )


def _two_parallel_loops() -> Parallel:
    """Two ``DynamicLoop``s side by side, each wrapping its own single
    GraphNode whose only output is its own loop-back edge.
      - loop_a: node_a, self-loop edge "a"
      - loop_b: node_b, self-loop edge "b"
    """
    return Parallel(sections=[
        DynamicLoop(
            name="loop_a",
            section=GraphNode(
                name="node_a",
                input_ids={"a"},
                outputs=[GraphEdge(next_node="node_a", name="a")],
            ),
            max_iters=10,
            outputs=[],
        ),
        DynamicLoop(
            name="loop_b",
            section=GraphNode(
                name="node_b",
                input_ids={"b"},
                outputs=[GraphEdge(next_node="node_b", name="b")],
            ),
            max_iters=10,
            outputs=[],
        ),
    ])


class TestStopLoopsAttribution:
    def test_returns_only_stopped_loops_loop_back_signals(self):
        """Stopping ``loop_a`` returns ``loop_a``'s loop-back signal and
        nothing from ``loop_b`` — even though both loops have self-loop
        edges and the over-broad pre-fix walker would have surfaced both.
        """
        queues = _make_queues(_two_parallel_loops())
        signals = queues.stop_loops("rid", {"loop_a"})
        assert signals == {("a", "node_a")}

    def test_returns_union_when_multiple_loops_stopped(self):
        queues = _make_queues(_two_parallel_loops())
        signals = queues.stop_loops("rid", {"loop_a", "loop_b"})
        assert signals == {("a", "node_a"), ("b", "node_b")}

    def test_returns_empty_when_no_loops_match(self):
        queues = _make_queues(_two_parallel_loops())
        signals = queues.stop_loops("rid", {"loop_does_not_exist"})
        assert signals == set()

    def test_register_finished_only_fires_on_matched_loops(self):
        """Side-effect verification: the unmatched loop's ``_finished`` flag
        must stay False, so the walker's filtering of loop-back signals is
        consistent with the actual stop side-effect (which the worker relies
        on for ``_is_done`` detection).
        """
        section = _two_parallel_loops()
        queues = _make_queues(section)
        queues.stop_loops("rid", {"loop_a"})
        loop_a, loop_b = section.sections
        assert loop_a._finished is True
        assert loop_b._finished is False


class TestStopLoopsWithSharedNodeName:
    """Regression for the docstring claim: ``stop_loops`` must drop only the
    stopped loop's loop-back, even when two loops contain GraphNodes with
    *the same* logical name and disjoint self-loop edges. This is the exact
    shape the PR-78 comment flagged: a node that "participates in two
    distinct loops with different self-loop edges".
    """

    def test_disjoint_self_loop_edges_attributed_per_loop(self):
        section = Parallel(sections=[
            DynamicLoop(
                name="loop_a",
                section=GraphNode(
                    name="n",
                    input_ids={"a"},
                    outputs=[GraphEdge(next_node="n", name="a")],
                ),
                max_iters=10,
                outputs=[],
            ),
            DynamicLoop(
                name="loop_b",
                section=GraphNode(
                    name="n",
                    input_ids={"b"},
                    outputs=[GraphEdge(next_node="n", name="b")],
                ),
                max_iters=10,
                outputs=[],
            ),
        ])
        queues = _make_queues(section)

        # Stop only loop_a. The signal set must NOT contain ("b", "n"),
        # because that's loop_b's loop-back — without per-loop attribution
        # the worker's routing filter would drop it and loop_b would lose
        # its loop-back tensor.
        signals = queues.stop_loops("rid", {"loop_a"})
        assert signals == {("a", "n")}
        assert ("b", "n") not in signals


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
