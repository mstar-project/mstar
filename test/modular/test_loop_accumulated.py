"""Unit tests for the Loop.accumulated_outputs primitive (Phase 2).

Drives Loop / DynamicLoop directly via ``cache_outputs`` + ``complete_loops``
rather than routing through the full worker/engine stack — keeps the tests
fast and focused on the new primitive's behavior.  A tiny ``MockTensorManager``
stands in for the real communication manager so we can assert refcount
balance without booting shared memory.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest

from mminf.graph.base import (
    DynamicLoop,
    GraphEdge,
    GraphNode,
    Loop,
    TensorPointerInfo,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockTensorManager:
    """Counts increment_ref / dereference calls per uuid."""

    def __init__(self):
        self.increments: dict[str, int] = {}
        self.decrements: dict[str, int] = {}

    def increment_ref(self, request_id: str, uuid: str) -> None:
        self.increments[uuid] = self.increments.get(uuid, 0) + 1

    def dereference(self, request_id: str, uuid: str) -> None:
        self.decrements[uuid] = self.decrements.get(uuid, 0) + 1

    def net_refs(self) -> dict[str, int]:
        all_uuids = set(self.increments) | set(self.decrements)
        return {u: self.increments.get(u, 0) - self.decrements.get(u, 0) for u in all_uuids}


def _make_info(uuid_str: str) -> TensorPointerInfo:
    return TensorPointerInfo(
        dims=[1],
        dtype="float32",
        nbytes=4,
        address=0,
        stride=[1],
        uuid=uuid_str,
        source_session_id="test",
        source_entity="worker",
    )


def _build_loop(
    max_iters: int = 3,
    include_terminal_output: bool = False,
) -> Loop:
    """A single-node loop whose section emits both a loop-back ``x`` and an
    emit-to-client ``pred``.  ``outputs`` is optionally populated with ``x``
    so tests can exercise the mixed-outputs path when needed.
    """
    section = GraphNode(
        name="node",
        input_ids={"x"},
        outputs=[
            GraphEdge(next_node="node", name="x"),
            GraphEdge(next_node="node", name="pred"),
        ],
    )
    outputs: list[GraphEdge] = []
    if include_terminal_output:
        outputs = [GraphEdge(next_node="client", name="x")]
    return Loop(
        section=section,
        max_iters=max_iters,
        outputs=outputs,
        accumulated_outputs=[GraphEdge(next_node="client", name="pred")],
    )


def _force_loop_completion_state(loop: Loop, final_iter: int) -> None:
    """Shortcut the Loop to the state ``complete_loops`` expects at terminal
    iteration so we can unit-test the emission path without simulating the
    full worker-driven advance cycle.
    """
    loop.curr_iter = final_iter
    loop._curr_iter_section = None
    loop._waiting_for_execution.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPostInit:
    def test_default_accumulated_outputs_empty(self):
        section = GraphNode(
            name="node",
            input_ids={"x"},
            outputs=[GraphEdge(next_node="node", name="y")],
        )
        loop = Loop(
            section=section,
            max_iters=2,
            outputs=[GraphEdge(next_node="client", name="y")],
        )
        # Default should be an empty list — every existing Loop call site
        # (bagel, pi0.5, orpheus, qwen3_omni, dummy) omits the new kwarg and
        # must still construct cleanly.
        assert loop.accumulated_outputs == []
        assert loop._accumulated_output_names == set()
        assert loop._accumulated_cache == {}

    def test_disjoint_with_outputs_raises(self):
        section = GraphNode(
            name="node",
            input_ids={"x"},
            outputs=[GraphEdge(next_node="node", name="y")],
        )
        with pytest.raises(ValueError, match="disjoint"):
            Loop(
                section=section,
                max_iters=2,
                outputs=[GraphEdge(next_node="client", name="y")],
                accumulated_outputs=[GraphEdge(next_node="client", name="y")],
            )

    def test_filters_edges_not_produced_by_section(self):
        section = GraphNode(
            name="node",
            input_ids={"x"},
            outputs=[GraphEdge(next_node="node", name="y")],
        )
        loop = Loop(
            section=section,
            max_iters=2,
            outputs=[],
            accumulated_outputs=[
                GraphEdge(next_node="client", name="y"),  # kept
                GraphEdge(next_node="client", name="ghost"),  # filtered out
            ],
        )
        assert [e.name for e in loop.accumulated_outputs] == ["y"]
        assert loop._accumulated_output_names == {"y"}


class TestAccumulation:
    def test_basic_per_iter_accumulation(self):
        loop = _build_loop(max_iters=3)
        tm = MockTensorManager()
        loop.register_communication_info(tm, "req1")

        # Simulate 3 iterations.  The between-iter ``_uncache_outputs`` call
        # mimics ``_advance_one_iter`` — it should clear ``_cached_outputs``
        # but leave ``_accumulated_cache`` intact.
        for k in range(3):
            loop.cache_outputs({"pred": [_make_info(f"pred_{k}")]})
            if k < 2:
                loop._uncache_outputs()

        _force_loop_completion_state(loop, final_iter=2)
        out = loop.complete_loops("node")

        pred_edges = [e for e in out.outputs if e.name == "pred"]
        assert len(pred_edges) == 1
        emitted = [i.uuid for i in pred_edges[0].tensor_info]
        assert emitted == ["pred_0", "pred_1", "pred_2"]

    def test_mixed_outputs_and_accumulated(self):
        """``outputs`` delivers last-iter only; ``accumulated_outputs`` the
        full per-iter list.  Exercises both sides of the cache split in
        ``cache_outputs``.
        """
        loop = _build_loop(max_iters=3, include_terminal_output=True)
        tm = MockTensorManager()
        loop.register_communication_info(tm, "req1")

        for k in range(3):
            loop.cache_outputs(
                {
                    "x": [_make_info(f"x_{k}")],
                    "pred": [_make_info(f"pred_{k}")],
                }
            )
            if k < 2:
                loop._uncache_outputs()

        _force_loop_completion_state(loop, final_iter=2)
        out = loop.complete_loops("node")

        x_edges = [e for e in out.outputs if e.name == "x"]
        pred_edges = [e for e in out.outputs if e.name == "pred"]
        assert len(x_edges) == 1 and len(pred_edges) == 1
        # ``outputs`` cache was wiped between iters → only last survives.
        assert [i.uuid for i in x_edges[0].tensor_info] == ["x_2"]
        # ``accumulated_outputs`` cache persists → all 3 survive.
        assert [i.uuid for i in pred_edges[0].tensor_info] == ["pred_0", "pred_1", "pred_2"]

    def test_no_emission_when_name_never_cached(self):
        """If no iteration produces the accumulated edge's name (rare but
        possible under conditional node execution), the edge is not included
        in the completion output."""
        loop = _build_loop(max_iters=2)
        tm = MockTensorManager()
        loop.register_communication_info(tm, "req1")

        # Never call cache_outputs with "pred"; cache only the loop-back "x".
        loop.cache_outputs({"x": [_make_info("x_0")]})
        loop._uncache_outputs()
        loop.cache_outputs({"x": [_make_info("x_1")]})

        _force_loop_completion_state(loop, final_iter=1)
        out = loop.complete_loops("node")

        assert [e.name for e in out.outputs] == []  # no pred, no x (x is only loop-back)


class TestDynamicLoopEarlyExit:
    def test_accumulated_length_matches_curr_iter(self):
        """DynamicLoop with early ``register_finished`` yields an accumulated
        list whose length is the number of iterations actually run, not
        ``max_iters``."""
        section = GraphNode(
            name="node",
            input_ids={"x"},
            outputs=[GraphEdge(next_node="node", name="pred")],
        )
        loop = DynamicLoop(
            name="rollout",
            section=section,
            max_iters=5,
            outputs=[],
            accumulated_outputs=[GraphEdge(next_node="client", name="pred")],
        )
        tm = MockTensorManager()
        loop.register_communication_info(tm, "req1")

        # Run 2 iters, then signal early-exit.
        for k in range(2):
            loop.cache_outputs({"pred": [_make_info(f"pred_{k}")]})
            if k < 1:
                loop._uncache_outputs()
            loop.curr_iter = k
        loop.register_finished()

        _force_loop_completion_state(loop, final_iter=1)
        out = loop.complete_loops("node")

        pred_edges = [e for e in out.outputs if e.name == "pred"]
        assert len(pred_edges) == 1
        assert len(pred_edges[0].tensor_info) == 2
        assert [i.uuid for i in pred_edges[0].tensor_info] == ["pred_0", "pred_1"]


class TestRefcounts:
    def test_increments_per_iter(self):
        """Every per-iter ``cache_outputs`` call on an accumulated name
        should bump the manager's increment counter for that uuid."""
        loop = _build_loop(max_iters=3)
        tm = MockTensorManager()
        loop.register_communication_info(tm, "req1")

        for k in range(3):
            loop.cache_outputs({"pred": [_make_info(f"pred_{k}")]})

        # Each of 3 distinct uuids incremented once.  No decrements yet.
        assert tm.increments == {"pred_0": 1, "pred_1": 1, "pred_2": 1}
        assert tm.decrements == {}

    def test_reset_dereferences_and_clears(self):
        """``reset`` dereferences every accumulated tensor_info and clears
        the cache — otherwise a reused Loop would leak refs."""
        loop = _build_loop(max_iters=3)
        tm = MockTensorManager()
        loop.register_communication_info(tm, "req1")

        for k in range(3):
            loop.cache_outputs({"pred": [_make_info(f"pred_{k}")]})

        assert len(loop._accumulated_cache["pred"]) == 3

        loop.reset()

        assert loop._accumulated_cache == {}
        # Every increment now paired with a dereference.
        net = tm.net_refs()
        assert all(v == 0 for v in net.values()), f"leaked refs: {net}"

    def test_uncache_outputs_leaves_accumulated_cache_alone(self):
        """``_uncache_outputs`` (called inside ``_advance_one_iter``) MUST
        NOT touch the accumulated cache — that's the whole reason it exists."""
        loop = _build_loop(max_iters=3)
        tm = MockTensorManager()
        loop.register_communication_info(tm, "req1")

        loop.cache_outputs({"pred": [_make_info("pred_0")]})
        loop._uncache_outputs()

        # The accumulated cache should still carry pred_0.
        assert "pred" in loop._accumulated_cache
        assert [i.uuid for i in loop._accumulated_cache["pred"]] == ["pred_0"]
        # And no dereference of pred_0 happened.
        assert tm.decrements.get("pred_0", 0) == 0

    def test_reset_safe_without_tensor_manager(self):
        """``reset`` should not crash if called before
        ``register_communication_info`` — the cache is empty at that point
        anyway, but the code path must tolerate ``_tensor_manager is None``."""
        loop = _build_loop(max_iters=2)
        # Don't call register_communication_info.
        loop.reset()
        assert loop._accumulated_cache == {}


# ---------------------------------------------------------------------------
# GraphNode.optional_input_ids tests (Phase 2 primitive #2)
# ---------------------------------------------------------------------------


class TestGraphNodeOptionalInputs:
    def test_is_ready_ignores_optional(self):
        """A node with required ``x`` and optional ``y`` is ready when only
        ``x`` has arrived — optional inputs don't block."""
        node = GraphNode(
            name="node",
            input_ids={"x"},
            optional_input_ids={"y"},
            outputs=[GraphEdge(next_node="node", name="out")],
        )
        node.ingest_inputs({"node": [GraphEdge(next_node="node", name="x")]})
        assert node.is_ready()
        assert "x" in node.ready_inputs and "y" not in node.ready_inputs

    def test_ingest_accepts_optional_when_present(self):
        """When the optional input *does* arrive, it lands in
        ``ready_inputs`` alongside the required ones."""
        node = GraphNode(
            name="node",
            input_ids={"x"},
            optional_input_ids={"y"},
            outputs=[GraphEdge(next_node="node", name="out")],
        )
        node.ingest_inputs(
            {
                "node": [
                    GraphEdge(next_node="node", name="x"),
                    GraphEdge(next_node="node", name="y"),
                ]
            }
        )
        assert "x" in node.ready_inputs
        assert "y" in node.ready_inputs

    def test_ingest_drops_unknown_names(self):
        """Names that are in neither ``input_ids`` nor ``optional_input_ids``
        are ignored — prior behavior preserved."""
        node = GraphNode(
            name="node",
            input_ids={"x"},
            optional_input_ids={"y"},
            outputs=[GraphEdge(next_node="node", name="out")],
        )
        node_to_inputs = {
            "node": [
                GraphEdge(next_node="node", name="x"),
                GraphEdge(next_node="node", name="zzz"),
            ]
        }
        node.ingest_inputs(node_to_inputs)
        assert "zzz" not in node.ready_inputs
        # The stray edge is left in node_to_inputs — downstream code is
        # free to route it elsewhere.
        assert any(e.name == "zzz" for e in node_to_inputs.get("node", []))

    def test_disjoint_validation_raises(self):
        with pytest.raises(ValueError, match="disjoint"):
            GraphNode(
                name="node",
                input_ids={"x"},
                optional_input_ids={"x"},
                outputs=[GraphEdge(next_node="node", name="out")],
            )

    def test_get_inputs_includes_optional(self):
        """``get_inputs`` returns both sets so the graph dispatcher knows
        to route optional edges here when they exist."""
        node = GraphNode(
            name="node",
            input_ids={"x"},
            optional_input_ids={"y", "z"},
            outputs=[],
        )
        names = {e.name for e in node.get_inputs()}
        assert names == {"x", "y", "z"}

    def test_default_optional_is_empty(self):
        """Existing call sites (no new kwarg) construct unchanged."""
        node = GraphNode(
            name="node",
            input_ids={"x"},
            outputs=[GraphEdge(next_node="node", name="out")],
        )
        assert node.optional_input_ids == set()
