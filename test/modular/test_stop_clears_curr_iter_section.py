"""Tests for the fix that clears ``_curr_iter_section`` when stops are
applied without speculation.

Background
----------
``_fast_postprocess`` calls ``apply_spec_consumption`` to clear the Loop's
``_curr_iter_section`` so the next ``complete_loops`` can finalize the
loop. That call only fires when ``spec_node_name is not None`` — i.e. when
this iter speculated a successor.

When a stop arrives on an iter that did NOT speculate, the prior iter's
``process_node_outputs`` had already advanced the Loop and queued the
next-iter body. ``register_finished`` sets ``_finished=True``, but
``_curr_iter_section`` still points at that next-iter ``GraphNode``.
``GraphNode.complete_loops`` returns ``LoopCompletionOutput(self)``, so
the Loop's recursion sets ``_curr_iter_section`` back to the GraphNode.
``_iter_done()`` returns False (curr_iter_section is non-None even though
``_finished=True`` and ``_waiting_for_execution`` is empty), the Loop
never reports done, the worker graph never emits WORKER_GRAPHS_DONE, and
the request hangs forever.

Repro: Q3-Omni Thinker hits ``<|im_end|>`` on an iter where the spec
chain happens not to fire (no spec built that iter, or alloc cap reached
under load). Both single and concurrent traffic hung indefinitely.

Fix: in ``_fast_postprocess``, if any rid in ``stopped_loop_backs`` is in
the batch, also call ``apply_spec_consumption(rid, batch.node_name)`` —
the same operation spec uses to mark the just-queued next-iter body as
consumed. This sets ``_curr_iter_section=None``; the subsequent
``complete_loops`` then sees ``_iter_done()=True`` AND ``_finished=True``,
returns ``new_waiting=None``, and the worker graph completes.

These tests drive the relevant graph primitives directly.
"""

from __future__ import annotations

import pytest

pytest.skip(
    "Targets the deleted DynamicLoop + _curr_iter_section + complete_loops "
    "machinery. The equivalent behavior (loop terminates cleanly when "
    "register_loop_finish_signal fires on a non-speculated iter) is covered "
    "by test_graph.py::test_eos_clears_ready_signals; the specific "
    "apply_spec_consumption call no longer exists.",
    allow_module_level=True,
)

import sys  # noqa: E402

sys.path.insert(0, ".")

from mminf.graph.base import (  # noqa: E402
    DynamicLoop,
    GraphEdge,
    GraphNode,
)


def _make_thinker_decode_loop() -> DynamicLoop:
    """Q3-Omni-shaped Thinker decode loop body: GraphNode("Thinker") with
    self-loop edge ``text_inputs`` and a streaming output to Talker.
    """
    body = GraphNode(
        name="Thinker",
        input_ids={"text_inputs"},
        outputs=[
            GraphEdge(next_node="emit_to_client", name="new_token"),
            GraphEdge(next_node="Thinker", name="text_inputs"),
        ],
    )
    return DynamicLoop(
        name="thinker_decode_loop",
        section=body,
        max_iters=2048,
        outputs=[],
    )


class TestStopClearsCurrIterSection:
    def test_loop_finalizes_when_stop_applied_with_spec_clear(self):
        """Mirrors what ``apply_spec_consumption`` does for stops.

        After a few iterations advancing normally, a stop is applied:
        ``register_finished()`` sets ``_finished=True``. WITHOUT clearing
        ``_curr_iter_section``, ``complete_loops`` recurses into the
        GraphNode (which returns itself), and ``_iter_done()`` returns
        False forever.

        Calling ``split_off_for_spec(self.section.name)`` — which is what
        ``apply_spec_consumption`` does internally — clears
        ``_curr_iter_section``. Then ``complete_loops`` sees the cleared
        state and finalizes.
        """
        loop = _make_thinker_decode_loop()

        # Simulate a completed iter: advance, then split_off_ready
        # which queues GraphNode in waiting_for_execution and clears
        # _curr_iter_section.
        loop._advance_one_iter()
        loop._curr_iter_section.ready_inputs["text_inputs"] = GraphEdge(
            next_node="Thinker", name="text_inputs",
        )
        ready, _ = loop.split_off_ready()
        assert len(ready) == 1
        assert ready[0].name == "Thinker"
        assert loop._curr_iter_section is None
        assert loop._waiting_for_execution == {"Thinker"}

        # Iter completes -> ``complete_loops("Thinker")`` runs.
        # process_node_outputs would then advance the loop, queueing the
        # NEXT iter body. Simulate that by calling complete_loops then
        # split_off_ready again (which advances).
        loop.complete_loops("Thinker")
        assert loop._waiting_for_execution == set()
        assert loop._curr_iter_section is None  # still empty after complete

        # Now route a new text_inputs (loop-back) and split_off_ready —
        # the Loop advances and the new GraphNode body becomes
        # _curr_iter_section.
        loop._curr_iter_section is None and loop.split_off_ready()  # noqa: B015
        assert loop._curr_iter_section is not None  # next-iter body queued
        assert isinstance(loop._curr_iter_section, GraphNode)

        # Stop arrives: register_finished sets _finished=True.
        loop.register_finished()
        assert loop._finished

        # Without clearing _curr_iter_section, complete_loops keeps the
        # GraphNode -> _iter_done()=False -> _is_done()=False.
        # This is the BUG path: simulate complete_loops fires (GraphNode
        # returns LoopCompletionOutput(self), so _curr_iter_section
        # stays as the GraphNode).
        out = loop.complete_loops("Thinker")
        assert out.new_waiting is loop  # NOT None — loop did NOT finalize
        assert loop._curr_iter_section is not None  # still a GraphNode
        assert not loop._is_done()  # would hang here

        # Apply the fix: split_off_for_spec("Thinker") clears
        # _curr_iter_section. Same thing apply_spec_consumption does.
        _, new_waiting = loop.split_off_for_spec("Thinker")
        assert new_waiting is loop  # outer Loop is still waiting
        assert loop._curr_iter_section is None  # cleared

        # NOW complete_loops finalizes.
        out = loop.complete_loops("Thinker")
        assert out.new_waiting is None  # loop terminated
        assert loop._is_done()

    def test_complete_loops_alone_does_not_finalize_after_finish(self):
        """Sanity: register_finished + complete_loops, with an active
        ``_curr_iter_section``, does NOT finalize. This is the BUG state
        that ``apply_spec_consumption`` (or its stop-path equivalent)
        prevents.
        """
        loop = _make_thinker_decode_loop()
        loop._advance_one_iter()  # _curr_iter_section = GraphNode

        loop.register_finished()
        assert loop._finished
        assert loop._curr_iter_section is not None
        assert not loop._is_done()  # _iter_done is False

        out = loop.complete_loops("Thinker")
        # Bug visible here: even with _finished=True, complete_loops
        # cannot finalize because GraphNode.complete_loops returns self.
        assert out.new_waiting is loop
        assert not loop._is_done()

    def test_split_off_for_spec_clears_iter_section_at_any_iter(self):
        """``split_off_for_spec`` is callable mid-loop and unconditionally
        clears ``_curr_iter_section`` when the spec node matches the body.
        """
        loop = _make_thinker_decode_loop()
        loop._advance_one_iter()
        assert loop._curr_iter_section is not None

        _, new_waiting = loop.split_off_for_spec("Thinker")
        assert new_waiting is loop
        assert loop._curr_iter_section is None

    def test_split_off_for_spec_no_match_preserves_state(self):
        """If the spec node name doesn't match the body, no clear."""
        loop = _make_thinker_decode_loop()
        loop._advance_one_iter()
        before = loop._curr_iter_section

        _, new_waiting = loop.split_off_for_spec("SomeOtherNode")
        assert new_waiting is loop
        assert loop._curr_iter_section is before  # unchanged


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
