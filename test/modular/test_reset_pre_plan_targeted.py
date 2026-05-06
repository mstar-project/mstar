"""Tests for slot-targeted ``reset_pre_plan_state_for_slot`` (PR #78 issue #5).

Old ``Worker._reset_skip_plan_flags`` walked every captured graph of every
engine across the worker on every speculation drop / pre-plan failure. With
``plan_executor.max_workers > 1``, that would stomp any sibling slot's
in-flight pre-plan whose ``_pre_planned_labels`` set hasn't yet been
consumed by the matching replay.

The fix added ``CudaGraphRunner.reset_pre_plan_state_for_slot`` to clear
just one (key, slot)'s pre-plan state. These tests drive the runner method
directly with a stub graphs dict, verifying that:

  - the targeted (key, slot)'s flags are cleared;
  - other slots of the SAME key are untouched;
  - other keys' slots are untouched;
  - mismatched / unknown shapes are no-ops (don't raise).
"""

from __future__ import annotations

import sys
import types

sys.path.insert(0, ".")

from mminf.engine.cuda_graph_runner import (
    CudaGraphData,
    CudaGraphKey,
    CudaGraphRunner,
    CudaGraphSlot,
)


def _make_stub_cm() -> types.SimpleNamespace:
    """Stand-in for ``BatchedCacheManager`` exposing only the two attrs the
    reset path touches. Tests don't need a real cache manager — the reset
    is pure dict/set mutation on the slot's static_cm.
    """
    cm = types.SimpleNamespace()
    cm._pre_planned_labels = set()
    cm._plan_done_event = None
    return cm


def _make_slot(label_set: set[str], event_marker: object | None) -> CudaGraphSlot:
    """Build a CudaGraphSlot with stub fields. The reset path only reads
    ``static_cache_manager``, so other fields are placeholder objects.
    """
    cm = _make_stub_cm()
    cm._pre_planned_labels = set(label_set)
    cm._plan_done_event = event_marker
    return CudaGraphSlot(
        graph=object(),
        static_inputs={},
        static_outputs={},
        static_cache_manager=cm,
    )


def _make_runner_with_two_keys() -> CudaGraphRunner:
    """Build a runner via ``__new__`` and populate ``graphs`` with two keys,
    each with two slots. Stub ``_get_basic_batched_key_for`` to do an
    exact-match lookup against (graph_walk, requires_cfg, bs) — skipping
    the padding/config-resolution logic that would require real capture
    configs. The reset path uses this helper (BASIC_BATCHED-only,
    num_tokens derived from config) since num_tokens is fully determined
    by the captured config for those graphs.
    """
    runner = CudaGraphRunner.__new__(CudaGraphRunner)
    runner.enable_nvtx = False

    key_a = CudaGraphKey(graph_walk="decode", requires_cfg=False, bs=4, num_tokens=4)
    key_b = CudaGraphKey(graph_walk="decode", requires_cfg=True, bs=2, num_tokens=2)

    runner.graphs = {
        key_a: CudaGraphData(
            config=object(),
            bs=4,
            slots=[
                _make_slot({"main"}, event_marker="A0"),
                _make_slot({"main"}, event_marker="A1"),
            ],
        ),
        key_b: CudaGraphData(
            config=object(),
            bs=2,
            slots=[
                _make_slot({"main", "cfg_img"}, event_marker="B0"),
                _make_slot({"main", "cfg_img"}, event_marker="B1"),
            ],
        ),
    }

    def _get_basic_batched_key_for(graph_walk, requires_cfg, batch_size):
        for key in runner.graphs:
            if (
                key.graph_walk == graph_walk
                and key.requires_cfg == requires_cfg
                and key.bs == batch_size
            ):
                return key
        return None

    runner._get_basic_batched_key_for = _get_basic_batched_key_for
    return runner


class TestResetPrePlanTargeted:
    def test_clears_only_targeted_slot(self):
        """Reset on (key_a, slot=0) clears that slot's flags only.
        key_a's slot 1 and key_b's both slots stay intact.
        """
        runner = _make_runner_with_two_keys()
        runner.reset_pre_plan_state_for_slot(
            graph_walk="decode", requires_cfg=False,
            batch_size=4, slot=0,
        )

        a0 = runner.graphs[CudaGraphKey("decode", False, 4, 4)].slots[0].static_cache_manager
        a1 = runner.graphs[CudaGraphKey("decode", False, 4, 4)].slots[1].static_cache_manager
        b0 = runner.graphs[CudaGraphKey("decode", True, 2, 2)].slots[0].static_cache_manager
        b1 = runner.graphs[CudaGraphKey("decode", True, 2, 2)].slots[1].static_cache_manager

        # Targeted slot wiped.
        assert a0._pre_planned_labels == set()
        assert a0._plan_done_event is None

        # Sibling slot of same key untouched.
        assert a1._pre_planned_labels == {"main"}
        assert a1._plan_done_event == "A1"

        # Other key's slots fully untouched.
        assert b0._pre_planned_labels == {"main", "cfg_img"}
        assert b0._plan_done_event == "B0"
        assert b1._pre_planned_labels == {"main", "cfg_img"}
        assert b1._plan_done_event == "B1"

    def test_clears_slot_1_independently(self):
        """Symmetry: slot=1 reset clears slot 1, leaves slot 0 alone."""
        runner = _make_runner_with_two_keys()
        runner.reset_pre_plan_state_for_slot(
            graph_walk="decode", requires_cfg=False,
            batch_size=4, slot=1,
        )
        a0 = runner.graphs[CudaGraphKey("decode", False, 4, 4)].slots[0].static_cache_manager
        a1 = runner.graphs[CudaGraphKey("decode", False, 4, 4)].slots[1].static_cache_manager

        assert a0._pre_planned_labels == {"main"}
        assert a0._plan_done_event == "A0"
        assert a1._pre_planned_labels == set()
        assert a1._plan_done_event is None

    def test_unknown_key_is_noop(self):
        """Reset for a (graph_walk, bs) that has no captured graph must
        not raise — pre_plan also no-ops on unknown keys, so the reset's
        contract is symmetric.
        """
        runner = _make_runner_with_two_keys()
        runner.reset_pre_plan_state_for_slot(
            graph_walk="decode", requires_cfg=False,
            batch_size=999, slot=0,
        )
        # All slots unchanged.
        a0 = runner.graphs[CudaGraphKey("decode", False, 4, 4)].slots[0].static_cache_manager
        assert a0._pre_planned_labels == {"main"}

    def test_slot_index_wraps(self):
        """``slot %= len(graph_data.slots)`` mirrors pre_plan's wrap so
        out-of-range slot indices map to a real slot rather than raising.
        """
        runner = _make_runner_with_two_keys()
        runner.reset_pre_plan_state_for_slot(
            graph_walk="decode", requires_cfg=False,
            batch_size=4, slot=3,  # wraps to 1
        )
        a0 = runner.graphs[CudaGraphKey("decode", False, 4, 4)].slots[0].static_cache_manager
        a1 = runner.graphs[CudaGraphKey("decode", False, 4, 4)].slots[1].static_cache_manager
        assert a0._pre_planned_labels == {"main"}  # untouched
        assert a1._pre_planned_labels == set()      # cleared via wrap

    def test_slot_none_defaults_to_zero(self):
        """``slot=None`` defaults to 0, matching pre_plan's contract."""
        runner = _make_runner_with_two_keys()
        runner.reset_pre_plan_state_for_slot(
            graph_walk="decode", requires_cfg=False,
            batch_size=4, slot=None,
        )
        a0 = runner.graphs[CudaGraphKey("decode", False, 4, 4)].slots[0].static_cache_manager
        a1 = runner.graphs[CudaGraphKey("decode", False, 4, 4)].slots[1].static_cache_manager
        assert a0._pre_planned_labels == set()
        assert a1._pre_planned_labels == {"main"}

    def test_empty_runner_is_noop(self):
        """A runner with no captured graphs must accept the reset call
        without raising — protects against early-startup or eager paths
        where pre_plan also no-ops.
        """
        runner = CudaGraphRunner.__new__(CudaGraphRunner)
        runner.enable_nvtx = False
        runner.graphs = {}
        runner._get_basic_batched_key_for = lambda *a, **k: None
        runner.reset_pre_plan_state_for_slot(
            graph_walk="decode", requires_cfg=False,
            batch_size=4, slot=0,
        )  # must not raise


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
