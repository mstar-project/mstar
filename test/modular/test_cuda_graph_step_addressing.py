"""Tests for the runner's slot step addressing.

Replay and pre-plan point a slot's cache manager at the step's request ids
(real ids first, the slot's capture-time ids for the padding tail) instead
of aliasing live request state onto dummy slots. These tests drive the
addressing helpers and the release paths directly with stub slots,
verifying that:

  - step ids compose real ids with the capture tail;
  - addressing and its reset write the cache manager's request_ids and
    active_labels and nothing else;
  - step metadata keys real info by real id and capture info by capture id;
  - the pre-plan drop path restores capture addressing and frees only
    capture-id pages.
"""

from __future__ import annotations

import sys
import types

sys.path.insert(0, ".")

from mstar.engine.cuda_graph_runner import (
    CudaGraphData,
    CudaGraphKey,
    CudaGraphRunner,
    CudaGraphSlot,
)


def _make_stub_cm() -> types.SimpleNamespace:
    cm = types.SimpleNamespace()
    cm.request_ids = []
    cm.active_labels = {}
    cm._pre_planned_labels = set()
    cm._plan_done_event = None
    return cm


def _make_slot(dummy_rids: list[str]) -> CudaGraphSlot:
    cm = _make_stub_cm()
    cm.request_ids = list(dummy_rids)
    cm.active_labels = {rid: "main" for rid in dummy_rids}
    return CudaGraphSlot(
        graph=object(),
        static_inputs={"dummy_rids": list(dummy_rids)},
        static_outputs={},
        static_cache_manager=cm,
    )


def _make_runner() -> CudaGraphRunner:
    runner = CudaGraphRunner.__new__(CudaGraphRunner)
    runner.enable_nvtx = False
    return runner


DUMMIES = ["__cg_d0__", "__cg_d1__", "__cg_d2__", "__cg_d3__"]


class TestSlotStepIds:
    def test_partial_batch_keeps_capture_tail(self):
        runner = _make_runner()
        slot = _make_slot(DUMMIES)
        step_ids = runner._slot_step_ids(slot, ["r0", "r1"])
        assert step_ids == ["r0", "r1", "__cg_d2__", "__cg_d3__"]

    def test_full_batch_has_no_tail(self):
        runner = _make_runner()
        slot = _make_slot(DUMMIES)
        step_ids = runner._slot_step_ids(slot, ["r0", "r1", "r2", "r3"])
        assert step_ids == ["r0", "r1", "r2", "r3"]

    def test_input_list_not_mutated(self):
        runner = _make_runner()
        slot = _make_slot(DUMMIES)
        real = ["r0"]
        runner._slot_step_ids(slot, real)
        assert real == ["r0"]


class TestAddressing:
    def test_address_sets_ids_and_labels(self):
        slot = _make_slot(DUMMIES)
        step_ids = ["r0", "r1", "__cg_d2__", "__cg_d3__"]
        CudaGraphRunner._address_slot(slot, step_ids)
        cm = slot.static_cache_manager
        assert cm.request_ids == step_ids
        assert cm.active_labels == {rid: "main" for rid in step_ids}

    def test_reset_restores_capture_addressing(self):
        slot = _make_slot(DUMMIES)
        CudaGraphRunner._address_slot(slot, ["r0", "r1", "__cg_d2__", "__cg_d3__"])
        CudaGraphRunner._reset_slot_addressing(slot)
        cm = slot.static_cache_manager
        assert cm.request_ids == DUMMIES
        assert cm.active_labels == {rid: "main" for rid in DUMMIES}

    def test_reset_noops_without_capture_ids(self):
        slot = CudaGraphSlot(
            graph=object(),
            static_inputs={},
            static_outputs={},
            static_cache_manager=_make_stub_cm(),
        )
        slot.static_cache_manager.request_ids = ["r0"]
        CudaGraphRunner._reset_slot_addressing(slot)
        assert slot.static_cache_manager.request_ids == ["r0"]


class TestStepMetadata:
    def test_real_head_and_capture_tail(self):
        runner = _make_runner()
        real_info = {"r0": "info0", "r1": "info1"}
        dummy_info = {d: f"cap_{d}" for d in DUMMIES}
        out = runner._build_step_metadata(
            ["r0", "r1", "__cg_d2__", "__cg_d3__"], 2, real_info, dummy_info,
        )
        assert out == {
            "r0": "info0",
            "r1": "info1",
            "__cg_d2__": "cap___cg_d2__",
            "__cg_d3__": "cap___cg_d3__",
        }


class _RecordingAllocManager:
    def __init__(self):
        self.resets: list[tuple[str, str, bool]] = []

    def reset_label(self, rid: str, label: str, free: bool = True) -> None:
        self.resets.append((rid, label, free))


class TestResetPrePlanRestoresAddressing:
    def _runner_with_slot(self):
        runner = _make_runner()
        runner.alloc_manager = _RecordingAllocManager()
        key = CudaGraphKey(graph_walk="decode", requires_cfg=False, bs=4, num_tokens=4)
        slot = _make_slot(DUMMIES)
        config = types.SimpleNamespace(labels=["main"])
        runner.graphs = {
            key: CudaGraphData(config=config, bs=4, index=0, slots=[slot]),
        }
        runner._get_basic_batched_key_for = lambda **kw: key
        return runner, slot

    def test_drop_path_restores_addressing_and_frees_capture_pages(self):
        runner, slot = self._runner_with_slot()
        CudaGraphRunner._address_slot(
            slot, ["r0", "r1", "__cg_d2__", "__cg_d3__"],
        )
        slot.static_cache_manager._pre_planned_labels = {"main"}
        runner.reset_pre_plan_state_for_slot(
            graph_walk="decode", requires_cfg=False, batch_size=4, slot=0,
        )
        cm = slot.static_cache_manager
        assert cm._pre_planned_labels == set()
        assert cm._plan_done_event is None
        assert cm.request_ids == DUMMIES
        assert cm.active_labels == {rid: "main" for rid in DUMMIES}
        assert runner.alloc_manager.resets == [
            (rid, "main", True) for rid in DUMMIES
        ]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
