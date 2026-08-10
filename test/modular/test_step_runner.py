"""Unit tests for the step runner's lifecycle sequencing.

``StepRunner`` drives the per-batch resource lifecycle: ``admit`` retrieves
published stream state into the pool before anything plans, ``plan`` builds
the step's plan surface for the chosen execution path, and ``publish``
describes each request's durable pool state outward after the step.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest

from mstar.conductor.request_info import PerLabelSeqInfo, SequenceInfo
from mstar.engine.resources import StepPlan, StepRunner


class _Info:
    """Just the slice of CurrentForwardPassInfo the runner touches."""

    def __init__(self, per_label_seq_info=None):
        self.per_label_seq_info = per_label_seq_info or PerLabelSeqInfo()


class _RecordingPool:
    def __init__(self):
        self.retrieved = []
        self.published = []

    def retrieve(self, request_id, label, seq_info):
        self.retrieved.append((request_id, label, seq_info))

    def publish(self, request_id):
        self.published.append(request_id)
        return {"main": SequenceInfo(seq_len=7, pos_id=7, latest_kv_transfer_info=None)}


def _seq_info(seq_len=4):
    return SequenceInfo(seq_len=seq_len, pos_id=seq_len, latest_kv_transfer_info=None)


class TestAdmit:
    def test_retrieves_every_published_stream(self):
        runner, pool = StepRunner(), _RecordingPool()
        info = _Info()
        info.per_label_seq_info.add("kv", 0, 1, {"main": _seq_info(), "cfg": _seq_info()})

        runner.admit(
            per_request_info={"r": info},
            pool=pool,
            kv_cache_string="kv",
            tp_rank=0,
            tp_world_size=1,
            needed_labels=None,
        )
        assert [(rid, label) for rid, label, _ in pool.retrieved] == [
            ("r", "main"), ("r", "cfg"),
        ]

    def test_needed_labels_filter(self):
        runner, pool = StepRunner(), _RecordingPool()
        info = _Info()
        info.per_label_seq_info.add("kv", 0, 1, {"main": _seq_info(), "cfg": _seq_info()})

        runner.admit(
            per_request_info={"r": info},
            pool=pool,
            kv_cache_string="kv",
            tp_rank=0,
            tp_world_size=1,
            needed_labels={"main"},
        )
        assert [(rid, label) for rid, label, _ in pool.retrieved] == [("r", "main")]

    def test_world_size_mismatch_is_rejected(self):
        runner, pool = StepRunner(), _RecordingPool()
        info = _Info()
        info.per_label_seq_info.add("kv", 0, 2, {"main": _seq_info()})

        with pytest.raises(RuntimeError, match="TP world size"):
            runner.admit(
                per_request_info={"r": info},
                pool=pool,
                kv_cache_string="kv",
                tp_rank=0,
                tp_world_size=1,
                needed_labels=None,
            )

    def test_unpublished_requests_retrieve_nothing(self):
        runner, pool = StepRunner(), _RecordingPool()
        runner.admit(
            per_request_info={"r": _Info()},
            pool=pool,
            kv_cache_string="kv",
            tp_rank=0,
            tp_world_size=1,
            needed_labels=None,
        )
        assert pool.retrieved == []


class TestPlan:
    def test_graph_mode_builds_no_surface(self):
        built = []
        step = StepRunner().plan(
            mode="graph", request_ids=["a", "b"], build_manager=built.append,
        )
        assert step == StepPlan(mode="graph")
        assert built == []

    def test_batched_mode_shares_one_surface(self):
        built = []

        def build(rids):
            built.append(rids)
            return f"manager{len(built)}"

        step = StepRunner().plan(
            mode="batched", request_ids=["a", "b"], build_manager=build,
        )
        assert built == [["a", "b"]]
        assert step.cache_manager == "manager1"
        assert step.per_request_managers == []

    def test_sequential_mode_builds_one_surface_per_request(self):
        built = []

        def build(rids):
            built.append(rids)
            return f"manager{len(built)}"

        step = StepRunner().plan(
            mode="sequential", request_ids=["a", "b"], build_manager=build,
        )
        assert built == [["a"], ["b"]]
        assert step.cache_manager is None
        assert step.per_request_managers == ["manager1", "manager2"]


class TestPublish:
    def test_publishes_onto_each_request_info(self):
        runner, pool = StepRunner(), _RecordingPool()
        infos = {"a": _Info(), "b": _Info()}

        runner.publish(
            request_ids=["a", "b"],
            per_request_info=infos,
            pool=pool,
            kv_cache_string="kv",
            tp_rank=0,
            tp_world_size=2,
        )
        assert pool.published == ["a", "b"]
        for info in infos.values():
            assert info.per_label_seq_info.get("kv", 0)["main"].seq_len == 7
            assert info.per_label_seq_info.world_size["kv"] == 2

    def test_requests_without_info_are_skipped(self):
        runner, pool = StepRunner(), _RecordingPool()
        runner.publish(
            request_ids=["a"],
            per_request_info={},
            pool=pool,
            kv_cache_string="kv",
            tp_rank=0,
            tp_world_size=1,
        )
        assert pool.published == []


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
