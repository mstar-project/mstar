"""Unit tests for the Phase 2 chunked-prefill scheduler. CPU-only."""
from __future__ import annotations

from mminf.conductor.request_info import CurrentForwardPassInfo
from mminf.worker.micro_scheduler import (
    DecodeReadyRequest,
    PrefillReadyRequest,
    plan_chunked_step,
)


def _make_info() -> CurrentForwardPassInfo:
    """Construct a minimal CurrentForwardPassInfo without GPU/model machinery."""
    info = CurrentForwardPassInfo.__new__(CurrentForwardPassInfo)
    # Initialise the dataclass fields that have no defaults so that
    # attribute access on *other* fields does not raise AttributeError.
    info.request_id = "test-req"
    info.graph_walk = "prefill"
    info.requires_cfg = False
    info.fwd_index = 0
    info.random_seed = 0
    info.max_tokens = 1
    info.sampling_config = {}
    # fields with default_factory — replicate the dataclass defaults
    info.step_metadata = {}
    from mminf.conductor.request_info import PerLabelSeqInfo
    info.per_label_seq_info = PerLabelSeqInfo()
    info.partition_name = "default"
    info.dynamic_loop_stop_signals = set()
    info.loop_stop_times = {}
    info.dynamic_loop_iter_counts = {}
    # Phase 2 chunked-prefill fields (defaults)
    info.prefill_tokens_total = 0
    info.prefill_tokens_consumed = 0
    return info


def test_prefill_progress_defaults():
    info = _make_info()
    assert info.prefill_tokens_total == 0
    assert info.prefill_tokens_consumed == 0
    assert info.is_prefill_complete is True  # 0 == 0 → trivially complete


def test_prefill_progress_in_flight():
    info = _make_info()
    info.prefill_tokens_total = 4096
    info.prefill_tokens_consumed = 1024
    assert info.is_prefill_complete is False


def test_prefill_progress_complete():
    info = _make_info()
    info.prefill_tokens_total = 4096
    info.prefill_tokens_consumed = 4096
    assert info.is_prefill_complete is True


# ---------------------------------------------------------------------------
# Phase 2 Task 2: plan_chunked_step tests
# ---------------------------------------------------------------------------


def test_decode_only_step_fills_budget():
    """3 decodes, budget=2048 → all 3 included."""
    plan = plan_chunked_step(
        ready_decodes=[DecodeReadyRequest(rid=f"d{i}") for i in range(3)],
        ready_prefills=[],
        max_step_tokens=2048,
    )
    assert plan.decode_rids == ["d0", "d1", "d2"]
    assert plan.prefill_allocations == {}
    assert plan.terminal_prefills == set()
    assert plan.total_tokens == 3


def test_prefill_only_step_chunks_to_budget():
    """1 prefill request with 8000 tokens left, budget=2048 → take 2048."""
    plan = plan_chunked_step(
        ready_decodes=[],
        ready_prefills=[PrefillReadyRequest(rid="p0", tokens_remaining=8000)],
        max_step_tokens=2048,
    )
    assert plan.decode_rids == []
    assert plan.prefill_allocations == {"p0": 2048}
    assert plan.terminal_prefills == set()  # 2048 < 8000, not terminal
    assert plan.total_tokens == 2048


def test_mixed_step_decode_first():
    """2 decodes + 1 prefill (8000 left), budget=2048 → 2 decodes, 2046 prefill."""
    plan = plan_chunked_step(
        ready_decodes=[DecodeReadyRequest(rid=f"d{i}") for i in range(2)],
        ready_prefills=[PrefillReadyRequest(rid="p0", tokens_remaining=8000)],
        max_step_tokens=2048,
    )
    assert plan.decode_rids == ["d0", "d1"]
    assert plan.prefill_allocations == {"p0": 2046}
    assert plan.total_tokens == 2048


def test_mixed_step_short_prefill_fits_entirely():
    """1 decode + 1 prefill (100 left), budget=2048 → 1 decode + 100 prefill (terminal)."""
    plan = plan_chunked_step(
        ready_decodes=[DecodeReadyRequest(rid="d0")],
        ready_prefills=[PrefillReadyRequest(rid="p0", tokens_remaining=100)],
        max_step_tokens=2048,
    )
    assert plan.decode_rids == ["d0"]
    assert plan.prefill_allocations == {"p0": 100}
    assert plan.terminal_prefills == {"p0"}  # 100 == 100, this chunk completes
    assert plan.total_tokens == 101


def test_overflow_decodes_drops_excess():
    """3000 decodes, budget=2048 → only 2048 included."""
    plan = plan_chunked_step(
        ready_decodes=[DecodeReadyRequest(rid=f"d{i}") for i in range(3000)],
        ready_prefills=[],
        max_step_tokens=2048,
    )
    assert len(plan.decode_rids) == 2048
    assert plan.total_tokens == 2048


def test_multiple_prefills_first_takes_all_budget():
    """2 long prefills, budget=2048 → first takes 2048, second deferred."""
    plan = plan_chunked_step(
        ready_decodes=[],
        ready_prefills=[
            PrefillReadyRequest(rid="p0", tokens_remaining=8000),
            PrefillReadyRequest(rid="p1", tokens_remaining=8000),
        ],
        max_step_tokens=2048,
    )
    assert plan.prefill_allocations == {"p0": 2048}


def test_empty_step_returns_empty_plan():
    plan = plan_chunked_step(ready_decodes=[], ready_prefills=[], max_step_tokens=2048)
    assert plan.decode_rids == []
    assert plan.prefill_allocations == {}
    assert plan.total_tokens == 0


def test_invalid_budget_raises():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        plan_chunked_step(ready_decodes=[], ready_prefills=[], max_step_tokens=0)
    with _pytest.raises(ValueError):
        plan_chunked_step(ready_decodes=[], ready_prefills=[], max_step_tokens=-1)


def test_prefill_with_zero_tokens_remaining_skipped():
    """Edge case: a prefill request with 0 tokens remaining should be skipped."""
    plan = plan_chunked_step(
        ready_decodes=[],
        ready_prefills=[
            PrefillReadyRequest(rid="p0", tokens_remaining=0),
            PrefillReadyRequest(rid="p1", tokens_remaining=100),
        ],
        max_step_tokens=2048,
    )
    assert plan.prefill_allocations == {"p1": 100}
    assert "p0" not in plan.prefill_allocations
