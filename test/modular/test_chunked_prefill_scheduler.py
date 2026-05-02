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


# ---------------------------------------------------------------------------
# Phase 2 Task 4: thinker_step graph walk + Thinker submodule routing
# ---------------------------------------------------------------------------

def test_thinker_step_walk_declared_in_source():
    """Qwen3OmniModel.get_graph_walk_graphs declares the thinker_step walk.

    Smoke test: full integration coverage with weights happens in Task 6.
    Here we just verify the source has the walk + the partition definitions
    include it so the conductor can route batches to that walk name.
    """
    import inspect

    from mminf.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

    src = inspect.getsource(Qwen3OmniModel.get_graph_walk_graphs)
    assert "thinker_step" in src, "thinker_step walk not declared in get_graph_walk_graphs"
    assert '"thinker_step": thinker_step' in src, (
        "thinker_step walk not registered in returned dict"
    )

    partitions_src = inspect.getsource(Qwen3OmniModel.get_partitions)
    assert "thinker_step" in partitions_src, (
        "thinker_step missing from Thinker partition's graph_walks set"
    )


def test_thinker_step_routed_to_prefill_mode():
    """ThinkerSubmodule.preprocess routes thinker_step to mode='prefill'.

    Avoids loading the 30B model — just inspects the source for the
    explicit mode-routing line to verify thinker_step doesn't fall through
    to mode='decode'. FlashInfer's prefill wrapper handles arbitrary
    per-request seq_lens (including seq_len=1 decode tokens) correctly,
    so the mixed-batch walk must use prefill mode.
    """
    import inspect

    from mminf.model.qwen3_omni.submodules import ThinkerSubmodule

    src = inspect.getsource(ThinkerSubmodule.preprocess)
    # The preprocess routing is `mode = "decode" if graph_walk == "thinker_decode" else "prefill"`.
    # Verify the routing line is intact (only thinker_decode -> decode; everything
    # else, including thinker_step, falls through to "prefill").
    assert 'graph_walk == "thinker_decode"' in src, (
        "preprocess no longer routes thinker_decode → decode mode"
    )


def test_thinker_step_per_request_lm_head_gating_in_source():
    """ThinkerSubmodule.forward_batched gates lm_head per-request for thinker_step.

    Verify the source contains the per-request terminal gating logic so
    non-terminal prefill chunks skip lm_head and emit no logits, while
    terminal requests (decode token OR final prefill chunk) get logits
    and are routed through the engine's per-rid sampling path.
    """
    import inspect

    from mminf.model.qwen3_omni.submodules import ThinkerSubmodule

    src = inspect.getsource(ThinkerSubmodule.forward_batched)
    assert "thinker_step" in src, "forward_batched has no thinker_step branch"
    assert "is_terminal_per_request" in src, (
        "forward_batched does not consult is_terminal_per_request for "
        "per-request lm_head gating"
    )


def test_thinker_step_can_batch():
    """ThinkerSubmodule.can_batch returns True for thinker_step batches."""
    import inspect

    from mminf.model.qwen3_omni.submodules import ThinkerSubmodule

    src = inspect.getsource(ThinkerSubmodule.can_batch)
    assert "thinker_step" in src, (
        "can_batch must accept thinker_step so the AR engine routes the "
        "mixed batch through forward_batched (not the per-request path)."
    )


def test_model_inputs_from_engine_carries_terminal_dict():
    """ModelInputsFromEngine exposes is_terminal_per_request for the submodule.

    The Thinker forward_batched needs per-request terminal flags to gate
    lm_head; adding the field to the engine-input dataclass (and populating
    it in AREngine._execute_batched from NodeBatch) is the plumbing path.
    """
    from mminf.model.submodule_base import ModelInputsFromEngine

    inp = ModelInputsFromEngine(
        request_ids=["a", "b"],
        per_request_info={},
        is_terminal_per_request={"a": True, "b": False},
    )
    assert inp.is_terminal_per_request == {"a": True, "b": False}

    # Backwards compat: defaults to empty dict ("all terminal").
    default_inp = ModelInputsFromEngine(
        request_ids=["x"], per_request_info={},
    )
    assert default_inp.is_terminal_per_request == {}


def test_thinker_step_per_request_gating_uses_terminal_dict():
    """Verify forward_batched's thinker_step branch reads is_terminal_per_request
    and emits logits only for terminal rids. Source-level check; full behavioral
    coverage comes via test_mixed_batch_correctness.py (Task 6)."""
    import inspect

    from mminf.model.qwen3_omni.submodules import ThinkerSubmodule

    src = inspect.getsource(ThinkerSubmodule.forward_batched)
    # The gating loop must:
    # 1. Read engine_inputs.is_terminal_per_request.
    assert "is_terminal_per_request" in src
    assert ".get(rid, True)" in src or "engine_inputs.is_terminal_per_request" in src
    # 2. Conditionally call lm_head.
    assert "lm_head" in src
    # 3. Conditionally emit logits.
    assert "logits" in src
