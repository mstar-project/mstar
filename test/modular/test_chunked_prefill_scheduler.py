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


def test_thinker_step_emits_batched_logits_for_cuda_graph_compat():
    """The thinker_step branch must emit __batched_logits__ (not per-rid
    logits) so output shape is fixed across terminal-flag distributions —
    a precondition for CUDA graph capture."""
    import inspect
    from mminf.model.qwen3_omni.submodules import ThinkerSubmodule
    src = inspect.getsource(ThinkerSubmodule.forward_batched)
    assert "__batched_logits__" in src
    assert 'graph_walk == "thinker_step"' in src or "thinker_step" in src


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


def test_thinker_step_per_request_gating_at_engine_level():
    """is_terminal_per_request gating moved from submodule to AREngine's
    batched-logits sampling fast path in Phase 2.1a (CUDA graph compat).
    """
    import inspect
    from mminf.engine.ar_engine import AREngine
    src = inspect.getsource(AREngine._execute_batched)
    assert "is_terminal_per_request" in src
    assert "new_token" in src


# ---------------------------------------------------------------------------
# Phase 2 Task 5: MicroScheduler chunked-step packing hook + worker bookkeeping
# ---------------------------------------------------------------------------


def test_micro_scheduler_accepts_max_step_tokens_param():
    """MicroScheduler.__init__ accepts max_step_tokens with default 2048."""
    import inspect

    from mminf.worker.micro_scheduler import MicroScheduler

    sig = inspect.signature(MicroScheduler.__init__)
    assert "max_step_tokens" in sig.parameters
    assert sig.parameters["max_step_tokens"].default == 2048


def test_micro_scheduler_exposes_chunked_step_method():
    """The new private packing method is in place on MicroScheduler.

    Source-level check; full behavioral coverage requires a real
    WorkerGraphsManager (Task 6). The method must:
      1. classify ready AR requests via ``is_prefill_complete``,
      2. call ``plan_chunked_step``,
      3. produce a ``ScheduledBatch`` with ``graph_walk='thinker_step'``
         and ``is_terminal_per_request`` populated.
    """
    import inspect

    from mminf.worker.micro_scheduler import MicroScheduler

    assert hasattr(MicroScheduler, "_get_chunked_step_batch")
    src = inspect.getsource(MicroScheduler._get_chunked_step_batch)
    assert "is_prefill_complete" in src
    assert "plan_chunked_step" in src
    assert '"thinker_step"' in src or "'thinker_step'" in src
    assert "is_terminal_per_request" in src
    assert "prefill_chunk_sizes" in src


def test_get_next_batch_short_circuits_when_owner_is_scheduler():
    """get_next_batch dispatches to the chunked-step path when
    ``scheduler_owns_chunking=True`` is set on the AR engine."""
    import inspect

    from mminf.worker.micro_scheduler import MicroScheduler

    src = inspect.getsource(MicroScheduler.get_next_batch)
    # Must check the flag and call the new method.
    assert "_ar_engine_owns_chunking" in src
    assert "_get_chunked_step_batch" in src
    # The flag check must come before the legacy node_name_to_requests dict
    # is built (so the new path takes precedence when active).
    flag_idx = src.index("_ar_engine_owns_chunking")
    legacy_idx = src.index("node_name_to_requests")
    assert flag_idx < legacy_idx


def test_scheduled_batch_carries_terminal_and_chunk_size_fields():
    """ScheduledBatch was extended with the chunked-step metadata fields."""
    from mminf.worker.micro_scheduler import ScheduledBatch

    batch = ScheduledBatch(
        node_name="Thinker",
        graph_walk="thinker_step",
        node_objects={},
        is_terminal_per_request={"a": True, "b": False},
        prefill_chunk_sizes={"b": 2048},
    )
    assert batch.is_terminal_per_request == {"a": True, "b": False}
    assert batch.prefill_chunk_sizes == {"b": 2048}

    # Backwards compat — both default to None.
    legacy = ScheduledBatch(
        node_name="Thinker", graph_walk="thinker_decode", node_objects={},
    )
    assert legacy.is_terminal_per_request is None
    assert legacy.prefill_chunk_sizes is None


def test_chunked_step_returns_none_when_no_ar_requests_ready():
    """With an empty WorkerGraphsManager, _get_chunked_step_batch returns
    None so callers fall through to the legacy scheduling path."""
    from dataclasses import dataclass, field
    from mminf.engine.base import EngineType
    from mminf.worker.engine_manager import EngineManager
    from mminf.worker.micro_scheduler import MicroScheduler

    @dataclass
    class _StubAR:
        scheduler_owns_chunking: bool = True

        def engine_type(self):
            return EngineType.AR

        def check_ready(self, *args, **kwargs):
            return True

    em = EngineManager(node_to_engine={"Thinker": _StubAR()})
    sched = MicroScheduler(em, max_step_tokens=2048)

    @dataclass
    class _StubWGM:
        queues: dict = field(default_factory=dict)
        per_request_info: dict = field(default_factory=dict)

        def get_partition_for_node(self, name):
            return "Thinker"

    out = sched._get_chunked_step_batch(_StubWGM())
    assert out is None


def test_worker_admission_initializes_prefill_total():
    """When scheduler_owns_chunking is on, _add_new_request primes
    prefill_tokens_total from the prompt tensor's leading dimension.

    Source-level check; behavioral coverage with real workers in Task 6.
    """
    import inspect

    from mminf.worker.worker import Worker

    src = inspect.getsource(Worker._add_new_request)
    # Must check the engine flag and read text_inputs.dims[0].
    assert "scheduler_owns_chunking" in src
    assert "text_inputs" in src
    assert "prefill_tokens_total" in src


def test_worker_advances_prefill_tokens_consumed_after_step():
    """The worker's post-step bookkeeping advances prefill_tokens_consumed
    for each prefill rid in the executed batch by the chunk size."""
    import inspect

    from mminf.worker.worker import Worker

    src = inspect.getsource(Worker._fast_postprocess)
    assert "prefill_chunk_sizes" in src
    assert "prefill_tokens_consumed" in src


def test_worker_propagates_is_terminal_per_request_into_node_batch():
    """_build_node_batch carries ScheduledBatch.is_terminal_per_request
    into NodeBatch so the AR engine + ThinkerSubmodule can gate lm_head."""
    import inspect

    from mminf.worker.worker import Worker

    src = inspect.getsource(Worker._build_node_batch)
    assert "is_terminal_per_request" in src
