"""Unit tests for the Phase 2 chunked-prefill scheduler. CPU-only."""
from __future__ import annotations

from mminf.conductor.request_info import CurrentForwardPassInfo


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
