"""Tests the chunked-prefill orchestrator with a stub inner_pass.

We don't need a real submodule or KV cache for these tests — the
orchestrator's contract is "given a way to run one forward pass, drive it
N times." A callable stub is sufficient to exercise it.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch

from mminf.engine.ar_engine import AREngine
from mminf.engine.base import NodeBatch, NodeOutput
from mminf.engine.chunked_prefill import execute_chunked_prefill
from mminf.model.submodule_base import ARNodeInputs


def _make_batch(seq_len: int, rid: str = "r0") -> tuple[NodeBatch, list[ARNodeInputs]]:
    batch = NodeBatch(
        node_name="LLM",
        graph_walk="prefill_text",
        request_ids=[rid],
        per_request_input_tensors={rid: {}},
        per_request_info={},
    )
    inputs = [
        ARNodeInputs(
            input_seq_len=seq_len,
            input_ids=torch.arange(seq_len).unsqueeze(0),
            custom_pos_ids=torch.arange(seq_len),
        )
    ]
    return batch, inputs


def test_executes_n_chunks_for_seq_len_evenly_divisible():
    batch, inputs = _make_batch(seq_len=8)
    calls = []

    def stub_inner_pass(b: NodeBatch, ins: list[ARNodeInputs]) -> NodeOutput:
        calls.append(ins[0].input_seq_len)
        return NodeOutput(per_request_output_tensors={"r0": {"sentinel": [torch.tensor([calls[-1]])]}})

    out = execute_chunked_prefill(batch, inputs, chunk_size=4, inner_pass=stub_inner_pass)
    assert calls == [4, 4]
    # Last chunk's output is what's returned.
    assert out.per_request_output_tensors["r0"]["sentinel"][0].item() == 4


def test_last_chunk_is_short_when_seq_len_not_divisible():
    batch, inputs = _make_batch(seq_len=10)
    seen_chunk_lens = []

    def stub(b, ins):
        seen_chunk_lens.append(ins[0].input_seq_len)
        return NodeOutput(per_request_output_tensors={"r0": {}})

    execute_chunked_prefill(batch, inputs, chunk_size=4, inner_pass=stub)
    assert seen_chunk_lens == [4, 4, 2]


def test_only_last_chunk_output_is_returned():
    batch, inputs = _make_batch(seq_len=6)
    chunk_idx = {"i": 0}

    def stub(b, ins):
        i = chunk_idx["i"]
        chunk_idx["i"] += 1
        return NodeOutput(per_request_output_tensors={"r0": {"chunk_id": [torch.tensor([i])]}})

    out = execute_chunked_prefill(batch, inputs, chunk_size=4, inner_pass=stub)
    assert out.per_request_output_tensors["r0"]["chunk_id"][0].item() == 1


def test_inner_pass_receives_token_axis_slice():
    batch, inputs = _make_batch(seq_len=10)
    seen_input_ids = []

    def stub(b, ins):
        seen_input_ids.append(ins[0].input_ids.clone())
        return NodeOutput(per_request_output_tensors={"r0": {}})

    execute_chunked_prefill(batch, inputs, chunk_size=4, inner_pass=stub)
    assert torch.equal(seen_input_ids[0], torch.arange(0, 4).unsqueeze(0))
    assert torch.equal(seen_input_ids[1], torch.arange(4, 8).unsqueeze(0))
    assert torch.equal(seen_input_ids[2], torch.arange(8, 10).unsqueeze(0))


def test_rejects_multi_request_batch():
    batch = NodeBatch(
        node_name="LLM",
        graph_walk="prefill_text",
        request_ids=["a", "b"],
        per_request_input_tensors={"a": {}, "b": {}},
        per_request_info={},
    )
    inputs = [
        ARNodeInputs(input_seq_len=8, input_ids=torch.arange(8).unsqueeze(0)),
        ARNodeInputs(input_seq_len=8, input_ids=torch.arange(8).unsqueeze(0)),
    ]

    with pytest.raises(ValueError, match="single-request"):
        execute_chunked_prefill(batch, inputs, chunk_size=4, inner_pass=lambda b, i: None)


def _ar_engine_with_chunk_size(chunk_size):
    return AREngine(max_prefill_chunk_size=chunk_size)


def _make_submodule(supports: bool):
    sub = MagicMock()
    sub.supports_chunked_prefill.return_value = supports
    return sub


def test_should_chunk_prefill_disabled_when_chunk_size_none():
    eng = _ar_engine_with_chunk_size(None)
    batch, inputs = _make_batch(seq_len=4096)
    sub = _make_submodule(supports=True)
    assert eng._should_chunk_prefill(batch, inputs, sub) is False


def test_should_chunk_prefill_disabled_when_submodule_does_not_opt_in():
    eng = _ar_engine_with_chunk_size(512)
    batch, inputs = _make_batch(seq_len=4096)
    sub = _make_submodule(supports=False)
    assert eng._should_chunk_prefill(batch, inputs, sub) is False


def test_should_chunk_prefill_disabled_for_short_prompts():
    eng = _ar_engine_with_chunk_size(512)
    batch, inputs = _make_batch(seq_len=100)
    sub = _make_submodule(supports=True)
    assert eng._should_chunk_prefill(batch, inputs, sub) is False


def test_should_chunk_prefill_disabled_when_prompt_equals_chunk_size():
    """Pin the `<=` boundary: a prompt of exactly chunk_size is not chunked."""
    eng = _ar_engine_with_chunk_size(512)
    batch, inputs = _make_batch(seq_len=512)
    sub = _make_submodule(supports=True)
    assert eng._should_chunk_prefill(batch, inputs, sub) is False


def test_should_chunk_prefill_disabled_for_multi_request_batches():
    eng = _ar_engine_with_chunk_size(512)
    batch = NodeBatch(
        node_name="LLM", graph_walk="prefill_text",
        request_ids=["a", "b"],
        per_request_input_tensors={"a": {}, "b": {}},
        per_request_info={},
    )
    inputs = [
        ARNodeInputs(input_seq_len=4096, input_ids=torch.arange(4096).unsqueeze(0)),
        ARNodeInputs(input_seq_len=4096, input_ids=torch.arange(4096).unsqueeze(0)),
    ]
    sub = _make_submodule(supports=True)
    assert eng._should_chunk_prefill(batch, inputs, sub) is False


def test_should_chunk_prefill_enabled_for_single_long_request():
    eng = _ar_engine_with_chunk_size(512)
    batch, inputs = _make_batch(seq_len=4096)
    sub = _make_submodule(supports=True)
    assert eng._should_chunk_prefill(batch, inputs, sub) is True


def test_dispatch_one_pass_method_exists():
    """Smoke test: _dispatch_one_pass exists and routes through the existing
    priority chain. Full integration coverage lives in test_chunked_prefill_equivalence.
    """
    eng = _ar_engine_with_chunk_size(None)
    assert hasattr(eng, "_dispatch_one_pass")


def test_scheduler_owns_chunking_default_off():
    """Default off — engine continues to chunk single-request batches per Phase 1."""
    eng = AREngine(max_prefill_chunk_size=512)
    assert eng.scheduler_owns_chunking is False


def test_scheduler_owns_chunking_disables_engine_chunking():
    """When scheduler owns chunking, engine's _should_chunk_prefill returns False
    even for batches that would otherwise be chunked."""
    eng = AREngine(max_prefill_chunk_size=512, scheduler_owns_chunking=True)
    batch, inputs = _make_batch(seq_len=4096)
    sub = _make_submodule(supports=True)
    assert eng._should_chunk_prefill(batch, inputs, sub) is False


def test_node_batch_terminal_flag_defaults_empty():
    """Backwards compat: existing batches don't set is_terminal_per_request,
    and default empty dict means 'all terminal' (existing single-walk behavior)."""
    batch = NodeBatch(
        node_name="LLM", graph_walk="prefill_text",
        request_ids=["a"], per_request_input_tensors={"a": {}},
        per_request_info={},
    )
    assert batch.is_terminal_per_request == {}


def test_node_batch_terminal_flag_explicit():
    """Constructor accepts an explicit is_terminal_per_request dict."""
    batch = NodeBatch(
        node_name="LLM", graph_walk="thinker_step",
        request_ids=["a", "b"],
        per_request_input_tensors={"a": {}, "b": {}},
        per_request_info={},
        is_terminal_per_request={"a": True, "b": False},
    )
    assert batch.is_terminal_per_request == {"a": True, "b": False}
