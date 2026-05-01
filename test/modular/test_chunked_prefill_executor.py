"""Tests the chunked-prefill orchestrator with a stub inner_pass.

We don't need a real submodule or KV cache for these tests — the
orchestrator's contract is "given a way to run one forward pass, drive it
N times." A callable stub is sufficient to exercise it.
"""
from __future__ import annotations

import pytest
import torch

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
