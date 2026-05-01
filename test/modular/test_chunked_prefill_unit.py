"""Unit tests for chunked prefill primitives. CPU-only, no model weights."""
from __future__ import annotations

import pytest
import torch

from mminf.engine.chunked_prefill import ChunkSlice, _plan_chunks, _slice_ar_inputs
from mminf.model.submodule_base import ARNodeInputs, NodeSubmodule


class _DummySubmodule(NodeSubmodule):
    """Concrete NodeSubmodule with the bare minimum to instantiate."""
    def prepare_inputs(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        raise NotImplementedError


def test_supports_chunked_prefill_default_false():
    sub = _DummySubmodule()
    assert sub.supports_chunked_prefill() is False


def _make_inputs(seq_len: int) -> ARNodeInputs:
    return ARNodeInputs(
        input_seq_len=seq_len,
        input_ids=torch.arange(seq_len).unsqueeze(0),  # [1, seq_len]
        custom_pos_ids=torch.arange(seq_len),  # [seq_len]
    )


def test_slice_input_ids_token_axis():
    inp = _make_inputs(seq_len=10)
    sliced = _slice_ar_inputs(inp, start=3, end=7)
    assert sliced.input_seq_len == 4
    assert torch.equal(sliced.input_ids, torch.arange(3, 7).unsqueeze(0))
    assert torch.equal(sliced.custom_pos_ids, torch.arange(3, 7))


def test_slice_preserves_tensor_inputs_and_kwargs_by_reference():
    inp = ARNodeInputs(
        input_seq_len=10,
        input_ids=torch.arange(10).unsqueeze(0),
        tensor_inputs={"foo": torch.zeros(3)},
        kwargs={"bar": "baz"},
    )
    sliced = _slice_ar_inputs(inp, start=0, end=5)
    # Non-token-axis tensors / kwargs pass through unchanged.
    assert sliced.tensor_inputs["foo"] is inp.tensor_inputs["foo"]
    assert sliced.kwargs["bar"] == "baz"


def test_slice_with_input_embeds():
    inp = ARNodeInputs(
        input_seq_len=8,
        input_embeds=torch.randn(1, 8, 16),  # [1, seq_len, hidden]
    )
    sliced = _slice_ar_inputs(inp, start=2, end=6)
    assert sliced.input_seq_len == 4
    assert sliced.input_embeds.shape == (1, 4, 16)
    assert torch.equal(sliced.input_embeds, inp.input_embeds[:, 2:6, :])


def test_slice_dict_custom_pos_ids():
    inp = ARNodeInputs(
        input_seq_len=10,
        input_ids=torch.arange(10).unsqueeze(0),
        custom_pos_ids={"a": torch.arange(10), "b": torch.arange(10) * 2},
    )
    sliced = _slice_ar_inputs(inp, start=4, end=10)
    assert sliced.input_seq_len == 6
    assert torch.equal(sliced.custom_pos_ids["a"], torch.arange(4, 10))
    assert torch.equal(sliced.custom_pos_ids["b"], torch.arange(4, 10) * 2)


def test_plan_chunks_evenly_divisible():
    plans = _plan_chunks(seq_len=8, chunk_size=4)
    assert plans == [
        ChunkSlice(index=0, start=0, end=4, is_last=False),
        ChunkSlice(index=1, start=4, end=8, is_last=True),
    ]


def test_plan_chunks_with_remainder():
    plans = _plan_chunks(seq_len=10, chunk_size=4)
    assert plans == [
        ChunkSlice(index=0, start=0, end=4, is_last=False),
        ChunkSlice(index=1, start=4, end=8, is_last=False),
        ChunkSlice(index=2, start=8, end=10, is_last=True),
    ]


def test_plan_chunks_seq_smaller_than_chunk():
    plans = _plan_chunks(seq_len=3, chunk_size=8)
    assert plans == [ChunkSlice(index=0, start=0, end=3, is_last=True)]


def test_plan_chunks_seq_equals_chunk():
    plans = _plan_chunks(seq_len=4, chunk_size=4)
    assert plans == [ChunkSlice(index=0, start=0, end=4, is_last=True)]


@pytest.mark.parametrize("seq_len", [0, -1])
def test_plan_chunks_rejects_non_positive_seq_len(seq_len):
    with pytest.raises(ValueError):
        _plan_chunks(seq_len=seq_len, chunk_size=4)


@pytest.mark.parametrize("chunk_size", [0, -1])
def test_plan_chunks_rejects_non_positive_chunk_size(chunk_size):
    with pytest.raises(ValueError):
        _plan_chunks(seq_len=8, chunk_size=chunk_size)


def test_qwen3_omni_thinker_opts_into_chunked_prefill():
    # Imported lazily because qwen3_omni instantiation may pull in heavy deps;
    # we only need the class.
    from mminf.model.qwen3_omni.submodules import ThinkerSubmodule
    # Override is on the class, not the instance — verify class-level method
    # returns True. We can't always instantiate without weights, so use a
    # dummy unbound-method check.
    instance = ThinkerSubmodule.__new__(ThinkerSubmodule)
    assert instance.supports_chunked_prefill() is True
