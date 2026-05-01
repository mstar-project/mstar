"""Unit tests for chunked prefill primitives. CPU-only, no model weights."""
from __future__ import annotations

import torch

from mminf.engine.chunked_prefill import _slice_ar_inputs
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
