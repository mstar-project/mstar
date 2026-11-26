"""CPU parity against a golden generated from the pinned official implementation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from mstar.model.components.deepseek_v41.compressor import DeepSeekV41Compressor


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads((Path(__file__).parent / "data/deepseek_v41_compressor_golden.json").read_text())


def _loaded(golden: dict, ratio: int) -> DeepSeekV41Compressor:
    model = DeepSeekV41Compressor(16, 8, ratio).eval()
    prefix = "one" if ratio == 1 else "two"
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            parameter.copy_(torch.tensor(golden["weights"][f"{prefix}.{name}"], dtype=parameter.dtype))
    return model


def _expected_state(golden: dict, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    kv, scores = (torch.tensor(part, dtype=torch.float32) for part in golden[name])
    return kv, scores.masked_fill(scores == -1e30, -torch.inf)


def _assert_state(actual: tuple[torch.Tensor, torch.Tensor], expected: tuple[torch.Tensor, torch.Tensor]) -> None:
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_ratio_one_prefill_and_decode_match_official_reference(golden: dict) -> None:
    model = _loaded(golden, 1)
    tokens = torch.tensor(golden["input"], dtype=torch.bfloat16)
    for start, end, expected in ((0, 5, "one_prefill"), (5, 6, "one_decode")):
        latent, state = model(tokens[:, start:end], start)
        assert latent is not None and state is None
        torch.testing.assert_close(latent, torch.tensor(golden[expected], dtype=torch.bfloat16), rtol=0, atol=0)


def test_ratio_two_prefill_and_decode_match_official_reference(golden: dict) -> None:
    model = _loaded(golden, 2)
    tokens = torch.tensor(golden["input"], dtype=torch.bfloat16)
    prefill, state = model(tokens[:, :5], 0)
    assert prefill is not None and state is not None
    # Five tokens make two complete groups and leave one token for the next call.
    torch.testing.assert_close(prefill, torch.tensor(golden["two_prefill"], dtype=torch.bfloat16), rtol=0, atol=0)
    _assert_state(state, _expected_state(golden, "after_prefill"))

    completed, next_state = model(tokens[:, 5:6], 5, state)
    assert completed is not None and next_state is not None
    torch.testing.assert_close(
        completed, torch.tensor(golden["two_complete5"], dtype=torch.bfloat16), rtol=0, atol=0,
    )
    _assert_state(next_state, _expected_state(golden, "after_complete5"))
    # State values belong to the caller; producing the next state cannot modify the prior one.
    _assert_state(state, _expected_state(golden, "after_prefill"))

    partial6, state6 = model(tokens[:, 6:7], 6, next_state)
    assert partial6 is None and state6 is not None
    _assert_state(state6, _expected_state(golden, "after_partial6"))
    completed7, state7 = model(tokens[:, 7:8], 7, state6)
    assert completed7 is not None and state7 is not None
    torch.testing.assert_close(
        completed7, torch.tensor(golden["two_complete7"], dtype=torch.bfloat16), rtol=0, atol=0,
    )
    _assert_state(state7, _expected_state(golden, "after_complete7"))


def test_prefill_on_exact_group_boundary_matches_reference(golden: dict) -> None:
    model = _loaded(golden, 2)
    tokens = torch.tensor(golden["input"], dtype=torch.bfloat16)
    latent, state = model(tokens[:, :4], 0)
    assert latent is not None and state is not None
    torch.testing.assert_close(latent, torch.tensor(golden["exact_prefill"], dtype=torch.bfloat16), rtol=0, atol=0)
    _assert_state(state, _expected_state(golden, "exact_state"))


def test_prefill_shorter_than_ratio_returns_state_only(golden: dict) -> None:
    model = _loaded(golden, 2)
    tokens = torch.tensor(golden["input"], dtype=torch.bfloat16)
    latent, state = model(tokens[:, :1], 0)
    assert latent is None and state is not None
    complete, _ = model(tokens[:, 1:2], 1, state)
    assert complete is not None and complete.shape == (2, 1, 8)
    torch.testing.assert_close(complete, torch.tensor(golden["short_complete"], dtype=torch.bfloat16), rtol=0, atol=0)


def test_decode_requires_caller_owned_state(golden: dict) -> None:
    model = _loaded(golden, 2)
    tokens = torch.tensor(golden["input"], dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="needs the request's partial state"):
        model(tokens[:, 5:6], 5)
    with pytest.raises(ValueError, match="wrong shape"):
        model(tokens[:, 5:6], 5, (torch.zeros(1, 2, 8), torch.zeros(1, 2, 8)))


def test_parent_dtype_cast_keeps_pooling_in_fp32() -> None:
    model = DeepSeekV41Compressor(16, 8, 2).to(torch.bfloat16)
    assert model.wkv.weight.dtype == torch.float32
    assert model.wgate is not None and model.wgate.weight.dtype == torch.float32
    assert model.norm.weight.dtype == torch.float32
    assert not list(model.buffers())


def test_parent_float_cast_keeps_ratio_one_projection_in_bf16(golden: dict) -> None:
    model = _loaded(golden, 1).float()
    assert model.wkv.weight.dtype == torch.bfloat16
    tokens = torch.tensor(golden["input"], dtype=torch.bfloat16)
    observed, _ = model(tokens[:, :5], 0)
    assert observed is not None
    torch.testing.assert_close(observed, torch.tensor(golden["one_prefill"], dtype=torch.bfloat16), rtol=0, atol=0)
