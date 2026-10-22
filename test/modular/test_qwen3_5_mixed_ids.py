"""A mixed step's input ids are each row's own ids, or its loop-back token,
in row order; runs of loop-back rows are packed as slices."""

from __future__ import annotations

import itertools

import pytest
import torch

from mstar.model.qwen3_5.submodules import LLMSubmodule
from mstar.model.submodule_base import ARNodeInputs


def _reference(ids: torch.Tensor, inputs: list[ARNodeInputs]) -> torch.Tensor:
    return torch.cat([
        inp.input_ids.to(ids.dtype) if inp.input_ids is not None else ids[i:i + 1]
        for i, inp in enumerate(inputs)
    ])


def _inputs(pattern: str) -> list[ARNodeInputs]:
    """'d' is a loop-back (decode) row, 'p' a prompt row of 1-3 tokens."""
    rows = []
    for i, kind in enumerate(pattern):
        if kind == "d":
            rows.append(ARNodeInputs(input_seq_len=1))
        else:
            n = 1 + i % 3
            rows.append(ARNodeInputs(
                input_seq_len=n, input_ids=torch.arange(1000 + 10 * i, 1000 + 10 * i + n, dtype=torch.int32),
            ))
    return rows


@pytest.mark.parametrize("pattern", [
    "d", "p", "ddd", "ppp", "pdd", "ddp", "dpd", "pdp", "dpdpd", "ppddpp",
    *("".join(c) for c in itertools.product("dp", repeat=5)),
])
@pytest.mark.parametrize("padding", [0, 3])
def test_packing_matches_one_piece_per_row(pattern, padding):
    inputs = _inputs(pattern)
    # loop-back tokens for every row of the step, plus a slot's padding rows past the inputs
    ids = torch.arange(len(inputs) + padding, dtype=torch.long) + 7
    got = LLMSubmodule._pack_mixed_ids(ids, inputs)
    want = _reference(ids, inputs)
    assert got.dtype == want.dtype == ids.dtype
    assert torch.equal(got, want)
