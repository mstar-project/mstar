"""A step's input ids are each row's loop-back token, or the ids the row
brought: a mixed step's prompt rows, or a decode row whose last step ran
elsewhere. The splice must equal the old per-row forms exactly."""

from __future__ import annotations

import itertools

import pytest
import torch

from mstar.model.qwen3_5.submodules import LLMSubmodule
from mstar.model.submodule_base import ARNodeInputs


def _mixed_reference(ids, inputs):
    return torch.cat([
        inp.input_ids.to(ids.dtype) if inp.input_ids is not None else ids[i:i + 1]
        for i, inp in enumerate(inputs)
    ])


def _decode_reference(ids, inputs):
    out = ids.clone()
    routed = [i for i, inp in enumerate(inputs) if inp.input_ids is not None and inp.input_seq_len]
    if routed:
        out.index_copy_(0, torch.tensor(routed), torch.cat([inputs[i].input_ids for i in routed]).to(ids.dtype))
    return out


def _inputs(pattern: str, decode: bool) -> list[ARNodeInputs]:
    """'d' a loop-back row; 'p' a row that brought ids (1-3 tokens for a mixed
    prompt, one token for a decode row); 'z' a row with ids but no tokens."""
    rows = []
    for i, kind in enumerate(pattern):
        if kind == "d":
            rows.append(ARNodeInputs(input_seq_len=1))
        elif kind == "z":
            rows.append(ARNodeInputs(input_seq_len=0, input_ids=torch.tensor([5], dtype=torch.int32)))
        else:
            n = 1 if decode else 1 + i % 3
            rows.append(ARNodeInputs(
                input_seq_len=n, input_ids=torch.arange(1000 + 10 * i, 1000 + 10 * i + n, dtype=torch.int32),
            ))
    return rows


PATTERNS = ["d", "p", "ddd", "ppp", "pdd", "ddp", "dpd", "pdp", "dpdpd", "ppddpp",
            *("".join(c) for c in itertools.product("dp", repeat=5))]


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("padding", [0, 3])
def test_mixed_splice_matches_one_piece_per_row(pattern, padding):
    inputs = _inputs(pattern, decode=False)
    ids = torch.arange(len(inputs) + padding, dtype=torch.long) + 7
    got = LLMSubmodule._splice_row_ids(ids, inputs, decode=False)
    want = _mixed_reference(ids, inputs)
    assert got.dtype == want.dtype == ids.dtype
    assert torch.equal(got, want)


@pytest.mark.parametrize("pattern", [*PATTERNS, "dzd", "zpz", "pzd"])
@pytest.mark.parametrize("padding", [0, 3])
def test_decode_splice_matches_index_copy(pattern, padding):
    inputs = _inputs(pattern, decode=True)
    ids = torch.arange(len(inputs) + padding, dtype=torch.long) + 7
    got = LLMSubmodule._splice_row_ids(ids, inputs, decode=True)
    want = _decode_reference(ids, inputs)
    assert got.shape == ids.shape  # the captured input's length
    assert torch.equal(got, want)
