"""Phi-4-mini rotates each request with one of two LongRoPE tables, picked by its prompt's length.

A declaration with no way to say so rotates every request with the plain table, and the output drifts from the reference
with no error. The two tables also give different KV for the same tokens, so a long request mustn't reuse a short one's
prefix pages.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.position.config import PositionReqConfig
from mstar.engine.state import MAX_POSITION_KWARG, Array, Attend, Causal, KVEntry, LongRoPE, Positions, Read, Rope
from mstar.engine.state.adapters import build_specs

_LONGROPE = LongRoPE(
    short_factor=(1.0,) * 8, long_factor=(4.0,) * 8, original_max_position_embeddings=64, max_position_embeddings=256,
)


def _positions(rope: Rope):
    kv = Array(
        "kv", nodes={"llm"}, layers={0, 1}, entry=KVEntry(heads=2, head_dim=16), max_len=256,
        positions=Positions(rope=rope), reads={"attn": Read(mask=Causal(), op=Attend(q_heads=4))},
    )
    (spec,) = [spec for spec in build_specs([kv]) if spec.resource_key == "rope"]
    return spec


def test_a_declared_longrope_reaches_the_position_resource():
    config = _positions(Rope(rotary_dim=16, table_dtype=torch.bfloat16, longrope=_LONGROPE)).config
    assert (config.rotary_dim, config.rope_table_dtype) == (16, torch.bfloat16)
    assert config.longrope == _LONGROPE, "the position resource would build the plain table alone"


def test_a_prompt_past_the_original_context_keys_its_prefix_apart():
    spec = _positions(Rope(longrope=_LONGROPE))
    assert spec.request_config({MAX_POSITION_KWARG: 65}) == PositionReqConfig(max_position=65)
    assert spec.prefix_variant({MAX_POSITION_KWARG: 64}) is None
    assert spec.prefix_variant({MAX_POSITION_KWARG: 65}) is not None, "a long request would reuse a short one's pages"


@pytest.mark.parametrize("scaling", [
    {"scale": 2.0}, {"low_freq_factor": 1.0, "high_freq_factor": 4.0, "old_context_len": 8192},
])
def test_longrope_refuses_any_other_scaling(scaling):
    with pytest.raises(ValueError, match="takes scale 1.0 and no Llama 3.1"):
        Rope(longrope=_LONGROPE, **scaling)


def test_longrope_refuses_a_factor_count_its_rotary_dim_does_not_take():
    with pytest.raises(ValueError, match="needs 4 factors per set for rotary_dim 8, and got 8"):
        Rope(rotary_dim=8, longrope=_LONGROPE)
