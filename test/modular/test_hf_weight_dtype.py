"""`hf_weight_dtype`: the weights' dtype a config.json names, under either key."""
from __future__ import annotations

import pytest

from mstar.model.hf_config import hf_weight_dtype


@pytest.mark.parametrize("hf, want", [
    ({"dtype": "float16"}, "float16"),
    ({"torch_dtype": "float32"}, "float32"),
    ({"dtype": "bfloat16", "torch_dtype": "bfloat16"}, "bfloat16"),
    ({}, "bfloat16"),
], ids=["dtype", "torch_dtype", "both_agree", "neither"])
def test_the_weights_dtype_comes_from_either_key(hf, want):
    assert hf_weight_dtype(hf) == want


def test_keys_that_disagree_are_refused():
    with pytest.raises(ValueError, match="dtype 'float16' and torch_dtype 'bfloat16'"):
        hf_weight_dtype({"dtype": "float16", "torch_dtype": "bfloat16"})
