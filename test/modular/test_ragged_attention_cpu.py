"""Ragged attention head-dimension padding checks that need no GPU."""

import pytest

from mstar.engine.resources.attn.ragged.wrappers import (
    SUPPORTED_HEAD_DIMS,
    padded_head_dim,
)


@pytest.mark.parametrize(
    ("head_dim", "expected"), [(64, 64), (72, 128), (128, 128), (129, 256), (256, 256)]
)
def test_padded_head_dim_rounds_up(head_dim, expected):
    assert padded_head_dim(head_dim) == expected


def test_padded_head_dim_rejects_oversized():
    with pytest.raises(ValueError, match="exceeds the largest supported"):
        padded_head_dim(SUPPORTED_HEAD_DIMS[-1] + 1)
