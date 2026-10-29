"""``PinnedStager`` copies raw bytes only into a destination of its own dtype
and layout; any other goes through the torch copy, which converts the dtype
and follows the destination's strides, instead of writing past it or
scrambling it."""
import numpy as np
import pytest
import torch

from mstar.utils.h2d import PinnedStager

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the direct copy is CUDA's")


def test_a_matching_destination_gets_the_values():
    dst = torch.zeros(6, dtype=torch.int64, device="cuda")
    PinnedStager(torch.int64).copy_(dst, np.arange(4, dtype=np.int64), pad_value=-1)
    assert dst.tolist() == [0, 1, 2, 3, -1, -1]


def test_another_dtype_is_converted_not_overrun():
    # an int64 stager's 8-byte words would run past a 4-element int32 tensor
    buf = torch.full((8,), 7, dtype=torch.int32, device="cuda")
    dst = buf[:4]
    PinnedStager(torch.int64).copy_(dst, np.array([1, 2, 3, 4], dtype=np.int64))
    assert buf.tolist() == [1, 2, 3, 4, 7, 7, 7, 7]


def test_a_strided_destination_gets_its_own_elements():
    buf = torch.zeros(4, 2, dtype=torch.int64, device="cuda")
    PinnedStager(torch.int64).copy_(buf[:, 0], np.arange(1, 5, dtype=np.int64))
    assert buf.tolist() == [[1, 0], [2, 0], [3, 0], [4, 0]]
