"""PinnedStager.to_device: a new device tensor of any shape, staged in one copy."""
import numpy as np
import pytest
import torch

from mstar.utils.h2d import PinnedStager


@pytest.mark.parametrize("values", [
    [3, 1, 4], np.array([[1, 2, 3], [4, 5, 6]]), torch.arange(5, dtype=torch.int32), [],
])
def test_to_device_holds_the_values_in_the_stager_dtype(values):
    stager = PinnedStager(torch.long, numel=2)  # smaller than some inputs: the ring grows
    out = stager.to_device(values, torch.device("cpu"))
    want = torch.as_tensor(np.asarray(values), dtype=torch.long)
    assert out.dtype == torch.long and out.shape == want.shape and torch.equal(out, want)


def test_to_device_is_a_fresh_tensor_each_time():
    stager = PinnedStager(torch.int32)
    a = stager.to_device([1, 2], torch.device("cpu"))
    b = stager.to_device([3, 4], torch.device("cpu"))
    assert a.tolist() == [1, 2] and b.tolist() == [3, 4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the pinned ring's non-blocking copy")
def test_to_device_on_cuda_survives_ring_reuse():
    stager = PinnedStager(torch.long, numel=4, depth=2)
    outs = [stager.to_device([i, i + 1, i + 2], torch.device("cuda")) for i in range(6)]
    torch.cuda.synchronize()
    assert [o.tolist() for o in outs] == [[i, i + 1, i + 2] for i in range(6)]
