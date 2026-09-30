"""PinnedStager (mstar/utils/h2d.py): copy-only H2D with host-side padding."""
import pytest
import torch

from mstar.utils.h2d import PinnedStager

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_values_and_host_padding_land_in_one_copy():
    dst = torch.full((8,), -7, dtype=torch.int32, device="cuda")
    st = PinnedStager(torch.int32, numel=4)  # grows past its initial size
    st.copy_(dst, [1, 2, 3], pad_value=0)
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 2, 3, 0, 0, 0, 0, 0]
    st.copy_(dst[:2], [9, 9])  # no padding: only the values' rows change
    torch.cuda.synchronize()
    assert dst.tolist() == [9, 9, 3, 0, 0, 0, 0, 0]


@cuda
def test_bool_and_long_dtypes():
    b = torch.ones(5, dtype=torch.bool, device="cuda")
    PinnedStager(torch.bool).copy_(b, [True, False], pad_value=False)
    x = torch.zeros(4, dtype=torch.long, device="cuda")
    PinnedStager(torch.long).copy_(x, [2 ** 40], pad_value=-1)
    torch.cuda.synchronize()
    assert b.tolist() == [True, False, False, False, False]
    assert x.tolist() == [2 ** 40, -1, -1, -1]


@cuda
def test_buffers_are_not_rewritten_under_an_in_flight_copy():
    """More copies than the ring holds, queued behind ~20 ms of GPU work: a
    buffer rewritten before its copy ran would land a later value early."""
    a = torch.randn(4096, 4096, device="cuda")
    b = torch.empty_like(a)
    st = PinnedStager(torch.int32, depth=2)
    dsts = [torch.zeros(4, dtype=torch.int32, device="cuda") for _ in range(6)]
    for _ in range(10):
        torch.mm(a, a, out=b)
    for i, d in enumerate(dsts):
        st.copy_(d, [i, i, i, i])
    torch.cuda.synchronize()
    assert [d.tolist() for d in dsts] == [[i] * 4 for i in range(6)]


def test_cpu_destination():
    dst = torch.zeros(4, dtype=torch.int32)
    PinnedStager(torch.int32).copy_(dst, [5, 6], pad_value=1)
    assert dst.tolist() == [5, 6, 1, 1]


@cuda
def test_too_many_values_rejected():
    with pytest.raises(ValueError):
        PinnedStager(torch.int32).copy_(torch.zeros(2, dtype=torch.int32, device="cuda"),
                                        [1, 2, 3], pad_value=0)
