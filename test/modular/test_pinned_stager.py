"""PinnedStager (mstar/utils/h2d.py): copy-only H2D with host-side padding."""
import pytest
import torch

from mstar.utils.h2d import H2DMirror, PinnedStager

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


@cuda
def test_mirror_skips_a_copy_the_destination_already_holds(monkeypatch):
    dst = torch.zeros(6, dtype=torch.int32, device="cuda")
    st, mirror = PinnedStager(torch.int32), H2DMirror()
    issued = []
    real_copy = torch.Tensor.copy_
    monkeypatch.setattr(
        torch.Tensor, "copy_",
        lambda self, src, non_blocking=False: (
            issued.append(1), real_copy(self, src, non_blocking=non_blocking)
        )[1],
    )
    st.copy_(dst, [1, 2], pad_value=0, mirror=mirror)
    st.copy_(dst, [1, 2], pad_value=0, mirror=mirror)   # same: skipped
    assert len(issued) == 1
    st.copy_(dst, [1, 3], pad_value=0, mirror=mirror)   # new values
    st.copy_(dst, [1, 3], pad_value=-1, mirror=mirror)  # same values, new padding
    st.copy_(dst, [1, 3, 0], pad_value=-1, mirror=mirror)  # different length
    assert len(issued) == 4
    monkeypatch.undo()
    torch.cuda.synchronize()
    assert dst.tolist() == [1, 3, 0, -1, -1, -1]


@cuda
def test_invalidated_mirror_copies_again():
    dst = torch.zeros(3, dtype=torch.int32, device="cuda")
    st, mirror = PinnedStager(torch.int32), H2DMirror()
    st.copy_(dst, [4, 5, 6], mirror=mirror)
    torch.cuda.synchronize()
    dst.zero_()                 # written some other way...
    mirror.invalidate()         # ...which its owner says
    st.copy_(dst, [4, 5, 6], mirror=mirror)
    torch.cuda.synchronize()
    assert dst.tolist() == [4, 5, 6]


@cuda
def test_mirror_compares_values_not_the_object_passed():
    """A caller passing a reused (and since rewritten) array still copies."""
    import numpy as np

    dst = torch.zeros(2, dtype=torch.int32, device="cuda")
    st, mirror = PinnedStager(torch.int32), H2DMirror()
    host = np.array([1, 2], dtype=np.int32)
    st.copy_(dst, host, mirror=mirror)
    host[:] = [7, 8]
    st.copy_(dst, host, mirror=mirror)
    torch.cuda.synchronize()
    assert dst.tolist() == [7, 8]
