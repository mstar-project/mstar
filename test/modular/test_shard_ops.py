"""The sharding computation in rust/src/shards.rs, through the torch wrapper and
through torch's Python calls (what ROCm, XPU and Mooncake run), against torch."""

import pytest
import torch

rs = pytest.importorskip("mstar_rust")

from mstar.communication.torch_abi import require_torch_abi  # noqa: E402

if not require_torch_abi("cpu"):
    pytest.skip("this platform has no torch wrapper", allow_module_level=True)

native = pytest.mark.parametrize("native", [True, False], ids=["wrapper", "torch-calls"])


def _tensor(*shape):
    return torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)


@native
@pytest.mark.parametrize("dim", [1, 2])
def test_lead_and_undo_round_trip(native, dim):
    t = _tensor(2, 3, 4)
    led = rs.shard_lead(t, dim, native)
    perm = [dim] + [i for i in range(3) if i != dim]
    assert torch.equal(led, t.permute(*perm))
    assert led.is_contiguous()
    back = rs.shard_undo_lead(led, dim, native)
    assert torch.equal(back, t) and back.is_contiguous()


@native
def test_a_slice_is_a_view_of_its_rows(native):
    t = _tensor(4, 3)
    view = rs.shard_slice_rows(t, 3 * t.element_size() * 1, 3 * t.element_size() * 2, 2, native)
    assert torch.equal(view, t[1:3])
    assert view.data_ptr() == t[1:].data_ptr(), "a slice must not copy"


@native
def test_a_slice_of_a_conjugate_view_finds_its_rows(native):
    t = torch.arange(12.0).to(torch.complex64).reshape(4, 3).conj()
    view = rs.shard_slice_rows(t, 3 * t.element_size(), 3 * t.element_size() * 2, 2, native)
    assert torch.equal(view, t[1:3])


@native
def test_rows_of_no_bytes_take_the_descriptors_row_count(native):
    t = torch.empty(4, 0)
    assert rs.shard_slice_rows(t, 0, 0, 2, native).shape == (2, 0)


@native
def test_a_misaligned_slice_is_refused(native):
    t = _tensor(4, 3)
    with pytest.raises(AssertionError, match="not aligned"):
        rs.shard_slice_rows(t, 1, 12, 1, native)


@native
def test_shards_join_in_rank_order_once_all_ranks_are_in(native):
    a, b = _tensor(2, 3), _tensor(1, 3) + 100
    buf = rs.BufferedShards(2, 0)
    buf.add(1, [b])
    assert not buf.is_done()
    with pytest.raises(AssertionError, match="until all"):
        buf.consolidate(native)
    buf.add(0, [a])
    assert buf.is_done()
    (joined,) = buf.consolidate(native)
    assert torch.equal(joined, torch.cat([a, b], dim=0))


@native
def test_unequal_tensor_counts_from_the_ranks_are_refused(native):
    buf = rs.BufferedShards(2, 0)
    buf.add(0, [_tensor(1), _tensor(1)])
    buf.add(1, [_tensor(1)])
    with pytest.raises(AssertionError, match="unequal tensor counts"):
        buf.consolidate(native)
