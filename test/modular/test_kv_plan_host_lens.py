"""The KV plan's host row of sequence lengths iterates as numpy scalars, so
FlashInfer's ``max(seq_lens).item()`` in its decode plan stays cheap."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mstar.engine.resources.kv.plan import HostLens, _host_lens  # noqa: E402


def test_host_lens_iterates_as_numpy_scalars():
    lens = _host_lens([3, 17, 9])
    assert isinstance(lens, HostLens) and lens.dtype == torch.int32
    items = list(lens)
    assert [int(x) for x in items] == [3, 17, 9]
    assert all(isinstance(x, np.integer) for x in items)
    # what FlashInfer does with it
    assert max(lens).item() == 17 and int(lens.min()) == 3 and len(lens) == 3
    assert lens.cpu() is lens


def test_host_lens_copies_into_a_plain_buffer():
    lens = _host_lens([5, 6, 7, 8])
    buf = torch.zeros(8, dtype=torch.int32)
    buf[: len(lens)].copy_(lens)
    assert buf.tolist() == [5, 6, 7, 8, 0, 0, 0, 0]
    assert torch.equal(lens, torch.tensor([5, 6, 7, 8], dtype=torch.int32))
