"""``mstar.utils.pinned_staging``: host lists staged through pinned memory."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mstar.utils.pinned_staging import pinned, to_device_async  # noqa: E402


def test_pinned_preserves_shape_dtype_values():
    t = pinned([[3, 4], [5, 6], [7, 8]], torch.long)
    assert t.shape == (3, 2) and t.dtype == torch.long
    assert t.tolist() == [[3, 4], [5, 6], [7, 8]]
    u = pinned([0, 4, 8], torch.int32)
    assert u.shape == (3,) and u.dtype == torch.int32 and u.tolist() == [0, 4, 8]
    e = pinned([], torch.int32)
    assert e.shape == (0,)
    if torch.cuda.is_available():
        assert t.is_pinned() and u.is_pinned()


def test_pinned_accepts_cpu_tensor_and_rejects_device():
    src = torch.tensor([1, 2, 3], dtype=torch.int64)
    t = pinned(src, torch.int32)
    assert t.dtype == torch.int32 and t.tolist() == [1, 2, 3]
    if torch.cuda.is_available():
        dev = torch.tensor([1], device="cuda")
        try:
            pinned(dev)
        except ValueError:
            pass
        else:
            raise AssertionError("pinned() must reject device tensors")


def test_to_device_async_cpu_roundtrip():
    t = to_device_async([5, 6, 7], torch.long, torch.device("cpu"))
    assert t.tolist() == [5, 6, 7] and t.dtype == torch.long
