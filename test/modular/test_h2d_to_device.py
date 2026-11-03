"""``to_device_nonblocking``: the pinned path must equal a pageable copy on any input
layout and dtype, without blocking the caller. The CPU cases run anywhere; the
pinned cases need an accelerator, since there is nothing to pin without one.
"""

from __future__ import annotations

import types

import pytest
import torch

from mstar.utils import h2d
from mstar.utils.h2d import to_device_nonblocking


@pytest.fixture(params=["cuda", "xpu"])
def accelerator_device(request):
    runtime = getattr(torch, request.param)
    if not runtime.is_available():
        pytest.skip(f"{request.param} is unavailable")
    return torch.device(request.param, 0)


def test_a_tensor_already_on_its_device_is_returned_as_is():
    tensor = torch.arange(6.0)
    assert to_device_nonblocking(tensor, torch.device("cpu")) is tensor


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.int64])
def test_the_queued_copy_equals_the_pageable_one(dtype, accelerator_device):
    tensor = (torch.randn(3, 5, 7) * 100).to(dtype)
    got = to_device_nonblocking(tensor, accelerator_device)
    torch.accelerator.synchronize(accelerator_device)
    assert got.device == accelerator_device and got.dtype == dtype and got.shape == tensor.shape
    assert torch.equal(got.cpu(), tensor)


def test_a_non_contiguous_input_is_copied_correctly(accelerator_device):
    tensor = torch.randn(8, 4).t()
    assert not tensor.is_contiguous()
    got = to_device_nonblocking(tensor, accelerator_device)
    torch.accelerator.synchronize(accelerator_device)
    assert torch.equal(got.cpu(), tensor)


def test_an_empty_tensor_arrives_with_its_shape(accelerator_device):
    got = to_device_nonblocking(torch.empty(0, 3), accelerator_device)
    assert got.shape == (0, 3) and got.device == accelerator_device


@pytest.mark.parametrize("contiguous", [True, False])
def test_an_already_pinned_tensor_is_copied_from_in_place(monkeypatch, contiguous):
    """A pinned contiguous source (``NemoLogMel.batch`` pads its clips into pinned
    memory) is copied from as is, not through a second pinned buffer. Runs on any
    host: the device is made to take pinned memory and the source to report pinned."""
    takes_pinned = types.SimpleNamespace(pin_memory=True)
    monkeypatch.setattr(h2d, "DeviceCapability", types.SimpleNamespace(of=lambda device: takes_pinned))
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: True)
    pinned_allocs = []
    empty = torch.empty

    def spy_empty(*args, pin_memory=False, **kwargs):
        if pin_memory:
            pinned_allocs.append(args)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", spy_empty)
    source = torch.arange(12.0).reshape(3, 4) if contiguous else torch.arange(12.0).reshape(4, 3).t()
    got = to_device_nonblocking(source, torch.device("cpu"))
    # a non-contiguous one is packed into a pinned buffer: ``.to`` would pack it pageable and sync
    assert (pinned_allocs == []) == contiguous and (got.data_ptr() == source.data_ptr()) == contiguous
    assert torch.equal(got, source)
