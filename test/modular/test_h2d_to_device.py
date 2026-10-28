"""``to_device_nonblocking``: the pinned path must equal a pageable copy on any input
layout and dtype, without blocking the caller. The CPU cases run anywhere; the
pinned cases need an accelerator, since there is nothing to pin without one.
"""

from __future__ import annotations

import pytest
import torch

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


def test_a_pinned_input_is_queued_as_it_is(accelerator_device, monkeypatch):
    # (address, non_blocking) of every copy issued from host memory
    copies: list[tuple[int, bool]] = []
    to = torch.Tensor.to

    def spy(self, *args, **kwargs):
        if self.device.type == "cpu":
            copies.append((self.data_ptr(), kwargs.get("non_blocking", False)))
        return to(self, *args, **kwargs)

    tensor = torch.arange(1 << 16, dtype=torch.float32).pin_memory()
    monkeypatch.setattr(torch.Tensor, "to", spy)
    got = to_device_nonblocking(tensor, accelerator_device)
    torch.accelerator.synchronize(accelerator_device)
    # no second host copy: the device copy is queued from the caller's own buffer
    assert copies == [(tensor.data_ptr(), True)]
    assert torch.equal(got.cpu(), tensor)
