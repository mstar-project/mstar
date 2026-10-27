"""``to_device_nonblocking``: a host tensor goes to the device through a pinned
buffer and a non-blocking copy, so the caller never waits on the stream, and the
pinned buffer is not handed out again until its copy has run.

The CPU case runs anywhere; the rest need an accelerator.
"""

from __future__ import annotations

import sys
import time

import pytest
import torch

sys.path.insert(0, ".")

from mstar.utils.h2d import to_device_nonblocking  # noqa: E402


@pytest.fixture(params=["cuda", "xpu"])
def accelerator_device(request):
    runtime = getattr(torch, request.param)
    if not runtime.is_available():
        pytest.skip(f"{request.param} is unavailable")
    return torch.device(request.param, 0)


def test_a_cpu_destination_gets_a_plain_copy():
    src = torch.randn(32)
    out = to_device_nonblocking(src, "cpu")
    assert out.device.type == "cpu" and torch.equal(out, src)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int64, torch.bool])
def test_the_copy_equals_the_pageable_one(dtype, accelerator_device):
    for shape in ((4096, 128), (1, 128, 64, 64), (7,), (), (0, 3)):
        src = (torch.randn(shape) * 100).to(dtype)
        out = to_device_nonblocking(src, accelerator_device)
        torch.accelerator.synchronize(accelerator_device)
        assert out.device == accelerator_device and out.dtype == dtype and out.shape == src.shape
        assert torch.equal(out.cpu(), src), f"copy differs for {shape}"


def test_a_non_contiguous_source_lands_contiguous(accelerator_device):
    src = torch.randn(16, 32).t()
    out = to_device_nonblocking(src, accelerator_device)
    torch.accelerator.synchronize(accelerator_device)
    assert out.is_contiguous() and torch.equal(out.cpu(), src)


def test_a_tensor_already_on_the_device_is_returned_as_is(accelerator_device):
    t = torch.ones(4, device=accelerator_device)
    assert to_device_nonblocking(t, accelerator_device) is t


def _hold_stream(ms: float) -> tuple[torch.cuda.Event, torch.cuda.Event]:
    """Queue about ``ms`` of spinning on the current stream, between the two events returned."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    # an idle GPU runs its first spin at a low clock, so let it ramp up before measuring it
    torch.cuda._sleep(50_000_000)
    start.record()
    torch.cuda._sleep(10_000_000)
    end.record()
    end.synchronize()
    cycles_per_ms = 10_000_000 / max(start.elapsed_time(end), 1e-3)
    start.record()
    torch.cuda._sleep(int(cycles_per_ms * ms))
    end.record()
    return start, end


@pytest.mark.skipif(not torch.cuda.is_available(), reason="holds the stream with torch.cuda._sleep")
def test_copies_queue_from_pinned_buffers_that_are_reused_only_after_they_ran(monkeypatch):
    device = torch.device("cuda", 0)
    # (pinned, non_blocking, address) of every copy issued from host memory
    copies: list[tuple[bool, bool, int]] = []
    to = torch.Tensor.to

    def spy(self, *args, **kwargs):
        if self.device.type == "cpu":
            copies.append((self.is_pinned(), kwargs.get("non_blocking", False), self.data_ptr()))
        return to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", spy)
    # 5 MiB, a pinned block size no other test here uses, so the reuse below is ours
    src = torch.zeros(5 << 18)
    torch.cuda.synchronize()
    start, end = _hold_stream(1000)
    t0 = time.perf_counter()
    outs = []
    for i in range(8):
        outs.append(to_device_nonblocking(src, device))
        src.fill_(i + 1)  # the caller's tensor is free as soon as the call returns
    calls_ms = (time.perf_counter() - t0) * 1e3
    busy = not torch.cuda.current_stream().query()
    end.synchronize()
    # none of the calls waited for the stream...
    assert busy, (
        f"the stream was idle after {calls_ms:.0f} ms of calls with {start.elapsed_time(end):.0f} ms queued:"
        " a copy waited for it, or the hold was too short"
    )
    # ...since each copy was queued from pinned memory, and no buffer was handed out
    # again while its copy was still waiting
    assert len(copies) == 8 and all(pinned and non_blocking for pinned, non_blocking, _ in copies)
    in_flight = {address for *_, address in copies}
    assert len(in_flight) == 8
    torch.cuda.synchronize()
    for i, out in enumerate(outs):
        assert torch.equal(out.cpu(), torch.full_like(src, i)), f"copy {i} saw another call's values"
    # once those copies have run, the host allocator hands their buffers out again
    to_device_nonblocking(src, device)
    assert copies[-1][2] in in_flight
