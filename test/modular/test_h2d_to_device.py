"""``to_device_nonblocking``: the pinned path must equal a pageable copy on any input
layout and dtype, without blocking the caller. The CPU cases run anywhere; the
pinned cases need an accelerator, since there is nothing to pin without one.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch

from mstar.utils import h2d
from mstar.utils.h2d import PinnedStager, to_device_nonblocking


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


def test_runs_fill_their_destinations_in_order():
    stager = PinnedStager(torch.long, numel=4)
    store = torch.zeros(10, dtype=torch.long)
    stager.copy_runs_([store[1:3], store[6:9]], [1, 2, 3, 4, 5])
    assert store.tolist() == [0, 1, 2, 0, 0, 0, 3, 4, 5, 0]


def test_runs_past_the_staging_buffer_grow_it(accelerator_device):
    stager = PinnedStager(torch.long, numel=4)
    # two runs that together are past one segment, so the claim grows the arena
    run = stager._seg_numel // 2 + 8
    store = torch.zeros(4 * run, dtype=torch.long, device=accelerator_device)
    values = np.arange(1, 2 * run + 1, dtype=np.int64)
    stager.copy_runs_([store[:run], store[2 * run:3 * run]], values)
    torch.accelerator.synchronize(accelerator_device)
    assert stager._seg_numel >= 2 * run
    assert store[:run].tolist() == values[:run].tolist()
    assert store[run:2 * run].tolist() == [0] * run
    assert store[2 * run:3 * run].tolist() == values[run:].tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA stream execution")
@pytest.mark.parametrize("kind", ["immune", "ring"])
def test_repeat_staging_does_not_wait_for_the_previous_run(kind):
    from mstar.engine.device_run import RunState
    from mstar.engine.engine import Engine
    from mstar.engine.resources.loopback_rows import LoopbackRows
    from mstar.model.submodule_base import BatchedModelOutput

    device = torch.device("cuda")
    state = RunState(2, "token", {9}, device)
    rows = LoopbackRows(device, capacity=4)
    rows.ask_ring("frames", depth=4)
    assert rows.ring_table("frames", torch.zeros(2, 2, device=device)) is None
    rows.add(1)
    items = torch.arange(4, device=device).reshape(2, 1, 2)
    batch = types.SimpleNamespace(request_ids=[2, 1], ring_bases={"frames": {1: 0}}, ring_width=1)
    output = BatchedModelOutput(row_outputs={"frames": items[:, 0]})
    mgmt = types.SimpleNamespace(loopback_rows=rows)
    # Warm allocation/indexing paths before checking a busy stream.
    state.stage(2, [False, False], 2)
    Engine._write_rings(None, mgmt, batch, output)
    torch.cuda.synchronize()
    done = torch.cuda.Event()
    torch.cuda._sleep(500_000_000)
    done.record()
    if kind == "immune":
        state.stage(2, [True, False], 3)
    else:
        Engine._write_rings(None, mgmt, batch, output)
    assert not done.query(), "staging waited for the preceding GPU run"
    torch.cuda.synchronize()
    if kind == "immune":
        assert state.immune.tolist() == [True, False]
    else:
        assert rows.ring_read("frames", 1, [0]).tolist() == [[2, 3]]
