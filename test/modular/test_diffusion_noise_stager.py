"""NoiseStager: the pinned-staging path must match the pageable one exactly, and must
never let a buffer be rewritten while its copy is still in flight.

The CPU cases run anywhere. The pinned/async cases need CUDA, since there is nothing
to pin and nothing to overlap without it.
"""

from __future__ import annotations

import sys

import pytest
import torch

sys.path.insert(0, ".")

from mstar.model.components.diffusion.noise import NoiseStager  # noqa: E402


@pytest.fixture(params=["cuda", "xpu"])
def accelerator_device(request):
    runtime = getattr(torch, request.param)
    if not runtime.is_available():
        pytest.skip(f"{request.param} is unavailable")
    return torch.device(request.param, 0)


def test_to_device_on_cpu_is_a_plain_copy():
    stager = NoiseStager(torch.float32)
    src = torch.randn(32)
    out = stager.to_device(src, torch.device("cpu"))
    assert out.device.type == "cpu" and torch.equal(out, src)


def test_dtype_mismatch_is_rejected(accelerator_device):
    stager = NoiseStager(torch.bfloat16)
    with pytest.raises(ValueError, match="holds"):
        stager.to_device(torch.randn(8, dtype=torch.float32), accelerator_device)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.int64, torch.bool])
def test_staged_copy_equals_the_pageable_copy(dtype, accelerator_device):
    """What the engine would have got from ``tensor.to(device)``, byte for byte."""
    stager = NoiseStager(dtype)
    device = accelerator_device
    for shape in ((4096, 128), (1, 128, 64, 64), (7,), ()):
        src = (torch.randn(shape) * 100).to(dtype)
        staged = stager.to_device(src, device)
        torch.accelerator.synchronize(device)
        assert staged.shape == src.shape and staged.dtype == dtype
        assert torch.equal(staged.cpu(), src), f"staged copy differs for {shape}"


def test_non_contiguous_input_is_staged_correctly(accelerator_device):
    stager = NoiseStager(torch.float32)
    src = torch.randn(16, 32).t()  # non-contiguous view
    staged = stager.to_device(src, accelerator_device)
    torch.accelerator.synchronize(accelerator_device)
    assert torch.equal(staged.cpu(), src)


def test_a_buffer_is_not_rewritten_while_its_copy_is_in_flight(accelerator_device):
    """The ring must wrap only onto buffers whose copy has retired.

    Issue more copies than the ring is deep, with distinct contents, and check every
    result. If a buffer were rewritten under an in-flight DMA the later values would
    bleed into the earlier tensors -- silently, and only under load.
    """
    depth = 3
    stager = NoiseStager(torch.float32, depth=depth)
    device = accelerator_device
    sources = [torch.full((1 << 18,), float(i)) for i in range(depth * 4)]
    staged = [stager.to_device(s, device) for s in sources]
    torch.accelerator.synchronize(device)
    for i, (out, src) in enumerate(zip(staged, sources, strict=True)):
        assert torch.equal(out.cpu(), src), f"copy {i} saw a rewritten staging buffer"


def test_growth_preserves_in_flight_copies(accelerator_device):
    """A copy issued before a grow must still land: _grow synchronises first."""
    stager = NoiseStager(torch.float32, depth=2, numel=16)
    device = accelerator_device
    small = torch.full((16,), 1.0)
    first = stager.to_device(small, device)
    big = torch.full((1 << 20,), 2.0)          # forces _grow
    second = stager.to_device(big, device)
    torch.accelerator.synchronize(device)
    assert torch.equal(first.cpu(), small), "a pre-grow copy was lost"
    assert torch.equal(second.cpu(), big)


def test_stage_all_covers_a_seed_set(accelerator_device):
    stager = NoiseStager(torch.bfloat16)
    device = accelerator_device
    seeds = {"latents": torch.randn(64, 8).to(torch.bfloat16),
             "solver": torch.randn(4, 8).to(torch.bfloat16)}
    out = stager.stage_all(seeds, device)
    torch.accelerator.synchronize(device)
    assert set(out) == set(seeds)
    for name, tensor in seeds.items():
        assert torch.equal(out[name].cpu(), tensor)
