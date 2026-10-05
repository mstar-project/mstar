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

CUDA = torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA, reason="requires CUDA")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_randn_matches_an_inline_draw(dtype):
    """The stager's generator must reproduce the inline draw bit for bit: the seeded
    noise is pinned to randn_tensor parity with the reference pipelines."""
    shape = (1, 128, 8, 8)
    stager = NoiseStager(dtype)
    mine = stager.randn(shape, seed=1234)
    theirs = torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(1234), dtype=dtype)
    assert mine.dtype == dtype and mine.shape == shape
    assert torch.equal(mine, theirs)


def test_randn_is_reproducible_across_calls_and_seeds():
    stager = NoiseStager(torch.bfloat16)
    a = stager.randn((4, 16), seed=7)
    b = stager.randn((4, 16), seed=8)
    c = stager.randn((4, 16), seed=7)
    assert torch.equal(a, c), "same seed must give the same draw after an intervening one"
    assert not torch.equal(a, b)


def test_to_device_on_cpu_is_a_plain_copy():
    stager = NoiseStager(torch.float32)
    src = torch.randn(32)
    out = stager.to_device(src, torch.device("cpu"))
    assert out.device.type == "cpu" and torch.equal(out, src)


def test_dtype_mismatch_is_rejected():
    stager = NoiseStager(torch.bfloat16)
    if not CUDA:
        pytest.skip("the check guards the CUDA path")
    with pytest.raises(ValueError, match="holds"):
        stager.to_device(torch.randn(8, dtype=torch.float32), torch.device("cuda"))


@cuda_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_staged_copy_equals_the_pageable_copy(dtype):
    """What the engine would have got from ``tensor.to(device)``, byte for byte."""
    stager = NoiseStager(dtype)
    device = torch.device("cuda")
    for shape in ((4096, 128), (1, 128, 64, 64), (7,)):
        src = torch.randn(shape).to(dtype)
        staged = stager.to_device(src, device)
        torch.cuda.synchronize()
        assert staged.shape == src.shape and staged.dtype == dtype
        assert torch.equal(staged.cpu(), src), f"staged copy differs for {shape}"


@cuda_only
def test_non_contiguous_input_is_staged_correctly():
    stager = NoiseStager(torch.float32)
    src = torch.randn(16, 32).t()  # non-contiguous view
    staged = stager.to_device(src, torch.device("cuda"))
    torch.cuda.synchronize()
    assert torch.equal(staged.cpu(), src)


@cuda_only
def test_a_buffer_is_not_rewritten_while_its_copy_is_in_flight():
    """The ring must wrap only onto buffers whose copy has retired.

    Issue more copies than the ring is deep, with distinct contents, and check every
    result. If a buffer were rewritten under an in-flight DMA the later values would
    bleed into the earlier tensors -- silently, and only under load.
    """
    depth = 3
    stager = NoiseStager(torch.float32, depth=depth)
    device = torch.device("cuda")
    sources = [torch.full((1 << 18,), float(i)) for i in range(depth * 4)]
    staged = [stager.to_device(s, device) for s in sources]
    torch.cuda.synchronize()
    for i, (out, src) in enumerate(zip(staged, sources, strict=True)):
        assert torch.equal(out.cpu(), src), f"copy {i} saw a rewritten staging buffer"


@cuda_only
def test_growth_preserves_in_flight_copies():
    """A copy issued before a grow must still land: _grow synchronises first."""
    stager = NoiseStager(torch.float32, depth=2, numel=16)
    device = torch.device("cuda")
    small = torch.full((16,), 1.0)
    first = stager.to_device(small, device)
    big = torch.full((1 << 20,), 2.0)          # forces _grow
    second = stager.to_device(big, device)
    torch.cuda.synchronize()
    assert torch.equal(first.cpu(), small), "a pre-grow copy was lost"
    assert torch.equal(second.cpu(), big)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_randn_to_device_matches_an_inline_draw(dtype):
    """Drawing with ``out=`` into the pinned buffer must not perturb the RNG stream.

    This is the gate on the host-memcpy-skipping path: if ``out=`` consumed the
    generator differently the seeded noise would silently stop matching the
    reference pipelines, which the reference-equivalence suite pins bit-exactly.
    """
    shape = (1, 128, 16, 16)
    stager = NoiseStager(dtype)
    staged = stager.randn_to_device(shape, seed=99, device=torch.device("cuda"))
    torch.cuda.synchronize()
    inline = torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(99), dtype=dtype)
    assert torch.equal(staged.cpu(), inline), "out= draw diverged from the inline draw"


@cuda_only
def test_randn_to_device_equals_todays_pack_then_copy():
    """The whole point, end to end: for layout-only post-processing, packing after
    the staged copy gives exactly what packing before a pageable copy gave."""
    from mstar.model.components.diffusion.image_io import pack_latents

    shape, seed, dtype = (1, 128, 16, 16), 4242, torch.bfloat16
    device = torch.device("cuda")
    today = pack_latents(
        torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(seed), dtype=dtype)
    )[0].to(device)
    staged = pack_latents(NoiseStager(dtype).randn_to_device(shape, seed, device))[0]
    torch.cuda.synchronize()
    assert staged.shape == today.shape
    assert torch.equal(staged, today), "packing on the device changed the seeded noise"


@cuda_only
def test_randn_to_device_respects_the_ring():
    stager = NoiseStager(torch.float32, depth=2)
    device = torch.device("cuda")
    drawn = [stager.randn_to_device((1024,), seed=i, device=device) for i in range(8)]
    torch.cuda.synchronize()
    for i, got in enumerate(drawn):
        want = torch.randn((1024,), generator=torch.Generator(device="cpu").manual_seed(i))
        assert torch.equal(got.cpu(), want), f"draw {i} saw a rewritten staging buffer"


@cuda_only
def test_stage_all_covers_a_seed_set():
    stager = NoiseStager(torch.bfloat16)
    device = torch.device("cuda")
    seeds = {"latents": torch.randn(64, 8).to(torch.bfloat16),
             "solver": torch.randn(4, 8).to(torch.bfloat16)}
    out = stager.stage_all(seeds, device)
    torch.cuda.synchronize()
    assert set(out) == set(seeds)
    for name, tensor in seeds.items():
        assert torch.equal(out[name].cpu(), tensor)
