"""HostCopier: one memcpy for a small copy, one more slice per MIN_SLICE_BYTES
up to its width for a large one, into a torch allocation it makes itself, with
the same bytes either way."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mstar.utils.cpus import CpuPlan
from mstar.utils.host_copy import MIN_SLICE_BYTES, HostCopier

MB = 1 << 20


@pytest.mark.parametrize(
    ("nbytes", "slices"),
    [(MIN_SLICE_BYTES - 1, 1), (3 * MB, 1), (8 * MB, 2), (16 * MB, 4), (40 * MB, 8), (145 * MB, 8)],
)
def test_the_slice_count_grows_with_the_copy(nbytes, slices):
    assert HostCopier(8).slices(nbytes) == slices


def test_a_small_copy_stays_on_the_calling_thread():
    copier = HostCopier(4)

    def no_pool(*args, **kwargs):
        raise AssertionError("a small copy must not use the pool")

    copier._pool.submit = no_pool
    src = np.arange(1 << 16, dtype=np.uint8)
    assert torch.equal(copier.copy(src), torch.from_numpy(src))


def test_one_worker_has_no_pool():
    copier = HostCopier(1, min_slice_bytes=1)
    assert copier._pool is None and copier.slices(1 << 20) == 1


@pytest.mark.parametrize("nbytes", [0, 1, 1 << 20, 3 * MB + 7])
def test_slices_copy_every_byte(nbytes):
    src = np.random.default_rng(0).integers(0, 256, nbytes, dtype=np.uint8)
    dst = HostCopier(4, min_slice_bytes=1).copy(src)
    assert isinstance(dst, torch.Tensor) and dst.dtype == torch.uint8 and dst.shape == (nbytes,)
    assert np.array_equal(dst.numpy(), src)
    assert nbytes == 0 or not np.shares_memory(dst.numpy(), src)


def test_the_bytes_view_as_the_callers_tensor():
    """The arena read copies the bytes and views them as the output tensor."""
    video = np.random.default_rng(1).standard_normal((3, 4, 6, 8), dtype=np.float32)
    flat = HostCopier(4, min_slice_bytes=1).copy(video.reshape(-1).view(np.uint8))
    assert torch.equal(flat.view(torch.float32).reshape(3, 4, 6, 8), torch.from_numpy(video))


@pytest.mark.parametrize("src", [np.zeros(4, np.float32), np.zeros((2, 2), np.uint8)])
def test_only_flat_bytes_are_copied(src):
    with pytest.raises(TypeError):
        HostCopier(1).copy(src)


def test_the_slices_run_on_the_pool():
    copier = HostCopier(4, min_slice_bytes=1)
    copier.copy(np.ones(1 << 20, np.uint8))
    assert copier._pool._threads and {t.name.split("_")[0] for t in copier._pool._threads} == {"mstar-copy"}


@pytest.mark.parametrize(
    ("cpus", "width"),
    [(1, 1), (4, 4), (8, 8), (12, 12), (16, 8), (32, 8), (41, 10), (64, 16), (256, 64)],
)
def test_the_copy_width_is_one_encodes_share_of_the_cpus(cpus, width):
    """The data worker's copier is as wide as one of the encode pool's slots:
    the pool times the width stays within the CPUs."""
    plan = CpuPlan(frozenset(range(cpus)), cpus)
    assert plan.copy_workers == width
    assert width * plan.encode_workers <= cpus
