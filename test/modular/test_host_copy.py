"""HostCopier: one memcpy for a small copy, one more slice per MIN_SLICE_BYTES
up to its width for a large one, into a torch allocation it makes itself, with
the same bytes either way."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mstar.api_server.data_worker import copy_workers, encode_workers
from mstar.utils.host_copy import MIN_SLICE_BYTES, HostCopier

MB = 1 << 20


@pytest.mark.parametrize(
    ("nbytes", "slices"),
    [(MIN_SLICE_BYTES - 1, 1), (3 * MB, 1), (8 * MB, 2), (16 * MB, 4), (40 * MB, 8), (145 * MB, 8)],
)
def test_the_slice_count_grows_with_the_copy(nbytes, slices):
    assert HostCopier(8).slices(np.empty(nbytes, np.uint8)) == slices


def test_a_slice_is_at_least_a_row():
    assert HostCopier(8, min_slice_bytes=1).slices(np.empty((3, 1024), np.uint8)) == 3


def test_a_small_copy_stays_on_the_calling_thread():
    copier = HostCopier(4)

    def no_pool(*args, **kwargs):
        raise AssertionError("a small copy must not use the pool")

    copier._pool.submit = no_pool
    src = np.arange(1 << 16, dtype=np.uint8)
    assert torch.equal(copier.copy(src), torch.from_numpy(src))


def test_one_worker_has_no_pool():
    copier = HostCopier(1, min_slice_bytes=1)
    assert copier._pool is None and copier.slices(np.empty(1 << 20, np.uint8)) == 1


@pytest.mark.parametrize("shape", [(1 << 20,), (3, 121, 4, 6), (5, 7), (1, 64), ()])
@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.bool_])
def test_slices_copy_every_byte(shape, dtype):
    src = np.random.default_rng(0).integers(0, 2 if dtype is np.bool_ else 256, shape).astype(dtype)
    dst = HostCopier(4, min_slice_bytes=1).copy(src)
    assert isinstance(dst, torch.Tensor) and dst.is_contiguous()
    assert dst.numpy().dtype == src.dtype and np.array_equal(dst.numpy(), src)
    assert not np.shares_memory(dst.numpy(), src)


def test_a_strided_source():
    """A [C, T, H, W] clip read out as [T, C, H, W] frames."""
    video = np.random.default_rng(1).integers(0, 256, (3, 121, 6, 8), dtype=np.uint8)
    frames = HostCopier(4, min_slice_bytes=1).copy(video.transpose(1, 0, 2, 3))
    assert frames.shape == (121, 3, 6, 8) and frames.is_contiguous()
    assert np.array_equal(frames.numpy(), np.ascontiguousarray(video.transpose(1, 0, 2, 3)))


def test_the_slices_run_on_the_pool():
    copier = HostCopier(4, min_slice_bytes=1)
    copier.copy(np.ones(1 << 20, np.uint8))
    assert copier._pool._threads and {t.name.split("_")[0] for t in copier._pool._threads} == {"mstar-copy"}


@pytest.mark.parametrize("cpus", [1, 4, 8, 12, 16, 32, 41, 64, 256])
def test_the_copy_width_is_one_encodes_share_of_the_cpus(cpus):
    """The data worker's copier is as wide as one of the encode pool's slots:
    the pool times the width stays within the CPUs."""
    workers = copy_workers(cpus, png_workers=min(8, cpus))
    assert workers >= min(8, cpus)
    assert workers * encode_workers(cpus, png_workers=min(8, cpus)) <= cpus
