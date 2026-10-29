"""copy_into: one memcpy below the threshold, slices on its own pool above it,
the same bytes either way."""

from __future__ import annotations

import numpy as np
import pytest

from mstar.utils import host_copy


@pytest.fixture
def parallel(monkeypatch):
    """Every copy takes a fresh four-thread pool, whatever its size."""
    monkeypatch.setattr(host_copy, "PARALLEL_COPY_BYTES", 0)
    monkeypatch.setattr(host_copy, "usable_cpus", lambda: 4)
    monkeypatch.setattr(host_copy, "_pool", None)
    monkeypatch.setattr(host_copy, "_workers", 0)
    yield
    if host_copy._pool is not None:
        host_copy._pool.shutdown()


def test_a_small_copy_stays_on_the_calling_thread(monkeypatch):
    def no_pool():
        raise AssertionError("a small copy must not use the pool")
    monkeypatch.setattr(host_copy, "_copy_pool", no_pool)
    src = np.arange(1 << 16, dtype=np.uint8)
    assert np.array_equal(host_copy.copy_into(np.empty_like(src), src), src)


@pytest.mark.parametrize("shape", [(1 << 20,), (3, 121, 4, 6), (5, 7), (1, 64)])
def test_slices_copy_every_byte(parallel, shape):
    src = np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)
    dst = host_copy.copy_into(np.empty_like(src), src)
    assert np.array_equal(dst, src)


def test_a_strided_source(parallel):
    """Cosmos3's [C, T, H, W] decode written out as [T, C, H, W] frames."""
    video = np.random.default_rng(1).integers(0, 256, (3, 121, 6, 8), dtype=np.uint8)
    frames = host_copy.copy_into(np.empty((121, 3, 6, 8), np.uint8), video.transpose(1, 0, 2, 3))
    assert np.array_equal(frames, np.ascontiguousarray(video.transpose(1, 0, 2, 3)))


def test_the_slices_run_on_the_pool(parallel):
    host_copy.copy_into(np.empty(1 << 20, np.uint8), np.ones(1 << 20, np.uint8))
    pool, workers = host_copy._copy_pool()
    assert workers == 4
    assert pool._threads and {t.name.split("_")[0] for t in pool._threads} == {"mstar-copy"}


def test_the_pool_is_capped(monkeypatch, parallel):
    monkeypatch.setattr(host_copy, "usable_cpus", lambda: 40)
    assert host_copy._copy_pool()[1] == host_copy.MAX_COPY_WORKERS


def test_a_shape_mismatch_raises():
    with pytest.raises(ValueError):
        host_copy.copy_into(np.empty(4, np.uint8), np.empty(5, np.uint8))
