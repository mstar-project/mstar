"""Large host copies split over a few threads of their own.

The API server sizes torch's intra-op pool for its small per-request ops
(``entrypoint.default_preprocess_threads``: two threads on an 8-core pin), and
one numpy memcpy runs on one core. A video output (121 frames at 832x480 is
145 MB) copied either way takes two to three times as long as it does over
the CPUs the process may use. numpy releases the GIL while it copies, so
slices on a small pool run in parallel without resizing the intra-op pool.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from mstar.utils.cpus import usable_cpus

# Smaller copies stay one memcpy. Slicing pays off somewhere past a few MB:
# a 3 MB image copies in ~0.7 ms on one core and ~1.7 ms in 8 slices, 40 MB
# in ~16 ms and ~10 ms (B300 host, Xeon 6740E).
PARALLEL_COPY_BYTES = 16 << 20
MAX_COPY_WORKERS = 8

_pool: ThreadPoolExecutor | None = None
_workers = 0
_pool_lock = threading.Lock()


def _copy_pool() -> tuple[ThreadPoolExecutor, int]:
    global _pool, _workers
    with _pool_lock:
        if _pool is None:
            _workers = max(1, min(MAX_COPY_WORKERS, usable_cpus()))
            _pool = ThreadPoolExecutor(max_workers=_workers, thread_name_prefix="mstar-copy")
        return _pool, _workers


def copy_into(dst: np.ndarray, src: np.ndarray) -> np.ndarray:
    """``dst[...] = src``, in slices along the first axis on the copy pool once
    ``dst`` is at least ``PARALLEL_COPY_BYTES``. Returns ``dst``.

    Give it a destination torch allocated (``torch.empty(...).numpy()``), not
    ``np.empty``: numpy madvises large arrays for huge pages, and under THP
    ``defrag=madvise`` their first-touch faults can stall in compaction."""
    if dst.shape != src.shape:
        raise ValueError(f"copy_into: shape {src.shape} into {dst.shape}")
    parts = 1
    if dst.nbytes >= PARALLEL_COPY_BYTES and dst.ndim:
        pool, workers = _copy_pool()
        parts = min(workers, dst.shape[0])
    if parts < 2:
        np.copyto(dst, src)
        return dst
    step = -(-dst.shape[0] // parts)
    futures = [
        pool.submit(np.copyto, dst[i:i + step], src[i:i + step])
        for i in range(0, dst.shape[0], step)
    ]
    for future in futures:
        future.result()
    return dst
