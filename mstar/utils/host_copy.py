"""Large host copies split over threads of their own.

The API server sizes torch's intra-op pool for its small per-request ops
(``entrypoint.default_preprocess_threads``), and one numpy memcpy runs on one
core. Most of a large copy into a fresh allocation is the destination's
first-touch page faults, which a copy split into slices takes in parallel too.
numpy releases the GIL while it copies, so slices on a pool of their own run in
parallel without resizing the intra-op pool.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

# The smallest slice worth a pool thread: below a few MB the handoff costs
# about what the slice saves, so a 3 MB image stays one memcpy and a copy gets
# one more slice per MIN_SLICE_BYTES, up to the copier's width.
MIN_SLICE_BYTES = 4 << 20


class HostCopier:
    """Copies flat uint8 host arrays into fresh torch allocations, a large one
    in slices on ``workers`` threads. The caller views and reshapes the bytes.

    The destination is torch's, never ``np.empty``: numpy madvises large arrays
    for huge pages, and under THP ``defrag=madvise`` their first-touch faults
    can stall in compaction."""

    def __init__(self, workers: int, min_slice_bytes: int | None = None):
        self.workers = max(1, workers)
        self.min_slice_bytes = MIN_SLICE_BYTES if min_slice_bytes is None else max(1, min_slice_bytes)
        self._pool = (
            ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="mstar-copy")
            if self.workers > 1 else None
        )

    def slices(self, nbytes: int) -> int:
        if self._pool is None:
            return 1
        return max(1, min(self.workers, nbytes // self.min_slice_bytes))

    def copy(self, src: np.ndarray) -> torch.Tensor:
        """A flat uint8 torch copy of the flat uint8 array ``src``."""
        if src.ndim != 1 or src.dtype != np.uint8:
            raise TypeError(f"HostCopier.copy takes a flat uint8 array, not {src.dtype} {src.shape}")
        dst = torch.empty(src.nbytes, dtype=torch.uint8)
        out = dst.numpy()
        if src.nbytes < 2 * self.min_slice_bytes or (parts := self.slices(src.nbytes)) == 1:
            np.copyto(out, src)
            return dst
        step = -(-src.nbytes // parts)
        futures = [
            self._pool.submit(np.copyto, out[i:i + step], src[i:i + step])
            for i in range(0, src.nbytes, step)
        ]
        for future in futures:
            future.result()
        return dst
