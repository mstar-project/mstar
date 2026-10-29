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

_TORCH_DTYPES: dict[np.dtype, torch.dtype] = {}


def _torch_dtype(dtype: np.dtype) -> torch.dtype:
    if dtype not in _TORCH_DTYPES:
        _TORCH_DTYPES[dtype] = torch.from_numpy(np.empty(0, dtype)).dtype
    return _TORCH_DTYPES[dtype]


class HostCopier:
    """Copies host arrays into fresh torch allocations, a large one in slices
    along its first axis on ``workers`` threads.

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

    def slices(self, src: np.ndarray) -> int:
        if self._pool is None or not src.ndim:
            return 1
        return max(1, min(self.workers, src.shape[0], src.nbytes // self.min_slice_bytes))

    def copy(self, src: np.ndarray) -> torch.Tensor:
        """A contiguous torch copy of ``src`` (which may be strided)."""
        # Allocated as bytes and viewed, which costs less than a typed,
        # shaped allocation on the many small reads.
        flat = torch.empty(src.nbytes, dtype=torch.uint8)
        if src.ndim == 1 and src.dtype == np.uint8:
            dst, out = flat, flat.numpy()
        else:
            dst = flat.view(_torch_dtype(src.dtype)).view(src.shape)
            out = flat.numpy().view(src.dtype).reshape(src.shape)
        if src.nbytes < 2 * self.min_slice_bytes or (parts := self.slices(src)) == 1:
            np.copyto(out, src)
            return dst
        step = -(-src.shape[0] // parts)
        futures = [
            self._pool.submit(np.copyto, out[i:i + step], src[i:i + step])
            for i in range(0, src.shape[0], step)
        ]
        for future in futures:
            future.result()
        return dst
