"""Two requests registered in the same step must each get their own sampling row.

``Buffer.write_master_row`` used to stage the value in one pinned ``[1]`` row
per Buffer and queue a non_blocking H2D from it. The copy engine reads that row
only when the stream reaches the copy -- behind whatever the GPU is still
running (step N-1 under the worker's 2-step launch bound) -- while the host has
long moved on to the next request's ``write_master_row`` on the same row. Rid A
then landed rid B's temperature / top_k / top_p / seed / penalty. With ~20 ms of
kernels queued ahead this failed 20/20 on an H100; a scalar ``fill_`` bakes the
value into the launch and has no host row to reuse.

GPU-only: the hazard needs a real asynchronous stream.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.engine.resources.sampler.utils import Buffer

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs an asynchronous CUDA stream"
)


def test_back_to_back_master_row_writes_keep_their_values():
    dev = torch.device("cuda")
    buf = Buffer.allocate(
        max_bs=4, capacity=8, device=dev, dtype=torch.float32, default=-1.0,
    )
    a = torch.randn(4096, 4096, device=dev)
    b = torch.empty_like(a)

    bad = 0
    for _ in range(20):
        buf.master.fill_(-1.0)
        torch.cuda.synchronize()
        for _ in range(10):  # ~20 ms of GPU work ahead of the writes: step N-1
            torch.mm(a, a, out=b)
        buf.write_master_row(0, 1.0)  # rid A registered
        buf.write_master_row(1, 2.0)  # rid B, a few microseconds later
        torch.cuda.synchronize()
        if buf.master[0].item() != 1.0 or buf.master[1].item() != 2.0:
            bad += 1
    assert bad == 0, f"{bad}/20 trials landed the second value in the first slot"
