"""Kept ids wait in host memory a worker sets aside once at startup.

A rebuild reads a request's prompt ids long after the step that read them, so the worker copies each prompt off
the device once. The pool is one pinned allocation handed out first fit, and a copy runs on the pool's own
stream after the step that wrote its ids completes, so the main thread never waits on it.
"""

from __future__ import annotations

import pytest
import torch

from mstar.worker.host_pool import HostPool, HostSlice

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="copies from a CUDA stream")


def test_a_slice_given_back_merges_with_its_free_neighbors():
    pool = HostPool(64, torch.device("cpu"))
    first, middle, last = pool.take(8), pool.take(8), pool.take(8)

    pool.give(first)
    pool.give(last)
    pool.give(middle)

    assert pool.take(64) == HostSlice(0, 64), "the pool stayed in pieces, so a long prompt would find no room"


@requires_cuda
def test_kept_ids_copy_after_the_steps_completion_event():
    device = torch.device("cuda")
    pool = HostPool(1 << 11, device)
    ids = torch.zeros(256, dtype=torch.long, device=device)
    torch.cuda._sleep(100_000_000)  # the step is still running when its ids are copied
    ids.fill_(7)
    step_done = torch.Event(device=device)
    step_done.record()

    copy = pool.copy_out(ids, step_done)
    copy.landed.synchronize()

    assert copy.view.tolist() == [7] * 256, (
        "the copy read the ids before the step wrote them"
    )
