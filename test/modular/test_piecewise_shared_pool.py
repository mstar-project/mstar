"""Piecewise regions of one node capture into one shared graph memory pool."""

import inspect

import pytest
import torch

from mstar.engine.cuda_graph_runner import PiecewiseCudaGraphRunner


def test_runner_accepts_a_shared_pool_and_keeps_it():
    assert "memory_pool" in inspect.signature(PiecewiseCudaGraphRunner.__init__).parameters
    runner = object.__new__(PiecewiseCudaGraphRunner)
    runner._memory_pool = None
    # a runner built without a pool allocates its own at capture time (CUDA only)
    src = inspect.getsource(PiecewiseCudaGraphRunner.warmup_and_capture)
    assert "if self._memory_pool is None" in src


@pytest.mark.skipif(not torch.cuda.is_available(), reason="graph pools need CUDA")
def test_two_runners_share_the_pool_handle():
    from unittest import mock

    pool = torch.cuda.graphs.graph_pool_handle()
    make = lambda: PiecewiseCudaGraphRunner(  # noqa: E731
        label="r", config=mock.Mock(capture_batch_sizes=[1], declare_step=None), resources={},
        step_runner=mock.Mock(), device=torch.device("cuda"), autocast_dtype=None, num_slots=1, memory_pool=pool,
    )
    a, b = make(), make()
    assert a._memory_pool is pool and b._memory_pool is pool
