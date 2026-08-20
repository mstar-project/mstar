"""Compatibility aliases for the accelerator graph runner API."""

from mstar.engine.accelerator_graph_runner import (
    AcceleratorGraphBucket as CudaGraphBucket,
    AcceleratorGraphRunner as CudaGraphRunner,
    AcceleratorGraphSlot as CudaGraphSlot,
    DummyRowPool,
    PiecewiseAcceleratorGraphRunner as PiecewiseCudaGraphRunner,
    PiecewiseGraphData,
    PiecewiseGraphKey,
    PiecewiseOutput,
    agree_across_ranks,
    autocast_scope,
)

__all__ = [
    "CudaGraphBucket",
    "CudaGraphRunner",
    "CudaGraphSlot",
    "DummyRowPool",
    "PiecewiseCudaGraphRunner",
    "PiecewiseGraphData",
    "PiecewiseGraphKey",
    "PiecewiseOutput",
    "agree_across_ranks",
    "autocast_scope",
]
