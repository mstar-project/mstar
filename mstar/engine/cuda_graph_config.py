"""Compatibility aliases for the accelerator graph configuration API."""

from mstar.engine.accelerator_graph_config import (
    AcceleratorGraphConfig as CudaGraphConfig,
    AcceleratorGraphConfigType as CudaGraphConfigType,
    BatchedAcceleratorGraphConfig as BasicBatchedCudaGraphConfig,
    BatchedAcceleratorGraphConfig as BatchedCudaGraphConfig,
    PackedAcceleratorGraphConfig as FlashInferPackedCudaGraphConfig,
    PackedAcceleratorGraphConfig as PackedCudaGraphConfig,
    PiecewiseAcceleratorGraphConfig as PiecewiseCudaGraphConfig,
    PiecewiseBatchedConfig,
    PiecewiseCallInputs,
    PiecewiseCaptureShape,
    PiecewiseConfigType,
    PiecewisePackedConfig,
    distribute_tokens,
)

__all__ = [
    "BasicBatchedCudaGraphConfig",
    "BatchedCudaGraphConfig",
    "CudaGraphConfig",
    "CudaGraphConfigType",
    "FlashInferPackedCudaGraphConfig",
    "PackedCudaGraphConfig",
    "PiecewiseBatchedConfig",
    "PiecewiseCallInputs",
    "PiecewiseCaptureShape",
    "PiecewiseConfigType",
    "PiecewiseCudaGraphConfig",
    "PiecewisePackedConfig",
    "distribute_tokens",
]
