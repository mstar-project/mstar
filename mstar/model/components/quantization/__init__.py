"""Model-agnostic quantization backends for mstar.

Currently provides the Marlin W4A16 routed-expert (fused-MoE) path. The seam
(:class:`FusedMoEQuantizeMethod`) and the generic post-load pass
(:func:`process_weights_after_loading`) are model-agnostic; Kimi-K2.7 is the first
consumer. See :mod:`mstar.model.components.quantization.base`.
"""
from mstar.model.components.quantization.base import (
    FusedMoEQuantizeMethod,
    process_weights_after_loading,
)
from mstar.model.components.quantization.marlin_moe import MarlinMoEMethod

__all__ = [
    "FusedMoEQuantizeMethod",
    "MarlinMoEMethod",
    "process_weights_after_loading",
]
