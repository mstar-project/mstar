"""Model-agnostic quantization backends for mstar."""
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
