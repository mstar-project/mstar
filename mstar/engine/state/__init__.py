"""The model-facing state contract. ``__all__`` is its whole surface, and after release the surface only grows."""

from mstar.engine.resources.kv.config import Span
from mstar.engine.resources.step import (
    Segment,
    SlotLease,
    SubmoduleStep,  # noqa: F401
)
from mstar.engine.state.contract import (
    Array,
    ArrayStep,
    Attend,
    Block,
    Causal,
    CausalConv,
    Entry,
    Fork,
    GatedDeltaNet,
    KVEntry,
    Layout,
    Positions,
    Reached,
    Read,
    Rope,
    Sampler,
    SampleSettings,
    SampleStep,
    TensorType,
    Value,
    Window,
)

# `SubmoduleStep` stays out of `__all__`, whose names can't be removed after release. It is the engine's own dataclass,
# and the rebuild after release replaces its `steps`, `cg_key_info` and `_ctx`.
__all__ = [
    "Array",
    "ArrayStep",
    "Attend",
    "Block",
    "Causal",
    "CausalConv",
    "Entry",
    "Fork",
    "GatedDeltaNet",
    "KVEntry",
    "Layout",
    "Positions",
    "Reached",
    "Read",
    "Rope",
    "SampleSettings",
    "SampleStep",
    "Sampler",
    "Segment",
    "SlotLease",
    "Span",
    "TensorType",
    "Value",
    "Window",
]
