"""The model-facing state contract. ``__all__`` is its whole surface, and after release the surface only grows."""

from mstar.engine.resources.kv.config import Span
from mstar.engine.resources.step import (
    Segment,
    SlotLease,
    SubmoduleStep,  # noqa: F401
)
from mstar.engine.state.calls import (
    add_read,
    at_last_token,
    causal_conv,
    gated_delta_net,
    host,
    kernel_setting,  # noqa: F401
    last_sampled,
    position,
    positions,
    rope,
    sample,
)
from mstar.engine.state.contract import (
    Array,
    ArrayStep,
    Attend,
    Bidirectional,
    Block,
    Causal,
    CausalConv,
    Entry,
    Fork,
    GatedDeltaNet,
    KVEntry,
    Layout,
    PagedNHD,
    PagedNHDTarget,
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
from mstar.engine.state.state_op import state_op

# `kernel_setting` and `SubmoduleStep` stay out of `__all__`, whose names can't be removed after release.
# `kernel_setting` is temporary. `SubmoduleStep` is the engine's own dataclass, and the rebuild after release replaces
# its `steps`, `cg_key_info` and `_ctx`.
__all__ = [
    "Array",
    "ArrayStep",
    "Attend",
    "Bidirectional",
    "Block",
    "Causal",
    "CausalConv",
    "Entry",
    "Fork",
    "GatedDeltaNet",
    "KVEntry",
    "Layout",
    "PagedNHD",
    "PagedNHDTarget",
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
    "add_read",
    "at_last_token",
    "causal_conv",
    "gated_delta_net",
    "host",
    "last_sampled",
    "position",
    "positions",
    "rope",
    "sample",
    "state_op",
]
