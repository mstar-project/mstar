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
    drew,  # noqa: F401
    fused_draw_args,  # noqa: F401
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
    FusedDrawArgs,  # noqa: F401
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

# `kernel_setting`, `SubmoduleStep` and the fused draw stay out of `__all__`, whose names can't be removed after
# release. `kernel_setting` is temporary. `SubmoduleStep` is the engine's own dataclass, and the rebuild after release
# replaces its `steps`, `cg_key_info` and `_ctx`. One model uses the fused draw: `FusedDrawArgs`, `fused_draw_args`,
# `drew`, `draw_row` and `draw_row_layout`. They're frozen together once a second model draws in place, or a state op
# on a sampler replaces `drew`.
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


def __getattr__(name: str):
    # imported on first use, so that importing the contract doesn't import Gluon
    if name in ("draw_row", "draw_row_layout"):
        from mstar.engine.resources.sampler import fused_draw

        return fused_draw.draw_row if name == "draw_row" else fused_draw.layout
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
