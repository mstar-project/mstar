"""What a model declares about a rolling KV store.

A request's keys and values are a stream it appends to step by step. The store
keeps the stream's first ``sink`` tokens and its last ``window`` (the step's own
included), in stream order and compacted: once a step would overflow, the
retained window shifts left behind the sink. A step's retained tokens and its
own are therefore always one contiguous range, which the model hands to
whatever attention it runs (a ragged kernel, say). The store runs no attention.

Unlike bounded KV, whose window is a ring that its own kernel reads in pieces,
the range here is one slice, so the same kernel over the same keys gives the
same bits as a cache the model kept itself.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import torch

from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


@dataclass
class RollingKVConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    # the most tokens one request may keep per layer, its ``sink + window``; a
    # request's K/V are allocated at its own size, so this bounds, not reserves
    max_request_tokens: int
    # requests resident at once; a request past them waits for one to end
    max_slots: int = 1
    # None takes the engine's KV dtype (``EngineResourceInfo.kv_dtype``)
    dtype: torch.dtype | None = None

    def __post_init__(self):
        if self.max_slots < 1:
            raise ValueError(f"max_slots must be >= 1, got {self.max_slots}")
        if self.max_request_tokens < 1:
            raise ValueError(f"max_request_tokens must be >= 1, got {self.max_request_tokens}")

    def request_bytes(self, tokens: int, dtype: torch.dtype) -> int:
        """What a request keeping ``tokens`` per layer holds, keys and values of every layer."""
        return self.num_layers * 2 * tokens * self.num_heads * self.head_dim * dtype.itemsize


class RollingPosition(NamedTuple):
    """Where a request's stream stands when a step starts, and what it keeps."""

    # tokens the request committed before this step
    written: int
    # leading tokens never evicted
    sink: int
    # trailing tokens kept, this step's own included
    window: int


@dataclass(frozen=True)
class RollingKVStep(ResourceStep):
    """Each segment's ``span`` is the tokens its request writes this step.

    The positions come from the model, not the store, so a plan never waits on
    the previous step's commit; ``admit`` checks them against what was committed.
    """

    positions: dict = field(default_factory=dict)


@dataclass
class RollingKVSpec(NodeResourceSpec):
    config: RollingKVConfig

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.kv.rolling.manager import RollingKVManager

        return RollingKVManager

    def apply_yaml_overrides(self, max_slots: int | None = None, max_request_tokens: int | None = None):
        """How many requests stay resident, and how large a window each may keep,
        are the deployment's memory trade."""
        if max_slots is not None:
            self.config.max_slots = max_slots
        if max_request_tokens is not None:
            self.config.max_request_tokens = max_request_tokens
        self.config.__post_init__()
