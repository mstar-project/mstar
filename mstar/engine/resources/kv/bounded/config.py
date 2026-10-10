"""What a model declares about a bounded KV cache.

A request's cache is a logical stream: an optional read-only source prefix
(a voice prompt's keys, say, shared by every request that uses it) followed by
the tokens the request writes. Retention keeps the stream's first ``sink``
entries and its last ``window`` (StreamingLLM's attention sinks plus a sliding
window), so a request's footprint is fixed and known when it starts: each slot
holds only the written part of the sink and a ring for the window, and source
entries are read where they live.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

import torch

from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


class SinkWindow(NamedTuple):
    """Keep the stream's first ``sink`` and last ``window`` entries."""

    sink: int
    window: int


@dataclass
class BoundedKVConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    # key/value rows a request owns per layer, e.g. 2 for batched CFG
    rows_per_request: int
    # the longest source prefix a request may have; sizes the slots
    max_source_len: int
    # source length -> what that request keeps; must not shrink as it grows
    retention: Callable[[int], SinkWindow]
    max_slots: int = 32
    dtype: torch.dtype = torch.float32
    # A step's tokens join the stream last-first. Only for attention that
    # ignores key order, where it decides which of the oldest step's tokens are
    # evicted first: Step-Audio2's flow decoder keeps the front of each chunk.
    reverse_step_order: bool = False

    def __post_init__(self):
        if self.max_slots < 1:
            raise ValueError(f"max_slots must be >= 1, got {self.max_slots}")
        policy = self.retention(self.max_source_len)
        if policy.sink < 0 or policy.window < 0:
            raise ValueError(f"retention for {self.max_source_len} source tokens is {policy}")

    @property
    def sink_capacity(self) -> int:
        """Written tokens a slot's sink region holds (the sink past the source)."""
        policy = self.retention(self.max_source_len)
        return max(0, policy.sink - self.max_source_len)

    @property
    def window_capacity(self) -> int:
        return self.retention(self.max_source_len).window

    @property
    def slot_tokens(self) -> int:
        return self.sink_capacity + self.window_capacity

    @property
    def slot_bytes(self) -> int:
        per_token = self.rows_per_request * self.num_heads * 2 * self.head_dim
        return self.num_layers * self.slot_tokens * per_token * torch.empty((), dtype=self.dtype).element_size()


@dataclass
class BoundedKVSpec(NodeResourceSpec):
    config: BoundedKVConfig

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.kv.bounded.manager import BoundedKVManager

        return BoundedKVManager

    def apply_yaml_overrides(self, max_slots: int | None = None):
        if max_slots is not None:
            self.config.max_slots = max_slots


class StreamPosition(NamedTuple):
    """Where a request's stream stands when a step starts."""

    source_len: int
    # tokens the request has written before this step
    written: int


@dataclass(frozen=True)
class BoundedKVStep(ResourceStep):
    """Each segment's ``span`` is what its request writes this step (0 reads only).

    The positions come from the model, not the resource, so a plan never waits
    on the previous step's commit and planning the same step twice (a piecewise
    region inside a forward that declared it too) is harmless.
    """

    positions: Mapping[str, StreamPosition] = field(default_factory=dict)
