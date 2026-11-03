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
    """Keep the stream's first ``sink`` and last ``window`` entries. With
    ``window_includes_step`` the window counts the step's own tokens, so a step
    that writes ``span`` attends only the last ``window - span`` retained ones."""

    sink: int
    window: int
    window_includes_step: bool = False


@dataclass
class BoundedKVConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    # key/value rows a request owns per layer, e.g. 2 for batched CFG
    rows_per_request: int
    # the longest source prefix a request may have; sizes the slots
    max_source_len: int
    # source length -> what that request keeps, or one policy for every length
    retention: Callable[[int], SinkWindow] | SinkWindow
    max_slots: int = 32
    # the shortest source prefix a request may have: the slots also hold the part
    # of a sink its source doesn't cover, which is largest for the shortest source
    min_source_len: int = 0
    # None is the engine's KV dtype (``EngineResourceInfo.kv_dtype``), settled when
    # the resource is built
    dtype: torch.dtype | None = torch.float32
    # A step's tokens join the stream last-first. Only for attention that
    # ignores key order, where it decides which of the oldest step's tokens are
    # evicted first: Step-Audio2's flow decoder keeps the front of each chunk.
    reverse_step_order: bool = False
    # None: the window is per step, every query of a step attending the same keys
    # (the retained ones and all of the step's own). Set, it is per query: each
    # query attends the keys at distances [0, query_window] from it, so a step of
    # several tokens is causal within itself (a streaming conformer's
    # ``att_context_size=[left, 0]``). Needs no sink and the stream in order.
    query_window: int | None = None

    def __post_init__(self):
        if self.max_slots < 1:
            raise ValueError(f"max_slots must be >= 1, got {self.max_slots}")
        if not 0 <= self.min_source_len <= self.max_source_len:
            raise ValueError(f"source lengths {self.min_source_len}..{self.max_source_len} are not a range")
        q_window = self.query_window
        if q_window is not None and (q_window < 0 or self.reverse_step_order):
            raise ValueError(f"query_window={q_window} needs to be >= 0, with the stream in order")
        r = self.retention
        # resolved once here: a plan looks one up for every row
        self._policies = tuple(r if isinstance(r, SinkWindow) else r(n)
                               for n in range(self.min_source_len, self.max_source_len + 1))
        sink = window = retained = 0
        for n, policy in enumerate(self._policies, self.min_source_len):
            if policy.sink < 0 or policy.window < 0:
                raise ValueError(f"retention for {n} source tokens is {policy}")
            if q_window is not None and (policy.sink or policy.window_includes_step or policy.window < q_window):
                raise ValueError(f"retention for {n} source tokens is {policy}: a per-query window of "
                                 f"{q_window} takes no sink and a window of at least {q_window}")
            sink = max(sink, policy.sink - n)
            window = max(window, policy.window)
            retained = max(retained, policy.sink + policy.window)
        self._sink_capacity, self._window_capacity, self._max_retained = sink, window, retained

    def retention_for(self, source_len: int) -> SinkWindow:
        return self._policies[source_len - self.min_source_len]

    @property
    def sink_capacity(self) -> int:
        """Written tokens a slot's sink region holds (the sink past the source)."""
        return self._sink_capacity

    @property
    def window_capacity(self) -> int:
        return self._window_capacity

    @property
    def max_retained(self) -> int:
        """The most entries a request retains, over every allowed source length."""
        return self._max_retained

    @property
    def slot_tokens(self) -> int:
        return self.sink_capacity + self.window_capacity


@dataclass
class BoundedKVSpec(NodeResourceSpec):
    config: BoundedKVConfig

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.kv.bounded.manager import BoundedKVManager

        return BoundedKVManager

    def apply_yaml_overrides(self, max_slots: int | None = None):
        if max_slots is not None:
            if max_slots < 1:
                raise ValueError(f"max_slots must be >= 1, got {max_slots}")
            self.config.max_slots = max_slots


class StreamPosition(NamedTuple):
    """Where a request's stream stands when a step starts."""

    source_len: int
    # tokens the request has written before this step
    written: int
    # which of the sources ``attend`` is given this request's starts with; -1 for its
    # own entry of the ``row_source`` it is given
    source: int = 0


@dataclass(frozen=True)
class BoundedKVStep(ResourceStep):
    """Each segment's ``span`` is what its request writes this step (0 reads only).

    The positions come from the model, not the resource, so a plan never waits
    on the previous step's commit and planning the same step twice (a piecewise
    region inside a forward that declared it too) is harmless.
    """

    positions: Mapping[str, StreamPosition] = field(default_factory=dict)
