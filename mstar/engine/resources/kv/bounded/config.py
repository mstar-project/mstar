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
    that writes ``span`` attends only the last ``window - span`` retained ones.

    By default the window is per step: every query of a step attends the same
    keys, the retained ones and all of the step's own. With ``right`` set it is
    per query: each query attends the keys at distances ``[-right, window]``
    from it, so a step of several tokens is causal within itself and each of
    them sees exactly ``window`` back (a streaming conformer's
    ``att_context_size``). Only ``right=0``, with no sink, is supported so far."""

    sink: int
    window: int
    window_includes_step: bool = False
    right: int | None = None


@dataclass
class BoundedKVConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    # key/value rows a request owns per layer, e.g. 2 for batched CFG
    rows_per_request: int
    # the longest source prefix a request may have; sizes the slots
    max_source_len: int
    # source length -> what that request keeps
    retention: Callable[[int], SinkWindow]
    max_slots: int = 32
    # the shortest source prefix a request may have: the slots also hold the part
    # of a sink its source doesn't cover, which is largest for the shortest source
    min_source_len: int = 0
    dtype: torch.dtype = torch.float32
    # A step's tokens join the stream last-first. Only for attention that
    # ignores key order, where it decides which of the oldest step's tokens are
    # evicted first: Step-Audio2's flow decoder keeps the front of each chunk.
    reverse_step_order: bool = False

    def __post_init__(self):
        if self.max_slots < 1:
            raise ValueError(f"max_slots must be >= 1, got {self.max_slots}")
        if not 0 <= self.min_source_len <= self.max_source_len:
            raise ValueError(f"source lengths {self.min_source_len}..{self.max_source_len} are not a range")
        sink = window = retained = 0
        query_windows = set()
        for n in range(self.min_source_len, self.max_source_len + 1):
            policy = self.retention(n)
            if policy.sink < 0 or policy.window < 0:
                raise ValueError(f"retention for {n} source tokens is {policy}")
            if policy.right is not None and (
                policy.right != 0 or policy.sink or policy.window_includes_step or self.reverse_step_order
            ):
                raise ValueError(f"retention for {n} source tokens is {policy}: a per-query window "
                                 "takes right=0, no sink, and the stream in order")
            query_windows.add(None if policy.right is None else policy.window)
            sink = max(sink, policy.sink - n)
            window = max(window, policy.window)
            retained = max(retained, policy.sink + policy.window)
        # the kernel takes a per-query window as one scalar for the whole step
        if len(query_windows) > 1:
            raise ValueError(f"retention mixes windows {sorted(query_windows, key=str)}; a per-query "
                             "window must be the same for every source length")
        # fixed once here: a plan reads them for every row
        self._sink_capacity, self._window_capacity, self._max_retained = sink, window, retained
        self._query_window = next(iter(query_windows))

    @property
    def sink_capacity(self) -> int:
        """Written tokens a slot's sink region holds (the sink past the source)."""
        return self._sink_capacity

    @property
    def window_capacity(self) -> int:
        return self._window_capacity

    @property
    def query_window(self) -> int | None:
        """Each query's window when retention is per query (``SinkWindow.right``), else None."""
        return self._query_window

    @property
    def max_retained(self) -> int:
        """The most entries a request retains, over every allowed source length."""
        return self._max_retained

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
