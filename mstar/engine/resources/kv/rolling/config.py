"""What a model declares about a rolling KV store.

A request's keys and values are a stream it appends to step by step. The store
keeps the stream's first ``sink`` tokens and its last ``window`` (the step's own
included), in stream order and compacted: once a step would overflow, the
retained window shifts left behind the sink. A step's retained tokens and its
own are therefore always one contiguous range, which the model hands to
whatever attention it runs (a ragged kernel, say). The store runs no attention.

The K/V live in one pool reserved at startup (``pool_tokens`` a layer). Each
request takes a contiguous region of exactly its ``sink + window`` there, so
requests of different sizes share the pool without each holding the largest.

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
    # tokens a layer the pool holds, shared by every request's sink + window
    pool_tokens: int
    # None takes the engine's KV dtype (``EngineResourceInfo.kv_dtype``)
    dtype: torch.dtype | None = None

    def __post_init__(self):
        if self.pool_tokens < 1:
            raise ValueError(f"pool_tokens must be >= 1, got {self.pool_tokens}")

    def token_bytes(self, tokens: int, dtype: torch.dtype) -> int:
        """``tokens`` a layer, keys and values of every layer."""
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

    def apply_yaml_overrides(self, pool_tokens: int | None = None):
        """The pool's size is the deployment's memory trade: what it reserves at
        startup against how many requests, and how large, it holds at once."""
        if pool_tokens is not None:
            self.config.pool_tokens = pool_tokens
        self.config.__post_init__()
