"""What a model declares about a rolling KV window.

A request's keys and values are a stream it appends to step by step. The window
keeps the stream's first ``sink`` tokens and its last ``window`` (the step's own
included), in stream order and compacted: once a step would overflow, the
retained window shifts left behind the sink. A step's retained tokens and its
own are therefore always one contiguous range, which the model hands to
whatever attention it runs (a ragged kernel, say). The window runs no attention.

It owns no memory either: each request's K/V live in a buffer of its region in
a request pool (``RollingKVConfig.pool``), named after this resource's key and
shaped ``[layers, 2, sink + window, heads, head_dim]``, which the model declares
with the request's other buffers (``RollingKVConfig.buffer``). The window is the
rules over that buffer, which it reads in its own plan (``RequestPool.buffer``)
rather than through the pool's plan: a region can hold many buffers, and the
window needs one.

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
    # the request pool whose regions hold each request's K/V
    pool: str
    dtype: torch.dtype = torch.bfloat16

    def buffer(self, tokens: int) -> tuple[tuple[int, ...], torch.dtype]:
        """The pool buffer a request keeping ``tokens`` a layer (its sink plus
        window) declares, under this resource's key."""
        return (self.num_layers, 2, tokens, self.num_heads, self.head_dim), self.dtype


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

    def depends_on(self) -> set[str]:
        return {self.config.pool}
