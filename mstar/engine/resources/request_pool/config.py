"""What a model declares about a request pool.

A request pool is storage only: one block of device memory reserved at
startup, from which each request takes one contiguous region for as long as it
runs. The region holds the named buffers the request declared when it opened
(a K/V stream, a decoder's causal-conv cache, ...), each a typed tensor the
model or a resource built on the pool reaches through the step's plan. Regions
are sized per request, so requests of different sizes share the pool instead
of each holding the largest.

The pool knows nothing of what a buffer means. A resource that needs rules
over a buffer (the rolling K/V window, say) depends on the pool and reads the
request's buffer directly in its own plan (``RequestPool.buffer``): a region
can hold many buffers, and such a resource needs one of them.
"""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig, check_sizes
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource

# Regions and the buffers in them start on this many bytes, so any dtype's view
# is aligned and a region keeps its alignment when the pool rearranges.
ALIGN = 256


def aligned(nbytes: int) -> int:
    return -(-nbytes // ALIGN) * ALIGN


@dataclass
class RequestPoolConfig:
    # bytes the pool reserves at startup, shared by every request's region
    pool_bytes: int

    def __post_init__(self):
        check_sizes({"pool_bytes": self.pool_bytes})


@dataclass
class RequestPoolReqConfig(ResourceReqConfig):
    """The buffers a request keeps in the pool, declared when it opens: it waits
    in line for a region that holds them all before its first step. Parallel
    lists, buffer by buffer, so the config crosses the wire typed."""

    names: list[str] = field(default_factory=list)
    shapes: list[list[int]] = field(default_factory=list)
    # torch dtype names, e.g. "bfloat16"
    dtypes: list[str] = field(default_factory=list)

    @classmethod
    def of(cls, buffers: "dict[str, tuple[tuple[int, ...], torch.dtype]]") -> "RequestPoolReqConfig":
        return cls(
            names=list(buffers),
            shapes=[list(shape) for shape, _ in buffers.values()],
            dtypes=[str(dtype).removeprefix("torch.") for _, dtype in buffers.values()],
        )

    def layout(self) -> "dict[str, tuple[int, tuple[int, ...], torch.dtype]]":
        """Each buffer's byte offset in the region, shape and dtype, in declared order."""
        out, offset = {}, 0
        for name, shape, dtype_name in zip(self.names, self.shapes, self.dtypes, strict=True):
            dtype = getattr(torch, dtype_name)
            out[name] = (offset, tuple(shape), dtype)
            offset += aligned(math.prod(shape) * dtype.itemsize)
        return out

    def nbytes(self) -> int:
        sizes = zip(self.shapes, self.dtypes, strict=True)
        return sum(aligned(math.prod(s) * getattr(torch, d).itemsize) for s, d in sizes)

    def validate(self, spec: "NodeResourceSpec | None" = None) -> None:
        if not self.names:
            raise ValueError("a request using the request pool declares at least one buffer")
        if not (len(self.names) == len(self.shapes) == len(self.dtypes)) or len(set(self.names)) != len(self.names):
            raise ValueError("request pool buffers need one shape and dtype each, under distinct names")
        for name, shape, dtype_name in zip(self.names, self.shapes, self.dtypes, strict=True):
            if not isinstance(getattr(torch, dtype_name, None), torch.dtype):
                raise ValueError(f"request pool buffer {name!r} has no dtype {dtype_name!r}")
            if not shape or any(not isinstance(n, int) or n < 1 for n in shape):
                raise ValueError(f"request pool buffer {name!r} has shape {shape}; every dim must be >= 1")
        config = getattr(spec, "config", None)
        if isinstance(config, RequestPoolConfig) and self.nbytes() > config.pool_bytes:
            raise ValueError(
                f"the request keeps {self.nbytes() / 2**30:.2f} GiB in the request pool; it holds "
                f"{config.pool_bytes / 2**30:.2f} GiB (resources: {spec.resource_key}: pool_bytes)"
            )


@dataclass(frozen=True)
class RequestPoolStep(ResourceStep):
    """The step's rows; each segment's request reads its region's buffers this step."""


@dataclass
class RequestPoolSpec(NodeResourceSpec):
    config: RequestPoolConfig

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.request_pool.manager import RequestPool

        return RequestPool

    def apply_yaml_overrides(self, pool_bytes: int | None = None):
        """The pool's size is the deployment's memory trade: what it reserves at
        startup against how many requests, and how large, it holds at once."""
        if pool_bytes is not None:
            self.config.pool_bytes = pool_bytes
        self.config.__post_init__()
