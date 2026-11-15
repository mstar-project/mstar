"""What a model declares about a store of tensors each request sizes.

Some per-request tensors follow sizes the request sets when it opens: a VAE's
causal-conv caches follow its latent's height and width. A pool of one shape
would hold every request at the largest. This store reserves nothing at
startup: a request is admitted by the bytes its fields take at its own sizes,
against ``max_bytes``, and the store keeps the tensors its steps write until
the request ends.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig, check_sizes
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


@dataclass(frozen=True)
class SizedField:
    # an int axis is fixed, and a ``(name, scale)`` axis is ``scale`` times the request's size ``name``
    shape: tuple[int | tuple[str, int], ...]
    dtype: torch.dtype

    def shape_at(self, sizes: Mapping[str, int]) -> tuple[int, ...]:
        return tuple(axis if isinstance(axis, int) else axis[1] * sizes[axis[0]] for axis in self.shape)

    def bytes_at(self, sizes: Mapping[str, int]) -> int:
        return math.prod(self.shape_at(sizes)) * self.dtype.itemsize


@dataclass
class SizedValueConfig:
    num_layers: int
    fields: dict[str, SizedField]
    # the sizes of the request whose fields take the most bytes
    largest: dict[str, int]
    # what every request's tensors may take at once; None holds one request at ``largest``
    max_bytes: int | None = None

    def __post_init__(self):
        if self.max_bytes is not None:
            check_sizes({"max_bytes": self.max_bytes})

    def bytes_at(self, sizes: Mapping[str, int]) -> int:
        """One label's fields at ``sizes``, on every layer."""
        return self.num_layers * sum(f.bytes_at(sizes) for f in self.fields.values())

    @property
    def budget(self) -> int:
        return self.bytes_at(self.largest) if self.max_bytes is None else self.max_bytes


@dataclass
class SizedValueReqConfig(ResourceReqConfig):
    """Each label's sizes for one request, declared when it opens, so the
    engine's admission queue holds it until the store has its bytes."""

    sizes: dict[str, dict[str, int]]

    def bytes(self, config: SizedValueConfig) -> int:
        return sum(config.bytes_at(sizes) for sizes in self.sizes.values())

    def validate(self, spec: "NodeResourceSpec | None" = None) -> None:
        config = getattr(spec, "config", None)
        if isinstance(config, SizedValueConfig) and self.bytes(config) > config.budget:
            raise ValueError(
                f"the request's tensors take {self.bytes(config)} bytes at {self.sizes}; the store holds "
                f"{config.budget} (resources: {spec.resource_key}: max_bytes)"
            )


@dataclass(frozen=True)
class SizedValueStep(ResourceStep):
    """The rows whose tensors the step reads and writes."""


@dataclass
class SizedValueSpec(NodeResourceSpec):
    config: SizedValueConfig

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.sized.store import SizedValueStore

        return SizedValueStore

    @property
    def restores_prefix(self) -> bool:
        # a prefix hit skips the steps that write a request's tensors, and the store keeps no cache of them
        return True

    def request_config(self, model_kwargs: dict | None) -> SizedValueReqConfig | None:
        """The sizes a request's ``state_sizes`` gives this store; None for a request with none."""
        sizes = ((model_kwargs or {}).get("state_sizes") or {}).get(self.resource_key)
        return SizedValueReqConfig(sizes={label: dict(by_name) for label, by_name in sizes.items()}) if sizes else None

    def apply_yaml_overrides(self, max_bytes: int | None = None):
        """The budget is the deployment's memory trade: how many requests, and
        how large, hold their tensors at once."""
        if max_bytes is not None:
            self.config.max_bytes = max_bytes
        self.config.__post_init__()
