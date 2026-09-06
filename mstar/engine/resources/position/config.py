"""What a model declares about positions: its scheme, its spec, its step.

Kept free of the manager and its kernels so a submodule can declare a step
without pulling FlashInfer in behind it.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import torch

from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


class PosBackend(Enum):
    ROPE = "rope"


class PosScheme(Enum):
    SEQUENTIAL = "sequential"
    BLOCK = "block"


@dataclass
class PositionConfig:
    # TODO: make this work without an upstream KV cache (e.g., always using
    # custom ids / advance)
    kv_cache: str  # name of upstream KV cache
    backend: PosBackend = PosBackend.ROPE
    scheme: PosScheme = PosScheme.SEQUENTIAL
    block_step: int = 1  # relevant to BLOCK `PosScheme` only

    # rope params; all overridable
    rotary_dim: int | None = None
    interleave: bool = False
    rope_scale: float = 1.0
    rope_theta: float = 10000.0
    rope_dtype: torch.dtype | None = None

    low_freq_factor: float | None = None
    high_freq_factor: float | None = None
    old_context_len: int | None = None

    @property
    def llama31_params(self) -> dict[str, float]:
        params = dict(
            low_freq_factor=self.low_freq_factor,
            high_freq_factor=self.high_freq_factor,
            old_context_len=self.old_context_len,
        )
        if any(value is None for value in params.values()):
            return {}
        return params


@dataclass
class PositionSpec(NodeResourceSpec):
    config: PositionConfig

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.position.manager import PositionManager

        return PositionManager


@dataclass(frozen=True)
class PositionStep(ResourceStep):
    # `pos_ids=None` derives from stream counters; otherwise
    # label -> ids for one step
    pos_ids: "dict[str, torch.Tensor] | torch.Tensor | None" = None
    advance: tuple[int, ...] | None = None  # `advance=None` means own rule
    # no combined_labels: positions take the packing off KV's plan output,
    # so the step declares the grouping once, on KVStep
