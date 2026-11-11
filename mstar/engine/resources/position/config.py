"""What a model declares about positions: its scheme, its spec, its step.

Kept free of the manager and its kernels so a submodule can declare a step
without pulling FlashInfer in behind it.
"""

import math
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


@dataclass(frozen=True)
class YaRNConfig:
    """Static YaRN frequency interpolation and cos/sin amplitude scaling."""

    factor: float
    original_max_position_embeddings: int
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 1.0
    attention_factor: float | None = None
    truncate: bool = True

    def __post_init__(self):
        for name in ("factor", "beta_fast", "beta_slow"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"YaRN {name} must be finite and positive")
        if self.original_max_position_embeddings <= 0:
            raise ValueError("YaRN original_max_position_embeddings must be positive")
        if self.beta_fast < self.beta_slow:
            raise ValueError("YaRN beta_fast must be >= beta_slow")
        for name in ("mscale", "mscale_all_dim", "attention_factor"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"YaRN {name} must be finite and positive")


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
    # Rotate with a cos/sin table rounded to this dtype (math stays float32),
    # for checkpoints trained with a low-precision table (Fish S2-Pro:
    # bfloat16). None computes the angles in the kernel with its fast sin/cos,
    # about a bf16 ulp from a float32 table. CUDA needs a rotary_dim that is a
    # multiple of 16 for a table. XPU rotates from a table in rope_dtype or the
    # KV dtype and refuses any other value.
    rope_table_dtype: torch.dtype | None = None

    low_freq_factor: float | None = None
    high_freq_factor: float | None = None
    old_context_len: int | None = None

    yarn: YaRNConfig | None = None

    def __post_init__(self):
        if self.yarn is not None:
            if self.rope_scale != 1.0 or any(value is not None for value in (
                self.low_freq_factor, self.high_freq_factor, self.old_context_len,
            )):
                raise ValueError("YaRN cannot be combined with linear or Llama 3.1 scaling")
            if not math.isfinite(self.rope_theta) or self.rope_theta <= 1:
                raise ValueError("YaRN rope_theta must be finite and > 1")

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

    def depends_on(self) -> set[str]:
        return {self.config.kv_cache}

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
