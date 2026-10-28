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


def yarn_mscale(factor: float, mscale: float = 1.0) -> float:
    """YaRN's magnitude correction for a context stretched ``factor`` times: ``0.1 * mscale *
    ln(factor) + 1`` (1 when nothing is stretched). DeepSeek-V3-family models also square it into
    their softmax scale: ``qk_head_dim ** -0.5 * yarn_mscale(factor, mscale_all_dim) ** 2``."""
    return 1.0 if factor <= 1 else 0.1 * mscale * math.log(factor) + 1.0


@dataclass(frozen=True)
class YarnScaling:
    """YaRN RoPE (DeepSeek-V3's form; Kimi K2.x, DeepSeek-V4.1's compressed layers, YaRN drafters).

    Frequencies whose wavelength fits ``beta_fast`` turns into the training context
    (``original_max_position_embeddings``) are kept, those under ``beta_slow`` turns are divided
    by ``factor``, and a linear ramp crosses the band between. ``attention_factor`` multiplies the
    cos/sin table, which checkpoints set differently, so it is stated rather than defaulted: 1 for
    DeepSeek-V4.1 and for Kimi K2.x (``yarn_mscale(f, mscale) / yarn_mscale(f, mscale_all_dim)``
    with both 1), ``yarn_mscale(f)`` for a Hugging Face ``rope_type: yarn`` config that names
    neither. Needs ``rope_table_dtype``: the kernels rotate from the table.
    """

    factor: float
    original_max_position_embeddings: int
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    attention_factor: float = 1.0

    def inv_freq(self, rotary_dim: int, theta: float, device=None) -> torch.Tensor:
        """The ``rotary_dim // 2`` frequencies, in fp32."""
        exponents = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim
        freq_extra = 1.0 / (theta ** exponents)
        freq_inter = 1.0 / (self.factor * theta ** exponents)

        def turns_dim(turns: float) -> float:
            # the dim whose wavelength completes ``turns`` rotations over the training context
            return (rotary_dim * math.log(self.original_max_position_embeddings
                                          / (turns * 2 * math.pi))) / (2 * math.log(theta))

        low = max(math.floor(turns_dim(self.beta_fast)), 0)
        high = min(math.ceil(turns_dim(self.beta_slow)), rotary_dim - 1)
        if low == high:
            high += 0.001  # one-dim band: avoid the division by zero
        ramp = ((torch.arange(rotary_dim // 2, dtype=torch.float32, device=device) - low)
                / (high - low)).clamp(0, 1)
        keep = 1.0 - ramp
        return freq_inter * (1 - keep) + freq_extra * keep


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

    # YaRN frequencies and table scale (rotates from the rope_table_dtype table only)
    yarn: YarnScaling | None = None

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
