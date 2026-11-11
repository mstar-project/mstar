"""What a model declares about positions: its scheme, its spec, its step.

Kept free of the manager and its kernels so a submodule can declare a step
without pulling FlashInfer in behind it.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


class PosBackend(Enum):
    ROPE = "rope"


class PosScheme(Enum):
    SEQUENTIAL = "sequential"
    BLOCK = "block"


@dataclass(frozen=True)
class LongRoPE:
    """LongRoPE scaling (Phi-3, Phi-4-mini): every rotary frequency divided by a
    per-dimension factor, from a short set or a long set, and cos/sin scaled by
    ``sqrt(1 + ln(max / original) / ln(original))``.

    A request rotates under one set for its whole life, chosen from the length
    the model passes (``PositionReqConfig.max_position``): the long set past
    ``original_max_position_embeddings``. Pass the length the model's reference
    chooses by: for Phi-3 and Phi-4 that is the prompt's, since the reference
    picks by the positions in its forward. Prompt plus ``max_tokens`` would put
    prompts the reference runs short on the long set. The reference switches
    mid-request, once its output crosses the threshold, and recomputes its cache;
    holding one set keeps a request's KV consistent without that.
    """

    short_factor: tuple[float, ...]
    long_factor: tuple[float, ...]
    original_max_position_embeddings: int
    max_position_embeddings: int

    def __post_init__(self):
        if len(self.short_factor) != len(self.long_factor):
            raise ValueError(
                f"LongRoPE short and long factors differ in length: "
                f"{len(self.short_factor)} vs {len(self.long_factor)}"
            )

    @classmethod
    def from_hf(cls, config: Mapping[str, Any]) -> "LongRoPE":
        """From a Hugging Face config with ``rope_scaling.type == "longrope"``."""
        scaling = config["rope_scaling"]
        if scaling.get("type", scaling.get("rope_type")) != "longrope":
            raise ValueError(f"not a LongRoPE config: rope_scaling={scaling!r}")
        return cls(
            short_factor=tuple(float(f) for f in scaling["short_factor"]),
            long_factor=tuple(float(f) for f in scaling["long_factor"]),
            original_max_position_embeddings=int(config["original_max_position_embeddings"]),
            max_position_embeddings=int(config["max_position_embeddings"]),
        )

    @property
    def attention_scale(self) -> float:
        scale = self.max_position_embeddings / self.original_max_position_embeddings
        if scale <= 1.0:
            return 1.0
        return math.sqrt(1 + math.log(scale) / math.log(self.original_max_position_embeddings))

    def is_long(self, max_position: int) -> bool:
        """Whether a request whose table is chosen by ``max_position`` rotates with
        the long factors."""
        return max_position > self.original_max_position_embeddings


# the length LongRoPE picks a request's table by, as `process_prompt` sets it
MAX_POSITION_KWARG = "max_position"


@dataclass
class PositionReqConfig(ResourceReqConfig):
    """The length a resource that picks its rotary table per request
    (``PositionConfig.longrope``) chooses by: the one the model's reference
    chooses by, which for Phi-3 and Phi-4 is the prompt's (`LongRoPE`)."""

    max_position: int

    def validate(self, spec: "NodeResourceSpec | None" = None) -> None:
        if isinstance(self.max_position, bool) or not isinstance(self.max_position, int) or self.max_position < 1:
            raise ValueError(f"max_position must be a positive int; got {self.max_position!r}")
        if isinstance(spec, PositionSpec) and spec.config.longrope is None:
            raise ValueError(
                f"position resource {spec.resource_key!r} has one rotary table; "
                "a per-request max_position needs PositionConfig.longrope"
            )


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
    # multiple of 32 for a table with NeoX pairing, of 16 with GPT-J pairing
    # (`check_table_rotary_dim`). XPU rotates from a table in rope_dtype or the
    # KV dtype and refuses any other value.
    rope_table_dtype: torch.dtype | None = None

    low_freq_factor: float | None = None
    high_freq_factor: float | None = None
    old_context_len: int | None = None

    # Per-request short/long rotary tables (`LongRoPE`). CUDA rotates from a
    # table, rounded to ``rope_table_dtype`` when that is set; a request picks
    # the long one with ``MAX_POSITION_KWARG`` (`PositionSpec.request_config`).
    longrope: LongRoPE | None = None

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
    # the sequence shard over its nodes, if any (`derive`): a split label's positions are its range's
    shard_key: str | None = field(default=None, init=False, repr=False)

    def depends_on(self) -> set[str]:
        keys = {self.config.kv_cache}
        if self.shard_key is not None:
            keys.add(self.shard_key)
        return keys

    def derive(self, specs: Mapping[str, NodeResourceSpec]) -> list[str]:
        from mstar.engine.resources.sequence_shard import take_shard_edge

        return take_shard_edge(self, specs)

    def request_config(self, model_kwargs: dict | None) -> "PositionReqConfig | None":
        """Under LongRoPE, the length the request's table is chosen by, from
        ``MAX_POSITION_KWARG``, which a model sets in ``process_prompt`` (the data worker
        drops a client's); without it, the short table."""
        length = (model_kwargs or {}).get(MAX_POSITION_KWARG)
        if length is None or self.config.longrope is None:
            return None
        return PositionReqConfig(max_position=length)

    def prefix_variant(self, model_kwargs: dict | None) -> bytes | None:
        """The long table gives other KV than the short one for the same tokens, so a
        request on it keys its prefix apart; derived from the same kwarg as its config,
        so the two cannot disagree."""
        config = self.request_config(model_kwargs)
        if config is None or not self.config.longrope.is_long(config.max_position):
            return None
        return b"longrope-long"

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
    # positions take their packing off KV's plan output, not off `combined_labels`
