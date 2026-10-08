"""What a model declares about sampling: its spec, its per-request config,
its step.

Kept free of the resource and its Triton kernels so a submodule can declare a
step without pulling them in behind it.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch

from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


@dataclass
class SamplerSpec(NodeResourceSpec):
    vocab_size: int | None # must be set for enabled_repetion_penalty
    # Capability, not intent: whether this node's sampling kernel is *able* to
    # apply a repetition penalty, which decides both the seen-token buffers and
    # the kernel variant baked into the captured graph. Whether the penalty
    # actually runs on a given step is settled per step from the resident
    # requests' `repetition_penalty` — see `SamplerResource.admit`.
    enable_repetion_penalty: bool = True
    # Same kind of capability for min-p: whether the node's sampler (eager and
    # graph-captured) carries the filter. It costs two passes over ``[B, V]``
    # per step, so nodes that never ask for it pay nothing; a request that
    # sets ``min_p`` on a node without it is refused at ingest.
    enable_min_p: bool = False

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.sampler.resource import SamplerResource

        return SamplerResource


@dataclass
class SamplingReqConfig(ResourceReqConfig):
    temperature: float = 0.6
    top_k: int = 0
    top_p: float = 1
    ignore_eos: bool = False # used for benchmark parity
    repetition_penalty: float = 1
    # Min-p (HF ``MinPLogitsWarper`` / vLLM ``min_p``): drop every token whose
    # probability is below ``min_p`` times the most likely token's, after the
    # penalty and temperature and before top-k/top-p. 0 disables.
    min_p: float = 0.0
    _seed: int = 0 # set by the conductor

    def apply_conductor_config(
        self, seed: int=0,
        **kwargs
    ):
        self._seed = seed

    @property
    def seed(self):
        return self._seed


@dataclass(frozen=True)
class SamplerStep(ResourceStep):
    apply_penalty: bool = True
    # rid -> prefill tokens for the repetition penalty
    prefill_tracked_tokens: dict[str, torch.Tensor] = field(default_factory=dict)
    # Rows whose sampled token is kept; None keeps every row. A dropped row
    # still draws, but its RNG offset and seen-token mask are not committed.
    kept_rids: frozenset[int] | None = None



def keep_final_chunk_samples(
    step: SamplerStep, request_ids: Sequence, inputs: Sequence,
) -> SamplerStep:
    """``step`` without the rows mid-prefill, for an autoregressive node that
    chunks: a non-final chunk's token is not a real sample, so its RNG offset
    and seen-token mask must not commit."""
    non_final = {
        rid for rid, inp in zip(request_ids, inputs, strict=True)
        if not getattr(inp, "is_final_chunk", True)
    }
    if not non_final:
        return step
    kept = frozenset(request_ids) if step.kept_rids is None else step.kept_rids
    return replace(step, kept_rids=kept - non_final)
