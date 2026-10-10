"""What a model declares about sampling: its spec, its per-request config,
its step.

Kept free of the resource and its Triton kernels so a submodule can declare a
step without pulling them in behind it.
"""

from dataclasses import dataclass, field
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
    # Capabilities for the generation-aware request knobs below; each off by
    # default, leaving the node's sampler as it was.
    # The longest ``repetition_window`` a request may ask for (sizes the history).
    max_repetition_window: int = 0
    # The ids ``min_tokens`` bars (the model's EOS ids).
    min_tokens_stop_ids: tuple[int, ...] = ()
    # Allows ``top_p_first`` / ``top_p_min_keep``.
    enable_top_p_first: bool = False

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
    # > 0: the penalty is ``repetition_penalty ** n``, n a token's count among
    # the last ``repetition_window`` generated tokens; 0: presence over prompt
    # and output.
    repetition_window: int = 0
    # The spec's stop ids get no probability until this many tokens exist
    # (applied after top-k/top-p).
    min_tokens: int = 0
    # HF's order, top-p then top-k, each on the full distribution (FlashInfer's
    # is top-k, then top-p over the renormalised top-k).
    top_p_first: bool = False
    # Top-p keeps at least this many tokens (HF's ``min_tokens_to_keep``).
    top_p_min_keep: int = 1
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
    # False turns top-k/top-p off for this step (e.g. a TTS's first code).
    # Captured graphs bake it: declare it the same way on every step of a walk.
    apply_filters: bool = True
    # rid -> prefill tokens for the repetition penalty
    prefill_tracked_tokens: dict[str, torch.Tensor] = field(default_factory=dict)
