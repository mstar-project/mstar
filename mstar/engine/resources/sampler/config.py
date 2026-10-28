"""What a model declares about sampling: its spec, its per-request config,
its step.

Kept free of the resource and its Triton kernels so a submodule can declare a
step without pulling them in behind it.
"""

import hashlib
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


@dataclass(frozen=True)
class LoopbackState:
    """Per-request state a decode loop carries from step to step on the
    device, beside its last sampled token: one row of ``width`` values per
    request, kept by sampler slot. A row starts (and restarts, when its slot
    is reused) at ``default``."""
    width: int
    dtype: torch.dtype
    default: tuple[int | float, ...]

    def __post_init__(self):
        if len(self.default) != self.width:
            raise ValueError(
                f"LoopbackState default has {len(self.default)} values for width {self.width}"
            )


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
    # name -> extra loop-back state the node keeps by slot beside the last
    # token (``SamplerResource.loopback_state`` / ``store_loopback_state``);
    # only used on the device loop-back path
    loopback_state: dict[str, LoopbackState] = field(default_factory=dict)
    # Capabilities for the generation-aware request knobs below; each off by
    # default, leaving the node's sampler as it was.
    # The longest ``repetition_window`` a request may ask for (sizes the history).
    max_repetition_window: int = 0
    # The ids ``min_tokens`` bars (the model's EOS ids).
    min_tokens_stop_ids: tuple[int, ...] = ()
    # Allows ``top_p_first`` / ``top_p_min_keep``.
    enable_top_p_first: bool = False
    # The node's decode verifies drafts (``NodeSubmodule.get_draft_policy``), greedily for
    # now, so a request whose sampling a greedy verify would change is refused.
    draft_verify: bool = False

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
    # Min-p (HF ``MinPLogitsWarper``): drop every token whose
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
    # Whether the repetition penalty also counts the prompt's tokens (HF does).
    # A node whose step declares no ``prefill_tracked_tokens`` only ever has the
    # generated ones to count.
    penalize_prompt: bool = True
    _seed: int = 0 # set by the conductor

    def apply_conductor_config(
        self, seed: int=0, resource_key: str | None = None,
        **kwargs
    ):
        # each sampler of a request draws its own stream from the request's seed
        self._seed = seed if resource_key is None else derive_seed(seed, resource_key)

    @property
    def seed(self):
        return self._seed

    def validate(self, spec: "SamplerSpec | None" = None) -> None:
        """Raise ``ValueError`` on a value outside the contract, or one ``spec``'s
        node cannot honour. None means the sampler's default."""
        _check_real("temperature", self.temperature, lambda v: v >= 0, ">= 0")
        _check_real("top_p", self.top_p, lambda v: 0 < v <= 1, "in (0, 1]")
        _check_real("min_p", self.min_p, lambda v: 0 <= v <= 1, "in [0, 1]")
        _check_real("repetition_penalty", self.repetition_penalty, lambda v: v > 0, "> 0")
        if self.top_k is not None and (
            isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0
        ):
            raise ValueError(f"top_k must be an integer >= 0 (0 disables it); got {self.top_k!r}")
        for name in ("ignore_eos", "penalize_prompt"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be a boolean; got {value!r}")
        if spec is None:
            return
        if spec.draft_verify:
            problem = draft_verify_problem(
                self.temperature or 0, self.repetition_penalty or 1, self.min_tokens or 0,
            )
            if problem is not None:
                raise ValueError(problem)
        # refused here, not ignored: the node's sampler has no such filter
        if self.min_p and not spec.enable_min_p:
            raise ValueError("this model does not support min_p; send 0 or leave it unset")
        # a windowed penalty counts the generation history, not the seen-token mask
        windowed = self.repetition_window > 0 and spec.max_repetition_window > 0
        if self.repetition_penalty not in (None, 1) and not (spec.enable_repetion_penalty or windowed):
            raise ValueError(
                "this model does not support repetition_penalty; send 1 or leave it unset"
            )


def draft_verify_problem(temperature: float, repetition_penalty: float, min_tokens: int) -> str | None:
    """Why a greedy draft verify can't serve these settings, if it can't. The verify keeps a
    draft only where it is the target's argmax, so temperature must be 0; a penalty or a
    min-tokens floor would move that argmax by the tokens before each draft, which the verify
    does not see."""
    asked = []
    if temperature != 0:
        asked.append(f"temperature={temperature}")
    if repetition_penalty != 1:
        asked.append(f"repetition_penalty={repetition_penalty}")
    if min_tokens:
        asked.append(f"min_tokens={min_tokens}")
    if not asked:
        return None
    return (
        f"this model decodes with draft verification (speculative decoding), which is greedy-only "
        f"so far, but the request asks for {', '.join(asked)}; send temperature=0 with no "
        "repetition_penalty or min_tokens"
    )


def derive_seed(seed: int, stream: str) -> int:
    """A seed for one named stream of a request, as a non-negative int64.

    Stable across processes (no ``hash()``), and distinct per stream, so a
    request's samplers do not draw correlated numbers from one seed.
    """
    digest = hashlib.blake2b(f"{seed}:{stream}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & (2**63 - 1)


def _check_real(name: str, value, ok, rule: str) -> None:
    """Raise ``ValueError`` unless ``value`` is None or a finite number satisfying ``ok``."""
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number; got {value!r}")
    if not math.isfinite(value) or not ok(value):
        raise ValueError(f"{name} must be a finite number {rule}; got {value!r}")


@dataclass(frozen=True)
class SamplerStep(ResourceStep):
    apply_penalty: bool = True
    # False turns top-k/top-p off for this step (e.g. a TTS's first code).
    # Captured graphs bake it: declare it the same way on every step of a walk.
    apply_filters: bool = True
    # rid -> prefill tokens for the repetition penalty
    prefill_tracked_tokens: dict[str, torch.Tensor] = field(default_factory=dict)
    # Rows whose sampled token is kept; None keeps every row. A dropped row
    # still draws, but its RNG offset and seen-token mask are not committed.
    kept_rids: frozenset[int] | None = None
    # The spec's loop-back states this step's forward stores
    # (``store_loopback_state``); commit persists them by slot.
    loopback_writes: frozenset[str] = frozenset()
