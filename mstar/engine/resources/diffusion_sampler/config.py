"""What a discrete-diffusion model declares about sampling.

Separate from ``sampler`` rather than a mode on it, for two reasons that are
not stylistic. The autoregressive sampler draws one token per request and
returns tokens only; a diffusion step scores every unrevealed position at once
and the caller needs the log-probabilities back to decide which of them to
keep. And ``SamplerResource`` bakes ``APPLY_PENALTY`` as a Triton constexpr at
capture, around seen-token buffers and a repetition penalty that have no
meaning over a canvas that is rewritten every step.

Kept free of the resource itself so a submodule can declare a step without
importing the kernels behind it.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


@dataclass
class DiffusionSamplerSpec(NodeResourceSpec):
    """Shapes the scoring kernel needs to know before any request arrives."""

    vocab_size: int
    # Rows per position. Diffusion over audio codes scores a [C, T] canvas, so
    # a position is a (codebook, frame) cell rather than a single token.
    num_rows: int = 1
    # Class the model must never emit, forced to -inf before the argmax.
    # ``None`` leaves every class eligible.
    forbidden_class: int | None = None

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.diffusion_sampler.resource import (
            DiffusionSamplerResource,
        )

        return DiffusionSamplerResource


@dataclass
class DiffusionSamplingReqConfig(ResourceReqConfig):
    """Per-request knobs, resolved once at ingest.

    ``guidance_scale`` 0 turns classifier-free guidance off, and the caller may
    then skip the unconditional forward entirely and pass ``None``.
    """

    guidance_scale: float = 0.0
    temperature: float = 0.0
    # Fraction of the vocabulary kept before the Gumbel draw. Only read when
    # temperature is above 0.
    top_ratio: float = 0.1
    _seed: int = 0  # set by the conductor

    def apply_conductor_config(self, seed: int = 0, **kwargs):
        self._seed = seed

    @property
    def seed(self) -> int:
        return self._seed


@dataclass(frozen=True)
class DiffusionSamplerStep(ResourceStep):
    """Nothing to stage: the sampler's whole input arrives with the call.

    Deliberately empty. An earlier draft carried the diffusion iteration here,
    which was the wrong shape: requests in one step are at *different*
    iterations, since each runs its own ``num_step``. The iteration is a
    per-request argument to ``sample`` instead.
    """
