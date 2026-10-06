"""What a model declares about linear attention: variant, backend, spec, step.

Kept free of the managers so declaring a step does not import FlashInfer. The
state lives in a ``RecurrentStatePool`` (as `kv/` is to `attn/`), and head
geometry is read off its blocks rather than declared here.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource


class LinearAttnVariant(Enum):
    # gated delta rule: scalar decay per head. Qwen3.5, Qwen3-Next.
    GDN = "gdn"
    # Kimi delta attention: diagonal decay per K channel. Kimi Linear, GLM-5.3.
    KDA = "kda"
    # Mamba-2 / SSD: per-head scalar decay exp(dt * A), grouped B/C. Nemotron-H.
    MAMBA2 = "mamba2"


class LinearAttnBackend(Enum):
    FLASHINFER = "flashinfer"


@dataclass
class LinearAttnConfig:
    recurrent_state: str  # name of the recurrent state pool
    variant: LinearAttnVariant
    backend: LinearAttnBackend = LinearAttnBackend.FLASHINFER

    # Defaults to head_k_dim ** -0.5 off the pool's geometry, as the kernels'
    # own default does.
    sm_scale: float | None = None

    # GLM-5.3 sets this (`gate_lower_bound: -5.0`), selecting a different gate
    # formula in the KDA kernel. None keeps the softplus one.
    gate_lower_bound: float | None = None

    # L2-normalise q and k for the delta rule: in the kernel where it can, by
    # `GDNManager.run` where it cannot (the SM90 chunked prefill). Qwen3.5 needs
    # it (HF and vLLM both pass it) and skipping it diverges silently.
    qk_l2norm: bool = True

    # Mamba-2 only: clamp on dt after softplus, as HF's ``time_step_limit``
    # (Nemotron-H leaves it open: (0, inf)).
    time_step_limit: tuple[float, float] = (0.0, float("inf"))


@dataclass
class LinearAttnSpec(NodeResourceSpec):
    config: LinearAttnConfig

    def depends_on(self) -> set[str]:
        return {self.config.recurrent_state}

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.linear_attn.base import LinearAttnManager

        return LinearAttnManager

    def apply_yaml_overrides(
        self, backend: str | LinearAttnBackend | None = None,
    ):
        """Let the deployment pick the backend; slot capacity is tuned on the pool."""
        if backend is not None:
            self.config.backend = LinearAttnBackend(backend)


@dataclass(frozen=True)
class LinearAttnStep(ResourceStep):
    """One step's work for a linear-attention layer stack.

    Carries no state semantics: the segments give the rows and spans, and the
    pool's own step says what becomes of the slots. The walk is derived from
    the spans.
    """
