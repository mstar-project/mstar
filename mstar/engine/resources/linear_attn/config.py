"""What a model declares about linear attention: its variant, its backend,
its spec, its step.

Kept free of the managers and their kernels so a submodule can declare a step
without pulling FlashInfer in behind it.

The wrapper half of the split `kv/` and `attn/` already use: the state lives in
a ``RecurrentStatePool``, and this resource plans and runs kernels against it.
Head geometry is not declared here — ``LinearAttnManager.build`` reads it off
the pool's blocks, so a model names its shapes once.
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


@dataclass
class LinearAttnSpec(NodeResourceSpec):
    config: LinearAttnConfig

    def depends_on(self) -> set[str]:
        # the kernels run against the pool's slots, and the geometry they
        # marshal for is read off that pool's blocks
        return {self.config.recurrent_state}

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.linear_attn.base import LinearAttnManager

        return LinearAttnManager

    def apply_yaml_overrides(
        self, backend: str | LinearAttnBackend | None = None,
    ):
        """Which kernel to run is the deployment's call as much as the model's.

        Slot capacity is not repeated here: it belongs to the pool this spec
        depends on, and is tuned under that resource's own block.
        """
        if backend is not None:
            self.config.backend = LinearAttnBackend(backend)


@dataclass(frozen=True)
class LinearAttnStep(ResourceStep):
    """One step's work for a linear-attention layer stack.

    Carries no state semantics: the segments say which rows run and how long
    each is, and the pool's own step says what becomes of the slots. Which rows
    take the recurrent path and which the chunked one is derived from the
    spans, not declared.
    """
