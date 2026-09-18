"""What a speculating node declares about its acceptance counts: the spec and the step.

Kept free of the resource so a submodule can declare the step without importing it."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import ResourceStep

if TYPE_CHECKING:
    from mstar.engine.resources.base import Resource

# the key other resources look the acceptance counts up under in ``ctx.plan_results``
SPEC_ACCEPTANCE = "spec_acceptance"


@dataclass
class SpecAcceptanceSpec(NodeResourceSpec):
    # drafted tokens per step: a verify step processes num_speculative + 1 tokens per request
    num_speculative: int = 7

    @property
    def resource_class(self) -> "type[Resource]":
        from mstar.engine.resources.speculative.resource import SpecAcceptance

        return SpecAcceptance

    def apply_yaml_overrides(self, num_speculative: int | None = None, **kwargs):
        if num_speculative is not None:
            self.num_speculative = int(num_speculative)


@dataclass(frozen=True)
class SpecStep(ResourceStep):
    # the step verifies a speculative block per row: its forward stages a verdict and its commit
    # registers the rows (a prefill step of the same node declares the resource without either)
    verify: bool = False


@dataclass(frozen=True)
class SpecAccepted:
    """One request's verdict from its last verify step, as published in the plan."""
    accepted: int  # drafted tokens the target agreed with, 0..num_speculative
    rejected: int  # num_speculative - accepted: the tail the step's resources must take back
    label: str = "main"
