"""What a target that hosts a drafter (Qwen3's DSpark) calls on it: the declaration,
built from config alone, and the module it builds on a worker."""
from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol

from mstar.engine.resources.draft.drafter import Drafter, DrafterDecl, PrefillInputs

if TYPE_CHECKING:
    import torch
    from torch import nn

    from mstar.engine.resources.step import ResourceStep


class HostedDrafter(Drafter, Protocol):
    @abstractmethod
    def prefill(self, inputs: PrefillInputs) -> torch.Tensor | None:
        """Its state over a prompt step's rows, after the target's forward wrote what it reads.
        A hosted drafter runs its own stack over the prompt, so it has to say what it keeps: for
        one that declares a carry (`DrafterNeeds.carry`), the rows' first carry, which the
        scaffold stores; None otherwise."""
        ...


class HostedDrafterDecl(DrafterDecl, Protocol):
    num_draft_tokens: int

    @abstractmethod
    def prefill_steps(self) -> dict[str, ResourceStep]:
        """Its resources' steps on a prompt step (`VerifyNode.prefill_steps`)."""
        ...

    @abstractmethod
    def verify_steps(self) -> dict[str, ResourceStep]:
        """Its resources' steps on a verify step (`VerifyNode.verify_steps`)."""
        ...

    @abstractmethod
    def build(
        self, *, device, comm_group=None, dtype: torch.dtype | None = None, target: nn.Module | None = None,
    ) -> HostedDrafter:
        ...
