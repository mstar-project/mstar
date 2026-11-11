"""What a target that hosts a drafter (Qwen3's DSpark) calls on it: the declaration,
built from config alone, and the module it builds on a worker."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from mstar.engine.resources.draft.drafter import Drafter, DrafterDecl

if TYPE_CHECKING:
    import torch
    from torch import nn

    from mstar.engine.resources.step import ResourceStep


class HostedDrafter(Drafter, Protocol):
    def prefill(self, engine_inputs, input_ids: torch.Tensor, sampled: torch.Tensor | None = None) -> None:
        """Its state over a prompt step's rows, after the target's forward wrote what it reads."""
        ...


class HostedDrafterDecl(DrafterDecl, Protocol):
    num_draft_tokens: int

    def prefill_steps(self) -> dict[str, ResourceStep]:
        """Its resources' steps on a prompt step (`VerifyNode.prefill_steps`)."""
        ...

    def verify_steps(self) -> dict[str, ResourceStep]:
        """Its resources' steps on a verify step (`VerifyNode.verify_steps`)."""
        ...

    def build(
        self, *, device, comm_group=None, dtype: torch.dtype | None = None, target: nn.Module | None = None,
    ) -> HostedDrafter:
        ...
