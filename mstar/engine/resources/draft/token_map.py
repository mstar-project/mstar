"""A drafter's vocabulary against its target's: draft id to target id.

An EAGLE-3 head drafts over a reduced vocabulary and the verify step needs target ids. `TokenMap` holds that one
direction as a device index tensor, built once when the drafter binds, and applies it with a gather, so a captured
step replays it. Nothing here runs on the host per step.
"""
from __future__ import annotations

import torch


class TokenMap:
    """``to_target_ids[d]`` is draft id ``d``'s target id."""

    def __init__(self, to_target_ids: torch.Tensor):
        self.to_target_ids = to_target_ids

    @classmethod
    def from_offsets(cls, d2t: torch.Tensor, target_vocab: int) -> TokenMap:
        """From an EAGLE-3 checkpoint's ``d2t``: draft id ``d`` is target id ``d + d2t[d]``. Each target id is
        named once and within the target's ``target_vocab`` ids."""
        ids = torch.arange(d2t.numel(), device=d2t.device) + d2t.long()
        host = ids.cpu()
        if host.numel() and (int(host.min()) < 0 or int(host.max()) >= target_vocab):
            raise ValueError(f"a draft id maps outside the target's {target_vocab} ids")
        if host.unique().numel() != host.numel():
            raise ValueError("two draft ids map to one target id")
        return cls(ids)

    def to(self, device: torch.device | str) -> TokenMap:
        return TokenMap(self.to_target_ids.to(device))

    def to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        """Target ids of ``draft_ids`` (any shape), a gather."""
        return self.to_target_ids[draft_ids.long()]
