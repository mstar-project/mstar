"""One verify step in a forward, whichever kind of drafter drafts for it.

Two kinds of drafter feed the draft-verify core:

- an engine-side proposer (``DraftPolicy.proposer`` names a ``DraftProposer`` resource, such as
  the n-gram proposer): the drafts are host work the resource staged in ``plan``, and the
  forward takes the block from ``DraftVerifyResource.propose``;
- a drafter in the forward (``Drafter``: a draft model, a DSpark drafter, later EAGLE-3): the
  forward runs it over the step's inputs and hands its drafts to ``agree``.

``draft_verify_step`` is the sequence both end in (drafts, agree, the target over the block,
verify, and a drafter's ``observe`` of the verified block) and returns the next step's inputs,
the loop-back edge. A drafter whose chain is interleaved with the target's own passes (GLM's
MTP) calls the resource's steps itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple, Protocol

import torch

if TYPE_CHECKING:
    from mstar.engine.resources.draft_verify.resource import DraftVerifyResource, VerdictTensors


class Drafts(NamedTuple):
    tokens: torch.Tensor  # [n, k]
    # [n, k, V]: the distributions sampled drafts were drawn from; None takes each as certain
    probs: torch.Tensor | None


class Drafter(Protocol):
    """A drafter that runs in the verify step's forward."""

    def propose(self, inputs: torch.Tensor, dv: DraftVerifyResource, request_ids: list[int]) -> Drafts:
        """``k`` drafts per row from the step's ``inputs`` (the rows' last kept tokens, the
        last of them the token the block starts with). Capturable."""
        ...

    def next_inputs(self, block: torch.Tensor, verdict: VerdictTensors) -> torch.Tensor:
        """The next step's inputs from this step's block ``[n, k + 1]`` and its verdict."""
        ...

    # Optional: ``observe(block, verdict)``, run after the verify, writes what the drafter keeps
    # of the verified block (a DSpark drafter's context KV from the target's taps); the
    # verdict's rewinds take its rejected rows back with the target's.


def draft_verify_step(
    dv: DraftVerifyResource,
    request_ids: list[int],
    inputs: torch.Tensor,
    target: Callable[[torch.Tensor], torch.Tensor],
    drafter: Drafter | None = None,
) -> torch.Tensor:
    """One verify step: the drafts (``drafter``'s, or with None the walk's engine-side
    proposer's), rank 0's on a lockstep instance; ``target`` over the block's ids, flat, to its
    logits ``[n * (k + 1), V]``; the verdict; a drafter's ``observe``. ``inputs`` holds a row's
    last kept tokens, the token its block starts with last (``[n]`` or ``[n, w]``). Returns the
    next step's inputs: the target's next token per row, or what ``drafter.next_inputs`` makes
    of the verdict. Capturable."""
    if drafter is None:
        block, probs = dv.propose(request_ids, inputs)
    else:
        drafts = drafter.propose(inputs, dv, request_ids)
        block = dv.block_ids(inputs.reshape(inputs.shape[0], -1)[:, -1], dv.agree(drafts.tokens))
        probs = drafts.probs
    verdict = dv.verify(request_ids, block, target(block.reshape(-1)), probs)
    if drafter is None:
        return verdict.next_inputs
    observe = getattr(drafter, "observe", None)
    if observe is not None:
        observe(block, verdict)
    return drafter.next_inputs(block, verdict)
