"""One verify step in a forward, whichever kind of drafter drafts for it.

Two kinds of drafter feed the draft-verify core:

- an engine-side proposer (``DraftPolicy.proposer`` names a ``DraftProposer`` resource, such as
  the n-gram proposer): the drafts are host work the resource staged in ``plan``, and the
  forward takes the block from ``DraftVerifyResource.propose``;
- a drafter in the forward (``Drafter``: a draft model, a DSpark drafter, later EAGLE-3): the
  forward runs it over the step's inputs and hands its drafts to ``agree``.

``draft_verify_step`` is the sequence both end in (the last verdict taken back on the device
under ``DraftPolicy.device_rewind``, drafts, agree, the target over the block, verify, and a
drafter's ``observe`` of the verified block) and returns the next step's inputs, the loop-back
edge. A drafter whose chain is interleaved with the target's own passes (GLM's
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
    # under device rewind, the rows the last step's verdict rejected leave this step's plan
    # before anything reads it
    dv.apply_previous()
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


class PackedDrafter(Protocol):
    """A drafter that can propose over a step mixing verify rows with prompt rows."""

    def propose_packed(self, ids: torch.Tensor, dv: DraftVerifyResource, request_ids: list[int]) -> Drafts:
        """``k`` drafts for each of the step's rows from its packed tokens ``ids``, the verify
        rows leading with ``k + 1`` tokens each (the anchor, then placeholders). A prompt or
        padding row's drafts are never read. Capturable."""
        ...


def leading_rows(x: torch.Tensor, rows: int, width: int) -> torch.Tensor:
    """``[rows, width, ...]``: the first ``rows * width`` rows of ``x``, zero-padded when a
    bucket holds fewer (a mixed step's last rows are prompts, shorter than a block)."""
    need = rows * width
    if x.shape[0] < need:
        x = torch.cat([x, x.new_zeros((need - x.shape[0], *x.shape[1:]))])
    return x[:need].view(rows, width, *x.shape[1:])


def draft_verify_mixed_step(
    dv: DraftVerifyResource,
    request_ids: list[int],
    ids: torch.Tensor,
    hidden: Callable[[torch.Tensor], torch.Tensor],
    logits: Callable[[torch.Tensor], torch.Tensor],
    last_hidden: Callable[[torch.Tensor], torch.Tensor],
    sample: Callable[[torch.Tensor], torch.Tensor],
    drafter: PackedDrafter,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One step of verify rows and prompt rows (``with_mixed_verify``), the target run once over
    both. ``ids`` are the step's packed tokens, the verify rows first, ``k + 1`` each; the step's
    ``n`` rows include its prompts and padding, and everything per verify row runs over all
    ``n``, so only the bucket's ``(rows, tokens)`` shape the graph. ``hidden`` is the target over
    packed ids, ``logits`` its head, ``last_hidden`` each row's last token's hidden state and
    ``sample`` the sampler over one row each. Returns the next step's inputs ``[n, 1]`` (the
    verify rows') and a token per row (the prompt rows' first). Capturable."""
    dv.apply_previous()
    n = len(request_ids)
    drafts = drafter.propose_packed(ids, dv, request_ids)
    width = drafts.tokens.shape[1] + 1
    block = dv.block_ids(leading_rows(ids, n, width)[:, 0], dv.agree(drafts.tokens))
    states = hidden(dv.place_block(ids, block))
    # one pass of the head over the blocks' rows and each row's last token: it reads its weights once
    scores = logits(torch.cat([leading_rows(states, n, width).reshape(n * width, -1), last_hidden(states)]))
    verdict = dv.verify(request_ids, block, scores[: n * width], drafts.probs)
    first = sample(scores[n * width:])
    # the drafter keeps the verified blocks' context and the prompts' alike, from the taps
    observe = getattr(drafter, "observe", None)
    if observe is not None:
        observe(block, verdict)
    return drafter.next_inputs(block, verdict), first

