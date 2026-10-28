"""An engine-side proposer: drafts for a verify step that the model's forward doesn't compute.

A walk whose ``DraftPolicy.proposer`` names one has its forward take the step's block from
``DraftVerifyResource.propose``, which asks the proposer for the drafts and runs them through
``agree``. The proposer learns each request's tokens from two places: its prompt, which the
prefill steps declare (``ProposeStep.history``), and each verdict, which the draft-verify
resource hands it once, as it reads it (``observe``). What it drafts from them is its own: a
lookup in the request's history (``NgramProposer``), or a forward of its own over weights the
engine loads (a draft model, later).

The other kind of drafter, one inside the model's forward (MTP, DSpark), declares no
proposer: its forward drafts and calls ``agree`` itself.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from mstar.engine.resources.base import Resource, RewindKind

if TYPE_CHECKING:
    from mstar.engine.resources.draft_verify.resource import DraftVerifyResource


class DraftProposer(Resource):
    # it learns emitted tokens only, from verdicts, so a rejected draft leaves nothing to take back
    rewind_kind = RewindKind.NONE

    # the drafts a row takes from it, which the walk's policy asks for
    max_drafts: int
    # it predicts what a row's next step keeps (`predict`, `expected_gain`), so an adaptive
    # walk (``DraftPolicy.adaptive``) can draft with it
    predicts: bool = False
    # the node's draft-verify resource, which hands it the verdicts; bound by the engine at load
    _verifier: DraftVerifyResource | None = None

    def bind_verifier(self, verifier: DraftVerifyResource) -> None:
        self._verifier = verifier

    @abstractmethod
    def observe(self, rid: int, row: np.ndarray) -> None:
        """A verdict of a request's verify step: ``row`` is ``[accepted, input, t_0 .. t_k]``,
        the drafts kept, the row's input token and the target's token at each row of the
        block, so the request emitted ``t_0 .. t_accepted``. Called once per verdict, on the
        thread that read it (plan thread or gpu thread), under the draft-verify resource's
        lock: no more than a note to take it in at the next plan."""

    def predict(self, request_ids: Sequence[int], drafts: int) -> None:
        """At the commit of a step of these rows (``drafts`` each, 0 for a step that drafted
        nothing), on the gpu thread, after every verdict before the step was handed to
        ``observe``: note what each row's next step is expected to keep, for
        ``expected_gain``. Only a proposer that ``predicts`` is asked."""
        del request_ids, drafts

    def expected_gain(self, request_ids: Sequence[int]) -> float:
        """The drafts these rows' next step is expected to keep, summed, as of their last
        ``predict``: what ``DraftVerifyResource.choose_drafts`` weighs against the policy's
        ``min_gain``."""
        raise NotImplementedError(f"{type(self).__name__} predicts nothing")

    @abstractmethod
    def propose(self, request_ids: Sequence[int], rows: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        """The drafts of the step being run, ``[rows, max_drafts]`` (a replay's padding rows
        included), and the distributions they were drawn from (``[rows, k, V]``), or None for
        drafts taken as certain. In the forward; capturable."""
