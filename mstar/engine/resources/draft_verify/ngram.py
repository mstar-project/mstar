"""Prompt-lookup drafting: a request's next tokens guessed from its own history.

A row's drafts are the tokens that followed the last earlier occurrence of its history's
suffix, trying the longest suffix first (``max_ngram`` tokens down to ``min_ngram``). Code
edits, refactors and summaries copy long runs of their prompt, so the guesses land there and
cost nothing to make: no weights, no forward, only host work.

Once a row drafts from a stretch, it keeps following it while the tokens it emits are that
stretch's: a file being rewritten repeats short n-grams (``self.``, an indent), whose last
occurrence is often another place in the file, but the place being copied is the one whose
tokens the reply has been matching.

The history is the prompt (``ProposeStep.history`` of the prefill steps, copied to the host
behind each prefill) and every emitted token (``observe``, once per verdict). A request's
n-grams are indexed in a dict, n-gram -> where it last ended, and only when it looks up: a row
that follows its stretch needs no lookup, so its tokens wait. Inserting the tokens appended
since is one ``dict.update`` over tuples built in C, so the work is in the tokens appended, not
the history; a step's other work is a few list operations per row.

For an adaptive walk the proposer also predicts, at each step's commit, how many drafts a row
would keep at its next step (``expected_gain``): none without a stretch to draft from, else
``p + p^2 + .. + p^k`` for ``p`` the rate at which the row's tokens have followed its
stretches lately (each verify step's kept drafts, and whether a plain step's one token
continued the stretch the row would have drafted from).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
from typing import NamedTuple

import numpy as np
import torch

from mstar.engine.resources.base import CGSlotSpec, EngineResourceInfo
from mstar.engine.resources.draft_verify.config import ProposeStep
from mstar.engine.resources.draft_verify.proposer import DraftProposer
from mstar.engine.resources.spec import NodeResourceSpec
from mstar.engine.resources.step import StepContext
from mstar.utils.h2d import PinnedStager

logger = logging.getLogger(__name__)

# what a row with no match drafts: any token is safe (the verify keeps what it emits exact)
_FILLER = 0
# how much of a row's past follow rate each new step leaves (a weight of 1 / (1 - DECAY) steps)
_DECAY = 0.8
# A row whose lookups keep finding nothing looks up less often on steps of no drafts: after
# its second miss in a row it skips 1 step, then 3, then 7 at most, so a chat row costs a
# lookup every 8 steps and a copy that starts mid-reply is found within 8 tokens.
_MAX_SKIP = 7


@dataclass
class NgramSpec(NodeResourceSpec):
    """Prompt-lookup drafting for a node's verify steps; ``DraftConfig.resources`` declares it."""
    max_drafts: int = 4
    min_ngram: int = 2
    max_ngram: int = 4

    @property
    def resource_class(self):
        return NgramProposer


class History:
    """One request's tokens, and where each of its n-grams last ended."""

    __slots__ = ("tokens", "fresh", "follow", "indexed", "ends", "hits", "trials", "predicted", "misses", "skip")

    def __init__(self):
        self.tokens: list[int] = []
        # its next verify step's input (the prefill's token) isn't in ``tokens`` yet
        self.fresh = True
        # where in ``tokens`` the next token is expected, while the tokens appended follow
        # the stretch the row last drafted from; -1 once they don't
        self.follow = -1
        # positions below this have their n-grams in ``ends``
        self.indexed = 0
        # n-gram (of any size) -> the position after its last earlier end
        self.ends: dict[tuple[int, ...], int] = {}
        # decayed counts of drafts kept (or, on a plain step, of a token that continued the
        # stretch) and of drafts tried, while the row had a stretch to draft from
        self.hits = 0.0
        self.trials = 0.0
        # the drafts its next step is expected to keep, as of its last commit
        self.predicted = 0.0
        # lookups in a row that found nothing, and the steps of no drafts left to skip one on
        self.misses = 0
        self.skip = 0


class NgramIndex:
    """Lookups over ``History`` objects. Host only; not thread-safe (``NgramProposer`` locks
    around it)."""

    def __init__(self, max_drafts: int, min_ngram: int, max_ngram: int):
        if not 1 <= min_ngram <= max_ngram:
            raise ValueError(f"n-gram sizes {min_ngram}..{max_ngram}")
        self.k = max_drafts
        self.sizes = tuple(range(min_ngram, max_ngram + 1))

    def open(self) -> History:
        return History()

    @staticmethod
    def extend(history: History, tokens: list[int]) -> None:
        """Append ``tokens``; the row stays on the stretch it follows while they are its."""
        h = history.tokens
        follow = history.follow
        h.extend(tokens)
        # the source starts before the old end, so each token compares with one before it
        if follow >= 0 and h[follow:follow + len(tokens)] == tokens:
            history.follow = follow + len(tokens)
        else:
            history.follow = -1

    def _index(self, history: History) -> None:
        """Insert the n-grams ending just before each position appended since the last lookup;
        a later end overwrites an earlier one."""
        h = history.tokens
        end = len(h)
        start = history.indexed
        ends = history.ends
        if end - start <= 16:
            # a step's few tokens: a plain loop costs less than setting up the bulk path
            for j in range(start, end):
                for n in self.sizes:
                    if j >= n:
                        ends[tuple(h[j - n:j])] = j
        else:
            ends.update(chain.from_iterable(
                zip(
                    zip(*(h[first - n + i:end - n + i] for i in range(n)), strict=True),
                    range(first, end), strict=True,
                )
                for n in self.sizes if (first := max(start, n)) < end
            ))
        history.indexed = end

    def locate(self, history: History) -> int:
        """Where the row's next token is expected in its history (``history.follow``): the
        stretch it follows, or the last earlier end of its longest suffix that has one; -1 when
        nothing matched."""
        best = history.follow
        if best < 0:
            h = history.tokens
            length = len(h)
            self._index(history)
            ends = history.ends
            for n in reversed(self.sizes):
                if length >= n:
                    found = ends.get(tuple(h[length - n:]))
                    if found is not None:
                        best = found
                        break
            history.follow = best
        return best

    def propose(self, history: History) -> list[int] | None:
        """``k`` drafts, or None when nothing matched. They follow the last earlier end of the
        longest suffix that has one; past the history's end they repeat the stretch since that
        end, as a sequence that just repeated would. A fresh row's next input (the prefill's
        token) isn't in its history yet: its drafts take that token to be the one the match
        continues with, and follow it."""
        best = self.locate(history)
        if best < 0:
            return None
        h = history.tokens
        length = len(h)
        skip = 1 if history.fresh else 0
        if best + skip + self.k <= length:
            return h[best + skip:best + skip + self.k]
        period = length - best
        return [h[best + (skip + i) % period] for i in range(self.k)]


class _Pending(NamedTuple):
    """A prompt's tokens on their way to the host: the copy, and the event after it."""
    host: torch.Tensor
    event: torch.Event | None


def expected_kept(history: History, k: int) -> float:
    """The drafts of ``k`` a row with a stretch to draft from is expected to keep: each one
    kept with the row's follow rate (Laplace-smoothed, so a row with no record keeps about one)
    given the ones before it were."""
    p = (history.hits + 1.0) / (history.trials + 2.0)
    return p * (1.0 - p ** k) / (1.0 - p)


class NgramProposer(DraftProposer):
    """``NgramIndex`` behind the resource lifecycle: a ``History`` per request handle, the
    prompts copied to the host at the prefill's commit, the verdicts noted by ``observe``, and
    each verify step's drafts staged into a device buffer in ``plan`` (per capture slot, so
    the next step's plan never writes the buffer a replay in flight reads).

    The maps are keyed by the worker's request handle and cleared at ingest and removal."""

    # its lookups say whether a row has drafts at all, and its verdicts how many stick
    predicts = True

    def __init__(self, device: torch.device, max_drafts: int, min_ngram: int, max_ngram: int):
        self._device = torch.device(device)
        self.max_drafts = max_drafts
        self._index = NgramIndex(max_drafts, min_ngram, max_ngram)
        self._lock = threading.Lock()
        # handle -> its history; -> prompt chunks not yet on the host; -> verdict rows not yet
        # taken in
        self._histories: dict[int, History] = {}
        self._prompts: dict[int, list[_Pending]] = {}
        self._verdicts: dict[int, list[np.ndarray]] = {}
        self._stager = PinnedStager(torch.int64)
        self._pin = self._device.type == "cuda" and torch.cuda.is_available()
        # (bucket, slot) -> the drafts buffer a captured graph reads
        self._buffers: dict[tuple, torch.Tensor] = {}
        self._cg_rows = 0
        self._current: torch.Tensor | None = None
        # rows -> the drafts of a step that runs none
        self._no_drafts: dict[int, torch.Tensor] = {}
        # a pre-plan's drafts, staged for the step it was planned for: (key, drafts)
        self._staged: tuple[tuple, torch.Tensor] | None = None
        self.rows_proposed = 0
        self.rows_matched = 0

    @classmethod
    def build(cls, spec: NgramSpec, info: EngineResourceInfo) -> NgramProposer:
        return cls(info.device, spec.max_drafts, spec.min_ngram, spec.max_ngram)

    # request lifecycle

    def ingest_request(self, rid: int, overrides=None):
        del overrides
        with self._lock:
            if rid in self._histories:
                return  # ingested once per partition; keep what it has
            self._histories[rid] = self._index.open()
            self._prompts.pop(rid, None)
            self._verdicts.pop(rid, None)

    def ingest_capture_row(self, rid: int):
        return

    def remove_request(self, rid: int):
        with self._lock:
            self._histories.pop(rid, None)
            self._prompts.pop(rid, None)
            self._verdicts.pop(rid, None)

    def reset_request(self, rid: int, free: bool = False):
        del free
        self.remove_request(rid)

    def cleanup(self):
        if self.rows_proposed:
            logger.info("n-gram drafts: %d of %d verify rows matched", self.rows_matched, self.rows_proposed)

    # history

    def observe(self, rid: int, row: np.ndarray) -> None:
        with self._lock:
            if rid in self._histories:
                self._verdicts.setdefault(rid, []).append(row)

    def _take_in(self, history: History, rid: int) -> None:
        """Append the request's prompt and the verdicts that came in since its last step, and
        count how far each followed the stretch the row had then. Under the lock."""
        for pending in self._prompts.pop(rid, ()):
            if pending.event is not None:
                pending.event.synchronize()  # copied behind the prefill, which has run
            history.tokens.extend(pending.host.tolist())
        for verdict in self._verdicts.pop(rid, ()):
            # [accepted, input, t_0 .. t_k]: it emitted t_0 .. t_accepted; a fresh row takes in
            # its input too (the prefill's token)
            accepted = int(verdict[0])
            drafts = len(verdict) - 3
            following = history.follow >= 0
            self._index.extend(history, verdict[1 if history.fresh else 2:3 + accepted].tolist())
            history.fresh = False
            if following:
                # a verify step tried its drafts until one failed; a plain step's token tried
                # the stretch's next one
                kept = accepted if drafts else float(history.follow >= 0)
                history.hits = _DECAY * history.hits + kept
                history.trials = _DECAY * history.trials + kept + (kept < max(drafts, 1))

    # step

    @property
    def supports_preplan(self):
        return True

    @property
    def force_double_buffer(self):
        # each verify step's drafts go into a device buffer its replay reads
        return True

    def clear_preplan(self):
        self._staged = None

    def build_accelerator_graph_buffers(self, slots: list[CGSlotSpec], max_bs: int, max_seq_len: int) -> None:
        del slots, max_seq_len
        self._cg_rows = max(self._cg_rows, max_bs)

    @staticmethod
    def _plan_key(ctx: StepContext) -> tuple:
        lease = ctx.slot_lease
        return tuple(ctx.padded_request_ids), None if lease is None else (lease.bucket, lease.slot)

    def plan(self, step: ProposeStep, ctx: StepContext) -> None:
        if not step.drafts:
            if step.verify and not ctx.is_preplan:
                # an adaptive walk's plain step: a block of no drafts
                rows = len(ctx.padded_request_ids)
                empty = self._no_drafts.get(rows)
                if empty is None:
                    empty = self._no_drafts[rows] = torch.empty(rows, 0, dtype=torch.int64, device=self._device)
                self._current = empty
            return  # a step that adds history does so at its commit
        if step.drafts != self.max_drafts:
            raise ValueError(f"a verify step of {step.drafts} drafts from an n-gram proposer of {self.max_drafts}")
        key = self._plan_key(ctx)
        if ctx.is_preplan:
            # The drafts need the rows' last verdicts, which a step planned ahead of them hasn't
            # read yet: `correct_preplan` stages them once it has, before the event the replay
            # waits on, so nothing here waits for the step in flight.
            self._staged = (key, None)
            return
        if self._staged is not None:
            staged, self._staged = self._staged, None
            if staged[0] == key and staged[1] is not None:
                self._current = staged[1]
                return
        self._current = self._stage(ctx, self._draft_rows(ctx))

    corrects_preplan = True

    def correct_preplan(self, verdicts, step: ProposeStep, ctx: StepContext) -> bool:
        """Stage the drafts of the verify step planned ahead, now that the runner has read its
        rows' last verdicts (or, for a row whose last step drafted nothing, reads it here)."""
        del verdicts
        staged = self._staged
        if not step.drafts or staged is None or staged[1] is not None:
            return True
        self._staged = (staged[0], self._stage(ctx, self._draft_rows(ctx)))
        return True

    def _draft_rows(self, ctx: StepContext) -> np.ndarray:
        """``[rows, k]``: each padded row's drafts, filler where nothing matched (and on a
        capture's or a replay's padding rows), from every verdict before the step."""
        padded = ctx.padded_request_ids
        drafts = np.full((len(padded), self.max_drafts), _FILLER, dtype=np.int64)
        if ctx.capture:
            return drafts
        if self._verifier is not None:
            # the runner's rewinds read most of these verdicts already; a row whose last step
            # drafted nothing is read here. Outside the lock (reading one calls `observe`)
            self._verifier.harvest(padded)
        with self._lock:
            histories, waiting = self._histories, self._prompts or self._verdicts
            at: list[int] = []
            guesses: list[int] = []
            proposed = 0
            for i, rid in enumerate(padded):
                history = histories.get(rid)
                if history is None:
                    continue  # a replay's padding row: filler
                if waiting:
                    self._take_in(history, rid)
                guess = self._index.propose(history)
                proposed += 1
                if guess is not None:
                    at.append(i)
                    guesses += guess
            if at:
                drafts[at] = np.array(guesses, dtype=np.int64).reshape(len(at), self.max_drafts)
            self.rows_proposed += proposed
            self.rows_matched += len(at)
        return drafts

    def _stage(self, ctx: StepContext, drafts: np.ndarray) -> torch.Tensor:
        """``drafts`` on the device, in the slot's buffer under a lease (one H2D on the current
        stream, the plan stream for a step planned ahead)."""
        lease = ctx.slot_lease
        if lease is None:
            return self._stager.to_device(drafts, self._device)
        rows = drafts.shape[0]
        out = self._buffer(lease, rows)
        self._stager.copy_(out, drafts.reshape(-1), pad_value=_FILLER)
        return out[:rows]

    def _buffer(self, lease, rows: int) -> torch.Tensor:
        key = (lease.bucket, lease.slot)
        buf = self._buffers.get(key)
        if buf is None:
            size = max(rows, self._cg_rows, lease.bucket.bs if lease.bucket is not None else 0)
            buf = self._buffers[key] = torch.full(
                (size, self.max_drafts), _FILLER, dtype=torch.int64, device=self._device,
            )
        if buf.shape[0] < rows:
            raise RuntimeError(f"n-gram drafts: slot {key} holds {buf.shape[0]} rows, a step needs {rows}")
        return buf

    def commit(self, step: ProposeStep, ctx: StepContext) -> None:
        if not step.history or ctx.capture:
            return
        # Behind the forward that read them, on its stream: a verify step of the request
        # waits for its prefill anyway, so the copy has landed by the time it is read.
        with self._lock:
            for rid, ids in step.history.items():
                if rid not in self._histories or ctx.is_padding_row(rid):
                    continue
                ids = ids.reshape(-1)
                if ids.device.type == "cpu":
                    pending = _Pending(ids.clone(), None)
                else:
                    host = torch.empty(ids.shape, dtype=ids.dtype, pin_memory=self._pin)
                    host.copy_(ids, non_blocking=True)
                    event = torch.Event(device=ids.device)
                    event.record(torch.accelerator.current_stream(ids.device))
                    pending = _Pending(host, event)
                self._prompts.setdefault(rid, []).append(pending)

    def predict(self, request_ids: Sequence[int], drafts: int) -> None:
        """At a verify-walk step's commit: each row's expected kept drafts for its next step.
        A step that drafted looked its rows up in its plan, from every verdict before it; a step
        of no drafts takes those verdicts in and looks up now. The next step's drafts are
        decided from these, so they change only here."""
        k = self.max_drafts
        with self._lock:
            histories, waiting = self._histories, self._prompts or self._verdicts
            for rid in request_ids:
                history = histories.get(rid)
                if history is None:
                    continue
                if not drafts:
                    if history.skip:
                        # no stretch, and none found lately: its verdicts wait for a later lookup
                        history.skip -= 1
                        history.predicted = 0.0
                        continue
                    if waiting:
                        self._take_in(history, rid)
                    if self._index.locate(history) < 0:
                        history.misses += 1
                        history.skip = min((1 << history.misses >> 1) - 1, _MAX_SKIP)
                    else:
                        history.misses = 0
                elif history.follow >= 0:
                    # a verify step's lookup found a stretch
                    history.misses = history.skip = 0
                history.predicted = expected_kept(history, k) if history.follow >= 0 else 0.0

    def expected_gain(self, request_ids: Sequence[int]) -> float:
        """The drafts these rows are expected to keep at their next step, summed."""
        with self._lock:
            histories = self._histories
            return sum(h.predicted for rid in request_ids if (h := histories.get(rid)) is not None)

    # forward side

    def propose(self, request_ids: Sequence[int], rows: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self._current is None:
            raise RuntimeError("NgramProposer.propose outside a planned verify step")
        return self._current[:rows], None
