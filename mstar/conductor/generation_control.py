"""Pausing generation and reloading weights in place, as the conductor sees it.

States, and what moves between them::

    serving --pause--> draining --no request left--> quiescing --every rank idle--> paused
    paused --update--> preparing --every rank checked--> applying --every rank copied--> paused
    paused --continue--> verifying --every rank on the committed version--> serving

Any state but ``serving`` refuses new requests (the conductor answers them with a
503). ``pause`` with mode ``abort`` also fails the requests in flight; ``wait`` lets
them finish. Each protocol step goes to every rank of every worker, and each rank
answers for itself, so a TP follower is asked directly rather than through rank 0.

The weight version commits only when every rank copied the checkpoint. A refusal
while checking leaves every rank on the version it had. A failure or a missing
answer while copying leaves the deployment paused, since some ranks may hold the
new weights and some not. ``continue`` resumes only when every rank reports the
committed version: its label and the epoch of the apply step that copied it, so a
label reused for other weights, or an apply that never ran on a rank or ran after
its step timed out, keeps the deployment paused. A pause that arrives while that
check is still running waits for the ranks to answer, and the deployment stays
paused.

A rank that does not answer within ``timeout_s`` fails the step it was in. Its late
answer is ignored: each step carries its own epoch.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from mstar.api_server.request_types import APIServerMessage, GenerationControlResult
from mstar.utils.ipc_format import (
    ContinueGeneration,
    GetGenerationState,
    PauseGeneration,
    UpdateWeightsFromDisk,
    WeightControl,
    WeightControlAck,
    WeightOp,
    WorkerMessage,
    WorkerMessageType,
)

logger = logging.getLogger(__name__)

PAUSE_MODES = ("abort", "wait")


class GenerationState(str, Enum):
    SERVING = "serving"
    DRAINING = "draining"
    QUIESCING = "quiescing"
    PAUSED = "paused"
    PREPARING = "preparing"
    APPLYING = "applying"
    VERIFYING = "verifying"


@dataclass(slots=True)
class _Step:
    """One protocol step sent to every worker and not yet answered by all."""

    op: WeightOp
    epoch: int
    waiting: set[str]
    deadline: float
    acks: dict[str, WeightControlAck] = field(default_factory=dict)
    timed_out: bool = False


class GenerationControl:
    """The conductor's half of pause, continue and in-place weight updates.

    ``workers`` are every worker rank; ``send(peer, message)`` is the conductor's
    communicator. ``reject_queued()`` fails the requests waiting for admission and
    ``abort_in_flight()`` the admitted ones; both are the conductor's. ``clock`` is
    a monotonic time.
    """

    def __init__(
        self,
        workers: list[str],
        send: Callable[[str, object], None],
        reject_queued: Callable[[], None],
        abort_in_flight: Callable[[], None],
        timeout_s: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._workers = list(workers)
        self._send = send
        self._reject_queued = reject_queued
        self._abort_in_flight = abort_in_flight
        self._timeout_s = timeout_s
        self._clock = clock

        self.state = GenerationState.SERVING
        self.mode: str | None = None
        # None is the checkpoint the deployment started with. The epoch of the
        # apply step that committed it (0 for the startup checkpoint) is what
        # each rank must also report: a label can be reused for other weights.
        self.weight_version: str | None = None
        self._weight_epoch = 0
        # every rank confirmed idle since the pause began
        self._quiesced = False
        self._epoch = 0
        self._step: _Step | None = None
        # the latest answer per worker, for reporting
        self._ranks: dict[str, str] = {}
        self._rank_versions: dict[str, str | None] = {}
        # op ids waiting on the pause to finish, and on the running update or continue
        self._pause_waiters: list[str] = []
        self._update: UpdateWeightsFromDisk | None = None
        self._continue_op: str | None = None

    @property
    def paused(self) -> bool:
        return self.state is not GenerationState.SERVING

    # ---- client operations ----------------------------------------------------

    def pause(self, body: PauseGeneration) -> None:
        if body.mode not in PAUSE_MODES:
            self._answer(body.op_id, 400, f"mode must be one of {list(PAUSE_MODES)}; got {body.mode!r}")
            return
        if self.state is GenerationState.SERVING:
            logger.info("Pausing generation (mode %s)", body.mode)
            self.state = GenerationState.DRAINING
            self.mode = body.mode
            self._quiesced = False
            self._pause_waiters.append(body.op_id)
            self._reject_queued()
            if body.mode == "abort":
                self._abort_in_flight()
            return
        if self.state is GenerationState.DRAINING:
            self._pause_waiters.append(body.op_id)
            if body.mode == "abort" and self.mode != "abort":
                logger.info("Pause escalated from wait to abort")
                self.mode = "abort"
                self._abort_in_flight()
            return
        if self.state is GenerationState.QUIESCING:
            self._pause_waiters.append(body.op_id)
            return
        if self.state is GenerationState.PAUSED and not self._quiesced:
            # the last confirmation timed out: ask every rank again
            self._pause_waiters.append(body.op_id)
            self._start(WeightOp.QUIESCE, GenerationState.QUIESCING)
            return
        if self.state is GenerationState.VERIFYING:
            # answering 200 here reports paused, then a successful verify resumes
            self._pause_waiters.append(body.op_id)
            return
        self._answer(body.op_id, 200)

    def update(self, body: UpdateWeightsFromDisk) -> None:
        if self.state is not GenerationState.PAUSED or not self._quiesced:
            self._answer(body.op_id, 409, (
                f"weights can be updated only while paused with every rank idle; "
                f"generation is {self.state.value}"
            ))
            return
        for name in ("model_path", "weight_version"):
            value = getattr(body, name)
            if not isinstance(value, str) or not value:
                self._answer(body.op_id, 400, f"{name} must be a non-empty string")
                return
        logger.info("Updating weights to version %r from %s", body.weight_version, body.model_path)
        self._update = body
        self._start(
            WeightOp.PREPARE, GenerationState.PREPARING,
            model_path=body.model_path, weight_version=body.weight_version,
        )

    def resume(self, body: ContinueGeneration) -> None:
        if self.state is GenerationState.SERVING:
            self._answer(body.op_id, 200)
            return
        if self.state is not GenerationState.PAUSED:
            self._answer(body.op_id, 409, f"generation is {self.state.value}; try again once it is paused")
            return
        self._continue_op = body.op_id
        self._start(WeightOp.VERIFY, GenerationState.VERIFYING)

    def report(self, body: GetGenerationState) -> None:
        self._answer(body.op_id, 200)

    # ---- worker answers and the run loop ----------------------------------------

    def ack(self, body: WeightControlAck) -> None:
        step = self._step
        if step is None or body.epoch != step.epoch:
            logger.info(
                "Ignoring %s answer from %s for epoch %d (current %s)",
                body.op, body.worker_id, body.epoch, None if step is None else step.epoch,
            )
            return
        if body.worker_id not in step.waiting:
            return
        step.waiting.discard(body.worker_id)
        step.acks[body.worker_id] = body
        if not step.waiting:
            self._finish(step)

    def tick(self, live_requests: int) -> None:
        """Once per conductor loop: start the idle check once every request is
        gone, and give up on a step past its deadline."""
        if self.state is GenerationState.DRAINING and live_requests == 0:
            logger.info("No request left; asking every rank to confirm it is idle")
            self._start(WeightOp.QUIESCE, GenerationState.QUIESCING)
        step = self._step
        if step is not None and step.waiting and self._clock() >= step.deadline:
            logger.error(
                "Generation control: no %s answer from %s within %.0f s",
                step.op.value, sorted(step.waiting), self._timeout_s,
            )
            step.timed_out = True
            self._finish(step)

    # ---- steps ---------------------------------------------------------------------

    def _start(
        self, op: WeightOp, state: GenerationState,
        model_path: str = "", weight_version: str | None = None,
    ) -> None:
        self._epoch += 1
        self.state = state
        step = _Step(
            op=op, epoch=self._epoch, waiting=set(self._workers),
            deadline=self._clock() + self._timeout_s,
        )
        self._step = step
        message = WorkerMessage(
            message_type=WorkerMessageType.WEIGHT_CONTROL,
            body=WeightControl(
                op=op.value, epoch=step.epoch,
                model_path=model_path, weight_version=weight_version,
            ),
        )
        for worker_id in self._workers:
            self._send(worker_id, message)
        if not step.waiting:
            self._finish(step)

    def _record(self, step: _Step) -> dict[str, str]:
        """Each worker's result for ``step``: ``"ok"`` or why not."""
        results: dict[str, str] = {}
        for worker_id in self._workers:
            ack = step.acks.get(worker_id)
            if ack is None:
                results[worker_id] = f"no answer within {self._timeout_s:.0f} s"
                continue
            results[worker_id] = "ok" if ack.error is None else ack.error
            self._rank_versions[worker_id] = None if ack.dirty else ack.weight_version
        self._ranks = results
        return results

    def _finish(self, step: _Step) -> None:
        self._step = None
        results = self._record(step)
        failed = {w: r for w, r in results.items() if r != "ok"}
        timeout_status = 504 if step.timed_out and all(w not in step.acks for w in failed) else None
        if step.op is WeightOp.QUIESCE:
            self.state = GenerationState.PAUSED
            self._quiesced = not failed
            if failed:
                self._answer_pause(timeout_status or 500, self._detail("not every rank confirmed idle", failed))
            else:
                logger.info("Generation paused; every rank is idle")
                self._answer_pause(200)
            return
        if step.op is WeightOp.PREPARE:
            if failed:
                self.state = GenerationState.PAUSED
                logger.warning("Weight update refused: %s", failed)
                self._answer_update(
                    timeout_status or 400,
                    self._detail("checkpoint refused, weights unchanged", failed),
                )
                return
            update = self._update
            self._start(
                WeightOp.APPLY, GenerationState.APPLYING,
                model_path=update.model_path, weight_version=update.weight_version,
            )
            return
        if step.op is WeightOp.APPLY:
            self.state = GenerationState.PAUSED
            if failed:
                logger.error(
                    "Weight update to %r failed on %s; staying paused",
                    self._update.weight_version, failed,
                )
                self._answer_update(timeout_status or 500, self._detail(
                    "weight update failed; ranks may hold different weights, so "
                    "generation stays paused until an update succeeds", failed,
                ))
                return
            self.weight_version = self._update.weight_version
            self._weight_epoch = step.epoch
            logger.info("Weight version %r committed on every rank", self.weight_version)
            self._answer_update(200)
            return
        # VERIFY: same label and same copy on every rank. A rank whose apply
        # never ran, or ran after its step timed out, reports another epoch.
        mismatched = {
            w: (
                f"holds {self._rank_versions.get(w)!r} copied at update step "
                f"{step.acks[w].weight_epoch}, committed step {self._weight_epoch}"
            )
            for w in self._workers
            if w not in failed and (
                step.acks[w].dirty
                or step.acks[w].weight_version != self.weight_version
                or step.acks[w].weight_epoch != self._weight_epoch
            )
        }
        op_id, self._continue_op = self._continue_op, None
        if failed or mismatched:
            self.state = GenerationState.PAUSED
            self._answer(op_id, timeout_status or 409, self._detail(
                f"not every rank holds the committed version {self.weight_version!r}",
                {**failed, **mismatched},
            ))
            self._answer_pause(200)
            return
        if self._pause_waiters:
            self.state = GenerationState.PAUSED
            logger.info("Pause arrived while resume was still being verified; staying paused")
            self._answer(op_id, 409, "generation was paused again before it resumed")
            self._answer_pause(200)
            return
        logger.info("Resuming generation on weight version %r", self.weight_version)
        self.state = GenerationState.SERVING
        self.mode = None
        self._quiesced = False
        self._answer(op_id, 200)

    @staticmethod
    def _detail(summary: str, per_rank: dict[str, str]) -> str:
        return summary + ": " + "; ".join(f"{w}: {r}" for w, r in sorted(per_rank.items()))

    # ---- answers -----------------------------------------------------------------

    def _answer_pause(self, status: int, error: str | None = None) -> None:
        waiters, self._pause_waiters = self._pause_waiters, []
        for op_id in waiters:
            self._answer(op_id, status, error)

    def _answer_update(self, status: int, error: str | None = None) -> None:
        update, self._update = self._update, None
        self._answer(update.op_id, status, error)

    def _answer(self, op_id: str, status: int, error: str | None = None) -> None:
        self._send("api_server", APIServerMessage(
            message_type="generation_control_result",
            body=GenerationControlResult(
                op_id=op_id, status=status, state=self.state.value, paused=self.paused,
                weight_version=self.weight_version, error_message=error,
                ranks=dict(self._ranks), rank_versions=dict(self._rank_versions),
            ),
        ))
