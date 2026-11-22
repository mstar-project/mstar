"""Weight-control steps at the worker's settled StepPipeline boundary.

The conductor sends the same ordered control step to every rank. This object
records the inbound step; StepPipeline alone decides when it is safe to run it.
"""
from __future__ import annotations

import logging

from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    WeightControl,
    WeightControlAck,
    WeightOp,
)
from mstar.worker.lockstep import Lockstep
from mstar.worker.request_lifecycle import RequestLifecycle
from mstar.worker.services import WorkerServices

logger = logging.getLogger(__name__)


class WeightOps:
    def __init__(self, services: WorkerServices, lifecycle: RequestLifecycle, lockstep: Lockstep):
        self.worker_id = services.worker_id
        self._runtime = services.graph_runtime
        self._engine = services.engine_manager.engine
        self._communicator = services.communicator
        self._rebuilds = services.rebuilds
        self._lifecycle = lifecycle
        self._lockstep = lockstep
        self.pending: WeightControl | None = None
        self.weight_version: str | None = None
        self.weight_epoch = 0
        self.dirty = False
        self._logged_epoch = -1

    def enqueue(self, op: WeightControl) -> None:
        # One step at a time: the conductor waits for every rank's ack before
        # sending the next step. A duplicate epoch replaces nothing newer.
        if self.pending is None or op.epoch >= self.pending.epoch:
            self.pending = op

    def idle_blockers(self, in_flight_rids: set[int]) -> list[str]:
        requests, parked_routes, staged_specs = self._runtime.idle_counts()
        blockers = []
        if requests:
            blockers.append(f"{requests} request(s) still held")
        if parked_routes:
            blockers.append(f"{parked_routes} step output(s) not yet sent")
        if staged_specs:
            blockers.append(f"{staged_specs} staged speculative ingest(s)")
        if in_flight_rids:
            blockers.append("a step in flight")
        teardowns = self._lifecycle.pending_teardown_count()
        if teardowns:
            blockers.append(f"{teardowns} teardown(s) not yet applied")
        follows = self._lockstep.pending_committed_steps()
        if follows:
            blockers.append(f"{follows} committed TP step(s) not yet run")
        if self._rebuilds.copies_queued:
            blockers.append("rebuild copy still in flight")
        return blockers

    def process(self, in_flight_rids: set[int]) -> bool:
        """Run one step when no GPU/preplan step remains. True when acknowledged.

        A quiesce with pending work leaves the op in place, allowing StepPipeline
        to keep driving already admitted requests and follower steps.
        """
        op = self.pending
        if op is None:
            return False
        blockers = self.idle_blockers(in_flight_rids)
        if op.op == WeightOp.QUIESCE and blockers:
            if self._logged_epoch != op.epoch:
                self._logged_epoch = op.epoch
                logger.info("Worker %s: waiting to pause: %s", self.worker_id, "; ".join(blockers))
            return False
        self.pending = None
        error = None
        if op.op in (WeightOp.QUIESCE, WeightOp.VERIFY):
            pass
        elif blockers:
            error = "not idle: " + "; ".join(blockers)
        elif op.op == WeightOp.PREPARE:
            try:
                self._engine.check_weight_reload(op.model_path)
            except Exception as exc:  # noqa: BLE001 - return a per-rank error to conductor
                logger.warning("Worker %s: checkpoint %s refused: %s", self.worker_id, op.model_path, exc)
                error = f"{type(exc).__name__}: {exc}"
        elif op.op == WeightOp.APPLY:
            self.dirty = True
            try:
                self._engine.reload_weights(op.model_path, op.weight_version)
            except Exception as exc:  # noqa: BLE001 - partial copy must leave the rank dirty
                logger.exception("Worker %s: reload from %s failed", self.worker_id, op.model_path)
                error = f"{type(exc).__name__}: {exc}"
            else:
                self.weight_version = op.weight_version
                self.weight_epoch = op.epoch
                self.dirty = False
                # outputs from here on carry it; nothing is in flight to mix
                self._runtime.set_weight_version(op.weight_version)
                logger.info(
                    "Worker %s: weights reloaded from %s as version %r",
                    self.worker_id, op.model_path, op.weight_version,
                )
        else:
            error = f"unknown weight control step {op.op!r}"
        self._communicator.send("conductor", ConductorMessage(
            message_type=ConductorMessageType.WEIGHT_CONTROL_ACK,
            body=WeightControlAck(
                epoch=op.epoch, worker_id=self.worker_id, op=op.op, error=error,
                weight_version=self.weight_version, weight_epoch=self.weight_epoch,
                dirty=self.dirty,
            ),
        ))
        return True
