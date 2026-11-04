"""Lanes: the steps of nodes placed off a worker's main chain, each lane on a
stream of its own.

A worker's GPU runs any number of lanes. Lane 0 is the chain (``CHAIN_LANE``):
step N on the device, N+1 speculated behind it. A step on the chain costs a
pass of the loop and holds the next step behind its kernels on the main stream,
which is what a codec window does to the decode steps around it. A node whose
node group in the serving config sets ``lane: <id>`` runs every walk of its on
that lane instead: the scheduler hands the lane only its nodes' keys
(``MicroScheduler.lane_of``), its steps run on the GPU thread between two chain
steps, and they land on the lane's stream while the chain carries on. Submit,
the refused-step tail and postprocess are the chain's own
(``Worker._submit_step``, ``Worker._settle_step_rows``, ``Worker._finish_step``),
keyed by lane.

Each lane has one step in flight; steps that come ready meanwhile batch into
the next one. ``EngineManager.build`` decides which nodes a lane can take and
refuses what it can't serve: a step here is admitted and planned inline, never
speculated, planned ahead or lockstep.

The engine owns each lane's stream, made at load, and puts on it everything a
lane issues (``Engine.on_lane``): the lane's graph captures, its steps, the
collate of their inputs and the teardown of what its nodes hold. A resource
a lane holds is that lane's alone (``Engine._lane_contexts``): the step admits,
plans and commits it on the GPU thread, as the chain's inline steps do theirs;
with one step in flight, the next one plans only after this one has landed, so
there is nothing to double-buffer, and it leaves the chain's staged pre-plan
alone (``StepRunner._beside_stage``).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, wait
from typing import TYPE_CHECKING

from mstar.utils.client_errors import client_error_text

if TYPE_CHECKING:
    from mstar.utils.threads import ThreadPool
    from mstar.worker.micro_scheduler import ScheduledBatch
    from mstar.worker.worker import PendingBatch, Worker


logger = logging.getLogger(__name__)

# How long the idle main loop sleeps between looks at several busy lanes
# once one has issued its step and only the device is left to wait for
_LAND_POLL_S = 50e-6


class Lane:
    """A lane off the chain, and the one step in flight on it."""

    def __init__(self, worker: Worker, lane: int):
        self._worker = worker
        self.lane = lane
        self.pending: PendingBatch | None = None

    def _on_lane(self):
        return self._worker.engine_manager.engine.on_lane(self.lane)

    def _fail(self, batch: ScheduledBatch, exc: Exception) -> None:
        """Fail a step's rows alone: the chain's rows in flight have nothing to
        do with it, so this doesn't go through the main loop's handler."""
        worker = self._worker
        worker._clear_in_flight_flag(batch)
        err, _ = client_error_text(exc, f"Worker {worker.worker_id} error in a step on lane {self.lane}", log=logger)
        worker._fail_requests(dict.fromkeys(batch.request_to_worker_graph, err))

    def execute(self, batch: ScheduledBatch, node_batch, plan_future):
        """The GPU thread's step, on the lane's stream, with nothing to wait for
        on the main one: the streamed items it reads had landed before they
        were routed to it (postprocess waits for the step that made them), and
        ``launch`` collated its inputs on this stream."""
        with self._on_lane():
            outputs = self._worker._execute_on_gpu_thread(batch, node_batch, plan_future)
        if outputs.host_rows is not None or outputs.host_spans is not None:
            # the pinned ring they stage through is reused by the chain's next
            # steps before this one lands (refused at startup where declared)
            raise RuntimeError(
                f"{batch.node_name} {batch.graph_walk!r} returned stop buffers or output spans on lane {self.lane}"
            )
        return outputs

    def launch(self, gpu_executor: ThreadPool) -> None:
        """Queue the lane's next step on the GPU thread, if the lane is free."""
        if self.pending is not None:
            return
        worker = self._worker
        batch = worker.scheduler.get_next_batch(worker.request_state, lane=self.lane)
        if batch is None:
            return
        try:
            # its inputs stack here, so the step is ordered behind them alone
            with self._on_lane():
                node_batch = worker._build_executing_batch(batch)
            self.pending = worker._submit_step(
                gpu_executor, batch, node_batch, lane=self.lane, execute=self.execute,
            )
        except Exception as exc:
            self._fail(batch, exc)
            return
        worker._hold_rows(self.lane, batch.request_to_worker_graph)

    @staticmethod
    def _landed(step: PendingBatch) -> bool:
        if not step.future.done():
            return False
        event = step.node_batch.completion_event
        return event is None or event.query()

    def finish(self, block: bool = False) -> None:
        """Postprocess the step in flight once it has landed; ``block`` waits.

        Never before the step has landed: postprocess frees its inputs first,
        and those are main-stream allocations a main-stream step could reuse
        while this one still reads them.
        """
        step = self.pending
        if step is None or not (block or self._landed(step)):
            return
        worker = self._worker
        try:
            outputs = step.future.result()
            worker._wait_for_step(step)
            worker._clear_in_flight_flag(step.batch)
            worker._count_turn(step)
            if not worker._settle_step_rows(step, outputs):
                worker._finish_step(step, outputs)
        except Exception as exc:
            self._fail(step.batch, exc)
        finally:
            self.pending = None
            worker._hold_rows(self.lane, ())


def finish_first_to_land(busy: Sequence[Lane]) -> None:
    """With the chain idle, postprocess the step of whichever of ``busy``'s
    lanes lands first. One lane blocks on its step; several block until the
    GPU thread has issued one of their steps and are then polled, so one
    lane's long step doesn't hold back another's that has landed."""
    if len(busy) == 1:
        busy[0].finish(block=True)
        return
    wait([lane.pending.future for lane in busy], return_when=FIRST_COMPLETED)
    while not any(Lane._landed(lane.pending) for lane in busy):
        time.sleep(_LAND_POLL_S)
    for lane in busy:
        lane.finish()
