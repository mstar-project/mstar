"""Steps that run beside a worker's main chain, on a stream of their own.

The main loop runs one chain: step N on the device, N+1 speculated behind it. A
step on that chain costs a pass of the loop and holds the next step behind its
kernels on the main stream, which is what a codec window does to the decode
steps around it. A walk that declares ``NodeSubmodule.side_stream_walks`` takes
neither: it leaves the chain's fairness check, runs on the GPU thread between
two chain steps, and lands on a second stream while the chain carries on.

One step in flight here at a time; windows that come ready meanwhile batch into
the next one. ``EngineManager.side_stream_keys`` decides which walks qualify,
and leaves out what this can't serve: a step here is never speculated, planned
ahead or lockstep.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import torch

from mstar.utils.client_errors import client_error_text

if TYPE_CHECKING:
    from mstar.utils.threads import ThreadPool
    from mstar.worker.micro_scheduler import ScheduledBatch
    from mstar.worker.worker import PendingBatch, Worker


logger = logging.getLogger(__name__)


class SideStepLane:
    """The worker's side stream and the one step in flight on it."""

    def __init__(self, worker: Worker, keys: list[tuple[str, str]]):
        self._worker = worker
        self.keys = keys
        # made on the GPU thread, the only thread that launches on it
        self._stream: torch.Stream | None = None
        self.pending: PendingBatch | None = None

    @property
    def rids(self) -> set[int]:
        """Rows in flight here, for the worker's teardown checks."""
        return set(self.pending.batch.request_to_worker_graph) if self.pending is not None else set()

    def owns(self, step: PendingBatch) -> bool:
        return step is self.pending

    def _fail(self, batch: ScheduledBatch, exc: Exception) -> None:
        """Fail a step's rows alone: the chain's rows in flight have nothing to
        do with it, so this doesn't go through the main loop's handler."""
        worker = self._worker
        worker._clear_in_flight_flag(batch)
        err, _ = client_error_text(exc, f"Worker {worker.worker_id} error in a side step", log=logger)
        worker._fail_requests(dict.fromkeys(batch.request_to_worker_graph, err))

    def execute(self, batch: ScheduledBatch, node_batch, plan_future) -> dict:
        """The GPU thread's step, on the side stream.

        The step's inputs reached the device on the main stream: the main thread
        clones each streamed item and stacks each window there, behind whatever
        ran then. The fence orders the step behind all of it, so it starts once
        the chain step ahead of it ends, and the steps launched after this one
        run beside it.
        """
        worker = self._worker
        if self._stream is None:
            self._stream = torch.Stream(device=worker.device)
        fence = torch.Event(device=worker.device)
        fence.record(torch.accelerator.current_stream(worker.device))
        self._stream.wait_event(fence)
        with self._stream:
            return worker._execute_on_gpu_thread(batch, node_batch, plan_future)

    def launch(self, gpu_executor: ThreadPool) -> None:
        """Queue the next step on the GPU thread, if the stream is free."""
        if self.pending is not None:
            return
        worker = self._worker
        for key in self.keys:
            batch = worker.scheduler.get_next_batch(worker.request_state, target=key)
            if batch is None:
                continue
            try:
                node_batch = worker._build_executing_batch(batch)
                worker._set_in_flight_flag(batch, True)
                submitted_at = time.perf_counter() if worker._metrics is not None else 0.0
                future = worker._submit_gpu_step(gpu_executor, batch, node_batch, None, execute=self.execute)
            except Exception as exc:
                self._fail(batch, exc)
                return
            worker.wakeup_event.register_future(future)
            self.pending = worker._side_pending(batch, node_batch, future, submitted_at)
            worker._in_flight_rids |= self.rids
            return

    def _landed(self) -> bool:
        step = self.pending
        if not step.future.done():
            return False
        event = step.node_batch.completion_event
        return event is None or event.query()

    def finish(self, block: bool = False) -> None:
        """Postprocess the step in flight once it has landed; ``block`` waits.

        Never before the step has landed: postprocess frees its inputs first,
        and those are main-stream allocations a main-stream step could reuse
        while this one still reads them. A step here holds no resource, so
        there is no admit to have refused it.
        """
        step = self.pending
        if step is None or not (block or self._landed()):
            return
        worker = self._worker
        rids = self.rids
        try:
            outputs = step.future.result()
            worker._wait_for_step(step)
            worker._clear_in_flight_flag(step.batch)
            if step.node_batch.failed_requests:
                failed = dict(step.node_batch.failed_requests)
                worker._drop_failed_rids(step, outputs, failed)
                worker._fail_requests(failed)
            worker.scheduler.clear_batch_limit(
                step.node_name, step.graph_walk, len(step.batch.request_to_worker_graph),
                walks=step.batch.rids_by_walk(),
            )
            with worker._span("worker.postprocess_side", cpu="worker.postprocess_side.cpu"):
                worker._postprocess_batch(step, outputs)
            if worker._metrics is not None:
                worker._metrics.step_done(step.node_name, step.graph_walk, step.node_batch.inputs, step.submitted_at)
        except Exception as exc:
            self._fail(step.batch, exc)
        finally:
            self.pending = None
            worker._in_flight_rids -= rids
