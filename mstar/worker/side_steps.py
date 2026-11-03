"""Steps that run beside a worker's main chain, on a stream of their own.

The main loop runs one chain (``CHAIN_LANE``): step N on the device, N+1
speculated behind it. A step on that chain costs a pass of the loop and holds
the next step behind its kernels on the main stream, which is what a codec
window does to the decode steps around it. A node whose node group sets
``side_stream: true`` runs its ``side_stream_walks`` on a lane of its own
instead: the scheduler hands that lane only its keys (``MicroScheduler.lane_of``),
its steps run on the GPU thread between two chain steps, and they land on a
second stream while the chain carries on. Submit and postprocess are the
chain's own (``Worker._submit_step``, ``Worker._finish_step``), keyed by lane.

One step in flight at a time; windows that come ready meanwhile batch into the
next one. ``EngineManager.build`` decides which node gets the lane and refuses
what it can't serve: a step here is admitted and planned inline, never
speculated, planned ahead or lockstep. Its graphs are captured on a stream of
their own (``Engine._capture_stream_for``), which is what makes overlapping
them safe.

A resourced node, or a second side node, would each take a lane of their own
with an id past this one: admit and plan on the lane's thread against its own
stream, and the same submit, finish and scheduler filter.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from mstar.utils.client_errors import client_error_text

if TYPE_CHECKING:
    from mstar.utils.threads import ThreadPool
    from mstar.worker.micro_scheduler import ScheduledBatch
    from mstar.worker.worker import PendingBatch, Worker


logger = logging.getLogger(__name__)

# The one side lane a worker runs; the chain is ``CHAIN_LANE``
SIDE_LANE = 1


class SideStepLane:
    """A side lane's stream and the one step in flight on it."""

    def __init__(self, worker: Worker, lane: int = SIDE_LANE):
        self._worker = worker
        self.lane = lane
        # made on first launch; the main thread collates on it, the GPU thread runs on it
        self._stream: torch.Stream | None = None
        self.pending: PendingBatch | None = None

    def _fail(self, batch: ScheduledBatch, exc: Exception) -> None:
        """Fail a step's rows alone: the chain's rows in flight have nothing to
        do with it, so this doesn't go through the main loop's handler."""
        worker = self._worker
        worker._clear_in_flight_flag(batch)
        err, _ = client_error_text(exc, f"Worker {worker.worker_id} error in a side step", log=logger)
        worker._fail_requests(dict.fromkeys(batch.request_to_worker_graph, err))

    def execute(self, batch: ScheduledBatch, node_batch, plan_future):
        """The GPU thread's step, on the lane's stream, with nothing to wait for
        on the main one: the streamed items it reads had landed before they
        were routed to it (postprocess waits for the step that made them), and
        ``launch`` collated its windows on this stream."""
        with self._stream:
            outputs = self._worker._execute_on_gpu_thread(batch, node_batch, plan_future)
        if outputs.host_rows is not None or outputs.host_spans is not None:
            # the pinned ring they stage through is reused by the chain's next
            # steps before this one lands (refused at startup where declared)
            raise RuntimeError(
                f"{batch.node_name} {batch.graph_walk!r} returned stop buffers or output spans in a side step"
            )
        return outputs

    def launch(self, gpu_executor: ThreadPool) -> None:
        """Queue the lane's next step on the GPU thread, if the stream is free."""
        if self.pending is not None:
            return
        worker = self._worker
        batch = worker.scheduler.get_next_batch(worker.request_state, lane=self.lane)
        if batch is None:
            return
        if self._stream is None:
            self._stream = torch.Stream(device=worker.device)
        try:
            # its windows stack here, so the step is ordered behind them alone
            with self._stream:
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
        while this one still reads them. A step here holds no resource, so
        there is no admit to have refused it.
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
            if step.node_batch.failed_requests:
                worker._fail_step_rows(step, outputs)
            worker._finish_step(step, outputs)
        except Exception as exc:
            self._fail(step.batch, exc)
        finally:
            self.pending = None
            worker._hold_rows(self.lane, ())
