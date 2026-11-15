"""The main loop: what is in flight across a pass, the order the components run
in, and the teardowns between them."""
from __future__ import annotations

import logging
import time as _time
from collections.abc import Collection, Iterable, Mapping
from functools import partial
from types import MappingProxyType

from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import AllocationFailed
from mstar.model.submodule_base import BatchedModelOutput
from mstar.profile.worker import WorkerProfileInfo
from mstar.utils.client_errors import client_error_text
from mstar.utils.ipc_format import (
    ConductorMessage,
    ConductorMessageType,
    DrainRequest,
    FailRequests,
    MessageSource,
    ReadsDone,
    RemoveRequest,
    WorkerMessage,
    WorkerMessageType,
)
from mstar.utils.profiler import PhaseLaps, range_pop, range_push, set_step_key
from mstar.worker.admission import Admission
from mstar.worker.executing_batch import build_executing_batch
from mstar.worker.input_ingest import InputIngest
from mstar.worker.lockstep import Lockstep
from mstar.worker.micro_scheduler import CHAIN_LANE, SIDE_LANE, MicroScheduler, ScheduledBatch
from mstar.worker.phase_timer import PhaseTimer, step_key
from mstar.worker.preemption import Preemption
from mstar.worker.request_lifecycle import RequestLifecycle
from mstar.worker.services import WorkerLayout, WorkerServices
from mstar.worker.speculation import Speculation, Speculator
from mstar.worker.step_executor import PendingBatch, StepExecutor
from mstar.worker.worker_metrics import WorkerMetrics

logger = logging.getLogger(__name__)

# Pipeline shape:
#   iter K (main thread):                          GPU thread
#     CPU preamble  ───────────────► overlaps with execute_batch(N)
#     speculate + build N+1
#     await GPU(N).future Python return
#     thread N's outputs → N+1's loop-back inputs
#     submit GPU(N+1) ───────────────► execute_batch(N+1)
#     postprocess(N) ─── ─────────► overlap with GPU(N+1)
#
# Speculation scope (currently): AR engine only, intra-worker, 1-deep,
# for rids whose loop is still continuing.

# How often the admit fence reports the step it is waiting for. The fence itself
# never gives up; see ``StepPipeline._await_admit_settled``.
_ADMIT_FENCE_WARN_S = 10.0


def repoll_nodes(consumers: frozenset[str], node_names: frozenset[str], parallel: frozenset[str]) -> frozenset[str]:
    """The stream consumers whose steps the loop follows with a poll of the stream
    buffers before it picks the next step, so their rows run every step: only on a
    worker that runs nothing but such consumers. A worker that also runs other nodes
    (a producer beside its consumer) keeps its order of turns between them, and a
    lockstep node is left out, since an extra poll on one rank would move when that
    rank ingests relative to its peers."""
    eligible = consumers - parallel
    return eligible if node_names and node_names <= eligible else frozenset()


class StepPipeline:
    def __init__(
        self, services: WorkerServices, layout: WorkerLayout, timer: PhaseTimer,
        lifecycle: RequestLifecycle, lockstep: Lockstep, admission: Admission,
        scheduler: MicroScheduler, executor: StepExecutor, ingest: InputIngest,
        speculator: Speculator, preemption: Preemption, *, metrics: WorkerMetrics | None = None,
        profile_info: WorkerProfileInfo | None = None, launch_wait_s: float = 0.0,
        enable_nvtx: bool = False,
    ):
        self.worker_id = services.worker_id
        self._services = services
        self._rebuilds = services.rebuilds
        self._graph_runtime = services.graph_runtime
        self.engine_manager = services.engine_manager
        self.tensor_manager = services.tensor_manager
        self.request_state = services.request_state
        self.streams = services.streams
        self._stream_consumer_nodes = repoll_nodes(
            frozenset(self.streams.consumer_node.values()) if self.streams is not None else frozenset(),
            layout.node_names, layout.parallel_nodes,
        )
        self.communicator = services.communicator
        self.is_tp_follower = layout.is_tp_follower
        self.timer = timer
        self.lifecycle = lifecycle
        self.lockstep = lockstep
        self.admission = admission
        self.scheduler = scheduler
        self.executor = executor
        self.ingest = ingest
        self.speculator = speculator
        self.preemption = preemption
        self._metrics = metrics
        self.profile_info = profile_info if profile_info is not None else WorkerProfileInfo()
        self._launch_wait_s = launch_wait_s
        self.enable_nvtx = enable_nvtx
        self._has_side_lane = bool(layout.side_keys)

        # step N, on the chain, and the side lane's step
        self.pending: PendingBatch | None = None
        self.side_pending: PendingBatch | None = None
        # this pass's: step N+1 once submitted, its speculation, a fresh batch
        self._spec_pending: PendingBatch | None = None
        self._speculation: Speculation | None = None
        self._batch: ScheduledBatch | None = None
        # The step whose decision on the next one (a head from it, or its marker)
        # ``Speculator.build`` settled, for a TP-async lead step. The error handler
        # owes the followers a marker for any other lead step it drops.
        self._decided: PendingBatch | None = None
        # The seq of a lockstep step this leader broadcast and has not submitted:
        # the followers run it unless N's verdict voids it on every rank, so a
        # raise that drops it must stop this rank (see ``run``). A follower's own
        # raises are not covered.
        self._committed: int | None = None
        # lane -> the rows its step in flight holds; replaced, never updated in place
        self._rows_by_lane: Mapping[int, frozenset[int]] = MappingProxyType({})
        # every lane's rows: removes and drains for these are deferred. Shared
        # with the scheduler and updated in place: it starts no rebuild for a
        # row in it.
        self.in_flight_rids: set[int] = set()
        scheduler.in_flight_rids = self.in_flight_rids

    # -- the loop --------------------------------------------------------------

    def run(self) -> None:
        # The main thread hands every step to the GPU and plan threads and back,
        # so all three move to the step CPUs (each executor thread in its initializer).
        self.executor.pin_this_thread()
        self.executor.start()
        while True:
            try:
                self._pass()
            except Exception as e:
                pending, spec_pending = self.pending, self._spec_pending
                self._handle_error(e, (pending, spec_pending), self._batch, self._speculation)
                if self._committed is not None:
                    # Failing the step's rows is all this rank can do: the followers
                    # are in its collectives, or will be, and no marker releases
                    # them. Stop, and the conductor's dead-worker check fails the
                    # rest instead of the group hanging.
                    raise SystemExit(
                        f"Worker {self.worker_id}: dropped lockstep step seq {self._committed} "
                        f"that the followers run ({e!r})"
                    ) from e
                # Follower: a head from a step that raised must not sit at the
                # FIFO front with failed rids.
                if pending is not None and self.speculator.is_tp_follow_pending(pending):
                    self.lockstep.close_follow_step(pending)
                # Leader: a dropped step whose decision never went out (the raise
                # came before or in the speculation build, or N+1 was already
                # submitted) leaves the followers awaiting it, and a drain
                # deferred behind it with them.
                else:
                    for dropped in (pending, spec_pending if spec_pending is not pending else None):
                        if (
                            dropped is not None and dropped is not self._decided
                            and self.speculator.can_speculate(dropped.batch)
                        ):
                            self.lockstep.maybe_broadcast_nospec(dropped, None)
                # Clear the in-flight step: its rows are failed and its future
                # drained, and it must not be post-processed again.
                self._set_pending(None)
                self.speculator.chain_ended()
                _time.sleep(0.01)

    def _set_pending(self, step: PendingBatch | None) -> None:
        self.pending = step
        self._hold_rows(CHAIN_LANE, step.batch.request_to_worker_graph if step else ())

    def _arm(self, speculation: Speculation) -> None:
        if self.speculator.plans_ahead(speculation):
            self.executor.arm_preplan(self.pending, speculation)

    def _pass(self) -> None:
        # Reset per pass so the error handler sees only this pass's work: a stale
        # ``batch`` from a previous pass would otherwise be failed twice.
        self._batch = self._spec_pending = self._speculation = None
        self._decided = self._committed = None
        timer = self.timer
        phase_period = timer.period
        iter_start = _time.perf_counter() if phase_period else 0.0
        iter_cpu = _time.thread_time() if phase_period else 0.0
        laps = PhaseLaps("worker.main.")

        self._deferred_teardown(laps)
        self._preamble(laps)

        # 2. Speculatively schedule + build N+1, overlapping GPU(N). Falls through
        # to the fresh path below for a step that can't (not AR, loop ended).
        speculation, yield_away_from_target = self.speculator.build(self.pending)
        self._speculation = speculation
        self._decided = self.pending
        if speculation is not None and speculation.tp_seq >= 0 and not self.is_tp_follower:
            self._committed = speculation.tp_seq
        # inclusive of `speculate` and `worker.schedule_yield_away`
        laps.lap("build_speculation")
        if speculation is not None:
            self._arm(speculation)
            laps.lap("arm_speculation")

        # 3. If pending: await GPU(N), submit speculated GPU(N+1) asap, then
        # post-process N overlapping with GPU(N+1).
        finished_node = self.pending.node_name if self.pending is not None else None
        if self.pending is not None:
            self._finish_chain_step(laps)

        spec_pending = self._spec_pending
        if spec_pending is not None:
            self.speculator.note_launched(self._speculation)
            if phase_period:
                timer.pass_done(len(spec_pending.batch.request_to_worker_graph), iter_start, iter_cpu)
            self._set_pending(spec_pending)
            return
        self.speculator.chain_ended()

        if finished_node in self._stream_consumer_nodes:
            # The preamble polled while N ran, and refused the next chunk of each
            # row N held. N is post-processed now, so offer those chunks again:
            # otherwise N+1 leaves out every row of N, and a stream consumer's
            # rows run every other step.
            self.ingest.poll_stream_buffers()

        # 4. Non-speculative path: nothing pending, or speculation skipped.
        with timer.span("worker.schedule"):
            batch = self._next_batch(yield_away_from_target)
        self._batch = batch
        if batch is None or batch.node_name not in self.lockstep.led:
            self.lockstep.flush_moves()
        if batch is None:
            if self.side_pending is not None:
                self._finish_side(block=True)
                return
            self._rebuilds.flush_idle()
            # the next pass's `flush_idle` waits for the copies just started, so it needn't wait the tick too
            self.communicator.wait_for_work(0 if self._rebuilds.copies_queued else 10)
            return

        if self.enable_nvtx:
            range_push("worker.build_node_batch", synchronize=False)
        node_batch = build_executing_batch(self._services, batch)
        batch_partition = self.request_state.get_partition_for_node(batch.node_name)
        if self.enable_nvtx:
            range_pop(synchronize=False)

        # Nothing speculated this batch, so prepare it here. The GPU thread then
        # admits, plans and runs it inline; the slot is leased inside exec, once
        # the token count is known. A leader stamps the seq it sends; a follower's
        # batch keeps the one it came off the FIFO with; everyone else -1.
        broadcast_seq = self.lockstep.broadcast(node_batch)
        if broadcast_seq >= 0:
            self._committed = broadcast_seq
        tp_seq = broadcast_seq if broadcast_seq >= 0 else batch.tp_seq

        # Unlike the spec submit, this path doesn't hold off the GIL for the
        # launch, but it still has to be fenceable, or the next pass schedules
        # against an admit that hasn't run.
        step = self.executor.submit(batch, node_batch, partition=batch_partition, tp_seq=tp_seq)
        self._committed = None
        logger.debug("Scheduling: %s %s", batch.node_name, node_batch.request_ids)
        self._set_pending(step)
        if phase_period:
            timer.pass_done(len(batch.request_to_worker_graph), iter_start, iter_cpu)

    def _next_batch(self, yield_away_from_target: tuple[str, str] | None) -> ScheduledBatch | None:
        """The non-speculative path's step: one away from ``yield_away_from_target`` when given, else any.

        A follower's first call can take a head of rank 0's with no rows, and rank 0 applied the removals
        stamped with that head before it sent the next step, so this rank applies them before building it."""
        if yield_away_from_target is not None:
            batch = self.scheduler.next_batch(exclude_target=yield_away_from_target)
            if batch is not None:
                return batch
            if self.is_tp_follower:
                self._apply_removes(self.in_flight_rids)
        return self.scheduler.next_batch()

    def _deferred_teardown(self, laps: PhaseLaps) -> None:
        # Everything below moves pages: the removes and drains directly, the
        # preamble through a follower's delta, the readiness scans through
        # ``check_ready``'s reload. None of it may run while N's admit is
        # outstanding. See ``_await_admit_settled``.
        self._await_admit_settled(self.pending)
        self._count_turn(self.pending)
        laps.skip()
        # removals held for the journal would otherwise wait out a chain of another node's steps
        self.lockstep.flush_moves(self.pending)
        self._apply_removes(self.in_flight_rids)
        # Requests a resource refused for good since last pass, in a readiness
        # scan or a step's admit (``Admission``), which failed them then.
        self._report_failures()
        self._apply_drains(self.in_flight_rids)
        self.preemption.make_room()
        laps.lap("teardown")

    def _preamble(self, laps: PhaseLaps) -> None:
        """1. CPU preamble, overlapping GPU(N). synchronize=False on every range so
        torch.cuda.synchronize() doesn't drain the in-flight GPU work."""
        enable_nvtx = self.enable_nvtx
        if enable_nvtx:
            range_push("worker.process_messages", synchronize=False)
        self.ingest.process_messages()
        if enable_nvtx:
            range_pop(synchronize=False)
        laps.lap("process_messages")

        if enable_nvtx:
            range_push("worker.check_ready_tensors", synchronize=False)
        self.ingest.check_ready_tensors()
        if enable_nvtx:
            range_pop(synchronize=False)
        laps.lap("check_ready_tensors")

        if enable_nvtx:
            range_push("worker.poll_stream_buffers", synchronize=False)
        self.ingest.poll_stream_buffers()
        if enable_nvtx:
            range_pop(synchronize=False)
        laps.lap("poll_stream_buffers")

        # What the preamble recorded, for requests nothing is using.
        self._report_failures()
        self._teardown(self.in_flight_rids)
        laps.lap("ingest_teardown")

        if self._has_side_lane:
            if self.pending is None:
                self._finish_side()
            self._launch_side()
            laps.lap("side_steps")

        # Reads only: whether a rank sends this changes nothing it runs.
        if self._metrics is not None:
            self._metrics.maybe_report()
            laps.lap("metrics")

    def _finish_chain_step(self, laps: PhaseLaps) -> None:
        """Stages 4 to 7, for step N."""
        pending = self.pending
        timer = self.timer
        phase_period = timer.period
        enable_nvtx = self.enable_nvtx
        outputs: BatchedModelOutput | None = None
        if self._speculation is None and self.speculator.is_tp_follow_pending(pending):
            # Follower: watch for the head while N runs and settle the leader's
            # decision before N is post-processed.
            if enable_nvtx:
                range_push("worker.follow_await", synchronize=False)
            t0 = _time.perf_counter() if phase_period else 0.0
            outputs, self._speculation = self._await_tp_follow_step(pending)
            if phase_period:
                timer.record("follow_await", _time.perf_counter() - t0)
            if enable_nvtx:
                range_pop(synchronize=False)

        if outputs is None:
            if enable_nvtx:
                range_push("worker.await_gpu", synchronize=False)
            t0 = _time.perf_counter() if phase_period else 0.0
            outputs = self._step_result(pending)
            if phase_period:
                timer.record("await_gpu", _time.perf_counter() - t0)
            if enable_nvtx:
                range_pop(synchronize=False)

        speculation = self._speculation
        if speculation is not None and not speculation.is_yield_away and self.speculator.step_voids_head(pending):
            self._committed = None  # when the verdict (a raise, say) is on every rank, every rank voids the head
        laps.skip()
        # the node has just completed
        self.executor.clear_in_flight(pending.batch)
        laps.lap("clear_in_flight")

        if pending.node_batch.admit_error is not None:
            # Admit refused pending, so no forward ran. ``_handle_admit_failure``
            # pushes the nodes back and on KV-cache OOM offloads or holds. Clearing
            # the speculation first drops its pre-plan, releasing pages the eviction
            # would otherwise go looking for. Nothing is speculated on a refusal, so
            # in practice there is nothing to clear.
            self.executor.clear_in_flight(pending.batch)
            self._clear_speculation()
            self._drop_rows_whose_loop_stopped(pending)
            self._handle_admit_failure(
                pending.batch, pending.node_batch,
                referenced_rids=self.speculator.speculated_rids(self._speculation),
            )

        if pending.node_batch.failed_requests:
            # Drop the speculation: it was built from pending's rids and may
            # thread outputs that the failed rids never produced.
            self._clear_speculation()
            self._fail_step_rows(pending, outputs)

        if self._speculation is not None:
            self._launch_speculation(outputs, laps)

        # Post-process N (routing stage), concurrently with GPU(N+1) if one was
        # submitted. Skipped on any admit failure since the output tensors aren't
        # valid; ``_handle_admit_failure`` already rehabilitated the failed rids.
        laps.skip()
        if pending.node_batch.admit_error is None:
            self._finish_step(pending, outputs)
            laps.skip()

        # After N, while N+1 runs: the same thread, but not ahead of the step the
        # chain is about to speculate.
        if self._has_side_lane:
            self._finish_side()
        laps.lap("step_done")
        # Before anything schedules N's rows again: otherwise a serial chain would
        # hold a drained rid at every teardown point.
        in_flight = self._held_beside(CHAIN_LANE)
        if self._spec_pending is not None:
            in_flight.update(self._spec_pending.batch.request_to_worker_graph)
        # drains only: removes wait for the next teardown point
        self._apply_drains(in_flight)
        laps.lap("post_removes")
        self._set_pending(None)
        if self._spec_pending is None:
            # N is done and nothing runs, so tear down what waited behind N
            # before the next step is built: its admit may need the pages. On a
            # follower these include the removes rank 0 stamped with N, or with
            # a head this rank dropped, which it tore down before sending any
            # later step; they have landed here (``removes_ready`` gates on
            # ``last_consumed_seq``).
            self._teardown(self.in_flight_rids)
        laps.lap("set_pending")

    def _clear_speculation(self) -> None:
        """N's outcome voids what was built on it. A speculation that depended on
        N's outputs (the plan thread already threaded them in) can't run at all. A
        yield-away is independent of N and still runs, but its pre-plan resets: on
        an admit refusal ``_handle_admit_failure`` may shift the engine's KV-cache
        state (paused or offloaded rids)."""
        speculation = self._speculation
        if speculation is None:
            return
        if speculation.plan_future is not None:
            speculation.plan_future.result()
            self.executor.reset_skip_plan_flags(speculation.node_batch)
            speculation.plan_future = None
        if speculation.is_yield_away:
            return
        # The step will not run at all: un-ingest every staged chunk and hand
        # them all back.
        sb = speculation.scheduled_batch
        self.speculator.settle(speculation, success=False)
        # Fresh rids are not re-readied by N's routing the way continuing rids
        # are: give them back.
        rows = sb.graph_rows()
        fresh = [
            rid for rid in rows
            if rid not in speculation.continuing_rids
            and rows.get(rid) is not None
        ]
        self._graph_runtime.push_back_node(
            sb.node_name, fresh, [rows[r] for r in fresh],
        )
        self._speculation = None

    def _launch_speculation(self, outputs: BatchedModelOutput, laps: PhaseLaps) -> None:
        """5 and 6. Reconcile N's outputs into N+1 and launch it."""
        speculation = self._speculation
        spec_batch = speculation.scheduled_batch
        spec_node_batch = speculation.node_batch
        timer = self.timer
        laps.lap("failure_checks")
        if not speculation.is_yield_away:
            self.speculator.thread_outputs(speculation, outputs)
            laps.lap("thread_outputs")
        # The one settle for a step that runs: marks the batch speculatively
        # scheduled (so its nodes are not put back on the ready queue while it
        # executes; not the dropped rids, which the threading already pruned), and
        # keeps the staged ingests except for those dropped rids, whose chunks are
        # un-ingested and go back to their buffers. Outside the submit guard below,
        # so an all-dropped batch is still settled.
        self.speculator.settle(speculation, success=True, dropped_rids=speculation.dropped)
        laps.lap("settle_speculation")

        if spec_batch.request_to_worker_graph:
            if self.enable_nvtx:
                range_push("worker.submit_spec", synchronize=False)
            t0 = _time.perf_counter() if timer.period else 0.0
            # Staleness is checked on the GPU thread, after the plan future
            # resolves and after prepare; see ``GpuStepBody``.
            self._spec_pending = self.executor.submit(
                spec_batch, spec_node_batch,
                plan_future=speculation.plan_future, speculated=True,
                partition=speculation.partition,
                speculative_new_iter=speculation.is_new_iter,
                loop_name=speculation.loop_name,
                tp_seq=speculation.tp_seq,
            )
            # the GPU thread owns the pre-plan and the rids from here; see _handle_error
            speculation.plan_future = None
            speculation.submitted = True
            self._committed = None
            if self.enable_nvtx:
                range_pop(synchronize=False)
                range_push("worker.gpu_submit_queued", synchronize=False)
            # Hold the main thread off the GIL until the GPU thread reaches the
            # forward launch.
            with timer.span("worker.submit_spec.launch_wait"):
                spec_node_batch.launch_started_event.wait(timeout=self._launch_wait_s)
            if timer.period:
                timer.record("submit_spec", _time.perf_counter() - t0)
            if self.enable_nvtx:
                range_pop(synchronize=False)
        else:
            # Every rank drops the same rows, so none runs it.
            self._committed = None
            if speculation.plan_future is not None:
                # All continuing rids were dropped, so no spec batch was submitted.
                # Drop the orphaned pre-plan so the next step to lease that slot
                # doesn't promote it. Await first: this batch never reaches the GPU
                # thread, so nothing else joins the future, and resetting under a
                # running plan races it.
                speculation.plan_future.result()
                self.executor.reset_skip_plan_flags(speculation.node_batch)
                speculation.plan_future = None

    # -- the side lane -----------------------------------------------------------
    #
    # The main loop runs one chain (``CHAIN_LANE``): step N on the device, N+1
    # speculated behind it. A step on that chain costs a pass of the loop and
    # holds the next step behind its kernels on the main stream, which is what a
    # codec window does to the decode steps around it. A node whose node group
    # sets ``side_stream: true`` runs its ``side_stream_walks`` on a lane of its
    # own instead: the scheduler hands that lane only its keys
    # (``MicroScheduler.lane_of``), its steps run on the GPU thread between two
    # chain steps, and they land on a second stream while the chain carries on.
    # Submit and postprocess are the chain's own (``StepExecutor.submit``,
    # ``_finish_step``), keyed by lane.
    #
    # One step in flight at a time; windows that come ready meanwhile batch into
    # the next one. ``EngineManager.build`` decides which node gets the lane and
    # refuses what it can't serve: a step here is admitted and planned inline,
    # never speculated, planned ahead or lockstep. Its graphs are captured on a
    # stream of their own (``Engine._capture_stream_for``), which is what makes
    # overlapping them safe.
    #
    # A resourced node, or a second side node, would each take a lane of their
    # own with an id past this one: admit and plan on the lane's thread against
    # its own stream, and the same submit, finish and scheduler filter.

    def _fail_side(self, batch: ScheduledBatch, exc: Exception) -> None:
        """Fail a side step's rows alone: the chain's rows in flight have nothing
        to do with it, so this doesn't go through ``_handle_error``."""
        self.executor.clear_in_flight(batch)
        err, _ = client_error_text(exc, f"Worker {self.worker_id} error in a side step", log=logger)
        self._fail_requests(dict.fromkeys(batch.request_to_worker_graph, err))

    def _launch_side(self) -> None:
        """Queue the side lane's next step on the GPU thread, if its stream is free."""
        if self.side_pending is not None:
            return
        batch = self.scheduler.next_batch(lane=SIDE_LANE)
        if batch is None:
            return
        try:
            # its windows stack here, so the step is ordered behind them alone
            with self.executor.lane_stream(SIDE_LANE):
                node_batch = build_executing_batch(self._services, batch)
            self.side_pending = self.executor.submit(batch, node_batch, lane=SIDE_LANE)
        except Exception as exc:
            self._fail_side(batch, exc)
            return
        self._hold_rows(SIDE_LANE, batch.request_to_worker_graph)

    def _finish_side(self, block: bool = False) -> None:
        """Postprocess the side step in flight once it has landed; ``block`` waits.

        Never before the step has landed: postprocess frees its inputs first, and
        those are main-stream allocations a main-stream step could reuse while
        this one still reads them. A step here holds no resource, so there is no
        admit to have refused it.
        """
        step = self.side_pending
        if step is None or not (block or self.executor.landed(step)):
            return
        try:
            outputs = step.future.result()
            self.executor.wait_for_step(step)
            self.executor.clear_in_flight(step.batch)
            self._count_turn(step)
            if step.node_batch.failed_requests:
                self._fail_step_rows(step, outputs)
            self._finish_step(step, outputs)
        except Exception as exc:
            self._fail_side(step.batch, exc)
        finally:
            self.side_pending = None
            self._hold_rows(SIDE_LANE, ())

    # -- steps -------------------------------------------------------------------

    def _await_admit_settled(self, pending: "PendingBatch | None") -> None:
        """Block until the in-flight step has admitted.

        Admit is on the GPU thread; every offload and reload is on this one. A
        page move in the window between a step's broadcast and its admit lands
        in the NEXT step's resident delta, so the follower applies it after
        admitting this step while the leader admitted with it already applied.
        The event is set at the step's last launch (just before a split replay's
        second graph), after every row has admitted, or when the step ends without launching
        (``_submit_gpu_step``). On the per-request path later rows plan after
        it lifts; their plans read only their own in-flight rows.

        It waits however long the admit takes: giving up lets the ranks split,
        and this thread blocks on the step's result later in the pass anyway.
        Separate from the submitter's ``MSTAR_LAUNCH_WAIT_MS`` wait on the same
        event, a GIL throttle that is meant to expire.
        """
        if pending is None:
            return
        event = pending.node_batch.launch_started_event
        if event is None or event.is_set():
            return
        waited = 0.0
        with self.timer.span("worker.await_admit_settled"):
            while not event.wait(timeout=_ADMIT_FENCE_WARN_S):
                waited += _ADMIT_FENCE_WARN_S
                logger.warning(
                    "Worker %s: still waiting for node=%s walk=%s to admit after "
                    "%.0f s; scheduling and teardowns wait for it",
                    self.worker_id, pending.node_name, pending.graph_walk, waited,
                )

    def _hold_rows(self, lane: int, rids: Iterable[int]) -> None:
        """The rows ``lane``'s step in flight holds are now ``rids``. Teardown
        defers for every lane's (``in_flight_rids``); a main-loop error fails
        the chain's."""
        rows = self._rows_by_lane = {**self._rows_by_lane, lane: frozenset(rids)}
        self.in_flight_rids.clear()
        self.in_flight_rids.update(*rows.values())

    def _held_beside(self, lane: int) -> set[int]:
        """The rows the other lanes' steps in flight hold."""
        return set().union(*(rows for other, rows in self._rows_by_lane.items() if other != lane))

    def _fail_step_rows(self, step: PendingBatch, outputs: BatchedModelOutput) -> None:
        """Fail the rows a per-rid stage (prepare_inputs, a per-request
        forward, postprocess) blamed; the rest of the step still postprocesses
        and routes."""
        failed = dict(step.node_batch.failed_requests)
        self.executor.drop_failed_rids(step, outputs, failed)
        self._fail_requests(failed)

    def _finish_step(self, step: PendingBatch, outputs: BatchedModelOutput) -> None:
        """A step that ran, on any lane: its batch limit lifts, its outputs
        are postprocessed and routed, and metrics count it."""
        self.admission.clear_batch_limit(
            step.node_name, step.graph_walk, len(step.batch.request_to_worker_graph),
            walks=step.batch.rids_by_walk(),
        )
        chain = step.lane == CHAIN_LANE
        span = "worker.postprocess_batch" if chain else "worker.postprocess_side"
        keyed = self.timer.period > 0
        if keyed:
            set_step_key(step_key(step.node_name, step.graph_walk))
        try:
            with self.timer.span(span, cpu=f"{span}.cpu"):
                self.executor.postprocess(step, outputs, clear_loop_stops=chain)
        finally:
            if keyed:
                set_step_key(None)
        # what postprocess blamed on a row
        self._report_failures()
        if self._metrics is not None:
            self._metrics.step_done(step.node_name, step.graph_walk, step.node_batch.inputs, step.submitted_at)

    def _count_turn(self, pending: "PendingBatch | None") -> None:
        """Once ``pending``'s admit has settled (``_await_admit_settled``): if
        it took the step, the step's rows have had their turn at its node. A
        refused step is retried, and its rows keep their place in line."""
        if pending is not None and pending.node_batch.admit_error is None:
            self.scheduler.note_ran(pending.batch)

    def _drop_rows_whose_loop_stopped(self, pending: PendingBatch) -> None:
        """Take a refused speculative step's rows whose loop stopped at N out
        of its batch. Postprocess drops them on a step that runs; pushed back,
        their emptied node would pop with no inputs."""
        if not pending.speculative_new_iter:
            return
        for rid in self._graph_runtime.pending_loop_stop_rids(
            list(pending.batch.rids_by_walk()), pending.loop_name,
        ):
            pending.batch.discard_rid(rid)

    def _handle_admit_failure(
        self, batch: ScheduledBatch, node_batch: ExecutingBatch,
        referenced_rids: frozenset[str] = frozenset(),
    ) -> None:
        """Re-queue a batch whose admit refused it, and for a lack of pages make
        room for the retry. Under TP this runs on every rank: page allocation is
        driven by rank 0's scheduler and replicated by the broadcast, so the
        ranks refuse the same batches (``KVCacheManager.post_warmup_validate``
        checks at startup that they begin with the same free pages). Only rank
        0 makes room (``Admission.relieve``); a follower re-queues and tears down.

        A teardown this rank has been holding releases pages, and spending one
        can mean no eviction is needed at all. After the push-back, so a rid torn
        down here cannot be pushed onto queues that have just been dismantled.
        ``referenced_rids``: what a speculation still touches; empty in practice,
        since nothing is speculated on a refusal. This step's own rids stay in
        ``in_flight_rids`` until the loop clears the pending step, and their
        removals wait for that.
        """
        self.admission.requeue(batch, node_batch)
        if not isinstance(node_batch.admit_error, AllocationFailed):
            return
        self._teardown(self.in_flight_rids.union(referenced_rids))
        node, walk = batch.node_name, batch.graph_walk
        batch_ids = set(batch.request_to_worker_graph)
        if not self.lockstep.follows(node) and self.preemption.preempt(node_batch):
            return
        self.admission.relieve(
            batch, node_batch, self.lifecycle.arrivals,
            last_in_line=partial(self.scheduler.last_in_line, node, walk),
            could_refill=partial(self.scheduler.could_refill, node, walk, batch_ids),
        )

    def _step_result(self, pending: PendingBatch) -> BatchedModelOutput:
        """N's outputs. A forward that raised becomes N's verdict instead: no
        outputs and every row failed (``failed_requests``), which voids a head
        built on N and lets a step independent of N run. When the raise is on
        every rank, every rank does so alike. A raise on one rank alone leaves
        the others in a step it won't run, and the group wedges (#1235)."""
        try:
            return pending.future.result()
        except Exception as exc:
            err, _ = client_error_text(
                exc, f"Worker {self.worker_id} step {pending.node_name}/{pending.graph_walk} failed",
                log=logger,
            )
            failed = pending.node_batch.failed_requests
            for rid in pending.batch.request_to_worker_graph:
                failed.setdefault(rid, f"Error in worker: {err}")
            return BatchedModelOutput()

    def _await_tp_follow_step(
        self, pending: PendingBatch,
    ) -> tuple[BatchedModelOutput | None, Speculation | None]:
        """Wait for step N and settle the leader's decision about N+1. Returns
        ``(outputs or None, armed head or None)``. Once N is done this never
        returns without a decision: going serial early strands the leader."""
        s = pending.tp_seq
        outputs: BatchedModelOutput | None = None
        t_done = last_warn = 0.0
        while True:
            if outputs is None and pending.future.done():
                outputs = self._step_result(pending)
                t_done = last_warn = _time.perf_counter()
            if self.lockstep.is_no_spec(s):
                return outputs, None
            head = self.lockstep.follow_head()
            if head is not None and head.speculative and head.spec_from_seq == s:
                void = self.speculator.step_voids_head(pending) if outputs is not None else None
                if void is not None and not self.speculator.prefix_cache_open(head.node_name):
                    # The leader clears its speculation on this verdict too.
                    self.lockstep.void_follow_head()
                    logger.debug(
                        "Worker %s: dropped speculative head seq %d (parent %d %s)",
                        self.worker_id, head.spec_seq, s, void,
                    )
                    return outputs, None
                # With the prefix cache open, build the head even when N's verdict voids it, since the
                # leader pre-plans every head it sends and this rank's pages must move as the leader's do.
                spec = self.speculator.follow(pending)
                if spec is not None:
                    # Its rids count as in flight now, so a remove landing before
                    # submit is deferred like any other; ``_set_pending`` re-derives
                    # the set on submit.
                    self._hold_rows(
                        CHAIN_LANE,
                        self._rows_by_lane.get(CHAIN_LANE, frozenset()).union(
                            spec.scheduled_batch.request_to_worker_graph,
                        ),
                    )
                    self._arm(spec)
                    return outputs, spec
                # Not buildable yet (fresh rid / stream chunk): poll and retry.
            elif head is not None and head.spec_seq > s:
                self.lockstep.leader_passed(s, head)
                return outputs, None
            self.communicator.wait_for_work(20)
            self.ingest.poll()
            self._report_failures()
            # Rank 0 released these before it sends the head this loop waits for.
            self._teardown(self.in_flight_rids)
            if outputs is not None:
                now = _time.perf_counter()
                if now - last_warn > 2.0:
                    last_warn = now
                    logger.warning(
                        "Worker %s: still waiting for the leader's decision on "
                        "step seq %d, %.1fs after it finished (head queued: %s)",
                        self.worker_id, s, now - t_done,
                        head is not None and head.spec_from_seq == s,
                    )

    def _handle_error(
        self,
        exc: Exception,
        in_flight: "tuple[PendingBatch | None, ...]",
        batch: ScheduledBatch | None,
        speculation: "Speculation | None" = None,
    ) -> None:
        """Fail everything the crashed iteration touched and drain its futures.

        Attribution is batch-granular here: a raise out of the batch build or
        output routing can't be pinned on one request (the
        stages that *can* attribute report through
        ``ExecutingBatch.failed_requests`` instead), so every rid this iteration
        touched fails together. Sequential retry of the batch — see the design
        discussion on #123 — would go here; today a batch-level crash is
        terminal for its rids.

        The caller still has to clear its own in-flight state; see the main
        loop's handler.
        """
        # Report the leaf exception to the client and keep the full chain in
        # the log. A forward's own raise doesn't come here (``_step_result``).
        err, _ = client_error_text(
            exc, f"Worker {self.worker_id} error in main loop", log=logger,
        )

        # the chain's rows: a side lane's step fails on its own (``_fail_side``)
        failed_rids: set[int] = set(self._rows_by_lane.get(CHAIN_LANE, ()))
        if speculation is not None and not speculation.submitted:
            # a follower's build holds the head's fresh rows with the chain's, and they go back as on rank 0
            failed_rids -= speculation.scheduled_batch.request_to_worker_graph.keys() - speculation.continuing_rids
        for stale in in_flight:
            if stale is None:
                continue
            failed_rids.update(stale.batch.request_to_worker_graph)
            self.executor.clear_in_flight(stale.batch)
            # Drain before dropping the reference: the future owns engine state
            # on the GPU thread, and an abandoned one leaves that thread writing
            # into a batch nobody will collect. Already-finished futures return
            # at once.
            if stale.future is None:
                continue
            try:
                stale.future.result()
            except Exception:
                logger.debug(
                    "Worker %s discarding failed batch for node %s",
                    self.worker_id, stale.node_name,
                )
        if batch is not None:
            failed_rids.update(batch.request_to_worker_graph)
            self.executor.clear_in_flight(batch)
        if speculation is not None and not speculation.submitted:
            # Built but never submitted. Its fresh rids came off their queues
            # (a yield-away step is all fresh rids), so they go back or nothing
            # would schedule them again; the continuing ones belong to the
            # pending batch and fail with it. Keyed on submission rather than
            # on an armed pre-plan: the plan future is None without pre-plan
            # and after a refusal reset it, and the rids are stranded either way.
            if speculation.plan_future is not None:
                # The plan thread may still be staging a pre-plan for a step
                # that will now never run. Drain it, then drop the stage, or
                # the next step to lease that slot would find it.
                try:
                    speculation.plan_future.result()
                except Exception:
                    logger.debug(
                        "Worker %s discarding the pre-plan of the failed iteration",
                        self.worker_id,
                    )
                speculation.plan_future = None
                self.executor.reset_skip_plan_flags(speculation.node_batch)
            sb = speculation.scheduled_batch
            # This step never ran, so its staged chunks go back to their
            # buffers -- in particular for the fresh rids pushed back below,
            # which survive this error and would otherwise hold a chunk in a
            # slot the next ingest can overtake.
            self.speculator.settle(speculation, success=False)
            self.executor.clear_in_flight(sb)
            fresh = {
                rid: wg_id
                for rid, wg_id in sb.graph_rows().items()
                if rid not in speculation.continuing_rids
                and rid not in failed_rids
            }
            if fresh:
                self._graph_runtime.push_back_node(
                    sb.node_name, list(fresh), list(fresh.values()),
                )

        self._fail_requests({rid: f"Error in worker: {err}" for rid in failed_rids})

    def _fail_requests(self, errors: dict[int, str]) -> None:
        """Report requests this worker can no longer serve to the conductor.
        The request stays until the conductor's REMOVE_REQUEST comes back."""
        self.lifecycle.fail(errors)
        self._report_failures()

    def _report_failures(self) -> None:
        """Tell the conductor of the failures ``RequestLifecycle`` has recorded,
        and stop what the scheduler has queued for them."""
        errors = self.lifecycle.take_failures()
        if not errors:
            return
        wire = self._graph_runtime.get_rid_string
        wire_errors = {wire(rid): msg for rid, msg in errors.items()}
        statuses = {wire(rid): status for rid, status in self.lifecycle.take_statuses(errors).items()}
        for wire_rid, msg in wire_errors.items():
            logger.error("Worker %s failing request %s: %s", self.worker_id, wire_rid, msg)
        self.scheduler.drop_backlogged(errors)
        self.communicator.send(
            "conductor",
            ConductorMessage(
                message_type=ConductorMessageType.FAIL_REQUESTS,
                body=FailRequests(errors=wire_errors, statuses=statuses),
            ),
        )

    def _teardown(self, in_flight: Collection[int]) -> None:
        """The teardown sequence: every remove and drain that no step in
        ``in_flight`` holds back."""
        self._apply_removes(in_flight)
        self._apply_drains(in_flight)

    def _apply_removes(self, in_flight: Collection[int]) -> None:
        # Rank 0 tears down nothing while its journal holds moves: each follower replays them with the next step and
        # tears down after the step a removal is stamped with, so the pages leave in the same order on every rank.
        if self.lockstep.owes_moves():
            return
        for rid in self.lifecycle.removes_ready(in_flight, self.lockstep.last_consumed_seq):
            self._tear_down_request(rid)

    def _tear_down_request(self, rid: int) -> None:
        # Rank 0 fans its removes out so that a follower tears down in the same
        # gap between steps, and can't tear down state rank 0 is still reading.
        wire_rid = self._graph_runtime.get_rid_string(rid)
        self.lockstep.forward_teardown(rid, WorkerMessage(
            message_type=WorkerMessageType.REMOVE_REQUEST,
            body=RemoveRequest(
                request_id=wire_rid, source=MessageSource.TP_RANK_0,
                after_tp_seq=self.lockstep.last_broadcast_seq,
            ),
        ))
        # Hard cleanup: force-drop every tensor for the rid (unlink SHM),
        # ignoring ref counts / persist. Safe because the conductor only sends
        # REMOVE_REQUEST once every reader has confirmed drained (READS_DONE):
        # on the abort/fail path via a prior DrainRequest, on the happy path
        # once the api server finished reading the outputs.
        laps = PhaseLaps("worker.msg.remove_request.")
        for message in self.lifecycle.removed(rid, wire_rid):
            self.ingest.drop_for_removed(message)
        try:
            self._release_request(rid, wire_rid, laps)
        except BaseException:
            self.lifecycle.teardown_failed(rid)
            raise

    def _release_request(self, rid: int, wire_rid: str, laps: PhaseLaps) -> None:
        freed = self.engine_manager.remove_request(rid)
        if freed:
            self.admission.lift_caps(freed)
        laps.lap("engine")
        self.streams.remove_request(rid)
        self.request_state.remove_request(rid)
        self.tensor_manager.force_cleanup_request(rid)
        laps.lap("worker_state_tensors")
        self.profile_info.pop_request(rid)
        self.scheduler.clear_rid(rid)
        self.admission.clear(rid)
        self.lockstep.clear(rid, wire_rid)
        self._rebuilds.remove_request(rid)
        self.preemption.remove_request(rid)
        laps.lap("misc")
        # DEBUG: once per request on the main thread, it cost as much as the
        # engine's own removal (test/waypoint/serve_rollout.py, which waits on
        # it, runs the server at DEBUG)
        logger.debug("Request cleanup complete: %s", wire_rid)
        laps.lap("log")
        # Last: frees the handle for reuse, so nothing above may run after it.
        self._graph_runtime.remove_request(rid)
        laps.lap("runtime")

    def _apply_drains(self, in_flight: Collection[int]) -> None:
        """Phase-1 teardown (abort/fail): stop scheduling a rid, then ACK
        READS_DONE once its in-flight reads finish. The hard cleanup waits for
        the conductor's REMOVE_REQUEST, sent only after every reader has ACKed."""
        lifecycle = self.lifecycle
        for wire_rid in lifecycle.drains_ready(in_flight):
            rid = lifecycle.begin_drain(wire_rid)
            if rid is not None:
                # each rank drains and ACKs its own READS_DONE (the conductor
                # waits on every rank)
                self.lockstep.forward_teardown(rid, WorkerMessage(
                    message_type=WorkerMessageType.DRAIN_REQUEST,
                    body=DrainRequest(request_id=wire_rid, source=MessageSource.TP_RANK_0),
                ))
                self.scheduler.drop_backlogged((rid,))
        for wire_rid, rid in lifecycle.open_drains():
            if rid is not None and self.tensor_manager.has_inflight_reads(rid):
                continue  # let get_ready_tensors resolve the futures; retry next pass
            # Not under the handle guard: a ScheduleTPNode can land before the
            # NEW_REQUEST that mints the handle, so the queued batch is counted
            # under the wire string while the handle is still None. ACKing here
            # would let the REMOVE tear the worker-graph queues out from under a
            # batch still sitting at the head of the TP FIFO.
            if self.lockstep.pending_follow_count(wire_rid) > 0:
                continue
            lifecycle.reads_done(wire_rid)
            self.communicator.send(
                "conductor",
                ConductorMessage(
                    message_type=ConductorMessageType.READS_DONE,
                    body=ReadsDone(request_id=wire_rid, entity_id=self.worker_id),
                ),
            )
