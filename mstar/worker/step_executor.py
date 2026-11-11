"""One step, from launch to publish: the GPU thread's body, the plan thread's
pre-plan, and the postprocess that waits on the GPU, checks stops and routes the
outputs. Owns both thread pools, since everything that runs on them is its code:
the pipeline gets back futures and events to wait on."""
from __future__ import annotations

import logging
import math
import threading
import time
import time as _time
from collections.abc import Collection, Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import torch

from mstar.communication.tensors import NameToTensorList, StoredOutputs
from mstar.engine import apply_dynamo_config
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources.sampler.config import SAMPLED_LOGPROB
from mstar.graph.runtime.base import (
    InlineColumn,
    PostprocessNeed,
    PostprocessRequest,
    PostprocessStep,
    RouteOutput,
    RouteTensors,
    SendInput,
)
from mstar.model.submodule_base import END_PARTITION, BatchedModelOutput, HostRows, Landing, chunk_loop_name
from mstar.profile.worker import WorkerProfileInfo
from mstar.utils import cpus, knobs
from mstar.utils.containers import ParallelList
from mstar.utils.device import DeviceCapability
from mstar.utils.profiler import PhaseLaps, phase_record, range_pop, range_push, set_step_key
from mstar.utils.threads import InlineExecutor, ThreadPool, pin_thread
from mstar.worker.host_copies import HostCopies, _host_column, _LazyHostOutputs, crop_to_spans
from mstar.worker.lockstep import Lockstep
from mstar.worker.micro_scheduler import CHAIN_LANE, ScheduledBatch
from mstar.worker.phase_timer import PhaseTimer, step_key
from mstar.worker.preemption import Preemption
from mstar.worker.request_lifecycle import RequestLifecycle
from mstar.worker.services import WorkerLayout, WorkerServices
from mstar.worker.speculation import Speculation
from mstar.worker.step_repeats import stream_run
from mstar.worker.worker_metrics import WorkerMetrics

logger = logging.getLogger(__name__)


@dataclass
class PendingBatch:
    batch: ScheduledBatch
    node_batch: ExecutingBatch
    node_name: str
    partition: str
    graph_walk: str
    future: Future
    speculative_new_iter: bool = False
    loop_name: str = None
    # The leader's broadcast seq for this batch (``ScheduleTPNode.spec_seq``).
    tp_seq: int = -1
    # perf_counter when the main thread handed it to the GPU thread, stamped
    # only while metrics are on and only off the speculative path: a
    # speculated step is handed off before the step it follows finishes, so
    # its step time starts at that finish anyway (see WorkerMetrics.step_done)
    submitted_at: float = 0.0
    # the chain's step, or a side lane's (``StepPipeline._launch_side``)
    lane: int = CHAIN_LANE


class ModelStops(NamedTuple):
    """``StepExecutor._check_stops``: the model's stops by rid, the rows it failed
    (which leave the batch), and the rows whose partition it ended."""
    stops: dict[int, set[str]]
    dropped: list[int]
    ended: list[int]


class GpuStepBody:
    """What runs on the GPU thread: wait for the pre-plan, read check, prepare,
    admit and forward, then publish."""

    def __init__(
        self, services: WorkerServices, timer: PhaseTimer, lockstep: Lockstep,
        lifecycle: RequestLifecycle, host_copies: HostCopies,
        device_capability: DeviceCapability, enable_nvtx: bool,
        tensor_streams: Mapping[tuple[str, str], frozenset[str]] | None = None,
    ):
        self.worker_id = services.worker_id
        self.device = services.device
        self.engine_manager = services.engine_manager
        self._graph_runtime = services.graph_runtime
        self.timer = timer
        self.lockstep = lockstep
        self.lifecycle = lifecycle
        self._host_copies = host_copies
        self.device_capability = device_capability
        self.enable_nvtx = enable_nvtx
        # (node, walk) -> the loop-back signals the node keeps on the device
        self._device_loopback: dict[tuple[str, str], frozenset[str]] = {}
        # (node, walk) -> its streamed signals that go as tensors (``WorkerLayout``)
        self._tensor_streams = tensor_streams or {}

    def __call__(
        self,
        batch: ScheduledBatch,
        node_batch: ExecutingBatch,
        plan_future: Future | None = None,
    ) -> dict[int, NameToTensorList]:
        """Run the engine on the GPU executor thread.

        The NVTX range bracketing this call is ``synchronize=False`` —
        adding a ``cudaDeviceSynchronize`` at the marker boundary would
        drain the GPU on every iter and hide the overlap between
        post-processing and the next step's kernel execution.

        Once the step's work is submitted we record a CUDA event on the
        default stream; anything that reads the output VALUES waits on it.
        """
        timed = self.timer.period
        t_start = _time.perf_counter() if timed else 0.0
        c_start = _time.thread_time() if timed else 0.0
        if timed:
            set_step_key(step_key(batch.node_name, batch.graph_walk))
        laps = PhaseLaps("worker.gpu_thread.")
        engine = self.engine_manager.get_engine(batch.node_name)
        if self.enable_nvtx:
            range_push("worker.gpu_thread_start", synchronize=False)
            range_pop(synchronize=False)
            range_push(
                f"worker[{self.worker_id}].node[{batch.node_name}].graph_walk[{batch.graph_walk}]",
                synchronize=False,
            )
        try:
            # Every output of this step stays on the device: the engine keeps
            # the rows and skips the per-request dicts and tail (cached per node
            # and walk, a dict read here).
            node_batch.rows_only = self.all_on_device(
                engine, batch.node_name, batch.graph_walk, batch.output_signals,
            )
            if batch.request_walks:
                node_batch.loopback_rids = self._loopback_rids(engine, batch)
            if logger.isEnabledFor(logging.DEBUG):
                # test/waypoint/serve_rollout.py parses this line (wire ids,
                # logged before prepare_inputs runs) to recover the DiT schedule.
                # Wire ids come from the infos; the runtime may be mid-call on
                # the main thread.
                infos = node_batch.per_request_info
                logger.debug(
                    "Executing: %s graph_walk=%s %s", node_batch.node_name,
                    batch.graph_walk, [infos[r].request_id for r in node_batch.request_ids],
                )
            laps.lap("setup")
            # The plan thread wrote this batch's plan into the resources; wait
            # for it before the forward reads them. Waiting releases the GIL —
            # which is the point: running prepare_inputs here instead just puts
            # the two threads in contention for it, and measured worse.
            if plan_future is not None:
                with self.timer.span("worker.gpu_thread.await_plan"):
                    plan_future.result()
            laps.skip()
            try:
                if node_batch.read_check_rids is not None:
                    with self.timer.span("worker.gpu_thread.read_check"):
                        self.lockstep.drop_rows_whose_read_failed(node_batch)
                with self.timer.span(
                    "worker.gpu_thread.prepare_inputs",
                    cpu="worker.gpu_thread.prepare_inputs.cpu",
                ):
                    engine.prepare_inputs(node_batch)
            except Exception:
                if plan_future is not None and node_batch.preplanned_rids is not None:
                    # The stage was for this batch, which now never runs.
                    engine.reset_pre_plan_for_batch(node_batch)
                raise
            # call is_stale after prepare_inputs because prepare_inputs may drop rids
            laps.skip()
            if plan_future is not None and engine.preplan_is_stale(node_batch):
                engine.reset_pre_plan_for_batch(node_batch)
            execution_stream = (
                torch.accelerator.current_stream(self.device)
                if self.device_capability.is_accelerator
                else None
            )
            # Device time of the step; the phase timers only see host submit
            # time, and both are needed to tell a starved iter from a busy one.
            gpu_start = None
            if self.timer.period and execution_stream is not None:
                gpu_start = torch.Event(device=self.device, enable_timing=True)
                gpu_start.record(execution_stream)
            if self.timer.period:
                # after prepare_inputs, which can drop rids — this is the row
                # count the forward really pays for
                self.timer.record(
                    f"rows.{batch.graph_walk}#", len(node_batch.request_ids),
                )
            laps.lap("pre_exec")
            with self.timer.span(
                "worker.gpu_thread.exec", cpu="worker.gpu_thread.exec.cpu",
            ):
                outputs = engine.exec_and_postprocess(node_batch)
            laps.skip()
            if execution_stream is not None:
                self._host_copies.stage_host_rows(outputs)
                laps.lap("stage_host_rows")
                event = torch.Event(device=self.device)
                event.record(execution_stream)
                node_batch.completion_event = event
                laps.lap("completion_event")
                if gpu_start is not None:
                    gpu_end = torch.Event(device=self.device, enable_timing=True)
                    gpu_end.record(execution_stream)
                    self.timer.record_gpu_span(gpu_start, gpu_end, step_key(batch.node_name, batch.graph_walk))
                    laps.lap("gpu_exec_events")
            return outputs
        finally:
            # Safety net: a step that raised before publishing outputs or
            # committing would otherwise strand the plan thread preparing the
            # next one. The engine does this too on its own paths; here covers
            # a raise outside them.
            node_batch.release_waiters()
            # Publish each resource's durable state onto per_request_info so
            # the next iter's prep and the conductor see it. Runs regardless
            # of success, allocation failure, or an uncaught raise —
            # finalize_batch reads whatever state the engine actually reached.
            laps.skip()
            publish_rids = self.lifecycle.publishable(node_batch)
            node_batch.resource_publish_info = engine.finalize_batch(
                node_batch, publish_request_ids=publish_rids,
            )
            laps.lap("finalize_batch")
            if timed:
                self.timer.record(
                    "worker.gpu_thread.total", _time.perf_counter() - t_start,
                )
                self.timer.record(
                    "worker.gpu_thread.total.cpu", _time.thread_time() - c_start,
                )
                set_step_key(None)
            if self.enable_nvtx:
                range_pop(synchronize=False)

    def all_on_device(self, engine, node_name: str, graph_walk: str, signals) -> bool:
        """Whether every output signal of this step is one the node keeps on
        the device (cached per node and walk: it depends on the node's
        resources, not on the step). Not one that streams as a tensor: its
        stream needs the tensor stored."""
        # without inline emit, a rows-only step has no path for the client's tokens
        if not signals or not self._graph_runtime.supports_inline_emit:
            return False
        key = (node_name, graph_walk)
        on_device = self._device_loopback.get(key)
        if on_device is None:
            on_device = self._device_loopback[key] = frozenset(
                engine.device_loopback_signals(node_name, graph_walk)
            ) - self._tensor_streams.get(key, frozenset())
        return all(s in on_device for s in signals)

    def walk_signals(self, batch: ScheduledBatch, walk: str):
        signals = batch.walk_output_signals.get(walk)
        if signals is not None:
            return signals
        return self._graph_runtime.get_output_signals(batch.node_name, walk)

    def _loopback_rids(self, engine, batch: ScheduledBatch) -> frozenset:
        """A mixed step's rows whose own walk keeps every output signal on
        the device. Asked per real walk: the combined label is no walk of the
        submodule's, and would keep none."""
        on_device = {
            walk for walk in set(batch.request_walks.values())
            if self.all_on_device(
                engine, batch.node_name, walk, self.walk_signals(batch, walk),
            )
        }
        if not on_device:
            return frozenset()
        return frozenset(
            rid for rid, walk in batch.request_walks.items() if walk in on_device
        )


class StepExecutor:
    def __init__(
        self, services: WorkerServices, layout: WorkerLayout, timer: PhaseTimer,
        lockstep: Lockstep, lifecycle: RequestLifecycle, preemption: Preemption, *,
        wakeup_event=None, metrics: WorkerMetrics | None = None,
        profile_info: WorkerProfileInfo | None = None,
        device_capability: DeviceCapability | None = None,
        enable_nvtx: bool = False, enable_prof: bool = False,
        step_cpus: cpus.StepCpus | None = None, process_cpus: frozenset[int] = frozenset(),
    ):
        self.worker_id = services.worker_id
        self.device = services.device
        self._graph_runtime = services.graph_runtime
        self.engine_manager = services.engine_manager
        self.tensor_manager = services.tensor_manager
        self.request_state = services.request_state
        self.streams = services.streams
        self._rebuilds = services.rebuilds
        self._layout = layout
        self.timer = timer
        self.lockstep = lockstep
        self.lifecycle = lifecycle
        self.preemption = preemption
        self.wakeup_event = wakeup_event
        self._metrics = metrics
        self.profile_info = profile_info if profile_info is not None else WorkerProfileInfo()
        self.device_capability = device_capability or DeviceCapability.of(services.device)
        self.enable_nvtx = enable_nvtx
        self.enable_prof = enable_prof
        self._step_cpus = step_cpus
        self._process_cpus = process_cpus
        self._host_copies = HostCopies(device=services.device, device_capability=self.device_capability)
        self._body = GpuStepBody(
            services=services, timer=timer, lockstep=lockstep, lifecycle=lifecycle,
            host_copies=self._host_copies, device_capability=self.device_capability, enable_nvtx=enable_nvtx,
            tensor_streams=layout.tensor_streams,
        )
        self._gpu_pool: ThreadPool | InlineExecutor | None = None
        self._plan_pool: ThreadPool | InlineExecutor | None = None
        # ``knobs: step_threads: serial``: the GPU and plan threads' work runs on the main thread
        self._serial_steps = False
        # lane -> its stream, made on the lane's first step
        self._lane_streams: dict[int, torch.Stream] = {}
        self._stop_token_signal: dict[tuple[str, str], str] = {}
        if services.engine_manager is not None and layout.my_walks:
            self._declare_stop_policies(layout.my_walks)

    # -- threads ---------------------------------------------------------------

    def start(self) -> None:
        """Create the GPU thread and, with pre-plan on, the plan thread. Nothing
        is spawned before this, so a test can build a StepExecutor without threads.

        The async worker path needs decode submission to return quickly so the
        main loop can overlap queue/tensor polling and post-processing with GPU
        execution, so the engine always runs on a dedicated 1-worker thread.

        The plan thread pre-plans the speculatively built next batch, concurrent
        with the main thread's await_gpu (which releases the GIL), so plan()'s
        Python work isn't contended by postprocess. With double-buffered wrappers
        plan(N+1) runs concurrent with replay(N) on the disjoint slot, gated on
        N's commit event (right after advance_seq_lens(N) on the GPU thread).

        Default on; ``knobs: {pre_plan: false}`` or MSTAR_PRE_PLAN_SPEC=0 falls
        back to the double-buffer-without-pre-plan baseline.
        """
        self._serial_steps = knobs.step_threads() == "serial"
        self._gpu_pool = self._engine_thread_pool("mstar-gpu")
        logger.info(
            "Worker %s: engine runs %s", self.worker_id,
            "on the main thread (step_threads: serial)" if self._serial_steps else "on dedicated GPU thread",
        )
        if knobs.pre_plan_enabled():
            self._plan_pool = self._engine_thread_pool("mstar-plan")
            if not self._serial_steps:
                logger.info(
                    "Worker %s: plan_executor enabled — speculative plan() "
                    "pre-runs on a dedicated thread",
                    self.worker_id,
                )

    def _engine_thread_pool(self, name: str) -> ThreadPool | InlineExecutor:
        """A one-thread pool for the GPU or plan thread, started here, before the loop: before torch
        2.12 dynamo's config is process-wide, and a plan thread whose initializer ran at its first
        pre-plan could read and write the recompile limit while the GPU thread's compiled call patches it
        (``with_recompile_limit``). ``apply_dynamo_config`` never lowering the limit covers a limit
        raised before the pools start (a DiT's, at compile time), not that interleaving. A failed
        initializer also stops the worker here rather than failing every step."""
        if self._serial_steps:
            self._init_engine_thread()
            return InlineExecutor()
        pool = ThreadPool(
            max_workers=1, thread_name_prefix=f"{name}-{self.worker_id}",
            initializer=self._init_engine_thread,
        )
        pool.submit(int).result()
        return pool

    def pin_this_thread(self) -> None:
        if self._step_cpus is not None:
            pin_thread(self._step_cpus.cpus, self._process_cpus)

    def _init_engine_thread(self) -> None:
        """Pin this executor thread to the worker's accelerator device.

        The CUDA current device is per-thread and defaults to 0. PyTorch
        ops carry per-tensor device guards, but raw Triton launches and
        bare ``torch.cuda.current_stream()`` / ``synchronize()`` calls
        resolve against the THREAD's device — on a worker whose model
        lives on a non-zero device, work issued from an unpinned thread
        lands on device 0's stream, unordered with the real compute.

        It also applies mstar's dynamo config, because dynamo config is
        per-thread since torch 2.12 and this thread compiles, and moves the
        thread to the step CPUs.
        """
        if self.device_capability.is_accelerator and self.device.index is not None:
            torch.accelerator.set_device_index(self.device)
        apply_dynamo_config()
        self.pin_this_thread()

    # -- launch ----------------------------------------------------------------

    def arm_preplan(self, pending: PendingBatch | None, speculation: Speculation) -> None:
        """Hand N+1 to the plan thread now. It waits on N's commit event and
        nothing else, then reserves a slot and pre-plans, while the main thread
        sits in await_gpu with the GIL released and N's kernels are still running."""
        if self._plan_pool is not None:
            speculation.plan_future = self._plan_pool.submit(self._preplan_spec, pending, speculation)

    def lane_stream(self, lane: int) -> torch.Stream:
        stream = self._lane_streams.get(lane)
        if stream is None:
            stream = self._lane_streams[lane] = torch.Stream(device=self.device)
        return stream

    def _execute_on_lane(self, lane, batch, node_batch, plan_future):
        """A side lane's step, on its stream, with nothing to wait for on the
        main one: the streamed items it reads had landed before they were routed
        to it (postprocess waits for the step that made them), and the pipeline
        collated its windows on this stream."""
        with self._lane_streams[lane]:
            outputs = self._body(batch, node_batch, plan_future)
        if outputs.host_rows is not None or outputs.host_spans is not None:
            # the pinned ring they stage through is reused by the chain's next
            # steps before this one lands (refused at startup where declared)
            raise RuntimeError(
                f"{batch.node_name} {batch.graph_walk!r} returned stop buffers or output spans in a side step"
            )
        return outputs

    @staticmethod
    def landed(step: PendingBatch) -> bool:
        """A step's future is done and its completion event has fired."""
        if not step.future.done():
            return False
        event = step.node_batch.completion_event
        return event is None or event.query()

    def _release_finished_prompts(self, batch: ScheduledBatch) -> None:
        """Let go of the prepared input of each prompt whose last chunk just
        landed; earlier chunks are cut from it (``Engine._prepare_chunk``)."""
        engine = self.engine_manager.get_engine(batch.node_name)
        for rid, chunk in batch.chunk_ranges.items():
            if chunk.is_final and rid in batch.request_to_worker_graph:
                engine.release_chunk_inputs(rid, batch.node_name)

    def _count_new_tokens(
        self,
        new_token_idxs: list[int],
        flat_rids: list[int],
        flat_uuids: list[int],
        signals: list[str],
        signal_idxs: list[int],
    ) -> dict[int, dict[str, int]]:
        """Token counts for the conductor, per rid. Needs numel(), hence the
        tensors -- which is why this stays here and not behind the contract.

        Indices into this batch's columns, which also carry the rid and the
        signal, so the runtime hands over no strings for this.

        First edge of a signal wins, per rid. The runtime does NOT drop repeat
        edges (``new_token_outputs`` is a bare filter), and one output routed
        to two destinations is two edges carrying the same tensors, so summing
        them would double every token.
        """
        counts: dict[int, dict[str, int]] = {}
        for idx in new_token_idxs:
            per_rid = counts.setdefault(flat_rids[idx], {})
            signal = signals[signal_idxs[idx]]
            if signal in per_rid:
                continue
            per_rid[signal] = self.tensor_manager.get_tensor(
                flat_uuids[idx]
            ).numel()
        return counts

    def _profiling_payloads(self, rids: list[int]) -> ParallelList:
        """rx/tx/timings for each rid's WORKER_GRAPHS_DONE."""
        return ParallelList(rids, [
            (
                self.tensor_manager.get_rx_info(rid),
                self.tensor_manager.get_tx_info(rid),
                self.profile_info.per_rid_graph_timings.get(rid, {}),
            ) for rid in rids
        ])

    def _register_outputs(
        self,
        route_output: RouteOutput,
    ) -> frozenset[int]:
        """Register the tensors the runtime says remote consumers (other
        workers, the api server, the conductor via persist) will read. It has
        already deduped them by uuid. A request whose tensors the transfer
        engine refuses fails alone; returns those rids, so that nothing of
        theirs is sent.
        """
        # Uuids, not descriptors: registration only ever reads ``uuid`` off
        # them and takes the tensor from the store.
        per_rid: dict[int, list[int]] = {}
        for uuid, request_id in zip(
            route_output.register_uuids,
            route_output.register_rids,
            strict=True,
        ):
            per_rid.setdefault(request_id, []).append(uuid)
        if not per_rid:
            return frozenset()
        # One staging pass for the batch: the arena collapses what would be one
        # host-blocking D2H stream sync per request into one.
        failed = self.tensor_manager.register_for_send_uuids(
            ParallelList(list(per_rid), list(per_rid.values())),
            skip_cuda_sync=True,
        )
        if not failed:
            return frozenset()
        # A follower still runs a failed rid while rank 0 sends it; report once.
        self.lifecycle.fail({
            rid: f"Error in worker: {msg}" for rid, msg in failed.items()
            if rid not in self.lifecycle.failed_rids
        })
        return frozenset(failed)

    def _preplan_spec(
        self,
        pending: PendingBatch | None,
        speculation: Speculation,
    ) -> bool:
        """Pre-plan the speculative batch on the plan thread.

        Waits on batch N's ``commit_done`` — its resource state has to be
        committed before N+1 can admit and plan against it — and nothing else.
        The batch is not prepared yet, so the plan runs against the capture
        config's shape for the batch size; only a batched capture can serve
        that. See ``Engine.pre_plan_for_batch`` for what moving
        ``prepare_inputs`` ahead of the forward would take (and unlock).

        Returns True when the batch was pre-planned; False means ``exec``
        plans it inline, which is always correct, just slower.
        """
        spec_batch = speculation.node_batch
        engine = self.engine_manager.get_engine(spec_batch.node_name)

        if not engine.can_pre_plan(spec_batch.node_name):
            return False
        if self.timer.period:
            sb = speculation.scheduled_batch
            set_step_key(step_key(sb.node_name, sb.graph_walk))
        try:
            if pending is not None:
                # A step N this slow (a long admit, say) skips the pre-plan
                # rather than holding plan_executor; exec plans N+1 inline.
                with self.timer.span("worker.plan_thread.await_commit"):
                    committed = pending.node_batch.commit_done.wait(timeout=10.0)
                if not committed:
                    logger.warning(
                        "Worker %s: plan_executor timed out waiting for "
                        "batch N commit; skipping pre-plan", self.worker_id,
                    )
                    return False
            with self.timer.span("worker.plan_thread.reserve_slot"):
                leased = engine.reserve_replay_slot(spec_batch) is not None
            if not leased:
                return False  # eager, or no batched capture for this shape
            with self.timer.span(
                "worker.plan_thread.pre_plan",
                cpu="worker.plan_thread.pre_plan.cpu",
            ):
                return engine.pre_plan_for_batch(spec_batch)
        except Exception:
            logger.exception("Worker %s: plan_executor pre-plan failed", self.worker_id)
            self.reset_skip_plan_flags(spec_batch)
            return False
        finally:
            if self.timer.period:
                set_step_key(None)

    def reset_skip_plan_flags(self, spec_node_batch: ExecutingBatch) -> None:
        """Drop the pre-plan staged for ``spec_node_batch``.

        Used when the pre-plan was dispatched but the batch never reached the
        GPU thread: the resources would otherwise promote that stale plan into
        the next step that leases the same slot. Targeted at this batch's own
        lease, so a different slot's valid in-flight pre-plan isn't stomped.
        """
        engine = self.engine_manager.get_engine(spec_node_batch.node_name)
        engine.reset_pre_plan_for_batch(spec_node_batch)

    def _submit_gpu_step(
        self, batch: ScheduledBatch, node_batch: ExecutingBatch,
        plan_future: Future | None, lane: int,
    ) -> Future:
        """Submit a step to the GPU thread with a fresh launch event.

        The engine sets the event at launch. A step that ends without one (a
        refused admit, a raise, an executor that never ran it) sets it on
        completion, so the admit fence and the submitter's wait always lift.
        """
        batch.chunk_loop_walks = self._layout.chunk_loops.walks_by_node.get(batch.node_name, frozenset())
        event = node_batch.launch_started_event = threading.Event()
        if lane == CHAIN_LANE:
            future = self._gpu_pool.submit(self._body, batch, node_batch, plan_future)
        else:
            future = self._gpu_pool.submit(self._execute_on_lane, lane, batch, node_batch, plan_future)
        future.add_done_callback(lambda _f: event.set())
        return future

    def submit(
        self, batch: ScheduledBatch, node_batch: ExecutingBatch, *,
        lane: int = CHAIN_LANE, plan_future: Future | None = None,
        speculated: bool = False, **step: Any,
    ) -> PendingBatch:
        """Hand a step to the GPU thread as ``lane``'s. A ``speculated`` step was
        flagged in flight when its speculation settled, and is not timed from here
        (see ``PendingBatch.submitted_at``). Which rows the lane holds is the
        pipeline's to record: the chain's step N+1 takes over only once N is
        postprocessed."""
        if not speculated:
            self._set_in_flight(batch, True)
        submitted_at = _time.perf_counter() if self._metrics is not None and not speculated else 0.0
        future = self._submit_gpu_step(batch, node_batch, plan_future, lane)
        self.wakeup_event.register_future(future)
        step.setdefault("partition", self.request_state.get_partition_for_node(batch.node_name))
        return PendingBatch(
            batch=batch, node_batch=node_batch, node_name=batch.node_name,
            graph_walk=batch.graph_walk, future=future, submitted_at=submitted_at,
            lane=lane, **step,
        )

    def _set_in_flight(self, batch: ScheduledBatch, value: bool) -> None:
        self._graph_runtime.set_in_flight(
            batch.node_name, list(batch.graph_rows()),
            batch.wg_ids(), value,
        )

    def clear_in_flight(self, batch: ScheduledBatch) -> None:
        """The graph runtime's flag that keeps the rows' node off the ready queue;
        not the pipeline's per-lane rows, which defer teardowns."""
        self._set_in_flight(batch, False)

    def wait_for_step(self, batch_N: PendingBatch) -> None:
        """Wait for batch N's GPU work: its completion event, or the whole
        device when it has none."""
        if self.device.type == "cpu":
            return
        if batch_N.node_batch.completion_event is not None:
            with self.timer.span("worker.postprocess.event_sync"):
                batch_N.node_batch.completion_event.synchronize()
        else:
            with self.timer.span("worker.postprocess.device_sync"):
                torch.accelerator.synchronize(self.device)

    def _finish_rebuild_rows(self, batch_N: PendingBatch, outputs: BatchedModelOutput) -> None:
        """Take the step's rebuild rows off it and advance their rebuilds. A row's last chunk moves its rebuild to
        the next row, and the last row's lands the rebuild and lets its request's decode row run."""
        batch, node_batch = batch_N.batch, batch_N.node_batch
        rebuild = batch.rebuild_rids
        if not rebuild:
            return
        node = batch_N.node_name
        # the engine also skips a row the prefix cache served whole, whose rebuild is done (below)
        skipped = (rebuild & node_batch.skipped_rids) - node_batch.cached_rids
        if skipped:
            get_rid_string = self._graph_runtime.get_rid_string
            self.lifecycle.fail({
                rid: f"Error in worker: {node} skipped the rebuild row of {get_rid_string(rid)}, whose inputs come "
                "from its kept ids" for rid in skipped
            })
        # a row the prefix cache served whole ran no forward, and its rebuild is done all the same
        ran = set(node_batch.request_ids) | node_batch.cached_rids
        ended = {
            rid: run for rid in rebuild
            if rid in ran and (run := self._rebuilds.land_chunk(rid, node, batch.chunk_ranges.get(rid)))
        }
        if ended:
            # the step's kernels read the prepared input released here
            self.wait_for_step(batch_N)
            engine = self.engine_manager.get_engine(node)
            for rid, run in ended.items():
                self.preemption.rebuilt(rid, node, run.ran)
                # A row from a record prepared its own input. A prompt's step in progress goes on from the
                # input the engine keeps for the prompt, which waited aside while the rows before it ran.
                if run.row.record is not None:
                    engine.release_chunk_inputs(rid, node)
                if self._rebuilds.rebuilding(rid, node):
                    if self._rebuilds.cursor(rid, node).resumes_prefill:
                        engine.restore_chunk_inputs(rid, node)
                    continue
                engine.land_rebuild(node, rid)
        for rid in rebuild:
            outputs.pop(rid, None)
            batch.discard_rid(rid)
            batch.chunk_ranges.pop(rid, None)
            node_batch.per_request_info.pop(rid, None)
        batch.incomplete_node_rids -= rebuild
        node_batch.cached_rids -= rebuild
        node_batch.request_ids = [rid for rid in node_batch.request_ids if rid not in rebuild]

    def postprocess(
        self, batch_N: PendingBatch,
        outputs: BatchedModelOutput,
        clear_loop_stops: bool = True,
    ):
        # Stage stopwatch named like the NVTX ranges below; a clock stamp, not
        # a nested `_span`, because of the two early returns.
        _pp_t = _time.perf_counter() if self.timer.period else 0.0
        def _pp_stage(name: str) -> None:
            nonlocal _pp_t
            if not self.timer.period:
                return
            now = _time.perf_counter()
            self.timer.buf[f"worker.postprocess.{name}"].append(now - _pp_t)
            _pp_t = now

        self._finish_rebuild_rows(batch_N, outputs)
        if self.enable_nvtx:
            range_push("worker.postprocess.cleanup_inputs", synchronize=False)

        # a row mid-prefill keeps its inputs: its next chunk reads them again
        held = batch_N.batch.incomplete_node_rids
        rids = [
            r for r in batch_N.batch.request_to_worker_graph if r not in held
        ] if held else list(batch_N.batch.request_to_worker_graph)
        self.lockstep.consume_reads(batch_N.node_batch)
        # What the runtime dereferenced to zero and cannot reclaim itself: a
        # runtime behind the contract has the bookkeeper, not the shm files or
        # the registered memory.
        self.tensor_manager.cleanup_collectable(
            *self._graph_runtime.cleanup_consumed_inputs(
                batch_N.batch.node_name, rids,
                [batch_N.batch.request_to_worker_graph[r] for r in rids],
            )
        )
        _pp_stage("cleanup_inputs")
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.pending_loop_stops", synchronize=False)
        # If any nodes in the batch have "overstayed" their loop stop, then make
        # sure to not route their outputs. A row the cache served, or one that
        # declined its stream's final chunk, ran no forward but completes its node.
        valid_rids = set(batch_N.node_batch.request_ids) | batch_N.node_batch.no_forward_rids
        if batch_N.speculative_new_iter:
            stopped_rids = self._graph_runtime.pending_loop_stop_rids(
                list(batch_N.batch.rids_by_walk()), batch_N.loop_name,
            )
            stopped_rids &= valid_rids
            for stopped_rid in stopped_rids:
                outputs.pop(stopped_rid, None)
                valid_rids.discard(stopped_rid)
                batch_N.batch.discard_rid(stopped_rid)
                batch_N.node_batch.per_request_info.pop(stopped_rid, None)
                # its loop already stopped; see ExecutingBatch.drop_rids
                batch_N.node_batch.final_stream_rids.discard(stopped_rid)
                batch_N.node_batch.partition_done_rids.discard(stopped_rid)
        # keep the forward's order: other code walks this list positionally
        batch_N.node_batch.request_ids = [
            rid for rid in batch_N.node_batch.request_ids if rid in valid_rids
        ]
        if not valid_rids:
            # no outputs leave, but the step's kernels may still be running
            # in the slot a later step reuses
            self.wait_for_step(batch_N)
            if self.enable_nvtx:
                range_pop(synchronize=False)
            return

        # pending stops are only needed for one iteration of the chain, so can
        # be cleared now; not by a side lane's step (``_finish_step``), which
        # lands between two chain steps: the chain's next one has yet to read them.
        if clear_loop_stops:
            self._graph_runtime.clear_pending_loop_stops()

        # A rid failed or drained after this step was built is torn down, so
        # none of its outputs may leave: the step can already be in flight
        # when the rid fails (the stop check of the step before it, say).
        # Not on a follower: rank 0 decides what runs there, and may still
        # send a step that needs this one's loop-back.
        if (
            self.lifecycle.failed_rids
            and not self.lockstep.follows(batch_N.node_name)
        ):
            torn_down = self.lifecycle.failed_rids & valid_rids
            if torn_down:
                self.drop_failed_rids(batch_N, outputs, torn_down)
                valid_rids -= torn_down
                if not valid_rids:
                    # no outputs leave, but the step's kernels may still be
                    # running in the capture slot a later step reuses
                    self.wait_for_step(batch_N)
                    if self.enable_nvtx:
                        range_pop(synchronize=False)
                    return

        # An engine can drop rids that were skipped during execution (a
        # submodule's prepare_inputs returned None) from node_batch.request_ids,
        # but it cannot reach the worker-side ScheduledBatch. Reconcile it here so
        # the routing/output loops below only touch rids that ran or that
        # complete without a forward.
        for rid in list(batch_N.batch.request_to_worker_graph):
            if rid not in valid_rids:
                batch_N.batch.discard_rid(rid)

        _pp_stage("pending_loop_stops")
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.synchronize_completion_event", synchronize=False)

        # Wait for batch N's completion event before proceeding
        # TODO: may need to refine this based on how it affects performance?
        inline_start = 0
        if batch_N.batch.request_to_worker_graph:
            inline_start = self._stream_run(batch_N, outputs)
            self.wait_for_step(batch_N)

        if self.enable_prof:
            batch_N.node_batch.exec_timings.fwd_end = time.perf_counter()

        _pp_stage("completion_event_sync")
        if batch_N.batch.chunk_ranges:
            self._release_finished_prompts(batch_N.batch)
        # Waited on above, so the spans have landed; cut before anything reads
        # the outputs: the stop check, the store, a streaming edge.
        empty_spans: dict[int, list[str]] = {}
        if outputs.output_spans and batch_N.batch.request_to_worker_graph:
            empty_spans = crop_to_spans(outputs)
            _pp_stage("crop_spans")
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.prematerialize", synchronize=False)

        # Prematerialising pulls the stop check's rows to the host, so this can
        # carry a device transfer.
        engine = self.engine_manager.get_engine(batch_N.node_name)
        cpu_outputs, host_rows = self._host_copies.prematerialize_for_check_stop(
            outputs, batch_N.node_batch.completion_event,
            request_ids=batch_N.node_batch.request_ids,
        )
        if host_rows is not None and outputs.landing is not None:
            self._settle_run(batch_N, host_rows, outputs.landing)
        self._rebuilds.record(batch_N.node_batch, {
            rid: self.request_state.get_fwd_info(rid, batch_N.partition) for rid in batch_N.node_batch.cached_rids
        })
        # Until its last chunk a row's sampled token is not a real sample, so it
        # keys no prefix and decides no stop (`_check_stops`).
        engine.step_finished(
            batch_N.node_batch, cpu_outputs, host_rows=host_rows,
            skip_rids=batch_N.batch.incomplete_node_rids,
        )
        forgot = self._rebuilds.keep_samples(
            batch_N.node_batch, cpu_outputs, host_rows, skip_rids=batch_N.batch.incomplete_node_rids,
        )
        self.preemption.refuse_drops(forgot, batch_N.node_name)
        _pp_stage("prematerialize")
        runtime_stops, tokens = self._runtime_stop_column(batch_N, host_rows)
        checked = None
        if not runtime_stops:
            # the model's check_stop decides, before the runtime, which then
            # routes without stopping for it
            checked = self._check_stops(batch_N, outputs, cpu_outputs, host_rows, engine).stops
            _pp_stage("check_stop")
        request = self._get_postprocess_request(
            batch_N, host_rows, engine, runtime_stops, tokens, checked, inline_start,
        )
        if self.enable_nvtx:
            range_pop(synchronize=False)
            range_push("worker.postprocess.runtime", synchronize=False)

        # The runtime checks the step's stops, routes its outputs and sends
        # them, stopping for what only the worker has (PostprocessNeed).
        # Main loop only: the runtime holds `&mut self` across its GIL
        # release, so a concurrent caller gets "Already mutably borrowed"
        # rather than blocking.
        if host_rows is not None and self._layout.step_repeats is not None:
            info_of = partial(self.request_state.get_fwd_info, partition_name=batch_N.partition)
            engine.record_tokens(batch_N.node_batch, host_rows, info_of)
        rt = self._graph_runtime
        _t_rt = _time.perf_counter() if self.timer.period else 0.0
        step = rt.begin_postprocess(request)
        rt_time = _time.perf_counter() - _t_rt if self.timer.period else 0.0
        stored = None
        sent_here = False
        while step.need is not PostprocessNeed.DONE:
            _t = _time.perf_counter() if self.timer.period else 0.0
            if step.need is PostprocessNeed.STOPS:
                model_stops = self._check_stops(batch_N, outputs, cpu_outputs, host_rows, engine)
                # after the request was built, which carries the step's other
                # partition ends; the runtime reports them on the same send
                for rid in model_stops.ended:
                    rt.mark_partition_done(rid, batch_N.partition)
                stage = "check_stop"
                _t_rt = _time.perf_counter() if self.timer.period else 0.0
                step = rt.resume_postprocess(stops=model_stops.stops, dropped=model_stops.dropped)
            elif step.need is PostprocessNeed.STORE:
                # What the stop check withheld (``StopCheck.withheld``), and an
                # output cut to an empty span, is not stored, so not routed: a
                # streaming edge then carries no item for the row, and the
                # producer's item count leaves it out as well.
                for withheld in (batch_N.node_batch.withheld_outputs, empty_spans):
                    for rid, names in withheld.items():
                        rid_outputs = outputs.get(rid)
                        if rid_outputs:
                            for name in names:
                                rid_outputs.pop(name, None)
                stored, route_tensors = self._store_outputs(
                    batch_N, outputs, request, step, cpu_outputs, host_rows, engine,
                )
                stage = "store_tensors"
                _t_rt = _time.perf_counter() if self.timer.period else 0.0
                step = rt.resume_postprocess(stored=route_tensors)
            else:
                send = self._send_input(batch_N, request, step, stored, engine)
                sent_here = True
                stage = "send_prep"
                _t_rt = _time.perf_counter() if self.timer.period else 0.0
                step = rt.resume_postprocess(send=send)
            if self.timer.period:
                now = _time.perf_counter()
                self.timer.record(f"worker.postprocess.{stage}", _t_rt - _t)
                rt_time += now - _t_rt
        if self.timer.period:
            self.timer.record("worker.postprocess.runtime", rt_time)
            _pp_t = _time.perf_counter()

        if self.enable_nvtx:
            range_pop(synchronize=False)
        if not sent_here and batch_N.batch.request_to_worker_graph:
            # sent without the worker (SEND does these before its send)
            self._buffer_publish(batch_N)
            rt.update_dynamic_loop_iters(
                batch_N.node_batch.per_request_info, batch_N.partition,
            )
        for rid in step.completed_rids:
            self.request_state.flush_publish_info(rid, batch_N.partition)
        # Those sends were the last readers of this step's host copies.
        self.tensor_manager.drop_host_copies()
        _pp_stage("tail")
        self._rebuilds.flush()
        engine.after_output_copy(batch_N.node_batch)
        _pp_stage("after_output_copy")

    @staticmethod
    def _settle_run(batch_N: PendingBatch, host_rows: HostRows, landing: Landing) -> None:
        """A run's landed width: a device run that ended short (every row
        stopped) wrote only the columns it ran, and a host run lands every
        replay, so its width stands. Everything after this reads the width the
        run really had, which the step's wait has landed. Only the run's own
        token columns are cut: a resource's host row (the log-probs of a
        one-iteration step) is one value a row, with no columns."""
        ran = landing.landed(0, 0)
        phase_record(f"repeats.{batch_N.graph_walk}.ran#", ran)
        for name, column in host_rows.buffers.items():
            if column.dim() > 1:
                host_rows.buffers[name] = column[:, :ran]
        batch_N.node_batch.step_context.repeats = ran

    def repeat_refusal(self, node_name: str, graph_walk: str) -> str | None:
        """Why no step of this walk can stand for several iterations, or None:
        its stop runs in the runtime and everything it routes stays on the device."""
        if self._stop_token_signal.get((node_name, graph_walk)) is None:
            return (
                "node: its stop is its own check_stop, which reads one token a step"
                " (declare a TokenStop without withhold)"
            )
        engine = self.engine_manager.get_engine(node_name)
        signals = self._graph_runtime.get_output_signals(node_name, graph_walk)
        if not self._body.all_on_device(engine, node_name, graph_walk, signals):
            return "node: an output of it doesn't stay on the device between steps"
        repeats = self._layout.step_repeats
        return engine.repeat_refusal(
            node_name, graph_walk, repeats.loop_back(node_name, graph_walk), repeats.limit(node_name),
        )

    def _runtime_stop_column(
        self, batch_N: PendingBatch, host_rows: "HostRows | None",
    ) -> tuple[bool, "torch.Tensor | None"]:
        """Whether the runtime checks the step's declared stop policies, and
        the token column they read (None when none reads one). A step that
        mixes walks, has no declared policies, or has no column keeps
        check_stop."""
        key = (batch_N.node_name, batch_N.graph_walk)
        if (
            key not in self._stop_token_signal or len(batch_N.batch.rids_by_walk()) > 1
            # a row mid-prefill decides no stop, so its step keeps check_stop
            or batch_N.batch.incomplete_node_rids
        ):
            return False, None
        token_signal = self._stop_token_signal[key]
        if token_signal is None:
            return True, None
        tokens = _host_column(host_rows, token_signal, batch_N.node_batch.step_context.repeats)
        return tokens is not None, tokens

    def _route_rows(self, batch_N: PendingBatch) -> tuple[list[int], list[int] | None, list[str], list[str]]:
        """The rows the step routes; each one's index into its walks when the
        step mixes walks (None for one walk); the walks; the signals they
        route under."""
        batch = batch_N.batch
        by_walk = batch.rids_by_walk()
        walks = list(by_walk) or [batch_N.graph_walk]
        if len(by_walk) <= 1:
            rids, rid_walk_idx = list(by_walk.get(walks[0], ())), None
            # a combined batch keeps its first walk's names, which may not be this walk's
            signals = (
                list(self._body.walk_signals(batch, walks[0])) if batch.request_walks
                else batch.output_signals
            )
        else:
            rids, rid_walk_idx = [], []
            for i, walk_rids in enumerate(by_walk.values()):
                rids.extend(walk_rids)
                rid_walk_idx.extend([i] * len(walk_rids))
            # several walks route under the union of their names; a walk's
            # node ignores the ones it lacks
            signals = list(dict.fromkeys(
                signal for walk in walks for signal in self._body.walk_signals(batch, walk)
            ))
        return rids, rid_walk_idx, walks, signals

    def _inline_column(
        self, batch_N: PendingBatch, host_rows: "HostRows | None", engine, signals: list[str],
        rid_walk_idx: list[int] | None,
    ) -> InlineColumn | None:
        """The step's client column, when it rides inline. Device loop-back:
        every routed signal stays on the device and the client's copy rides
        inline, so the route stores nothing. A step with log-probs stores
        instead: STORE passes them to the route."""
        if not (
            host_rows is not None and rid_walk_idx is None
            and SAMPLED_LOGPROB not in host_rows.buffers
            # an inline route carries every output, a withheld one too
            and not batch_N.node_batch.withheld_outputs
            and self._body.all_on_device(engine, batch_N.node_name, batch_N.graph_walk, signals)
        ):
            return None
        client = engine.inline_signals(batch_N.node_name, batch_N.graph_walk)
        signal = next((s for s in client if s in signals), None)
        column = None if signal is None else _host_column(
            host_rows, client[signal], batch_N.node_batch.step_context.repeats,
        )
        return None if column is None else InlineColumn(signal, column)

    def _stream_run(self, batch_N: PendingBatch, outputs: BatchedModelOutput) -> int:
        """While a repeated step runs, send the client its tokens as they land
        (``stream_run``): how many iterations went out, from which the step's
        postprocess sends the rest. Before the step's wait, over the rows its
        route takes: those the worker dropped (stopped a step ago, torn down)
        are gone from it already."""
        host_rows, event = outputs.host_rows, batch_N.node_batch.completion_event
        if (
            host_rows is None or event is None or outputs.landing is None
            or batch_N.node_batch.step_context.repeats < 2
        ):
            return 0
        runtime_stops, tokens = self._runtime_stop_column(batch_N, host_rows)
        if not runtime_stops or tokens is None:
            return 0
        rids, rid_walk_idx, walks, signals = self._route_rows(batch_N)
        engine = self.engine_manager.get_engine(batch_N.node_name)
        inline = self._inline_column(batch_N, host_rows, engine, signals, rid_walk_idx)
        if inline is None:
            return 0
        with self.timer.span("worker.postprocess.stream_run"):
            return stream_run(
                self._graph_runtime, batch_N.partition, batch_N.node_name, walks[0], rids, host_rows.request_ids,
                tokens, inline, outputs.landing, event,
            )

    def _get_postprocess_request(
        self, batch_N: PendingBatch, host_rows: "HostRows | None", engine,
        runtime_stops: bool, tokens: "torch.Tensor | None",
        checked: dict[int, set[str]] | None, inline_start: int = 0,
    ) -> PostprocessRequest:
        """The step's ``PostprocessRequest``: its rows, its stops known before
        the runtime's check (``checked``: the model's, when the runtime does
        not check), and the host columns the runtime reads in place.
        ``inline_start``: the iterations of the client column that went out
        while the step ran (``_stream_run``)."""
        batch = batch_N.batch
        repeats = batch_N.node_batch.step_context.repeats
        rids, rid_walk_idx, walks, signals = self._route_rows(batch_N)
        # Stream-terminated loop: a stream-consuming node inside a loop has no
        # internal stop signal (unlike a self-EOS loop), so when it consumes
        # the terminal chunk of its stream (``final_stream_rids``: the
        # StreamBuffer popped ``is_final``, which a ``continue_after_done``
        # policy never sets) end its innermost enclosing loop. Without this
        # the loop would spin to ``max_iters`` and the partition would never
        # report done.
        stops: dict[int, set[str]] = {}
        held = batch.incomplete_node_rids  # mid-prefill: decides no stop
        for rid in batch_N.node_batch.final_stream_rids:
            if rid in held:
                continue
            loop_name = self._layout.innermost_loop_by_node.get((batch.walk_of(rid), batch_N.node_name))
            if loop_name is not None:
                stops.setdefault(rid, set()).add(loop_name)
        for rid, loops in (checked or {}).items():
            stops.setdefault(rid, set()).update(loops)
        inline = self._inline_column(batch_N, host_rows, engine, signals, rid_walk_idx)
        if inline_start:
            if inline is None:
                raise RuntimeError(f"{batch_N.node_name}/{walks[0]}: tokens went out while the step ran, but its "
                                   "route no longer rides inline")
            inline = inline._replace(start=inline_start)
        return PostprocessRequest(
            partition=batch_N.partition, node_name=batch_N.node_name, graph_walk=walks[0],
            output_signals=signals,
            wg_ids=ParallelList(rids, [batch.request_to_worker_graph[r] for r in rids]),
            walks=walks if rid_walk_idx is not None else None, rid_walk_idx=rid_walk_idx,
            stops=stops, runtime_stops=runtime_stops, tokens=tokens,
            row_rids=host_rows.request_ids if host_rows is not None else (),
            inline=inline,
            # the consuming pass (not the earlier ingest) reports the partition
            # done, so it rides the WGD that also counts this pass's last chunks
            partition_done=list(batch_N.node_batch.partition_done_rids),
            python_send=self.enable_prof,
            loop_strides=batch.loop_strides(rids, repeats),
            token_width=repeats,
        )

    def _check_stops(
        self, batch_N: PendingBatch, outputs: BatchedModelOutput, cpu_outputs,
        host_rows: "HostRows | None", engine,
    ) -> ModelStops:
        """The model's ``check_stop``, less the ``END_PARTITION`` stops, which
        end their rows' partitions instead (``partition_done_rids``)."""
        stops = engine.check_stop_for_batch(batch_N.node_batch, cpu_outputs, host_rows=host_rows)
        # Until its last chunk a row's sampled token is not a real sample: a
        # stop would cut the prompt short. Its chunk Loop ends by its own count.
        for rid in batch_N.batch.incomplete_node_rids:
            stops.pop(rid, None)
        ended = []
        # Only a row outside any loop can end its pass here: inside one it
        # would stop only the innermost loop, and a speculated next iteration
        # would carry no report of it. Returning it from a loop is a model
        # bug, so the client sees an internal error. A row whose innermost
        # loop is its own chunk Loop is on its last chunk (above), where that
        # Loop ends by its count, so it counts as outside one.
        for rid in [rid for rid, s in stops.items() if END_PARTITION in s]:
            walk = batch_N.batch.walk_of(rid)
            loop = self._layout.innermost_loop_by_node.get((walk, batch_N.node_name))
            if loop is not None and loop != chunk_loop_name(batch_N.node_name, walk):
                batch_N.node_batch.register_failure(rid, RuntimeError(
                    f"check_stop of node {batch_N.node_name!r} returned END_PARTITION "
                    f"inside a loop of walk {walk!r}; it may end a partition only "
                    "from a row outside any loop"
                ), walk)
            else:
                del stops[rid]
                batch_N.node_batch.partition_done_rids.add(rid)
                ended.append(rid)
        if not batch_N.node_batch.failed_requests:
            return ModelStops(stops, [], ended)
        # A rid whose stop check raised (or was refused above) has no
        # trustworthy stop decision: routing it would either run its loop
        # forever or end it early. Only ever the ones this stop stage just
        # added: the caller already reported (and cleared) the rids that
        # failed in prepare_inputs / their forward / postprocess.
        failed = dict(batch_N.node_batch.failed_requests)
        self.drop_failed_rids(batch_N, outputs, failed)
        self.lifecycle.fail(failed)
        for rid in failed:
            stops.pop(rid, None)
        return ModelStops(stops, list(failed), ended)

    def _declare_stop_policies(self, walks: set[tuple[str, str]]) -> None:
        """Give the graph runtime each local (node, walk)'s declared stop
        policies (``NodeSubmodule.get_stop_policies``), and note the token
        signal each one reads. A walk whose policies read two token signals
        keeps ``check_stop``."""
        # (node, walk) -> the token signal its policies read, or None
        self._stop_token_signal: dict[tuple[str, str], str | None] = {}
        for node, walk in sorted(walks):
            policies = self.engine_manager.get_engine(node).get_stop_policies(node, walk)
            if policies is None:
                continue
            signals = {p.token_signal for p in policies.values() if p.reads_token}
            if len(signals) > 1:
                logger.warning(
                    "%s/%s: stop policies read several token signals %s; check_stop decides",
                    node, walk, sorted(signals),
                )
                continue
            self._graph_runtime.set_stop_policies(node, walk, policies)
            self._stop_token_signal[(node, walk)] = next(iter(signals), None)

    def _buffer_publish(self, batch_N: PendingBatch) -> set[int]:
        """Buffer this step's publication for the conductor; the rids that
        published. CurrentForwardPassInfo also contains publication inherited
        from peer ranks, so only what this worker produced goes."""
        buffered: set[int] = set()
        for rid, published in batch_N.node_batch.resource_publish_info.items():
            if published:
                self.request_state.buffer_publish_info(rid, batch_N.partition, published)
                buffered.add(rid)
        return buffered

    def _store_outputs(
        self, batch_N: PendingBatch, outputs: BatchedModelOutput,
        request: PostprocessRequest, step: PostprocessStep, cpu_outputs,
        host_rows: "HostRows | None" = None, engine=None,
    ) -> tuple[StoredOutputs, RouteTensors]:
        """STORE: store the step's routed rows' outputs, held for the route;
        what the route takes of them."""
        rids = step.wg_ids.keys
        signals = request.output_signals
        graph_walk = request.graph_walk
        # One store call for the batch rather than one per request, and the
        # flat columns come back already built: the manager fills them as it
        # mints, so nothing is keyed by request and signal only to be taken
        # apart again here.
        # Any of the walks names the source's TP group: a combined walk's walks
        # share one (refuse_combined_walks_across_tp_groups).
        # Pass the stop check's host copies so a host-memory transport needn't
        # copy the rows again. They are views of pinned buffers the next step
        # reuses, so they are dropped once this step's sends are done.
        # Row views of one batch clone (a decode step) are described from one
        # row; anything else goes through the general per-tensor path. Both
        # take the real walk, not the combined label: the runtime keys the
        # store and the route on it.
        # The client copy first: whether it rides inline decides whether the
        # tensors are needed at all.
        inline_signal, inline_values = self._inline_emit_values(
            engine, batch_N, host_rows, rids, signals,
        )
        logprobs = self._row_logprobs(host_rows, rids)

        def store(rids):
            stored = None
            if outputs.row_views and outputs.row_request_ids is not None:
                stored = self.tensor_manager.store_row_outputs_batch(
                    rids, outputs, signals, outputs.row_views, outputs.row_request_ids,
                    node_name=batch_N.node_name,
                    graph_walk=graph_walk,
                    host_rows=host_rows,
                )
            if stored is None:
                stored = self.tensor_manager.store_and_return_tensor_info_batch(
                    rids, outputs, signals,
                    node_name=batch_N.node_name,
                    graph_walk=graph_walk,
                    skip_cuda_sync=True,
                    cpu_tensors=(
                        cpu_outputs.landed(signals if self.tensor_manager.needs_cpu_tensor else ())
                        if isinstance(cpu_outputs, _LazyHostOutputs) else cpu_outputs
                    ),
                    ready_event=batch_N.node_batch.completion_event,
                )
            return stored

        loopback = outputs.loopback_rids if inline_signal is not None else ()
        if inline_signal is not None and self._body.all_on_device(
            engine, batch_N.node_name, batch_N.graph_walk, signals,
        ):
            # Device loop-back: the node keeps every routed signal's value on
            # the device (the sampler's slot master) and the client copy rides
            # inline, so nothing is stored, held or freed. The route still
            # runs per request for the loop counters and the fallbacks; it
            # carries zero tensors per (request, signal).
            stored = StoredOutputs([], [], [], [0] * (len(rids) * len(signals)))
        elif loopback:
            # A mixed step: the rows whose walk loops back on the device carry
            # zero tensors, as they would in a step of their walk alone; only
            # the others are stored.
            routed = [rid for rid in rids if rid not in loopback]
            stored = store(routed) if routed else StoredOutputs([], [], [], [])
            n_sig = len(signals)
            counts, j = [], 0
            for rid in rids:
                if rid in loopback:
                    counts.extend([0] * n_sig)
                else:
                    counts.extend(stored.num_tensors[j:j + n_sig])
                    j += n_sig
            stored = stored._replace(num_tensors=counts)
        else:
            stored = store(rids)
        self.lockstep.note_local_inputs(
            stored, signals, batch_N.node_name, rids,
            request.walks or [graph_walk], step.rid_walk_idx,
        )
        flat_uuids = stored.flat_uuids
        num_tensors = stored.num_tensors
        # Safety hold: ref=1 until the real fanout is known, which
        # complete_and_route_batch settles. One call for the batch rather than
        # one per tensor -- with a Rust bookkeeper each is a boundary crossing,
        # and a 128-request batch has hundreds of them. The count is uniform,
        # so it crosses as a scalar rather than a list built per batch.
        if flat_uuids:
            self.tensor_manager.increment_ref_batch_uniform(flat_uuids, 1)

        return stored, RouteTensors(flat_uuids, num_tensors, inline_signal, inline_values, logprobs)

    def _send_input(
        self, batch_N: PendingBatch, request: PostprocessRequest, step: PostprocessStep,
        stored: StoredOutputs | None, engine,
    ) -> SendInput:
        """SEND: what the send needs of the worker, after the stopped rows'
        finalize, the step's publication and the route's tensor work."""
        rids = step.wg_ids.keys
        signals = request.output_signals
        route_output = step.route_output
        if stored is None:  # routed inline: nothing stored
            stored = StoredOutputs([], [], [], [0] * (len(rids) * len(signals)))
        # Ordinary publication precedes stop detection on the GPU thread.
        # Export final-only state now, after the last iteration committed and
        # before the send reports the completed loop to the conductor. The
        # ordinary publication of a stopping request is finished first, so the
        # completion carries both, as it did when every step published.
        if step.stopped_rids:
            engine.materialize_publish(batch_N.node_batch, step.stopped_rids)
            engine.finalize_stopped_requests(batch_N.node_batch, step.stopped_rids)
        buffered_publish = self._buffer_publish(batch_N)
        # One runtime call rather than a Python loop over the rows (#368).
        self._graph_runtime.update_dynamic_loop_iters(
            batch_N.node_batch.per_request_info, batch_N.partition,
        )
        # Normally empty: cleanup_consumed_inputs took them before the route.
        # Not empty if a completion ever precedes it, and then nobody else will.
        if route_output.freed_inputs.uuids:
            self.tensor_manager.cleanup_collectable(*route_output.freed_inputs)
        # a request whose outputs the transfer engine refused failed alone: none of it is sent
        failed_rids = self._register_outputs(route_output)
        if self.enable_prof:
            self.profile_info.register_end(
                batch_N.node_batch.node_name,
                batch_N.node_batch.graph_walk,
                batch_N.node_batch.request_ids,
                batch_N.node_batch.exec_timings,
            )
        laps = PhaseLaps("worker.postprocess.stream_send.")
        new_token_counts = self._count_new_tokens(
            route_output.new_token_output_idxs,
            stored.flat_rids, stored.flat_uuids, signals, stored.signal_idxs,
        )
        laps.lap("count_tokens")

        # Before the sends below: remote consumers read the walks off the
        # fwd_info that rides with the streamed edges.
        self.streams.assign_consumer_walks(batch_N)

        # The buffer holds each item by its uuid, taking over the routing
        # reference the runtime counted for the local stream destination.
        for signal, per_signal in route_output.local_streaming_by_signal.items():
            consumer = self.streams.walk_driven_edges.get(signal)
            by_rid: dict[int, list[int]] = {}
            for rid, uuid in per_signal:
                by_rid.setdefault(rid, []).append(uuid)
            for rid, uuids in by_rid.items():
                req_info = self.request_state.per_request_info[rid]
                stream_buf = req_info.stream_buffers[signal]
                walks = req_info.per_partition_info[batch_N.partition].current_fwd_info.stream_consumer_walks
                stream_buf.pre_read_register_joined(
                    uuids, walks.get(consumer) if consumer else None,
                )
                for uuid in uuids:
                    stream_buf.put(uuid, uuid)
        for signal, per_signal in route_output.local_stream_values_by_signal.items():
            buffers = self.request_state.per_request_info
            for rid, value in per_signal:
                buffers[rid].stream_buffers[signal].put_values([value])
        laps.lap("local_streams")

        send_rids = [rid for rid in rids if rid not in failed_rids] if failed_rids else list(rids)
        needs_info = route_output.rids_needing_request_info
        info_rids = (
            send_rids if needs_info is None
            else [rid for rid in send_rids if rid in needs_info]
        )
        # A frame to a peer or the conductor carries this step's publication:
        # finish the snapshot for those rids (the stopped ones already are).
        # The snapshot pins the lengths and page counts of step N, so the
        # commit of N+1 that may have landed since does not leak in; only a
        # reset or fork in between moves a stream past its snapshot, and the
        # KV manager publishes such a stream as it stands (its generation
        # says so).
        if info_rids:
            engine.materialize_publish(batch_N.node_batch, info_rids)
            for rid in info_rids:
                if rid in buffered_publish:
                    continue
                published = batch_N.node_batch.resource_publish_info.get(rid)
                if published:
                    self.request_state.buffer_publish_info(
                        rid, batch_N.partition, published,
                    )
                    buffered_publish.add(rid)
        laps.lap("publish")
        # Only the rids with something to report cross: a decode step has no
        # new-token counts (the inline token is not counted), and an empty
        # entry per rid cost a dict each to build and to extract on the
        # runtime side. A rid left out gets nothing added,
        # which is what an empty entry did.
        ntc_rids = [rid for rid in send_rids if new_token_counts.get(rid)]
        return SendInput(
            completion_id=route_output.completion_id,
            per_request_info=ParallelList(
                info_rids,
                [
                    self.request_state.get_fwd_info(rid, batch_N.partition)
                    for rid in info_rids
                ],
            ),
            new_token_counts=ParallelList(
                ntc_rids, [new_token_counts[rid] for rid in ntc_rids],
            ),
            profiling=self._profiling_payloads(send_rids) if self.enable_prof
            else None,
            resource_publish_info=ParallelList(
                info_rids,
                [
                    self.request_state.get_pending_publish_info(
                        rid, batch_N.partition,
                    )
                    for rid in info_rids
                ],
            ),
            failed_rids=failed_rids,
        )

    def _inline_emit_values(
        self, engine, batch_N: PendingBatch, host_rows: "HostRows | None",
        rids: list[int], signals,
    ) -> tuple[str | None, list[int | None]]:
        """The values of a client signal that can ride inline this step.

        Taken off the stop check's host copy, one ``tolist`` for the batch,
        in ``rids`` order. A row with no host row ran no forward (see
        ``ExecutingBatch.no_forward_rids``): its value is None, so it alone
        takes the tensor path.
        """
        if host_rows is None or not self._graph_runtime.supports_inline_emit:
            return None, []
        if not batch_N.batch.request_walks:
            return self._inline_emit_one_walk(engine, batch_N, host_rows, rids, signals)
        # Per real walk: a combined walk is no walk of the submodule's.
        walks = list(batch_N.batch.rids_by_walk()) or [batch_N.graph_walk]
        candidates: dict[str, str] = {}
        for walk in walks:
            for signal, buffer_name in engine.inline_signals(
                batch_N.node_name, walk,
            ).items():
                candidates.setdefault(signal, buffer_name)
        for signal, buffer_name in candidates.items():
            if signal not in signals:
                continue
            buf = host_rows.buffers.get(buffer_name)
            if not torch.is_tensor(buf) or buf.dim() == 0 or buf.numel() == 0:
                continue
            values = buf.reshape(buf.shape[0], -1)[:, 0].tolist()
            return signal, self._values_by_rid(values, host_rows, rids)
        return None, []

    @staticmethod
    def _values_by_rid(values: list[int], host_rows: "HostRows", rids: list[int]) -> list[int | None]:
        """``values``, in host-row order, in ``rids`` order; None for a rid with no host row."""
        row_of = {rid: i for i, rid in enumerate(host_rows.request_ids[:len(values)])}
        return [values[row_of[rid]] if rid in row_of else None for rid in rids]

    def _inline_emit_one_walk(
        self, engine, batch_N: PendingBatch, host_rows: "HostRows",
        rids: list[int], signals,
    ) -> tuple[str | None, list[int | None]]:
        """`_inline_emit_values` for a step of one walk: every row takes the
        value, so one comprehension, and a slice when the host rows are in
        the batch's order (the usual case)."""
        candidates = engine.inline_signals(batch_N.node_name, batch_N.graph_walk)
        for signal, buffer_name in candidates.items():
            if signal not in signals:
                continue
            buf = host_rows.buffers.get(buffer_name)
            if not torch.is_tensor(buf) or buf.dim() == 0 or buf.numel() == 0:
                continue
            values = buf.reshape(buf.shape[0], -1)[:, 0].tolist()
            n = len(rids)
            if len(values) >= n and list(host_rows.request_ids[:n]) == rids:
                return signal, values[:n]
            return signal, self._values_by_rid(values, host_rows, rids)
        return None, []

    @staticmethod
    def _row_logprobs(host_rows: "HostRows | None", rids: list[int]) -> list[float | None]:
        """Each rid's sampled-token log-prob off the stop check's host copy,
        None where the sampler left NaN (the row did not ask). Empty when no
        rid has one."""
        buf = None if host_rows is None else host_rows.buffers.get(SAMPLED_LOGPROB)
        if buf is None:
            return []
        values = dict(zip(host_rows.request_ids, buf.tolist(), strict=False))
        out = [None if math.isnan(v := values.get(rid, math.nan)) else v for rid in rids]
        return out if any(v is not None for v in out) else []

    def drop_failed_rids(
        self, pending: PendingBatch,
        outputs: BatchedModelOutput,
        failed_requests: Collection[int],
    ) -> None:
        """Excise ``failed_requests`` (rids, or a rid -> error map) from a finished batch.

        A rid that raised in its per-request forward or in ``postprocess`` is
        still carried in the batch (the engine only recorded the error), and one
        that raised in ``prepare_inputs`` is already out of
        ``node_batch.request_ids`` but not out of the worker-side
        ``ScheduledBatch``. Either way we must not route its outputs or mark its
        node complete — that's how a request that blew up mid-walk ends up
        reported to the client as a successful empty response.
        ``postprocess`` reconciles the remaining structures from
        ``node_batch.request_ids``.
        """
        for rid in failed_requests:
            outputs.pop(rid, None)
            pending.batch.discard_rid(rid)
            pending.node_batch.per_request_info.pop(rid, None)
            pending.node_batch.cached_rids.discard(rid)
            # Clear what we just reported, so a later stage that fails more
            # rids (check_stop, below the forward) can tell its own from these
            # and doesn't report them to the conductor twice.
            pending.node_batch.failed_requests.pop(rid, None)
            # its streams do not end here; see ExecutingBatch.drop_rids
            pending.node_batch.final_stream_rids.discard(rid)
            pending.node_batch.partition_done_rids.discard(rid)
        pending.node_batch.request_ids = [
            rid for rid in pending.node_batch.request_ids
            if rid not in failed_requests
        ]
