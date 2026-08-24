

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import JointGroups, WorkerParallelGroups
from mstar.engine.resources.base import Resource, build_resource
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig
from mstar.engine.resources.step import (
    AdmitFailedReason,
    AllocationFailed,
    SlotLease,
    StepContext,
    SubmoduleStep,
)
from mstar.engine.v1.cuda_graph_runner import (
    CudaGraphRunner,
    PiecewiseCudaGraphRunner,
    autocast_scope,
)
from mstar.engine.v1.kv_transfer import TransferEngineInfo
from mstar.engine.v1.sampler import SamplerResource
from mstar.model.submodule_base import (
    LazyRequestStates,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)
from mstar.profile.worker import ExecTimings
from mstar.utils.profiler import mark, range_pop, range_push

logger = logging.getLogger(__name__)


@dataclass
class SubmoduleManagement:
    submodule: NodeSubmodule
    forward: Callable
    forward_batched: Callable
    joint_comm_group: JointGroups
    resources: dict[str, Resource]
    cuda_graph_runner: CudaGraphRunner | None = None

    # label -> PiecewiseCudaGraphRunner for inner-loop capture; spread into
    # ModelInputsFromEngine so the submodule's forward can look them up
    piecewise_runners: dict[str, PiecewiseCudaGraphRunner] = field(
        default_factory=dict
    )

    @property
    def sampler(self) -> SamplerResource | None:
        """DEPRECATED: engine-side sampling. A node may now have more than one
        sampler resource, so which one applies is the forward's call — it
        should sample there instead of handing logits back."""
        for resource in self.resources.values():
            if isinstance(resource, SamplerResource):
                return resource
        return None


@dataclass
class ExecutingBatch:
    node_name: str

    # real request ids (the ones in step_context may have dummy req ids)
    request_ids: list[str]
    per_request_info: Mapping[str, CurrentForwardPassInfo]
    step_context: StepContext

    running_batched: bool = False

    # Selects among a walk's capture buckets; matches SubmoduleStep.cg_key_info
    cg_key_info: Any | None = None

    # Populated on preplan
    preplan_event: torch.cuda.Event | None = None
    # The rids the staged plan was built over. The plan is theirs exactly —
    # order included — so it is stale the moment this stops matching
    # ``request_ids`` (a request dropped while threading outputs or preparing).
    preplanned_rids: tuple[str, ...] | None = None

    # Declared once for the batch (pre-plan declares it first when it runs)
    # and driven from here on
    step: SubmoduleStep | None = None

    # {request_id: {input_name: [tensor]}}, what prepare_inputs reads
    per_request_input_tensors: Mapping[str, NameToTensorList] = field(
        default_factory=dict
    )
    # rids whose consumed streaming input was the final chunk — this step
    # reports the partition done
    final_stream_rids: set[str] = field(default_factory=set)

    # Populated on batch preparation
    inputs: list[NodeInputs] | None = None
    # rids the submodule declined this step — e.g. a speculatively scheduled
    # flow step for a request already past its own max iters
    skipped_rids: set[str] = field(default_factory=set)
    # rid -> error, for per-rid stages that raised. The rid leaves the batch;
    # the rest of it runs.
    failed_requests: dict[str, str] = field(default_factory=dict)

    allocation_failed: AllocationFailed | None = None
    admit_error: AdmitFailedReason | None = None

    # This step's per-rid outputs, published as soon as the forward has been
    # submitted — the tensors exist then, even though their values land later.
    outputs: dict[str, NameToTensorList] = field(default_factory=dict)

    # The next step reads N's outputs, and plans against N's committed state.
    # Two separate dependencies, so two events: whoever prepares N+1 can start
    # threading N's outputs while N is still committing.
    outputs_ready: threading.Event = field(default_factory=threading.Event)
    commit_done: threading.Event = field(default_factory=threading.Event)

    # Recorded on the default stream once this step's GPU work is submitted;
    # what a reader of the output values has to wait on.
    completion_event: torch.cuda.Event | None = None

    # Set by exec right before the forward's CUDA launch (where torch drops the
    # GIL). A worker that submitted this batch to the GPU thread and then wants
    # to do its own Python work waits on this first, so its GIL grab doesn't
    # stall the GPU thread's path to graph.replay(). None => nobody is waiting.
    launch_started_event: threading.Event | None = None

    # Per-step wall-clock, for the worker's profiler
    exec_timings: ExecTimings = field(default_factory=ExecTimings)

    @property
    def graph_walk(self) -> str:
        return self.step_context.graph_walk

    def register_prepare_batch(self, inputs: list[NodeInputs]):
        self.inputs = inputs

    def release_waiters(self):
        """Let anything waiting on this step proceed.

        Called on every exit from ``exec``, so a step that raised before
        publishing outputs or committing doesn't strand the thread preparing
        the next one.
        """
        self.outputs_ready.set()
        self.commit_done.set()

    def register_failure(self, rid: str, error: Exception):
        self.failed_requests[rid] = f"{type(error).__name__}: {error}"

    def drop_rids(self, rids: set[str]):
        """Take rids out of the step, leaving the rest of the batch intact."""
        if not rids:
            return
        self.request_ids = [rid for rid in self.request_ids if rid not in rids]
        self.per_request_info = {
            rid: info for rid, info in self.per_request_info.items()
            if rid not in rids
        }

    def register_admit_error(self, reason: AdmitFailedReason):
        self.admit_error = reason
        if isinstance(reason, AllocationFailed):
            self.allocation_failed = reason

    def lease_slot(self, slot_lease: SlotLease):
        self.step_context.slot_lease = slot_lease


class Engine:
    def __init__(
        self, autocast_dtype=torch.bfloat16,
        enable_nvtx: bool = False,
        enable_profile: bool=False,
    ):
        self._device = None
        self._autocast_dtype = autocast_dtype
        self._resources: dict[str, Resource] = {}
        self._submodules: dict[str, SubmoduleManagement] = {}
        self._runner: StepRunner = None

        self._enable_nvtx = enable_nvtx
        self._enable_profile = enable_profile

    def load_model(
        self,
        submodules: dict[str, NodeSubmodule],
        specs: list[NodeResourceSpec],
        parallel_groups: WorkerParallelGroups,
        device: torch.device,
        transfer_engine_info: TransferEngineInfo,
        kv_cache_type=None,
    ):
        self._device = device
        if kv_cache_type is None:
            kv_cache_type = self._autocast_dtype

        node_names = set(submodules.keys())
        node_to_resources = {}
        for spec in specs:
            relevant_nodes = spec.nodes & node_names
            if len(relevant_nodes) == 0:
                continue # resource not needed

            if not parallel_groups.all_in_same_group(spec.nodes):
                raise ValueError(
                    f"Resource spec {spec.label} nodes {spec.nodes} "
                    f"must all be in the same parallel (tp x sp) group"
                )
            joint_comm_group = parallel_groups.get_joint_group_for_node(
                next(iter(relevant_nodes))
            )
            self._resources[spec.label] = build_resource(
                spec=spec,
                device=device,
                joint_comm_group=joint_comm_group,
                transfer_engine_info=transfer_engine_info,
                kv_dtype=kv_cache_type
            )

            for node in relevant_nodes:
                node_to_resources.setdefault(node, []).append(spec.label)

        self._runner = StepRunner(
            self._resources,
            # every node, including one that owns nothing (Code2Wav) — an
            # absent node would fall back to the full sweep
            node_resources={n: node_to_resources.get(n, []) for n in node_names},
            enable_nvtx=self._enable_nvtx,
        )

        for node_name, submodule in submodules.items():
            resources = {
                label: self._resources[label] for label in node_to_resources.get(node_name, [])
            }
            self._submodules[node_name] = SubmoduleManagement(
                submodule=submodule,
                forward=submodule.forward,
                forward_batched=submodule.forward_batched,
                joint_comm_group=parallel_groups.get_joint_group_for_node(node_name),
                resources=resources
            )
            submodule.bind_node_resources(resources)

    def _compile_submodules(self) -> None:
        """Apply torch.compile to submodule forward paths.

        Compiles each submodule's ``forward`` and ``forward_batched`` with the
        default mode (fullgraph=False, dynamic=None), which in general provides
        performance gains without frequent slow recompiles.
        """
        if not torch.cuda.is_available():
            return

        for node_name, submodule_mgmt in self._submodules.items():
            submodule = submodule_mgmt.submodule

            if getattr(submodule, "disable_torch_compile", False):
                logger.info("Engine: torch.compile disabled for %s (submodule opt-out)", node_name)
                continue

            try:
                submodule_mgmt.forward = torch.compile(
                    submodule.forward,
                    fullgraph=False,
                    dynamic=None,
                )
                submodule_mgmt.forward_batched = torch.compile(
                    submodule.forward_batched,
                    fullgraph=False,
                    dynamic=None,
                )
                logger.info("Engine: torch.compile applied to %s", node_name)
            except Exception:
                logger.warning(
                    "Engine: torch.compile failed for %s, using eager mode",
                    node_name, exc_info=True
                )

    def _autocast_dtype_for(self, submodule: NodeSubmodule) -> torch.dtype | None:
        """This node's autocast dtype, or None for one that opted out."""
        return None if submodule.disable_autocast else self._autocast_dtype

    def warmup(self) -> None:
        for node_name, submodule_mgmt in self._submodules.items():
            submodule = submodule_mgmt.submodule
            runner = CudaGraphRunner(
                submodule_name=node_name,
                submodule=submodule,
                resources=submodule_mgmt.resources,
                step_runner=self._runner,
                device=self._device,
                autocast_dtype=self._autocast_dtype_for(submodule),
                joint_comm_group=submodule_mgmt.joint_comm_group,
                enable_nvtx=self._enable_nvtx
            )
            runner.warmup_and_capture()
            if runner.any_graphs:
                submodule_mgmt.cuda_graph_runner = runner

            submodule_mgmt.piecewise_runners = self._build_piecewise_runners(
                node_name, submodule_mgmt
            )

        # torch.compile applied after CUDA graph capture because the cuda
        # graph runner compiles internally
        self._compile_submodules()

        for resource in self._resources.values():
            resource.post_warmup_validate()

    def _build_piecewise_runners(
        self, node_name: str, submodule_mgmt: SubmoduleManagement,
    ) -> dict[str, PiecewiseCudaGraphRunner]:
        """One runner per region the submodule declares, warmed up.

        A region whose capture fails is left out, so its forward takes the
        eager path for that label.
        """
        node_dtype = self._autocast_dtype_for(submodule_mgmt.submodule)
        configs = submodule_mgmt.submodule.get_piecewise_cuda_graph_configs(
            self._device,
            # the dtype the region runs in; an opted-out node keeps its params'
            node_dtype or torch.float32,
            submodule_mgmt.joint_comm_group.world_size,
        )
        runners: dict[str, PiecewiseCudaGraphRunner] = {}
        for label, config in configs.items():
            runner = PiecewiseCudaGraphRunner(
                label=f"{node_name}_{label}",
                config=config,
                resources=submodule_mgmt.resources,
                step_runner=self._runner,
                device=self._device,
                autocast_dtype=node_dtype,
                joint_comm_group=submodule_mgmt.joint_comm_group,
            )
            runner.warmup_and_capture()
            if runner.any_graphs:
                runners[label] = runner
        return runners

    def prepare_inputs(self, batch: ExecutingBatch) -> None:
        """Per-rid ``submodule.prepare_inputs``, onto ``batch.inputs``.

        Per-rid, so a raise is attributable to one request: record it and take
        that rid out rather than losing the batch. A rid the submodule declines
        (returns None) leaves the same way, without being an error.
        """
        if self._enable_nvtx:
            range_push(f"engine.prepare_inputs.bs{len(batch.request_ids)}")
        try:
            self._prepare_inputs(batch)
        finally:
            if self._enable_nvtx:
                range_pop()

    def _prepare_inputs(self, batch: ExecutingBatch) -> None:
        submodule = self._submodules[batch.node_name].submodule
        node_inputs: list[NodeInputs] = []
        for rid in batch.request_ids:
            try:
                req_inputs = submodule.prepare_inputs(
                    graph_walk=batch.step_context.graph_walk,
                    fwd_info=batch.per_request_info[rid],
                    inputs=batch.per_request_input_tensors.get(rid, {}),
                    resources=self._submodules[batch.node_name].resources,
                )
            except Exception as error:
                logger.exception(
                    "prepare_inputs failed for request %s (node=%s, walk=%s)",
                    rid, batch.node_name, batch.step_context.graph_walk,
                )
                batch.register_failure(rid, error)
                continue
            if req_inputs is None:
                batch.skipped_rids.add(rid)
            else:
                node_inputs.append(req_inputs)

        batch.register_prepare_batch(node_inputs)
        batch.running_batched = submodule.can_batch(
            batch=batch, model_inputs=node_inputs
        )
        batch.drop_rids(batch.skipped_rids | batch.failed_requests.keys())

    def exec(
        self, batch: ExecutingBatch
    ) -> dict[str, NameToTensorList]:
        """Run one step: declare → admit → plan → forward → commit.

        Captured replay and the eager forward are the same path. Under a lease
        the batch is padded to the slot's shape, preprocess output is staged
        into the static buffers, and the launch is a replay; everything above
        the launch is identical.
        """
        nvtx = self._enable_nvtx
        if self._enable_profile and batch.exec_timings.start is None:
            batch.exec_timings.start = time.perf_counter()
        if nvtx:
            range_push(
                f"engine.{batch.node_name}.{batch.step_context.graph_walk}"
                f".bs{len(batch.request_ids)}"
            )
        try:
            # inference-only, under the same scope the capture path used
            with torch.no_grad(), autocast_scope(
                self._autocast_dtype_for(self._submodules[batch.node_name].submodule)
            ):
                # a caller that pre-planned already reserved one; otherwise take it here
                lease = batch.step_context.slot_lease or self.reserve_replay_slot(batch)  
                # admit/plan/commit assume the whole batch reaches the forward
                # in order; an unbatchable walk with >1 request can't, so each
                # request runs its own full cycle and merges.
                if not batch.running_batched and len(batch.request_ids) > 1 and lease is None:
                    batch.outputs = self._exec_per_request(batch)
                else:
                    batch.outputs = self._exec_single(batch)
            batch.outputs_ready.set()
            return batch.outputs
        finally:
            batch.preplan_event = None
            batch.release_waiters()
            if nvtx:
                range_pop()

    def _exec_single(
        self, batch: ExecutingBatch
    ) -> dict[str, NameToTensorList]:
        """The one-forward path: batched (a lease replay or ``forward_batched``)
        or a single eager request."""
        submodule_mgmt = self._submodules[batch.node_name]
        cg_runner = submodule_mgmt.cuda_graph_runner
        lease = batch.step_context.slot_lease
        real_bs = len(batch.request_ids)

        inputs = batch.inputs
        req_info = batch.per_request_info
        if lease is not None:
            inputs = cg_runner.pad_inputs(lease, inputs)
            req_info = cg_runner.step_metadata(
                lease, batch.request_ids, batch.per_request_info
            )
            batch.step_context.request_ids = cg_runner.step_ids(lease, batch.request_ids)

        try:
            raw, batch.step = self._drive_step(
                batch, submodule_mgmt, batch.request_ids, inputs, req_info,
                batch.step_context, lease, batch.running_batched,
                step=batch.step, set_launch=True,
            )
            if raw is None:
                return {rid: {} for rid in batch.request_ids}
            # Commit first: releasing `commit_done` here is what lets a pre-plan
            # of N+1 overlap this step's per-request tail.
            batch.commit_done.set()
            if self._enable_nvtx:
                range_push("engine.collect_outputs")
            try:
                return self._collect_outputs(
                    batch, submodule_mgmt, lease, raw, inputs, req_info,
                    request_ids=batch.request_ids,
                    step_request_ids=batch.step_context.request_ids,
                )
            finally:
                if self._enable_nvtx:
                    range_pop()
        finally:
            if lease is not None:
                cg_runner.release(lease, real_bs)

    def _exec_per_request(
        self, batch: ExecutingBatch
    ) -> dict[str, NameToTensorList]:
        """Full step cycle per request, eager (no lease), outputs merged.

        The unbatchable fallback: no capture applies (a lease would have set
        ``running_batched``), so each request declares/admits/plans/commits on
        its own, keeping the plan matched to its single-request forward.
        """
        nvtx = self._enable_nvtx
        submodule_mgmt = self._submodules[batch.node_name]
        merged: dict[str, NameToTensorList] = {}
        launched = False
        for rid, inp in zip(batch.request_ids, batch.inputs, strict=True):
            req_info = {rid: batch.per_request_info[rid]}
            ctx = StepContext(
                request_ids=(rid,),
                graph_walk=batch.step_context.graph_walk,
                slot=0, capture=False,
            )
            if nvtx:
                range_push(f"engine.per_request.{rid}")
            try:
                raw, _ = self._drive_step(
                    batch, submodule_mgmt, [rid], [inp], req_info, ctx,
                    lease=None, running_batched=False,
                    step=None, set_launch=not launched,
                )
                if raw is None:
                    merged[rid] = {}
                    continue
                launched = True
                merged.update(self._collect_outputs(
                    batch, submodule_mgmt, None, raw, [inp], req_info,
                    request_ids=[rid], step_request_ids=(rid,),
                ))
            finally:
                if nvtx:
                    range_pop()
        batch.commit_done.set()
        return merged

    def _drive_step(
        self,
        batch: ExecutingBatch,
        submodule_mgmt: SubmoduleManagement,
        request_ids: list[str],
        inputs: list[NodeInputs],
        req_info: Mapping[str, CurrentForwardPassInfo],
        ctx: StepContext,
        lease: SlotLease | None,
        running_batched: bool,
        step: SubmoduleStep | None,
        set_launch: bool,
    ) -> tuple[dict | None, SubmoduleStep | None]:
        """declare → admit → plan → preprocess → forward → commit for one forward.

        ``request_ids`` are the real rids (for admit-fail / collect); the model
        runs over ``ctx.request_ids`` (padded under a lease). Returns
        ``(raw_outputs, step)``; ``raw_outputs`` is None when admit failed.
        """
        nvtx = self._enable_nvtx
        cg_runner = submodule_mgmt.cuda_graph_runner
        submodule = submodule_mgmt.submodule
        rids = list(ctx.request_ids)

        if step is None:
            if nvtx:
                range_push("engine.declare_step")
            try:
                step = submodule.declare_step(
                    graph_walk=ctx.graph_walk, request_ids=rids, inputs=inputs,
                )
            finally:
                if nvtx:
                    range_pop()

        if step is not None:
            if lease is not None and step.cg_key_info != lease.bucket.cg_key_info:
                # The slot was leased from `cg_key_info` before the step was
                # declared. If the two disagree the replay runs a graph that
                # was planned for a different declaration — silently wrong
                # output, so fail here instead. Derive both from one place.
                raise RuntimeError(
                    f"{batch.node_name}: leased {lease.bucket} but the step "
                    f"declares cg_key_info={step.cg_key_info!r}; "
                    "cg_key_info() and declare_step disagree"
                )
            step.set_ctx(ctx)
            # TODO: skip the resources a pre-plan already admitted and planned.
            # They no-op (and promote) internally today, so this re-drives the
            # whole sweep to get the rest.
            if nvtx:
                range_push("engine.admit")
            try:
                admit_outcome = self._runner.admit(step)
            finally:
                if nvtx:
                    range_pop()
            if not admit_outcome.ok:
                batch.register_admit_error(admit_outcome.reason)
                return None, step
            if nvtx:
                # promoted = a pre-plan was consumed; fresh = planned inline
                range_push(
                    "engine.plan.promoted" if batch.preplan_event is not None
                    else "engine.plan.fresh"
                )
            try:
                self._runner.plan(step)
            finally:
                if nvtx:
                    range_pop()

        engine_inputs = ModelInputsFromEngine(
            request_ids=rids,
            per_request_info=req_info,
            resources=submodule_mgmt.resources,
            piecewise_runners=submodule_mgmt.piecewise_runners,
            # padding rows get their own states, like their cache streams:
            # the submodule indexes this by step id, not by real rid
            per_request_states=LazyRequestStates(submodule, rids),
            captured=lease is not None,
            step=step,
        )
        if nvtx:
            range_push("engine.preprocess")
        try:
            preprocessed = submodule.preprocess(
                ctx.graph_walk, engine_inputs=engine_inputs, inputs=inputs,
            )
        finally:
            if nvtx:
                range_pop()

        if self._enable_profile and batch.exec_timings.fwd_start is None:
            batch.exec_timings.fwd_start = time.perf_counter()
        # The waiter is released inside the forward, immediately before the
        # launch that drops the GIL — not here, which is still several GIL-held
        # staging copies away from it.
        release_event = batch.launch_started_event if set_launch else None
        if nvtx:
            # the launch/enqueue span, not the GPU work: `synchronize=True`
            # here would drain the stream and destroy the overlap
            range_push("engine.forward")
        try:
            raw = self._forward(
                batch, submodule_mgmt, cg_runner, engine_inputs, preprocessed,
                lease, running_batched, request_ids, release_event,
            )
        finally:
            if nvtx:
                range_pop()

        if step is not None:
            if nvtx:
                range_push("engine.commit")
            try:
                self._runner.commit(step)
            finally:
                if nvtx:
                    range_pop()
        return raw, step

    def _forward(
        self,
        batch: ExecutingBatch,
        submodule_mgmt: SubmoduleManagement,
        cg_runner,
        engine_inputs: ModelInputsFromEngine,
        preprocessed: dict[str, Any],
        lease: SlotLease | None,
        running_batched: bool,
        request_ids: list[str],
        release_event: "threading.Event | None" = None,
    ) -> dict:
        """Replay the leased slot, or run the eager forward.

        ``release_event`` is set as late as possible before the call that drops
        the GIL — the replay, or the submodule forward on the eager path."""
        graph_walk = batch.step_context.graph_walk
        if lease is not None:
            return cg_runner.run_forward(
                lease, preprocessed, plan_done_event=batch.preplan_event,
                launch_started_event=release_event,
            )
        if release_event is not None:
            release_event.set()
        if running_batched:
            return submodule_mgmt.forward_batched(
                graph_walk, engine_inputs=engine_inputs, **preprocessed
            )
        assert len(request_ids) == 1, (
            "the unbatched forward takes one request; batch of "
            f"{len(request_ids)} needs running_batched"
        )
        return {request_ids[0]: submodule_mgmt.forward(
            graph_walk, engine_inputs=engine_inputs, **preprocessed
        )}

    def postprocess_batch(
        self, batch: ExecutingBatch, outputs: dict[str, NameToTensorList],
    ) -> None:
        """Per-rid ``submodule.postprocess``, e.g. the non-capturable tail of a
        walk whose graph covered only the forward.

        Per-rid like ``prepare_inputs``: a raise fails that request and leaves
        the rest of the batch to route normally.
        """
        if self._enable_nvtx:
            range_push("engine.postprocess")
        try:
            self._postprocess_batch(batch, outputs)
        finally:
            if self._enable_nvtx:
                range_pop()

    def _postprocess_batch(
        self, batch: ExecutingBatch, outputs: dict[str, NameToTensorList],
    ) -> None:
        submodule = self._submodules[batch.node_name].submodule
        for rid, node_inputs in zip(batch.request_ids, batch.inputs, strict=True):
            try:
                submodule.postprocess(
                    request_id=rid,
                    request_info=batch.per_request_info[rid],
                    outputs=outputs.get(rid, {}),
                    inputs=node_inputs,
                )
            except Exception as error:
                logger.exception(
                    "postprocess failed for request %s (node=%s, walk=%s)",
                    rid, batch.node_name, batch.step_context.graph_walk,
                )
                batch.register_failure(rid, error)

    def exec_and_postprocess(
        self, batch: ExecutingBatch
    ) -> dict[str, NameToTensorList]:
        """The forward and its per-rid tail, which belong to the same step: a
        walk that captured only its forward finishes in ``postprocess``.

        ``prepare_inputs`` and the stop check stay outside — the worker has to
        place those itself.
        """
        outputs = self.exec(batch)
        self.postprocess_batch(batch, outputs)
        return outputs

    def check_stop_for_batch(
        self, batch: ExecutingBatch, outputs: dict[str, NameToTensorList],
    ) -> dict[str, set[str]]:
        """Each rid's ``submodule.check_stop``, as rid -> loops that should stop.

        Reads tensor values, so it belongs on the caller's slow-postprocess
        path rather than in ``exec``, where the ``.item()`` / ``.cpu()`` would
        block the GPU thread. Per-rid like the other stages: a raise fails that
        request onto the batch, because letting it escape would abandon the
        check for the rest of the batch and leave their loops running past
        their stop condition.
        """
        submodule = self._submodules[batch.node_name].submodule
        stops: dict[str, set[str]] = {}
        for rid in batch.request_ids:
            rid_outputs = outputs.get(rid)
            if not rid_outputs:
                continue
            try:
                rid_stops = submodule.check_stop(
                    rid, batch.per_request_info[rid], rid_outputs
                )
            except Exception as error:
                logger.exception(
                    "check_stop failed for request %s (node=%s, walk=%s)",
                    rid, batch.node_name, batch.step_context.graph_walk,
                )
                batch.register_failure(rid, error)
                continue
            if rid_stops:
                stops[rid] = rid_stops
        return stops

    def finalize_batch(
        self, batch: ExecutingBatch
    ):
        if self._enable_nvtx:
            range_push("engine.finalize_batch")
        try:
            # Returns rid -> {resource label -> published info}
            published = self._runner.publish(
                batch.request_ids, node_name=batch.node_name,
            )
            for rid, info in batch.per_request_info.items():
                if rid not in published:
                    continue
                info.update_publish_info(published[rid])
        finally:
            if self._enable_nvtx:
                range_pop()

    def _collect_outputs(
        self,
        batch: ExecutingBatch,
        submodule_mgmt: SubmoduleManagement,
        lease: SlotLease | None,
        raw_outputs: dict,
        inputs: list[NodeInputs],
        req_info: Mapping[str, CurrentForwardPassInfo],
        request_ids: list[str],
        step_request_ids: tuple[str, ...],
    ) -> dict[str, NameToTensorList]:
        """Per-rid outputs for the real requests: sample logits, drop the
        padding rows, and map a captured graph's keys back to real ids.

        A captured forward emits its per-rid entries under the slot's padding
        ids (those were the batch at capture time), so entry ``i`` belongs to
        ``request_ids[i]`` on either path.
        """
        submodule = submodule_mgmt.submodule
        out_ids = (
            step_request_ids if lease is None
            else submodule_mgmt.cuda_graph_runner.slot_for(lease).dummy_rids
        )
        sampler = submodule_mgmt.sampler
        outputs: dict[str, NameToTensorList] = {}

        # Fast path: the submodule stacked the batch's logits under a sentinel,
        # so the whole batch samples in one call instead of per-rid.
        batched_logits = raw_outputs.get("__batched_logits__")
        if batched_logits is not None and sampler is not None:
            if lease is not None and sampler.expects_padded_batch:
                # The graph sampler's params cover the whole padded slot, so
                # the padding rows have to go in with the real ones.
                rows = lease.bucket.bs
                sample_ids = out_ids[:rows]
            else:
                rows = len(request_ids)
                sample_ids = request_ids
            # FlashInfer reuses its sampling output buffer across calls, so a
            # view held past the next step aliases that step's token. The clone
            # snapshots this step's value.
            sampled = sampler.sample(
                sample_ids, batched_logits[:rows]
            )[:len(request_ids)].clone()
            outputs = {
                rid: {"new_token": [view]}
                for rid, view in zip(request_ids, sampled.split(1), strict=True)
            }

        needs_per_rid = batched_logits is None or lease is None or \
            submodule_mgmt.cuda_graph_runner.slot_for(lease).has_non_logit_outputs
        if needs_per_rid:
            self._merge_per_rid(
                outputs, raw_outputs, request_ids, out_ids,
                submodule, sampler, req_info,
                sample_logits=batched_logits is None,
            )
        self._merge_unpacked(
            outputs, raw_outputs, request_ids, submodule,
            inputs[:len(request_ids)], req_info,
        )
        return outputs

    def _merge_per_rid(
        self,
        outputs: dict[str, NameToTensorList],
        raw_outputs: dict,
        request_ids: list[str],
        out_ids: list[str],
        submodule: NodeSubmodule,
        sampler: SamplerResource | None,
        req_info: Mapping[str, CurrentForwardPassInfo],
        sample_logits: bool,
    ) -> None:
        """Fold the forward's per-rid entries into ``outputs``, sampling the
        logits the ``__batched_logits__`` fast path did not already cover.

        NOTE: the automatic sampling path is DEPRECATED; models should use
        the injected sampler resource in forward pass.
        """
        all_logits = []
        for rid, out_id in zip(request_ids, out_ids, strict=False):
            rid_out = raw_outputs.get(out_id)
            if not isinstance(rid_out, dict):
                continue
            # captured output keys are fixed for graph compat; the submodule
            # decides which of them this real request should receive
            rid_out = submodule.filter_batched_output(req_info.get(rid), rid_out)
            merged = outputs.setdefault(rid, {})
            for key, value in rid_out.items():
                if key == "logits":
                    all_logits.append(value[0] if isinstance(value, list) else value)
                elif isinstance(value, list):
                    merged[key] = [t.clone() for t in value]
                elif isinstance(value, torch.Tensor):
                    merged[key] = [value.clone()]
                else:
                    merged[key] = value

        if not (sample_logits and all_logits and sampler is not None):
            return
        sampled = sampler.sample(request_ids, torch.cat(all_logits, dim=0)).clone()
        for i, rid in enumerate(request_ids):
            outputs.setdefault(rid, {})["new_token"] = [sampled[i:i+1]]

    def _merge_unpacked(
        self,
        outputs: dict[str, NameToTensorList],
        raw_outputs: dict,
        request_ids: list[str],
        submodule: NodeSubmodule,
        real_inputs: list[NodeInputs],
        req_info: Mapping[str, CurrentForwardPassInfo],
    ) -> None:
        """Let the submodule slice any batch-wide packed sentinels.

        A captured region can't do this itself: the per-request slice ends
        depend on the real seq_lens, which only reach it through the plan.
        """
        unpacked = submodule.unpack_packed_outputs(
            static_output=raw_outputs,
            request_ids=request_ids,
            real_seq_lens=[inp.input_seq_len for inp in real_inputs],
            inputs=real_inputs,
            per_request_info=req_info,
        )
        for rid, rid_out in unpacked.items():
            outputs.setdefault(rid, {}).update(rid_out)


    def get_max_batch_size(self, node_name: str, graph_walk: str) -> int | None:
        """Most requests this node will take in one step, or None for no cap.

        Two sources: what the submodule says it can batch, and the largest
        batch this walk captured a graph for — going past that would drop the
        step to eager, so the scheduler splits instead. Splitting is the
        scheduler's job, not the engine's: the pieces then pipeline like any
        other batch instead of running back to back.
        """
        submodule_mgmt = self._submodules[node_name]
        caps = [submodule_mgmt.submodule.max_batch_size(graph_walk)]
        if submodule_mgmt.cuda_graph_runner is not None:
            caps.append(
                submodule_mgmt.cuda_graph_runner.max_batch_size_for(graph_walk)
            )
        capped = [cap for cap in caps if cap is not None]
        return min(capped) if capped else None

    def check_ready(
        self, node_name: str, request_id: str,
        request_info: CurrentForwardPassInfo,
    ) -> bool:
        """Whether this node can run the request now.

        An offloaded request is brought back first; it stays not-ready until
        that fits, which is the scheduler's cue to run something else (and,
        on OOM, to evict). Then each resource takes in whatever the request
        published elsewhere — a KV transfer from a prefill worker, say — and
        reports whether that has landed.
        """
        if self.is_offloaded(node_name, request_id) and not self.reload_request(
            node_name, request_id
        ):
            return False
        out = self._runner.admit_retrieve(
            rid=request_id, node_name=node_name,
            graph_walk=request_info.graph_walk,
            published=request_info.resource_publish_info,
        )
        return out.ok and out.ready

    def reserve_replay_slot(self, batch: ExecutingBatch) -> SlotLease | None:
        """Lease the slot this batch will replay on, before it is dispatched.

        Reserving up front is what puts pre-plan(N+1) and replay(N) on
        different slots. No captured graph for the batch's shape leaves the
        lease unset, i.e. the step runs eager.

        A batch that hasn't been through ``prepare_inputs`` has no token count
        yet, so only a batched capture can serve it — see
        ``CudaGraphRunner.select_batched_bucket``.
        """
        submodule_mgmt = self._submodules[batch.node_name]
        cg_runner = submodule_mgmt.cuda_graph_runner
        if cg_runner is None:
            return None
        # Which of the walk's captures this batch belongs to. The engine can't
        # derive it — what separates two captures of one walk is the model's
        # business (bagel: guidance on/off) — and the lease is taken before the
        # step is declared, so the submodule is asked directly. It answers from
        # the same per-request facts its `declare_step` stamps on the step.
        batch.cg_key_info = submodule_mgmt.submodule.cg_key_info(
            batch.step_context.graph_walk, batch.per_request_info,
        )
        lease = cg_runner.lease_slot(
            graph_walk=batch.step_context.graph_walk,
            bs=len(batch.request_ids),
            num_tokens=(
                None if batch.inputs is None
                else sum(inp.input_seq_len for inp in batch.inputs)
            ),
            cg_key_info=batch.cg_key_info,
        )
        if lease is not None:
            batch.lease_slot(lease)
        return lease

    def can_pre_plan(self, node_name: str) -> bool:
        """Whether pre-planning this node could stage anything at all.

        A node with no pre-planning resource has nothing to stage: its step
        declaration is empty, so `pre_plan_for_batch` is guaranteed to bail.
        Asked BEFORE the plan thread waits on the previous batch's commit, so
        a stateless node does not pay that wait to reach a foregone conclusion.
        """
        mgmt = self._submodules.get(node_name)
        if mgmt is None or mgmt.cuda_graph_runner is None:
            return False
        return any(r.supports_preplan for r in mgmt.resources.values())

    def pre_plan_for_batch(self, batch: ExecutingBatch) -> bool:
        """Admit and plan the pre-planning resources a step ahead.

        Worth doing only under a lease: the reserved slot is not the one the
        in-flight replay reads, so this can write its plan buffers on the plan
        stream while that replay runs. ``exec`` then re-drives the full sweep,
        where each pre-planned resource promotes what was staged here, and the
        replay waits on the event recorded here.

        An unprepared batch declares its step over the capture config's inputs
        for the leased bucket, which is why the lease had to come from a
        batched capture. That step is not cached: ``exec`` re-declares it over
        the real inputs.

        TODO: pre-planning a packed capture (and chunked prefill) needs the
        real token counts here, i.e. ``prepare_inputs`` run ahead of the
        forward. That needs every submodule's ``prepare_inputs`` to be
        async-safe — no ``.item()`` on a tensor the in-flight step is still
        producing — which some are not (e.g. qwen3_tts).
        ``ExecutingBatch.outputs_ready`` is the hook for it.

        Returns False when nothing was planned ahead, in which case ``exec``
        plans inline.
        """
        submodule_mgmt = self._submodules[batch.node_name]
        cg_runner = submodule_mgmt.cuda_graph_runner
        lease = batch.step_context.slot_lease
        if cg_runner is None or lease is None:
            return False
        if self._enable_nvtx:
            range_push(f"engine.pre_plan.bs{len(batch.request_ids)}")
        try:
            return self._pre_plan_for_batch(batch, submodule_mgmt, cg_runner, lease)
        finally:
            if self._enable_nvtx:
                range_pop()

    def _pre_plan_for_batch(
        self, batch: ExecutingBatch, submodule_mgmt: SubmoduleManagement,
        cg_runner, lease: SlotLease,
    ) -> bool:

        # Declare-only, so the unprepared case takes the bucket's shared
        # template rows rather than cloning a fresh set per pre-plan.
        inputs = (
            cg_runner.declare_inputs_for(lease)
            if batch.inputs is None
            else cg_runner.pad_inputs(lease, batch.inputs)
        )
        batch.step_context.request_ids = cg_runner.step_ids(lease, batch.request_ids)
        step = submodule_mgmt.submodule.declare_step(
            graph_walk=batch.step_context.graph_walk,
            request_ids=batch.step_context.request_ids,
            inputs=inputs,
        )
        if step is None:
            return False
        if batch.inputs is not None:
            batch.step = step

        batch.step_context.is_preplan = True
        step.set_ctx(batch.step_context)
        try:
            admit_outcome = self._runner.pre_admit(step)
            if not admit_outcome.ok:
                # exec re-drives the step and reports the failure from there
                self.reset_pre_plan_for_batch(batch)
                return False
            stream = cg_runner.plan_stream()
            with torch.cuda.stream(stream):
                self._runner.pre_plan(step)
            batch.preplan_event = torch.cuda.Event()
            batch.preplan_event.record(stream)
            batch.preplanned_rids = tuple(batch.request_ids)
        finally:
            batch.step_context.is_preplan = False
        return True

    def preplan_is_stale(self, batch: ExecutingBatch) -> bool:
        """Whether the staged plan no longer describes this batch."""
        return (
            batch.preplanned_rids is not None
            and batch.preplanned_rids != tuple(batch.request_ids)
        )

    def reset_pre_plan_for_batch(self, batch: ExecutingBatch | None = None) -> None:
        """Drop planned-ahead state, e.g. when its batch never dispatched.

        The resources hold the plan itself. The batch keeps whatever
        ``prepare_inputs`` produced — only the plan is redone — but the leased
        slot's padding rows are returned so their pages aren't left attributed
        to a step that never ran.
        """
        if batch is not None:
            if batch.preplanned_rids is not None and self._enable_nvtx:
                # a discarded pre-plan falls back to inline plan; mark it so the
                # cost is visible next to the nvtx ranges
                mark(f"engine.preplan_discarded.{batch.node_name}")
            batch.preplan_event = None
            batch.preplanned_rids = None
            lease = batch.step_context.slot_lease
            cg_runner = self._submodules[batch.node_name].cuda_graph_runner
            if lease is not None and cg_runner is not None:
                cg_runner.release(lease, len(batch.request_ids))
        for resource in self._resources.values():
            resource.clear_preplan()

    # ── Eviction ────────────────────────────────────────────────────────
    #
    # Which requests exist, how recently each ran, and when to reclaim are the
    # worker's; the resources only move their own state. A node's resources
    # are reclaimed together, so a request is either resident or not.

    def evictable(self, node_name: str) -> bool:
        return any(
            resource.supports_eviction
            for resource in self._submodules[node_name].resources.values()
        )

    def is_offloaded(self, node_name: str, request_id: str) -> bool:
        return any(
            resource.is_offloaded(request_id)
            for resource in self._submodules[node_name].resources.values()
        )

    def offload_request(self, node_name: str, request_id: str) -> int:
        """Move the request off-device across this node's resources.

        Returns what was reclaimed, in whatever each resource counts (pages,
        today); 0 means nothing moved and the caller should pick another
        victim.
        """
        return sum(
            resource.offload(request_id)
            for resource in self._submodules[node_name].resources.values()
            if resource.supports_eviction
        )

    def reload_request(self, node_name: str, request_id: str) -> bool:
        """Bring it back. False when any resource can't fit it yet, in which
        case the request stays offloaded and the caller retries later."""
        return all(
            resource.reload(request_id)
            for resource in self._submodules[node_name].resources.values()
            if resource.supports_eviction and resource.is_offloaded(request_id)
        )

    def offload_priority(
        self, node_name: str, request_id: str, resource_label: str,
    ) -> float:
        """How much one named resource wants this request gone.

        Only the PRIORITY eviction policy asks; it names the resource to
        consult, since "most worth reclaiming" means something different per
        resource.
        """
        resource = self._submodules[node_name].resources.get(resource_label)
        return 0.0 if resource is None else resource.get_offload_priority(request_id)

    def add_request(
        self, request_id: str,
        overrides: Mapping[str, ResourceReqConfig] | None = None,
    ) -> None:
        self._runner.ingest_request(request_id, overrides)

    def remove_request(self, request_id: str) -> None:
        self._runner.remove_request(request_id)

    def shutdown(self):
        for resource in self._resources.values():
            resource.cleanup()
