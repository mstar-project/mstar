

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import JointGroups, WorkerParallelGroups
from mstar.engine.kv_store import TransferEngineInfo
from mstar.engine.resources.base import PublishedInfo, Resource, build_resource
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
)
from mstar.engine.v1.sampler import SamplerResource
from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs, NodeSubmodule

logger = logging.getLogger(__name__)


@dataclass
class SubmoduleManagement:
    submodule: NodeSubmodule
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

    # Declared once for the batch (pre-plan declares it first when it runs)
    # and driven from here on
    step: SubmoduleStep | None = None

    # Populated on batch preparation
    inputs: list[NodeInputs] | None = None

    allocation_failed: AllocationFailed | None = None
    admit_error: AdmitFailedReason | None = None

    def register_prepare_batch(self, inputs: list[NodeInputs]):
        self.inputs = inputs

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

        self._runner = StepRunner(self._resources)

        for node_name, submodule in submodules.items():
            self._submodules[node_name] = SubmoduleManagement(
                submodule=submodule,
                joint_comm_group=parallel_groups.get_joint_group_for_node(node_name),
                resources={
                    label: self._resources[label] for label in node_to_resources[node_name]
                }
            )

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
                submodule.forward = torch.compile(
                    submodule.forward,
                    fullgraph=False,
                    dynamic=None,
                )
                submodule.forward_batched = torch.compile(
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

    def warmup(self) -> None:
        for node_name, submodule_mgmt in self._submodules.items():
            submodule = submodule_mgmt.submodule
            runner = CudaGraphRunner(
                submodule_name=node_name,
                submodule=submodule,
                resources=submodule_mgmt.resources,
                step_runner=self._runner,
                device=self._device,
                autocast_dtype=self._autocast_dtype,
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
        configs = submodule_mgmt.submodule.get_piecewise_cuda_graph_configs(
            self._device,
            self._autocast_dtype,
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
                autocast_dtype=self._autocast_dtype,
                joint_comm_group=submodule_mgmt.joint_comm_group,
            )
            runner.warmup_and_capture()
            if runner.any_graphs:
                runners[label] = runner
        return runners

    def exec(
        self,
        batch: ExecutingBatch
    ) -> dict[str, NameToTensorList]:
        """Run one step: declare → admit → plan → forward → commit.

        Captured replay and the eager forward are the same path. Under a lease
        the batch is padded to the slot's shape, preprocess output is staged
        into the static buffers, and the launch is a replay; everything above
        the launch is identical.
        """
        submodule_mgmt = self._submodules[batch.node_name]
        cg_runner = submodule_mgmt.cuda_graph_runner
        submodule = submodule_mgmt.submodule
        # a caller that pre-planned already reserved one; otherwise take it here
        lease = batch.step_context.slot_lease or self.reserve_replay_slot(batch)
        real_bs = len(batch.request_ids)

        inputs = batch.inputs
        req_info = batch.per_request_info
        if lease is not None:
            inputs = cg_runner.pad_inputs(lease, inputs)
            req_info = cg_runner.step_metadata(
                lease, batch.request_ids, batch.per_request_info
            )
            batch.step_context.request_ids = cg_runner.step_ids(lease, batch.request_ids)
        rids = batch.step_context.request_ids

        try:
            step = batch.step
            if step is None:
                step = batch.step = submodule.declare_step(
                    graph_walk=batch.step_context.graph_walk,
                    request_ids=rids,
                    inputs=inputs
                )

            if step is not None:
                step.set_ctx(batch.step_context)
                # TODO: skip the resources a pre-plan already admitted and
                # planned. They no-op (and promote) internally today, so this
                # re-drives the whole sweep to get the rest.
                admit_outcome = self._runner.admit(step)
                if not admit_outcome.ok:
                    batch.register_admit_error(admit_outcome.reason)
                    return {rid: {} for rid in batch.request_ids}
                self._runner.plan(step)

            engine_inputs = ModelInputsFromEngine(
                request_ids=rids,
                per_request_info=req_info,
                resources=submodule_mgmt.resources,
                piecewise_runners=submodule_mgmt.piecewise_runners,
                # TODO: per-request state
            )
            preprocessed = submodule.preprocess(
                batch.step_context.graph_walk,
                engine_inputs=engine_inputs,
                inputs=inputs
            )

            if lease is not None:
                outputs = cg_runner.run_forward(
                    lease, preprocessed, plan_done_event=batch.preplan_event,
                )
            elif batch.running_batched:
                outputs = submodule.forward_batched(
                    batch.step_context.graph_walk,
                    engine_inputs=engine_inputs,
                    **preprocessed
                )
            else:
                assert real_bs == 1, (
                    "the unbatched forward takes one request; batch of "
                    f"{real_bs} needs running_batched"
                )
                outputs = {batch.request_ids[0]: submodule.forward(
                    batch.step_context.graph_walk,
                    engine_inputs=engine_inputs,
                    **preprocessed
                )}

            if step is not None:
                self._runner.commit(step)
            return self._collect_outputs(
                batch, submodule_mgmt, lease, outputs, inputs, req_info,
            )
        finally:
            batch.preplan_event = None
            if lease is not None:
                cg_runner.release(lease, real_bs)

    def _collect_outputs(
        self,
        batch: ExecutingBatch,
        submodule_mgmt: SubmoduleManagement,
        lease: SlotLease | None,
        raw_outputs: dict,
        inputs: list[NodeInputs],
        req_info: Mapping[str, CurrentForwardPassInfo],
    ) -> dict[str, NameToTensorList]:
        """Per-rid outputs for the real requests: sample logits, drop the
        padding rows, and map a captured graph's keys back to real ids.

        A captured forward emits its per-rid entries under the slot's padding
        ids (those were the batch at capture time), so entry ``i`` belongs to
        ``request_ids[i]`` on either path.
        """
        submodule = submodule_mgmt.submodule
        request_ids = batch.request_ids
        out_ids = (
            batch.step_context.request_ids if lease is None
            else submodule_mgmt.cuda_graph_runner.slot_for(lease).dummy_rids
        )
        sampler = submodule_mgmt.sampler
        outputs: dict[str, NameToTensorList] = {}

        # Fast path: the submodule stacked the batch's logits under a sentinel,
        # so the whole batch samples in one call instead of per-rid.
        batched_logits = raw_outputs.get("__batched_logits__")
        if batched_logits is not None and sampler is not None:
            # FlashInfer reuses its sampling output buffer across calls, so a
            # view held past the next step aliases that step's token. The clone
            # snapshots this step's value.
            sampled = sampler.sample(
                request_ids, batched_logits[:len(request_ids)]
            ).clone()
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
        logits the ``__batched_logits__`` fast path did not already cover."""
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

    # TODO: all running stuff / check stop

    def check_ready(
        self, node_name: str, request_id: str,
        request_info: CurrentForwardPassInfo,
        published_info: Mapping[str, PublishedInfo]
    ) -> bool:
        out = self._runner.admit_retrieve(
            rid=request_id, node_name=node_name,
            graph_walk=request_info.graph_walk,
            published=published_info
        )
        return out.ok and out.ready

    def reserve_replay_slot(self, batch: ExecutingBatch) -> SlotLease | None:
        """Lease the slot this batch will replay on, before it is dispatched.

        Reserving up front is what puts pre-plan(N+1) and replay(N) on
        different slots. No captured graph for the batch's shape leaves the
        lease unset, i.e. the step runs eager.
        """
        cg_runner = self._submodules[batch.node_name].cuda_graph_runner
        if cg_runner is None or batch.inputs is None:
            return None
        lease = cg_runner.lease_slot(
            graph_walk=batch.step_context.graph_walk,
            bs=len(batch.request_ids),
            num_tokens=sum(inp.input_seq_len for inp in batch.inputs),
            cg_key_info=batch.cg_key_info,
        )
        if lease is not None:
            batch.lease_slot(lease)
        return lease

    def pre_plan_for_batch(self, batch: ExecutingBatch) -> bool:
        """Admit and plan the pre-planning resources a step ahead.

        Worth doing only under a lease: the reserved slot is not the one the
        in-flight replay reads, so this can write its plan buffers on the plan
        stream while that replay runs. ``exec`` then re-drives the full sweep,
        where each pre-planned resource promotes what was staged here, and the
        replay waits on the event recorded here.

        Returns False when nothing was planned ahead, in which case ``exec``
        plans inline.
        """
        submodule_mgmt = self._submodules[batch.node_name]
        cg_runner = submodule_mgmt.cuda_graph_runner
        lease = batch.step_context.slot_lease
        if cg_runner is None or lease is None:
            return False

        inputs = cg_runner.pad_inputs(lease, batch.inputs)
        batch.step_context.request_ids = cg_runner.step_ids(lease, batch.request_ids)
        step = batch.step = submodule_mgmt.submodule.declare_step(
            graph_walk=batch.step_context.graph_walk,
            request_ids=batch.step_context.request_ids,
            inputs=inputs,
        )
        if step is None:
            return False

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
        finally:
            batch.step_context.is_preplan = False
        return True

    def reset_pre_plan_for_batch(self, batch: ExecutingBatch | None = None) -> None:
        """Drop planned-ahead state, e.g. when its batch never dispatched.

        The resources hold the plan itself. The batch keeps whatever
        ``prepare_inputs`` produced — only the plan is redone — but the leased
        slot's padding rows are returned so their pages aren't left attributed
        to a step that never ran.
        """
        if batch is not None:
            batch.preplan_event = None
            lease = batch.step_context.slot_lease
            cg_runner = self._submodules[batch.node_name].cuda_graph_runner
            if lease is not None and cg_runner is not None:
                cg_runner.release(lease, len(batch.request_ids))
        for resource in self._resources.values():
            resource.clear_preplan()

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
