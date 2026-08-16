

from dataclasses import dataclass
import logging
from typing import Mapping

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import JointGroups, WorkerParallelGroups
from mstar.engine.base import NodeBatch
from mstar.engine.kv_store import TransferEngineInfo
from mstar.engine.resources.base import PublishedInfo, Resource, build_resource
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig
from mstar.engine.resources.step import AdmitOutcome
from mstar.engine.v1.cuda_graph_runner import CudaGraphRunner
from mstar.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)


@dataclass
class SubmoduleManagement:
    submodule: NodeSubmodule
    joint_comm_group: JointGroups
    resources: dict[str, Resource]
    cuda_graph_runner: CudaGraphRunner | None = None

    # TODO: PW cuda graph runner
    # # label -> PiecewiseCudaGraphRunner for inner-loop capture; spread into
    # # ModelInputsFromEngine so the submodule's forward can look them up.
    # piecewise_runners: dict[str, "PiecewiseCudaGraphRunner"] = field(default_factory=dict)


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
        self.device = device
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

            # TODO: piecewise cuda graph capture

        # torch.compile applied after CUDA graph capture because the cuda
        # graph runner compiles internally
        self._compile_submodules()
        for resource in self._resources.values():
            resource.post_warmup_validate()


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

    # TODO: CG runner part of reserve replay slot, preplan
    
    def reset_pre_plan_for_batch(self) -> None:
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