

from typing import Mapping

import torch

from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.distributed.communication import WorkerParallelGroups
from mstar.engine.base import NodeBatch
from mstar.engine.kv_store import TransferEngineInfo
from mstar.engine.resources.base import PublishedInfo, Resource, build_resource
from mstar.engine.resources.runner import StepRunner
from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig
from mstar.engine.resources.step import AdmitOutcome
from mstar.model.submodule_base import NodeSubmodule


class Engine:
    def __init__(
        self, autocast_dtype=torch.bfloat16,
        enable_nvtx: bool = False,
        enable_profile: bool=False,
    ):
        self._device = None
        self._autocast_dtype = autocast_dtype
        self._resources: dict[str, Resource] = {}
        self._node_to_resources: dict[str, list[str]] = {}
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
                self._node_to_resources.setdefault(node, []).append(spec.label)

        self._runner = StepRunner(self._resources)

    # TODO: compile submodules / CG runner  / warmup / all running stuff / check stop

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

    # TODO: CG runner end of reserve replay slot, preplan
    
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