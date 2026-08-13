from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from mstar.distributed.communication import CommGroup
from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig
from mstar.engine.resources.step import AdmitOutcome, BucketKey, ResourceStep, StepContext


@dataclass
class CGSlotSpec:
    bucket: BucketKey
    slot: int


class Resource(ABC):
    @classmethod
    @abstractmethod
    def build(
        cls, spec: NodeResourceSpec,
        device: torch.device,
        comm_group: CommGroup | None,
        **engine_kwargs
    ) -> "Resource":
        ...

    def depends_on(self) -> set[str]:
        return set()


    def ingest_request(self, rid: str, overrides: ResourceReqConfig | None):
        return

    def remove_request(self, rid: str):
        return

    def admit_retrieve(
        self, rid: str,
        node_name: str,
        graph_walk: str,
        published: "PublishedInfo | None"
    ) -> AdmitOutcome:
        """
        Takes the output of publish, possibly from another device, and kicks
        of a retrieval if needed (e.g., PD disaggregation KV transfer).
        Returns whether the retrieve has completed.
        """
        return AdmitOutcome(ok=True, ready=True)

    def admit(self, step: ResourceStep, ctx: StepContext) -> AdmitOutcome:
        return AdmitOutcome(ok=True)

    def plan(self, step: ResourceStep, ctx: StepContext) -> Any:
        """ret is immutable and opaque to runner; only gives to `ctx.plan_results`"""
        return None

    def commit(self, step: ResourceStep, ctx: StepContext) -> None:
        """record cstep consumption"""
        return

    def publish(self, request_id: str) -> "PublishedInfo | None":
        return None

    def build_cuda_graph_buffers(
        self, slots: list[CGSlotSpec],
        max_bs: int, max_seq_len: int
    ) -> None:
        # NOTE @nsagan: this should probably be refined; it was just the first
        # thing that came to mind
        return


class PublishedInfo(ABC):
    @abstractmethod
    def update(self, other: "PublishedInfo") -> None:
        ...
