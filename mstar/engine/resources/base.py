
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

	@abstractmethod
	def ingest_request(self, rid: str, overrides: ResourceReqConfig):
		...

	@abstractmethod
	def remove_request(self, rid: str):
		...

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

	@abstractmethod
	def admit(self, step: ResourceStep, ctx: StepContext) -> AdmitOutcome:
		...

	@abstractmethod
	def plan(self, step: ResourceStep, ctx: StepContext) -> None:
		...

	@abstractmethod
	def commit(self, step: ResourceStep, ctx: StepContext) -> None:
		...

	def publish(self, request_id: str) -> "PublishedInfo" | None:
		return

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