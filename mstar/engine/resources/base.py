
from abc import ABC, abstractmethod
from typing import Any

import torch

from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig
from mstar.engine.resources.step import AdmitOutcome, ResourceStep, StepContext


class Resource(ABC):
	@classmethod
	@abstractmethod
	def build(
		cls, spec: NodeResourceSpec,
		device: torch.device,
		parallel_rank: int,
		world_size: int,
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
	def plan(self, key: str, step: ResourceStep, ctx: StepContext) -> None:
		...

	@abstractmethod
	def commit(self, step: ResourceStep) -> None:
		...

	def publish(self, request_id: str) -> "PublishedInfo" | None:
		return


class PublishedInfo(ABC):
	@abstractmethod
	def update(self, other: "PublishedInfo") -> None:
		...