
from abc import ABC, abstractmethod
from typing import Any

import torch

from mstar.engine.resources.spec import NodeResourceSpec, ResourceReqConfig
from mstar.engine.resources.step import ResourceStep


class Resource(ABC):
	@classmethod
	@abstractmethod
	def build(
		cls, spec: NodeResourceSpec,
		device: torch.device,
		**engine_kwargs
	) -> "Resource":
		...

	@abstractmethod
	def ingest_request(self, rid: str, overrides: ResourceReqConfig):
		...

	@abstractmethod
	def remove_request(self, rid: str):
		...

	def admit_retrieve(self, rid: str, published: Any) -> bool:
		"""
		Takes the output of publish, possibly from another device, and kicks
		of a retrieval if needed (e.g., PD disaggregation KV transfer).
		Returns whether the retrieve has completed.
		"""
		return True

	@abstractmethod
	def admit(self, step: ResourceStep):
		...

	@abstractmethod
	def plan(self, key: str, step: ResourceStep) -> None:
		...

	@abstractmethod
	def commit(self, step: ResourceStep) -> None:
		...

	@abstractmethod
	def publish(self, request_id: str) -> object | None:
		...